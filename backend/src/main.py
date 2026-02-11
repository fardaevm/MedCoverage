from dotenv import load_dotenv
load_dotenv()

from pathlib import Path
from typing import List, Optional
import json

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.matcher import Matcher
from src.pricing import PricingLookup

from src.rag_pipeline import RAGPipeline
from src.rag import Datastore, Indexer, Retriever, ResponseGenerator
from src.eligibility_confidence import rule_confidence


app = FastAPI(title="Medical Cost Estimator API")

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# -------------------------
# Frontend
# -------------------------
FRONTEND_DIR = PROJECT_ROOT / "backend" / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/home", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")

# -------------------------
# Step 1 init
# -------------------------
pricing_lookup = PricingLookup(PROJECT_ROOT)
matcher = Matcher(PROJECT_ROOT, pricing_lookup)

# -------------------------
# Step 2 init (RAG pipeline)
# -------------------------
DEFAULT_SOURCE_PATH = PROJECT_ROOT / "data" / "sample_data" / "source"

def create_pipeline() -> RAGPipeline:
    datastore = Datastore()                      # LanceDB store for policy chunks
    indexer = Indexer()                          # Docling PDF -> chunks -> DataItems
    retriever = Retriever(datastore=datastore)   # retrieve + optional rerank
    response_generator = ResponseGenerator()     # OpenAI: generate YES/NO questions
    return RAGPipeline(datastore, indexer, retriever, response_generator)

rag_pipeline = create_pipeline()

# -------------------------
# API Models
# -------------------------
class MatchRequest(BaseModel):
    text: str
    top_k: int = 10


class SelectedProcedure(BaseModel):
    faiss_id: int
    code: str
    title: str
    category: Optional[str] = None
    description: Optional[str] = None
    basic_rate: Optional[float] = None
    rank: Optional[float] = None
    rerank_score: Optional[float] = None
   


class EligibilityRequest(BaseModel):
    user_text: Optional[str] = None
    selected: SelectedProcedure
    top_k: int = 10


class EligibilityDecisionRequest(BaseModel):
    user_text: Optional[str] = None
    selected: SelectedProcedure
    questions: List[str] = Field(default_factory=list)
    answers: List[bool] = Field(default_factory=list)
    none_apply: bool = False
    top_k: int = 10


class IndexRequest(BaseModel):
    paths: Optional[List[str]] = Field(
        default=None,
        description="Optional list of PDF file paths to index. If omitted, indexes DEFAULT_SOURCE_PATH."
    )

# -------------------------
# Routes
# -------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/match")
def match(req: MatchRequest):
    return {"input": req.text, "candidates": matcher.search(req.text, req.top_k)}


@app.post("/eligibility/questions")
def eligibility_questions(req: EligibilityRequest):
    selected = req.selected
    query_used = _build_eligibility_query(selected, req.user_text)
    questions_text = rag_pipeline.process_query(query_used)
    return {"selected": selected.model_dump(), "query_used": query_used, "questions_text": questions_text}


@app.post("/eligibility/decision")
def eligibility_decision(req: EligibilityDecisionRequest):
    selected = req.selected
    query_used = f"""
        Procedure title: {selected.title}
        Procedure code: {selected.code}
        Description: {selected.description}
        """
    context = rag_pipeline.retrieve_context(query_used, top_k=req.top_k)
    rg = rag_pipeline.response_generator

    raw = rg.generate_decision(
        query=query_used,
        context=context,
        questions=req.questions,
        answers=req.answers,
        none_apply=req.none_apply,
    )

    decision = "uncertain"
    reason = "Could not determine coverage from the available policy text."
    missing_info = []

    try:
        data = json.loads(raw)
        d = data.get("decision")
        if d in {"covered", "not_covered", "uncertain"}:
            decision = d
        reason = str(data.get("reason", reason)).strip()
        missing_info = data.get("missing_info", []) or []
    except Exception:
        pass

    confidence = rule_confidence(
    questions=req.questions,
    answers=req.answers,
    none_apply=req.none_apply,
    llm_decision=decision,
    )

    return {
        "selected": selected.model_dump(),
        "query_used": query_used,
        "decision": decision,
        "confidence": confidence,
        "reason": reason,
        "missing_info": missing_info,
        "raw": raw,
    }

@app.post("/rag/reset")
def rag_reset():
    rag_pipeline.reset()
    return {"status": "ok", "message": "RAG datastore reset."}



@app.post("/rag/index")
def rag_index():
    if not DEFAULT_SOURCE_PATH.exists():
        raise HTTPException(
            status_code=400,
            detail=f"Default source path not found: {DEFAULT_SOURCE_PATH}"
        )

    paths: List[str] = [
        str(p)
        for p in DEFAULT_SOURCE_PATH.iterdir()
        if p.is_file() and p.suffix.lower() == ".pdf" and not p.name.startswith(".")
    ]

    if not paths:
        raise HTTPException(
            status_code=400,
            detail=f"No PDF documents found in: {DEFAULT_SOURCE_PATH}"
        )

    rag_pipeline.add_documents(paths)
    return {"status": "ok", "indexed": len(paths)}


def _build_eligibility_query(selected: SelectedProcedure, user_text: str | None) -> str:
    parts = [
        "You are generating eligibility questions for a specific medical procedure.",
        f"Procedure title: {selected.title}",
        f"Procedure code: {selected.code}",
    ]
    if selected.description:
        parts.append(f"Procedure description: {selected.description}")
    if user_text:
        parts.append(f"Patient request (free text): {user_text}")
    parts += [
        "Use ONLY the policy context to form questions.",
        "Ask about clinical facts that affect coverage (symptoms, timing, frequency, age/sex if relevant, prior procedure dates, screening vs diagnostic, etc.).",
        "Do NOT ask admin workflow questions (prior auth, referrals, doctor requests).",
        "If the patient request implies a different category than the code (e.g., symptoms => diagnostic vs screening), ask 1-2 clarifying questions.",
    ]
    return "\n".join(parts)
    