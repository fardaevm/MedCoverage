from dotenv import load_dotenv
load_dotenv()

from pathlib import Path
from typing import List, Optional
import json
import re
import os

from fastapi import FastAPI, HTTPException, Response, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.matcher import Matcher
from src.pricing import PricingLookup

from src.rag_pipeline import RAGPipeline
from src.rag import Datastore, Indexer, Retriever, ResponseGenerator
from src.eligibility_confidence import rule_confidence

from src.cache import RedisJSONCache

app = FastAPI(title="Medical Cost Estimator API")

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Disable browser caching for /home/* in dev
if os.getenv("DISABLE_STATIC_CACHE", "").strip() in {"1", "true", "yes"}:
    @app.middleware("http")
    async def _no_cache_static(request: Request, call_next):
        resp = await call_next(request)
        if request.url.path.startswith("/home/"):
            resp.headers["Cache-Control"] = "no-store, max-age=0"
            resp.headers["Pragma"] = "no-cache"
            resp.headers["Expires"] = "0"
        return resp
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

# cache
questions_cache = RedisJSONCache()
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


class QuestionItem(BaseModel):
    text: str
    label: int = 1  # 0 = standalone (alone decides coverage), 1 = conditional (needs combination)


class EligibilityDecisionRequest(BaseModel):
    user_text: Optional[str] = None
    selected: SelectedProcedure
    questions: List[QuestionItem] = Field(default_factory=list)
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
@app.get("/")
def landing():
    landing_path = FRONTEND_DIR / "landing.html"
    if landing_path.exists():
        return FileResponse(str(landing_path))
    return FileResponse(str(FRONTEND_DIR / "index.html"))


@app.get("/health")
def health():
    return {
        "status": "ok",
        "redis_enabled": questions_cache.enabled(),
        "redis_url": getattr(questions_cache, "url", None),
        "redis_error": getattr(questions_cache, "last_error", None),
    }


@app.post("/match")
def match(req: MatchRequest):
    return {"input": req.text, "candidates": matcher.search(req.text, req.top_k)}


@app.post("/eligibility/questions")
def eligibility_questions(req: EligibilityRequest):
    selected = req.selected

    query_used = _build_eligibility_query(selected, req.user_text)

    cache_key = questions_cache.make_key(
        f"questions: {selected.code}",
        query_used,
    )

    cached = questions_cache.get(cache_key)

    if cached:
        return {
            "selected": selected.model_dump(),
            **cached,
        }

    
    questions_text = rag_pipeline.process_query(query_used)
    parsed = _parse_questions_text(questions_text)

    payload = {
        "query_used": query_used,
        "question_text": questions_text,
        "title": parsed["title"],
        "questions": parsed["questions"],
    }

    questions_cache.set(cache_key, payload)

    return {
        "selected": selected.model_dump(),
        **payload,
    }


@app.post("/eligibility/decision")
def eligibility_decision(req: EligibilityDecisionRequest):
    selected = req.selected
    query_used = f"""
        Procedure title: {selected.title}
        Procedure code: {selected.code}
        Description: {selected.description}
        """

    question_texts = [q.text for q in req.questions]
    question_labels = [q.label for q in req.questions]

    # Pre-check: any standalone (label=0) question answered YES → deterministically covered
    label0_covered = any(
        label == 0 and answer
        for label, answer in zip(question_labels, req.answers)
    )

    context = rag_pipeline.retrieve_context(query_used, top_k=req.top_k)
    rg = rag_pipeline.response_generator

    raw = rg.generate_decision(
        query=query_used,
        context=context,
        questions=question_texts,
        labels=question_labels,
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

    # Override: a standalone (label=0) question answered YES guarantees coverage unconditionally
    if label0_covered:
        decision = "covered"

    confidence = rule_confidence(
        questions=question_texts,
        answers=req.answers,
        none_apply=req.none_apply,
        llm_decision=decision,
    )

    # Boost confidence for deterministic label=0 coverage
    if label0_covered and decision == "covered":
        confidence = max(confidence, 0.90)

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


def _extract_question_text(content: str) -> str:
    """Extract plain-text question from various LLM output formats.

    Handles:
      - Full JSON object:     {"label": 1, "text": "question?"}
      - Quoted text field:    "text": "question?"
      - Unquoted text field:  "text": question?
      - Plain text fallback:  question text
    """
    stripped = content.strip()
    # Try full JSON object
    try:
        obj = json.loads(stripped)
        if isinstance(obj, dict) and "text" in obj:
            return str(obj["text"]).strip()
    except Exception:
        pass
    # Try "text": "..." pattern
    m = re.search(r'"text"\s*:\s*"([^"]+)"', stripped)
    if m:
        return m.group(1).strip()
    # Try "text": value (unquoted)
    m = re.search(r'"text"\s*:\s*(.+?)(?:,\s*"|\s*\}|$)', stripped)
    if m:
        return m.group(1).strip().strip('"')
    return stripped


def _parse_questions_text(raw: str) -> dict:
    """Parse LLM output into {title, questions: [{text, label}]}.

    Handles new Pathways: format (all label=0) and old Questions: JSON format as fallback.
    """
    lines = [l.strip() for l in raw.split("\n") if l.strip()]

    title = "Coverage pathways"
    for line in lines:
        if line.lower().startswith("title:"):
            title = line.split(":", 1)[1].strip()
            break

    # Try new Pathways: format first
    questions: List[dict] = []
    in_pathways = False
    for line in lines:
        low = line.lower().rstrip(":")
        if low == "pathways":
            in_pathways = True
            continue
        if low in {"questions", "title"}:
            in_pathways = False
            continue
        if in_pathways and line.startswith("- "):
            text = line[2:].strip()
            if text:
                questions.append({"text": text, "label": 0})

    if questions:
        return {"title": title, "questions": questions}

    # Fallback: old Questions: JSON format {"label": X, "text": "..."}
    for line in lines:
        if not line.startswith("- "):
            continue
        content = line[2:].strip()
        if content.lower().rstrip(":") in {"questions", "title"}:
            continue
        try:
            obj = json.loads(content)
            if isinstance(obj, dict):
                text = str(obj.get("text", content)).strip()
                label = int(obj.get("label", 1))
                if text:
                    questions.append({"text": text, "label": label})
                continue
        except Exception:
            pass
        m = re.search(r'"label"\s*:\s*(\d+)', content)
        label = int(m.group(1)) if m else 1
        text = _extract_question_text(content)
        if text:
            questions.append({"text": text, "label": label})

    return {"title": title, "questions": questions}


def _build_eligibility_query(selected: SelectedProcedure, user_text: str | None) -> str:
    parts = [
        f"Procedure title: {selected.title}",
        f"Procedure code: {selected.code}",
    ]
    if selected.description:
        parts.append(f"Description: {selected.description}")
    if user_text:
        parts.append(f"Patient note: {user_text}")
    return "\n".join(parts)
    