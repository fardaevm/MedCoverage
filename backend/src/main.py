# backend/src/main.py
from dotenv import load_dotenv
load_dotenv()

from pathlib import Path
from typing import List, Optional, Dict, Any
import json
import os

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.matcher import Matcher
from src.pricing import PricingLookup
from src.rag_pipeline import RAGPipeline
from src.rag import Datastore, Indexer, Retriever, ResponseGenerator
from src.cache import RedisJSONCache
from src.eligibility_evaluator import evaluate_answers

app = FastAPI(title="Medical Cost Estimator API")
PROJECT_ROOT = Path(__file__).resolve().parents[2]

if os.getenv("DISABLE_STATIC_CACHE", "").strip() in {"1", "true", "yes"}:
    @app.middleware("http")
    async def _no_cache_static(request: Request, call_next):
        resp = await call_next(request)
        if request.url.path.startswith("/home/"):
            resp.headers["Cache-Control"] = "no-store, max-age=0"
            resp.headers["Pragma"] = "no-cache"
            resp.headers["Expires"] = "0"
        return resp

FRONTEND_DIR = PROJECT_ROOT / "backend" / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/home", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")

pricing_lookup = PricingLookup(PROJECT_ROOT)
matcher = Matcher(PROJECT_ROOT, pricing_lookup)
DEFAULT_SOURCE_PATH = PROJECT_ROOT / "data" / "sample_data" / "source"

def create_pipeline() -> RAGPipeline:
    datastore = Datastore()
    indexer = Indexer()
    retriever = Retriever(datastore=datastore)
    response_generator = ResponseGenerator()
    return RAGPipeline(datastore, indexer, retriever, response_generator)

rag_pipeline = create_pipeline()
questions_cache = RedisJSONCache()

# ── Models ─────────────────────────────────────────────────────────

class MatchRequest(BaseModel):
    text: str
    top_k: int = 10

class SelectedProcedure(BaseModel):
    faiss_id: int
    code: str
    title: str
    category: Optional[str] = None
    description: Optional[str] = None
    rank: Optional[float] = None
    rerank_score: Optional[float] = None

class EligibilityStartRequest(BaseModel):
    user_text: Optional[str] = None
    selected: SelectedProcedure
    top_k: int = 10

class QAItem(BaseModel):
    q: str
    a: bool

class EligibilityNextRequest(BaseModel):
    user_text: Optional[str] = None
    selected: SelectedProcedure
    pathways: List[str] = Field(default_factory=list)
    questions_per_pathway: List[List[str]] = Field(default_factory=list)
    logic_tree: Optional[Dict[str, Any]] = None
    logic_trees: Optional[List[Dict[str, Any]]] = None
    question_map: Optional[Dict[str, str]] = None
    qa_so_far: List[QAItem] = Field(default_factory=list)
    top_k: int = 10

# ── Routes ─────────────────────────────────────────────────────────

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

# ── Eligibility: START ─────────────────────────────────────────────

@app.post("/eligibility/start")
def eligibility_start(req: EligibilityStartRequest):
    selected = req.selected
    query_used = _build_eligibility_query(selected, req.user_text)

    key_payload = json.dumps({"code": selected.code, "query": query_used}, ensure_ascii=False)
    cache_key = questions_cache.make_key("tree:start:v5", key_payload)
    cached = questions_cache.get(cache_key)
    if cached:
        return {"selected": selected.model_dump(), **cached}

    context = rag_pipeline.retrieve_context(query_used, top_k=req.top_k)
    rg = rag_pipeline.response_generator
    parsed = rg.generate_pathways_and_questions(query=query_used, context=context)

    pathways = parsed.get("pathways") or []
    questions_per_pathway = parsed.get("questions_per_pathway") or []
    logic_tree = parsed.get("logic_tree")        # new: single tree
    logic_trees = parsed.get("logic_trees")      # old: per-pathway
    question_map = parsed.get("question_map")

    has_pathways = any(str(p or "").strip() for p in pathways)
    has_tree = logic_tree is not None or (logic_trees and len(logic_trees) > 0)
    has_questions = any(len(qs) > 0 for qs in questions_per_pathway)
    no_pathways = not has_pathways or (not has_tree and not has_questions)
    no_pathways_reason = "We couldn't find coverage pathways for this procedure in our policy."

    # Pick first question
    first_question = ""
    if not no_pathways:
        result = evaluate_answers(
            pathways, questions_per_pathway, [],
            logic_tree=logic_tree, logic_trees=logic_trees, question_map=question_map,
        )
        first_question = result.get("next_question") or ""

    payload = {
        "query_used": query_used,
        "title": parsed.get("title") or "Coverage pathways",
        "pathways": pathways,
        "questions_per_pathway": questions_per_pathway,
        "logic_tree": logic_tree,
        "logic_trees": logic_trees,
        "question_map": question_map,
        "question": first_question,
        "no_pathways": no_pathways,
        "no_pathways_reason": no_pathways_reason,
    }
    questions_cache.set(cache_key, payload)
    return {"selected": selected.model_dump(), **payload}

# ── Eligibility: NEXT ──────────────────────────────────────────────

@app.post("/eligibility/next")
def eligibility_next(req: EligibilityNextRequest):
    selected = req.selected
    query_used = _build_eligibility_query(selected, req.user_text)

    pathways = [str(p or "").strip() for p in (req.pathways or [])]
    questions_per_pathway = req.questions_per_pathway or []
    logic_tree = req.logic_tree
    logic_trees = req.logic_trees
    question_map = req.question_map

    if not any(pathways) and not logic_tree:
        return {
            "selected": selected.model_dump(),
            "query_used": query_used,
            "decision": "not_covered",
            "reason": "No coverage pathways found for this procedure.",
            "next_question": None,
            "completeness": 0.0,
        }

    qa_so_far = [{"q": x.q, "a": bool(x.a)} for x in (req.qa_so_far or [])]

    result = evaluate_answers(
        pathways, questions_per_pathway, qa_so_far,
        logic_tree=logic_tree, logic_trees=logic_trees, question_map=question_map,
    )

    # Fallback LLM planner if no next question
    if result["decision"] == "uncertain" and not result.get("next_question"):
        context = rag_pipeline.retrieve_context(query_used, top_k=req.top_k)
        rg = rag_pipeline.response_generator
        next_q = rg.plan_next_question(
            query=query_used, context=context,
            pathways=pathways, qa_so_far=qa_so_far,
        )
        result["next_question"] = next_q

    return {"selected": selected.model_dump(), "query_used": query_used, **result}

# ── RAG admin ──────────────────────────────────────────────────────

@app.post("/rag/reset")
def rag_reset():
    rag_pipeline.reset()
    return {"status": "ok", "message": "RAG datastore reset."}

@app.post("/rag/index")
def rag_index():
    if not DEFAULT_SOURCE_PATH.exists():
        raise HTTPException(status_code=400, detail=f"Default source path not found: {DEFAULT_SOURCE_PATH}")
    paths = [str(p) for p in DEFAULT_SOURCE_PATH.iterdir() if p.is_file() and p.suffix.lower() == ".pdf" and not p.name.startswith(".")]
    if not paths:
        raise HTTPException(status_code=400, detail=f"No PDF documents found in: {DEFAULT_SOURCE_PATH}")
    rag_pipeline.add_documents(paths)
    return {"status": "ok", "indexed": len(paths)}

# ── Helpers ────────────────────────────────────────────────────────

def _build_eligibility_query(selected: SelectedProcedure, user_text: str | None) -> str:
    parts = [f"Procedure title: {selected.title}", f"Procedure code: {selected.code}"]
    if selected.description:
        parts.append(f"Description: {selected.description}")
    if user_text:
        parts.append(f"Patient note: {user_text}")
    return "\n".join(parts)