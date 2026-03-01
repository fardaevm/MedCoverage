# backend/src/main.py
from dotenv import load_dotenv
load_dotenv()

from pathlib import Path
from typing import List, Optional, Dict, Any
import json
import re
import os
from hashlib import sha1

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.matcher import Matcher
from src.pricing import PricingLookup
from src.rag_pipeline import RAGPipeline
from src.rag import Datastore, Indexer, Retriever, ResponseGenerator
from src.cache import RedisJSONCache

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
    qa_so_far: List[QAItem] = Field(default_factory=list)
    top_k: int = 10

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

@app.post("/eligibility/start")
def eligibility_start(req: EligibilityStartRequest):
    selected = req.selected
    query_used = _build_eligibility_query(selected, req.user_text)

    # cache pathways per (code + query_used)
    key_payload = json.dumps({"code": selected.code, "query": query_used}, ensure_ascii=False)
    cache_key = questions_cache.make_key("tree:start:v2", key_payload)
    cached = questions_cache.get(cache_key)
    if cached:
        return {"selected": selected.model_dump(), **cached}

    context = rag_pipeline.retrieve_context(query_used, top_k=req.top_k)
    rg = rag_pipeline.response_generator
    parsed = rg.generate_pathways_and_questions(query=query_used, context=context)

    pathways = (parsed.get("pathways") or [])[:2]
    if len(pathways) < 2:
        while len(pathways) < 2:
            pathways.append("")

    # No meaningful pathways => don't generate questions; stop flow
    has_pathways = any(str(p or "").strip() for p in pathways)
    no_pathways_reason = "We couldn't find coverage pathways for this procedure in our policy."

    questions_per_pathway = parsed.get("questions_per_pathway") or []
    if len(questions_per_pathway) != len(pathways):
        questions_per_pathway = [[] for _ in pathways]

    q1 = ""
    if has_pathways:
        if questions_per_pathway and questions_per_pathway[0]:
            q1 = questions_per_pathway[0][0]
        if not q1:
            q1 = rg.plan_next_question(
                query=query_used,
                context=context,
                pathways=pathways,
                qa_so_far=[],
            )

    payload = {
        "query_used": query_used,
        "title": parsed.get("title") or "Coverage pathways",
        "pathways": pathways,
        "questions_per_pathway": questions_per_pathway,
        "question": q1,
        "no_pathways": not has_pathways,
        "no_pathways_reason": no_pathways_reason,
    }
    questions_cache.set(cache_key, payload)
    return {"selected": selected.model_dump(), **payload}

@app.post("/eligibility/next")
def eligibility_next(req: EligibilityNextRequest):
    selected = req.selected
    query_used = _build_eligibility_query(selected, req.user_text)

    pathways = [str(p or "").strip() for p in (req.pathways or [])][:2]
    while len(pathways) < 2:
        pathways.append("")

    # No meaningful pathways => don't ask or decide from tree; return not_covered
    if not any(pathways):
        return {
            "selected": selected.model_dump(),
            "query_used": query_used,
            "decision": "not_covered",
            "confidence": 0.0,
            "reason": "No coverage pathways found for this procedure.",
            "missing_info": [],
            "raw": None,
        }

    qa_so_far = [{"q": x.q, "a": bool(x.a)} for x in (req.qa_so_far or [])]
    if len(qa_so_far) > 20:
        qa_so_far = qa_so_far[:20]

    context = rag_pipeline.retrieve_context(query_used, top_k=req.top_k)
    rg = rag_pipeline.response_generator

    raw = rg.decide_from_tree(query=query_used, context=context, pathways=pathways, qa_so_far=qa_so_far)

    decision = "uncertain"
    reason = "Need more information to determine coverage from the policy text."
    missing_info: List[str] = []

    try:
        data = json.loads(raw)
        d = str(data.get("decision", "")).strip()
        if d in {"covered", "not_covered", "uncertain"}:
            decision = d
        reason = str(data.get("reason", reason)).strip()
        mi = data.get("missing_info", []) or []
        if isinstance(mi, list):
            missing_info = [str(x).strip() for x in mi if str(x).strip()]
    except Exception:
        pass

    # stop conditions
    if decision in {"covered", "not_covered"}:
        return {
            "selected": selected.model_dump(),
            "query_used": query_used,
            "decision": decision,
            "confidence": _tree_confidence(decision, len(qa_so_far)),
            "reason": reason,
            "missing_info": missing_info,
            "raw": raw,
        }

    # plan next question
    next_q = rg.plan_next_question(
        query=query_used,
        context=context,
        pathways=pathways,
        qa_so_far=qa_so_far,
    )

    return {
        "selected": selected.model_dump(),
        "query_used": query_used,
        "decision": "uncertain",
        "confidence": _tree_confidence("uncertain", len(qa_so_far)),
        "reason": reason,
        "missing_info": missing_info,
        "next_question": next_q,
        "raw": raw,
    }

@app.post("/rag/reset")
def rag_reset():
    rag_pipeline.reset()
    return {"status": "ok", "message": "RAG datastore reset."}

@app.post("/rag/index")
def rag_index():
    if not DEFAULT_SOURCE_PATH.exists():
        raise HTTPException(status_code=400, detail=f"Default source path not found: {DEFAULT_SOURCE_PATH}")

    paths: List[str] = [
        str(p)
        for p in DEFAULT_SOURCE_PATH.iterdir()
        if p.is_file() and p.suffix.lower() == ".pdf" and not p.name.startswith(".")
    ]
    if not paths:
        raise HTTPException(status_code=400, detail=f"No PDF documents found in: {DEFAULT_SOURCE_PATH}")

    rag_pipeline.add_documents(paths)
    return {"status": "ok", "indexed": len(paths)}

def _parse_pathways(raw: str) -> dict:
    lines = [l.strip() for l in (raw or "").split("\n") if l.strip()]
    title = "Coverage pathways"
    for line in lines:
        if line.lower().startswith("title:"):
            title = line.split(":", 1)[1].strip() or title
            break

    pathways: List[str] = []
    in_pathways = False
    for line in lines:
        low = line.lower().rstrip(":")
        if low == "pathways":
            in_pathways = True
            continue
        if in_pathways and line.startswith("- "):
            txt = line[2:].strip()
            if txt:
                pathways.append(txt)

    return {"title": title, "pathways": pathways}

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

def _tree_confidence(decision: str, turns: int) -> float:
    t = max(0, min(int(turns or 0), 12))
    if decision == "covered":
        return min(0.92, 0.78 + 0.02 * t)
    if decision == "not_covered":
        return min(0.90, 0.74 + 0.02 * t)
    return min(0.62, 0.42 + 0.015 * t)