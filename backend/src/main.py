from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pathlib import Path
from .matcher import Matcher
from .pricing import PricingLookup


app = FastAPI(title="Medical Cost Estimator API")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
pricing_lookup = PricingLookup(PROJECT_ROOT)
matcher = Matcher(PROJECT_ROOT, pricing_lookup)


FRONTEND_DIR = PROJECT_ROOT / "backend" / "frontend"
app.mount("/home", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")

@app.get("/health")
def health():
    return {"status": "ok"}

class MatchRequest(BaseModel):
    text: str
    top_k: int = 10

@app.post("/match")
def match(req: MatchRequest):
    return {"input": req.text, "candidates": matcher.search(req.text, req.top_k)}
