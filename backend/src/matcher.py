from pathlib import Path
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from .pricing import PricingLookup

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

class Matcher:
    def __init__(self, project_root: Path, pricing_lookup: PricingLookup):
        idx_path = project_root / "data" / "embeddings" / "cpt_faiss.index"
        meta_path = project_root / "data" / "embeddings" / "cpt_meta.parquet"

        self.index = faiss.read_index(str(idx_path))
        self.meta = pd.read_parquet(meta_path)
        self.model = SentenceTransformer(MODEL_NAME)
        self.pricing = pricing_lookup

    def search(self, text: str, top_k: int = 10):
        q = self.model.encode([text], normalize_embeddings=True).astype("float32")
        scores, ids = self.index.search(q, top_k)

        out = []
        for faiss_id, score in zip(ids[0].tolist(), scores[0].tolist()):
            row = self.meta.iloc[int(faiss_id)]
            code = row.get("code")
            pricing = self.pricing.lookup(code) if code else None
            candidate = {
            "faiss_id": int(faiss_id),
            "score": float(score),
            "code": code,
            "title": str(row.get("title")).strip('"').rstrip('.'),
            "category": row.get("category"),
            "description": str(row.get("description")).strip('"'),
            "basic_rate": pricing["basic_rate"] if pricing else None,
            }
            print("CANDIDATE:", candidate)
            out.append(candidate)
        return out
