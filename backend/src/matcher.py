from pathlib import Path
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from src.pricing import PricingLookup
import os
import cohere

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

class Matcher:
    def __init__(self, project_root: Path, pricing_lookup: PricingLookup):
        idx_path = project_root / "data" / "embeddings" / "cpt_faiss.index"
        meta_path = project_root / "data" / "embeddings" / "cpt_meta.parquet"

        self.index = faiss.read_index(str(idx_path))
        self.meta = pd.read_parquet(meta_path)
        self.model = SentenceTransformer(MODEL_NAME)
        self.pricing = pricing_lookup


        self.co = None
        api_key = os.getenv("COHERE_API_KEY")
        if api_key:
            self.co = cohere.ClientV2(api_key=api_key)



    def _rerank(self, query: str, candidates: list[dict], top_k: int) -> list[dict]:
        if not self.co or not candidates:
            for i, c in enumerate(candidates[:top_k]):
                c["rank"] = i + 1
            return candidates[:top_k]

        docs = [
            f"{c.get('title','')}\n{c.get('description','')}\nCode:{c.get('code','')}".strip()
            for c in candidates
        ]

        resp = self.co.rerank(
            model="rerank-v3.5",
            query=query,
            documents=docs,
            top_n=min(top_k, len(docs)),
        )

        out = []
        for r in resp.results:
            c = candidates[r.index]
            c["rank"] = len(out) + 1
            c["rerank_score"] = float(getattr(r, "relevance_score", 0.0))
            out.append(c)

        return out


    def search(self, text: str, top_k: int = 10):
        pre_k = min(max(top_k * 10, 50), 200)

        q = self.model.encode([text], normalize_embeddings=True).astype("float32")
        scores, ids = self.index.search(q, pre_k)

        candidates = []
        for faiss_id, _score in zip(ids[0].tolist(), scores[0].tolist()):
            if faiss_id < 0:
                continue
            row = self.meta.iloc[int(faiss_id)]
            code = row.get("code")
            pricing = self.pricing.lookup(code) if code else None

            raw_desc = row.get("description")
            desc = None if raw_desc is None else str(raw_desc).strip().strip('"')
            
            raw_title = row.get("title")
            title = None if raw_title is None else str(raw_title).strip().strip('"').rstrip(".")

            candidates.append({
                "faiss_id": int(faiss_id),
                "code": code,
                "title": title,
                "category": row.get("category"),
                "description": desc,  
                "basic_rate": pricing["basic_rate"] if pricing else None,
            })


        return self._rerank(text, candidates, top_k)
