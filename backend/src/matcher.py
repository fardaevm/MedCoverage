from pathlib import Path
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

class Matcher:
    def __init__(self, project_root: Path):
        idx_path = project_root / "data" / "embeddings" / "cpt_faiss.index"
        meta_path = project_root / "data" / "embeddings" / "cpt_meta.parquet"

        self.index = faiss.read_index(str(idx_path))
        self.meta = pd.read_parquet(meta_path)
        self.model = SentenceTransformer(MODEL_NAME)

    def search(self, text: str, top_k: int = 10):
        q = self.model.encode([text], normalize_embeddings=True).astype("float32")
        scores, ids = self.index.search(q, top_k)

        out = []
        for faiss_id, score in zip(ids[0].tolist(), scores[0].tolist()):
            row = self.meta.iloc[int(faiss_id)]
            out.append({
                "faiss_id": int(faiss_id),
                "score": float(score),
                "code": row.get("code"),
                "title": row.get("title"),
                "search_text": row.get("search_text"),
                "category": row.get("category"),
            })
        return out
