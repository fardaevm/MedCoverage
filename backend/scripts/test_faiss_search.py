from pathlib import Path
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 10

def main():
    project_root = Path(__file__).resolve().parents[2]
    idx_path = project_root / "data" / "embeddings" / "cpt_faiss.index"
    meta_path = project_root / "data" / "embeddings" / "cpt_meta.parquet"

    index = faiss.read_index(str(idx_path))
    meta = pd.read_parquet(meta_path)

    model = SentenceTransformer(MODEL_NAME)

    query = "MRI brain with contrast"
    q = model.encode([query], normalize_embeddings=True).astype("float32")

    scores, ids = index.search(q, TOP_K)

    print("QUERY:", query)
    for rank, (i, s) in enumerate(zip(ids[0], scores[0]), start=1):
        row = meta.iloc[int(i)]
        print(f"{rank:02d}  score={s:.3f}  code={row.get('code')}  title={row.get('title')}")

if __name__ == "__main__":
    main()
