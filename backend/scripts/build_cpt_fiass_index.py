from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 256
TOP_TEXT_COL = "search_text"


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]

    in_path = project_root / "data" / "processed" / "cpt_hcpcs_clean.parquet"
    out_dir = project_root / "data" / "embeddings"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not in_path.exists():
        raise FileNotFoundError(f"Missing processed CPT file: {in_path}")

    df = pd.read_parquet(in_path)

    if TOP_TEXT_COL not in df.columns:
        raise ValueError(f"Expected column '{TOP_TEXT_COL}' not found. Columns: {df.columns.tolist()}")

    # Ensure text is str and non-null
    texts = df[TOP_TEXT_COL].fillna("").astype(str).tolist()

    # Embeddings (local)
    model = SentenceTransformer(MODEL_NAME)

    # Encode in batches; normalize embeddings for cosine similarity via dot product
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,  
    )

    # Convert to float32 for FAISS
    xb = np.asarray(embeddings, dtype="float32")
    dim = xb.shape[1]

    #FAISS index (cosine similarity)
    # With normalized vectors, inner product == cosine similarity
    index = faiss.IndexFlatIP(dim)
    index.add(xb)

    # Save index and metadata
    index_path = out_dir / "cpt_faiss.index"
    meta_path = out_dir / "cpt_meta.parquet"

    faiss.write_index(index, str(index_path))

    # Keep only what you need at inference time
    meta_cols = [c for c in ["search_text", "description","code", "title", "type", "category"] if c in df.columns]
    meta = df[meta_cols].copy()
    meta.insert(0, "faiss_id", np.arange(len(meta), dtype=np.int64))

    meta.to_parquet(meta_path, index=False)

    print(f"[OK] FAISS index saved: {index_path}")
    print(f"[OK] Metadata saved:   {meta_path}")
    print(f"[INFO] Rows indexed:   {index.ntotal:,}")
    print(f"[INFO] Embedding dim:  {dim}")


if __name__ == "__main__":
    main()
