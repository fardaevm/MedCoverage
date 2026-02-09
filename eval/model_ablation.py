"""
Model ablation: which embedding model produces best match quality?

Tests multiple embedding models against the eval harness using the
current enriched search_text (title + category + description + keywords)
with admin code filtering.

Usage:
    python -m eval.model_ablation
    python -m eval.model_ablation --save eval/results/model_ablation.json
"""

import argparse
import json
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from eval.match_quality_eval import TEST_QUERIES, normalize_code, ADMIN_CODE_PREFIXES

MODELS = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-mpnet-base-v2",
    "NeuML/pubmedbert-base-embeddings",
    "pritamdeka/S-PubMedBert-MS-MARCO",
]


def build_index(texts, model):
    embeddings = model.encode(
        texts, batch_size=256, show_progress_bar=True, normalize_embeddings=True,
    )
    xb = np.asarray(embeddings, dtype="float32")
    index = faiss.IndexFlatIP(xb.shape[1])
    index.add(xb)
    return index


def evaluate_model(index, meta, model, top_k=10):
    results = []
    for test in TEST_QUERIES:
        query = test["query"]
        expected = set(normalize_code(c) for c in test["expected"])

        fetch_k = top_k * 3
        q = model.encode([query], normalize_embeddings=True).astype("float32")
        scores, ids = index.search(q, fetch_k)

        returned_codes = []
        top_score = None
        for fid, score in zip(ids[0].tolist(), scores[0].tolist()):
            if len(returned_codes) >= top_k:
                break
            code = normalize_code(meta.iloc[int(fid)]["code"])
            if code.startswith(ADMIN_CODE_PREFIXES):
                continue
            if top_score is None:
                top_score = float(score)
            returned_codes.append(code)

        hit5 = any(c in expected for c in returned_codes[:5])
        hit10 = any(c in expected for c in returned_codes[:top_k])

        rr = 0.0
        for rank, code in enumerate(returned_codes, 1):
            if code in expected:
                rr = 1.0 / rank
                break

        found = set(returned_codes[:top_k]) & expected
        recall = len(found) / len(expected)

        results.append({
            "query": query,
            "verbosity": test["verbosity"],
            "hit@5": hit5,
            "hit@10": hit10,
            "mrr": rr,
            "recall@k": recall,
            "top_score": top_score,
        })

    n = len(results)
    return {
        "hit@5": sum(r["hit@5"] for r in results) / n,
        "hit@10": sum(r["hit@10"] for r in results) / n,
        "mrr": sum(r["mrr"] for r in results) / n,
        "recall@10": sum(r["recall@k"] for r in results) / n,
        "avg_top_score": sum(r["top_score"] for r in results) / n,
        "per_verbosity": per_verbosity(results),
        "details": results,
    }


def per_verbosity(results):
    out = {}
    for v in ["direct", "short", "conversational", "verbose"]:
        vr = [r for r in results if r["verbosity"] == v]
        if not vr:
            continue
        n = len(vr)
        out[v] = {
            "hit@5": sum(r["hit@5"] for r in vr) / n,
            "hit@10": sum(r["hit@10"] for r in vr) / n,
            "mrr": sum(r["mrr"] for r in vr) / n,
            "recall@10": sum(r["recall@k"] for r in vr) / n,
        }
    return out


def main():
    parser = argparse.ArgumentParser(description="Embedding model ablation study")
    parser.add_argument("--save", type=str, default=None)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    df = pd.read_parquet(project_root / "data" / "processed" / "cpt_hcpcs_clean.parquet")

    for col in ["title", "category", "description", "keywords"]:
        df[col] = df[col].fillna("").astype(str)

    texts = df["search_text"].fillna("").astype(str).tolist()
    all_results = {}

    for model_name in MODELS:
        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        print(f"{'='*60}")

        model = SentenceTransformer(model_name)
        print(f"  Embedding dim: {model.get_sentence_embedding_dimension()}")
        print(f"  Building index ({len(texts):,} rows) ...")
        index = build_index(texts, model)
        print(f"  Evaluating ...")
        summary = evaluate_model(index, df, model)

        print(f"  Hit@5:      {summary['hit@5']:.1%}")
        print(f"  Hit@10:     {summary['hit@10']:.1%}")
        print(f"  MRR:        {summary['mrr']:.3f}")
        print(f"  Recall@10:  {summary['recall@10']:.1%}")
        print(f"  Avg top score: {summary['avg_top_score']:.3f}")

        all_results[model_name] = summary

    # Comparison table
    print(f"\n\n{'='*90}")
    print("COMPARISON")
    print(f"{'='*90}")
    print(f"{'Model':<45} {'Hit@5':>7} {'Hit@10':>7} {'MRR':>7} {'Recall':>7} {'AvgScore':>9}")
    print("-" * 90)
    for name, s in all_results.items():
        short = name.split("/")[-1]
        print(f"{short:<45} {s['hit@5']:>6.0%} {s['hit@10']:>6.0%} {s['mrr']:>7.3f} {s['recall@10']:>6.0%} {s['avg_top_score']:>9.3f}")

    # Per-verbosity MRR
    print(f"\n{'='*90}")
    print("PER-VERBOSITY MRR")
    print(f"{'='*90}")
    print(f"{'Model':<45} {'direct':>8} {'short':>8} {'conv':>8} {'verbose':>8}")
    print("-" * 90)
    for name, s in all_results.items():
        short = name.split("/")[-1]
        pv = s["per_verbosity"]
        print(
            f"{short:<45}"
            f" {pv.get('direct',{}).get('mrr',0):>8.3f}"
            f" {pv.get('short',{}).get('mrr',0):>8.3f}"
            f" {pv.get('conversational',{}).get('mrr',0):>8.3f}"
            f" {pv.get('verbose',{}).get('mrr',0):>8.3f}"
        )

    if args.save:
        save_path = Path(args.save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_data = {
            name: {k: v for k, v in s.items() if k != "details"}
            for name, s in all_results.items()
        }
        save_path.write_text(json.dumps(save_data, indent=2))
        print(f"\nResults saved to {save_path}")


if __name__ == "__main__":
    main()
