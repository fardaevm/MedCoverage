"""
Ablation study: which search_text composition works best?

Tests multiple variants of search_text construction against the
match quality eval harness, building a temporary FAISS index in
memory for each variant.

Usage:
    python -m eval.search_text_ablation
    python -m eval.search_text_ablation --save eval/results/ablation.json
"""

import argparse
import json
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from eval.match_quality_eval import TEST_QUERIES, normalize_code

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

VARIANTS = {
    "A: title+cat+desc (baseline)": lambda r: f"{r['title']} {r['category']} {r['description']}",
    "B: title+cat+desc+kw (current)": lambda r: f"{r['title']} {r['category']} {r['description']} {r['keywords']}",
    "C: title+cat+kw": lambda r: f"{r['title']} {r['category']} {r['keywords']}",
    "D: title+kw": lambda r: f"{r['title']} {r['keywords']}",
    "E: labeled title+cat+kw": lambda r: f"Title: {r['title']}. Category: {r['category']}. Keywords: {r['keywords']}",
    "F: labeled all": lambda r: f"Title: {r['title']}. Category: {r['category']}. Description: {r['description']}. Keywords: {r['keywords']}",
}


def build_index(texts, model):
    embeddings = model.encode(
        texts, batch_size=256, show_progress_bar=False, normalize_embeddings=True,
    )
    xb = np.asarray(embeddings, dtype="float32")
    index = faiss.IndexFlatIP(xb.shape[1])
    index.add(xb)
    return index


def evaluate_variant(index, meta, model, top_k=10):
    results = []
    for test in TEST_QUERIES:
        query = test["query"]
        expected = set(normalize_code(c) for c in test["expected"])

        q = model.encode([query], normalize_embeddings=True).astype("float32")
        scores, ids = index.search(q, top_k)

        returned_codes = []
        for fid in ids[0].tolist():
            code = normalize_code(meta.iloc[int(fid)]["code"])
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
        })

    n = len(results)
    return {
        "hit@5": sum(r["hit@5"] for r in results) / n,
        "hit@10": sum(r["hit@10"] for r in results) / n,
        "mrr": sum(r["mrr"] for r in results) / n,
        "recall@10": sum(r["recall@k"] for r in results) / n,
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
    parser = argparse.ArgumentParser(description="search_text ablation study")
    parser.add_argument("--save", type=str, default=None)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    df = pd.read_parquet(project_root / "data" / "processed" / "cpt_hcpcs_clean.parquet")

    # Fill NaN to avoid string concat issues
    for col in ["title", "category", "description", "keywords"]:
        df[col] = df[col].fillna("").astype(str)

    print(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    all_results = {}

    for name, fn in VARIANTS.items():
        print(f"\n{'='*60}")
        print(f"Variant: {name}")
        print(f"{'='*60}")

        texts = df.apply(fn, axis=1).str.strip().tolist()
        print(f"  Building index ({len(texts):,} rows) ...")
        index = build_index(texts, model)
        print(f"  Evaluating ...")
        summary = evaluate_variant(index, df, model)

        print(f"  Hit@5:  {summary['hit@5']:.1%}")
        print(f"  Hit@10: {summary['hit@10']:.1%}")
        print(f"  MRR:    {summary['mrr']:.3f}")
        print(f"  Recall: {summary['recall@10']:.1%}")

        all_results[name] = summary

    # Comparison table
    print(f"\n\n{'='*80}")
    print("COMPARISON")
    print(f"{'='*80}")
    print(f"{'Variant':<35} {'Hit@5':>7} {'Hit@10':>7} {'MRR':>7} {'Recall':>7}")
    print("-" * 80)
    for name, s in all_results.items():
        print(f"{name:<35} {s['hit@5']:>6.0%} {s['hit@10']:>6.0%} {s['mrr']:>7.3f} {s['recall@10']:>6.0%}")

    # Per-verbosity breakdown for each variant
    print(f"\n\n{'='*80}")
    print("PER-VERBOSITY MRR")
    print(f"{'='*80}")
    print(f"{'Variant':<35} {'direct':>8} {'short':>8} {'conv':>8} {'verbose':>8}")
    print("-" * 80)
    for name, s in all_results.items():
        pv = s["per_verbosity"]
        print(
            f"{name:<35}"
            f" {pv.get('direct',{}).get('mrr',0):>8.3f}"
            f" {pv.get('short',{}).get('mrr',0):>8.3f}"
            f" {pv.get('conversational',{}).get('mrr',0):>8.3f}"
            f" {pv.get('verbose',{}).get('mrr',0):>8.3f}"
        )

    if args.save:
        save_path = Path(args.save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        # Strip details for cleaner JSON
        save_data = {
            name: {k: v for k, v in s.items() if k != "details"}
            for name, s in all_results.items()
        }
        save_path.write_text(json.dumps(save_data, indent=2))
        print(f"\nResults saved to {save_path}")


if __name__ == "__main__":
    main()
