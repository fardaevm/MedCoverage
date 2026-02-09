"""
CPT Code Match Quality Evaluation
==================================
Evaluates how well the FAISS-based matcher retrieves correct CPT codes
for plain-English user queries. This measures the first stage of the
pipeline (semantic search → CPT match) independent of RAG or pricing.

Metrics:
  - Hit@K    : Is any expected code in the top-K results?
  - MRR      : Mean Reciprocal Rank of the first expected code found.
  - Recall@K : Fraction of expected codes that appear in the top-K.

Usage:
    python -m eval.match_quality_eval
    python -m eval.match_quality_eval --top_k 20
    python -m eval.match_quality_eval --save eval/results/baseline.json
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Test set: mammogram-focused queries with expected CPT codes.
# A "hit" means ANY of the expected codes appears in the results.
# ---------------------------------------------------------------------------
TEST_QUERIES = [
    # --- Direct: short procedure lookups ---
    {
        "query": "mammogram",
        "expected": ["77067", "77066", "77065"],
        "verbosity": "direct",
    },
    {
        "query": "breast MRI",
        "expected": ["77046", "77047"],
        "verbosity": "direct",
    },
    {
        "query": "breast ultrasound",
        "expected": ["76641", "76642"],
        "verbosity": "direct",
    },
    {
        "query": "breast biopsy",
        "expected": ["19081", "19082", "19083", "19084"],
        "verbosity": "direct",
    },
    {
        "query": "3D mammogram",
        "expected": ["77063"],
        "verbosity": "direct",
    },
    {
        "query": "screening mammogram",
        "expected": ["77067", "77066", "77065"],
        "verbosity": "direct",
    },
    {
        "query": "diagnostic mammogram",
        "expected": ["77065", "77066"],
        "verbosity": "direct",
    },
    # --- Short sentence: brief coverage questions ---
    {
        "query": "is a mammogram covered",
        "expected": ["77067", "77066", "77065"],
        "verbosity": "short",
    },
    {
        "query": "what does medi-cal pay for a breast MRI",
        "expected": ["77046", "77047"],
        "verbosity": "short",
    },
    {
        "query": "is a breast biopsy covered by medi-cal",
        "expected": ["19081", "19082", "19083", "19084"],
        "verbosity": "short",
    },
    {
        "query": "does medi-cal cover breast ultrasound",
        "expected": ["76641", "76642"],
        "verbosity": "short",
    },
    {
        "query": "is a 3D mammogram covered",
        "expected": ["77063"],
        "verbosity": "short",
    },
    {
        "query": "what does medi-cal pay for a mammogram with contrast",
        "expected": ["77048", "77049"],
        "verbosity": "short",
    },
    {
        "query": "how much does medi-cal cover for a screening mammogram",
        "expected": ["77067", "77066", "77065"],
        "verbosity": "short",
    },
    # --- Conversational: real user phrasing with extra context ---
    {
        "query": "I need a mammogram, my last one was 3 years ago, is it covered?",
        "expected": ["77067", "77066", "77065"],
        "verbosity": "conversational",
    },
    {
        "query": "my doctor wants me to get a breast MRI, does medi-cal cover that?",
        "expected": ["77046", "77047"],
        "verbosity": "conversational",
    },
    {
        "query": "I was told I need a biopsy on my breast, is that covered by medi-cal?",
        "expected": ["19081", "19082", "19083", "19084"],
        "verbosity": "conversational",
    },
    {
        "query": "I need to get a follow up mammogram after some abnormal results, is it covered?",
        "expected": ["77065", "77066"],
        "verbosity": "conversational",
    },
    {
        "query": "my doctor recommended a breast ultrasound to check something, will medi-cal pay for it?",
        "expected": ["76641", "76642"],
        "verbosity": "conversational",
    },
    {
        "query": "I'm 45 and haven't had a mammogram in a while, does medi-cal cover annual screening?",
        "expected": ["77067", "77066", "77065"],
        "verbosity": "conversational",
    },
    {
        "query": "I got referred for a 3D mammogram, is that something medi-cal pays for?",
        "expected": ["77063"],
        "verbosity": "conversational",
    },
    # --- Verbose: longer rambling queries ---
    {
        "query": "I have a family history of breast cancer and my doctor said I should get a breast MRI every year, I have medi-cal, is that covered?",
        "expected": ["77046", "77047"],
        "verbosity": "verbose",
    },
    {
        "query": "I just turned 40 and I know I'm supposed to start getting mammograms, I'm on medi-cal and want to know if they cover it and how much they pay",
        "expected": ["77067", "77066", "77065"],
        "verbosity": "verbose",
    },
    {
        "query": "my last mammogram showed something and now they want me to do an ultrasound guided biopsy, will medi-cal cover the biopsy?",
        "expected": ["19083", "19084"],
        "verbosity": "verbose",
    },
    {
        "query": "I need to get a mammogram done and also maybe a breast ultrasound, I want to know what medi-cal will pay for both of those",
        "expected": ["77067", "77066", "77065", "76641", "76642"],
        "verbosity": "verbose",
    },
]


def normalize_code(code: str) -> str:
    """Strip leading zeros so codes match consistently."""
    code = str(code).strip().upper()
    if code.isdigit():
        code = code.lstrip("0") or "0"
    return code


def load_matcher(project_root: Path):
    """Load the FAISS index, metadata, and embedding model."""
    idx_path = project_root / "data" / "embeddings" / "cpt_faiss.index"
    meta_path = project_root / "data" / "embeddings" / "cpt_meta.parquet"

    index = faiss.read_index(str(idx_path))
    meta = pd.read_parquet(meta_path)
    meta["code_norm"] = meta["code"].apply(normalize_code)
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return index, meta, model


ADMIN_CODE_PREFIXES = ("G", "M", "Q")


def search(index, meta, model, query: str, top_k: int = 10):
    """Run a single query and return list of result dicts.

    Over-fetches from FAISS and filters out admin/quality-measure
    HCPCS codes (G, M, Q prefixes) to match production behaviour.
    """
    fetch_k = top_k * 3
    q = model.encode([query], normalize_embeddings=True).astype("float32")
    scores, ids = index.search(q, fetch_k)

    results = []
    for faiss_id, score in zip(ids[0].tolist(), scores[0].tolist()):
        if len(results) >= top_k:
            break
        row = meta.iloc[int(faiss_id)]
        code = normalize_code(row["code"])
        if code.startswith(ADMIN_CODE_PREFIXES):
            continue
        results.append({
            "code": code,
            "score": float(score),
            "title": str(row.get("title", "")).strip('"').rstrip("."),
        })
    return results


def evaluate(index, meta, model, top_k: int = 10):
    """Run all test queries and compute metrics."""
    results = []

    for test in TEST_QUERIES:
        query = test["query"]
        expected = set(normalize_code(c) for c in test["expected"])
        matches = search(index, meta, model, query, top_k)
        returned_codes = [m["code"] for m in matches]

        hit_at_5 = any(c in expected for c in returned_codes[:5])
        hit_at_10 = any(c in expected for c in returned_codes[:top_k])

        # Reciprocal rank of first expected code found
        rr = 0.0
        for rank, code in enumerate(returned_codes, start=1):
            if code in expected:
                rr = 1.0 / rank
                break

        found = set(returned_codes[:top_k]) & expected
        recall_at_k = len(found) / len(expected)

        results.append({
            "query": query,
            "verbosity": test["verbosity"],
            "expected": sorted(expected),
            "returned_top5": returned_codes[:5],
            "hit@5": hit_at_5,
            "hit@10": hit_at_10,
            "mrr": rr,
            "recall@k": recall_at_k,
            "first_hit_rank": next(
                (i + 1 for i, c in enumerate(returned_codes) if c in expected),
                None,
            ),
        })

    return results


def print_report(results, top_k: int):
    """Print a formatted evaluation report."""
    n = len(results)
    avg_hit5 = sum(r["hit@5"] for r in results) / n
    avg_hit10 = sum(r["hit@10"] for r in results) / n
    avg_mrr = sum(r["mrr"] for r in results) / n
    avg_recall = sum(r["recall@k"] for r in results) / n

    print("=" * 70)
    print("CPT MATCH QUALITY EVALUATION")
    print("=" * 70)
    print(f"Model        : all-MiniLM-L6-v2")
    print(f"Queries      : {n}")
    print(f"Top-K        : {top_k}")
    print()
    print(f"  Hit@5      : {avg_hit5:.1%}  ({sum(r['hit@5'] for r in results)}/{n})")
    print(f"  Hit@10     : {avg_hit10:.1%}  ({sum(r['hit@10'] for r in results)}/{n})")
    print(f"  MRR        : {avg_mrr:.3f}")
    print(f"  Recall@{top_k:<3} : {avg_recall:.1%}")
    print()

    # Per-verbosity breakdown
    verbosities = ["direct", "short", "conversational", "verbose"]
    print("-" * 70)
    print(f"{'Verbosity':<18} {'Hit@5':>7} {'Hit@10':>7} {'MRR':>7} {'Recall':>7}")
    print("-" * 70)
    for v in verbosities:
        v_results = [r for r in results if r["verbosity"] == v]
        if not v_results:
            continue
        cn = len(v_results)
        print(
            f"{v:<18} "
            f"{sum(r['hit@5'] for r in v_results)/cn:>6.0%} "
            f"{sum(r['hit@10'] for r in v_results)/cn:>6.0%} "
            f"{sum(r['mrr'] for r in v_results)/cn:>7.3f} "
            f"{sum(r['recall@k'] for r in v_results)/cn:>6.0%}"
        )
    print()

    # Per-query detail
    print("-" * 70)
    print("Per-Query Results:")
    print("-" * 70)
    for r in results:
        status = "HIT" if r["hit@10"] else "MISS"
        rank_str = f"rank {r['first_hit_rank']}" if r["first_hit_rank"] else "not found"
        print(f"  [{status:>4}] \"{r['query']}\"")
        print(f"         expected: {r['expected']}")
        print(f"         top 5:   {r['returned_top5']}")
        print(f"         first hit: {rank_str}  |  recall: {r['recall@k']:.0%}  |  mrr: {r['mrr']:.3f}")
        print()

    return {
        "hit@5": avg_hit5,
        "hit@10": avg_hit10,
        "mrr": avg_mrr,
        f"recall@{top_k}": avg_recall,
    }


def main():
    parser = argparse.ArgumentParser(description="CPT match quality evaluation")
    parser.add_argument("--top_k", type=int, default=10, help="Number of results to retrieve")
    parser.add_argument("--save", type=str, default=None, help="Save results to JSON file")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    print(f"Loading model and index from {project_root} ...")
    index, meta, model = load_matcher(project_root)
    print(f"Loaded {index.ntotal} vectors, {len(meta)} metadata rows.\n")

    results = evaluate(index, meta, model, top_k=args.top_k)
    summary = print_report(results, top_k=args.top_k)

    if args.save:
        save_path = Path(args.save)
        save_data = {
            "config": {
                "model": "sentence-transformers/all-MiniLM-L6-v2",
                "top_k": args.top_k,
                "n_queries": len(TEST_QUERIES),
                "label": "baseline",
            },
            "summary": summary,
            "results": results,
        }
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(json.dumps(save_data, indent=2))
        print(f"Results saved to {save_path}")


if __name__ == "__main__":
    main()
