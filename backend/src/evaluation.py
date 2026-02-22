# backend/src/evaluation.py
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.matcher import Matcher
from src.pricing import PricingLookup

ADMIN_CODE_PREFIXES = ("G", "M", "Q")


def _normalize_code(code: str) -> str:
    code = str(code or "").strip().upper()
    if code.isdigit():
        code = code.lstrip("0") or "0"
    return code


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


@dataclass
class MatchCase:
    query: str
    expected: List[str]


DEFAULT_MATCH_CASES: List[MatchCase] = [
    MatchCase("mammogram", ["77067", "77066", "77065"]),
    MatchCase("breast MRI", ["77046", "77047"]),
    MatchCase("breast ultrasound", ["76641", "76642"]),
    MatchCase("breast biopsy", ["19081", "19082", "19083", "19084"]),
    MatchCase("3D mammogram", ["77063"]),
    MatchCase("screening mammogram", ["77067", "77066", "77065"]),
    MatchCase("diagnostic mammogram", ["77065", "77066"]),
    MatchCase("is a mammogram covered", ["77067", "77066", "77065"]),
    MatchCase("what does medi-cal pay for a breast MRI", ["77046", "77047"]),
    MatchCase("is a breast biopsy covered by medi-cal", ["19081", "19082", "19083", "19084"]),
    MatchCase("does medi-cal cover breast ultrasound", ["76641", "76642"]),
    MatchCase("is a 3D mammogram covered", ["77063"]),
    MatchCase("what does medi-cal pay for a mammogram with contrast", ["77048", "77049"]),
    MatchCase("how much does medi-cal cover for a screening mammogram", ["77067", "77066", "77065"]),
    MatchCase("I need a mammogram, my last one was 3 years ago, is it covered?", ["77067", "77066", "77065"]),
    MatchCase("my doctor wants me to get a breast MRI, does medi-cal cover that?", ["77046", "77047"]),
    MatchCase("I was told I need a biopsy on my breast, is it covered by medi-cal?", ["19081", "19082", "19083", "19084"]),
    MatchCase("I need to get a follow up mammogram after some abnormal results, is it covered?", ["77065", "77066"]),
    MatchCase("my doctor recommended a breast ultrasound to check something, will medi-cal pay for it?", ["76641", "76642"]),
    MatchCase("I'm 45 and haven't had a mammogram in a while, does medi-cal cover annual screening?", ["77067", "77066", "77065"]),
    MatchCase("I got referred for a 3D mammogram, is that something medi-cal pays for?", ["77063"]),
    MatchCase("I have a family history of breast cancer and my doctor said I should get a breast MRI every year, I have medi-cal, is that covered?", ["77046", "77047"]),
    MatchCase("I just turned 40 and I know I'm supposed to start getting mammograms, I'm on medi-cal and want to know if they cover it and how much they pay", ["77067", "77066", "77065"]),
    MatchCase("my last mammogram showed something and now they want me to do an ultrasound guided biopsy, will medi-cal cover the biopsy?", ["19083", "19084"]),
    MatchCase("I need to get a mammogram done and also maybe a breast ultrasound, I want to know what medi-cal will pay for both of those", ["77067", "77066", "77065", "76641", "76642"]),
]


def _load_match_cases(path: Path) -> List[MatchCase]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        cases_raw = data
    elif isinstance(data, dict) and isinstance(data.get("cases"), list):
        cases_raw = data["cases"]
    else:
        raise ValueError("Match eval JSON must be a list or {\"cases\": [...]}")

    out: List[MatchCase] = []
    for i, x in enumerate(cases_raw):
        if not isinstance(x, dict):
            raise ValueError(f"Case {i} must be an object")
        q = str(x.get("query", "")).strip()
        exp = x.get("expected", [])
        if not q or not isinstance(exp, list) or not exp:
            raise ValueError(f"Case {i} requires non-empty query and expected[]")
        out.append(MatchCase(query=q, expected=[str(c) for c in exp]))
    return out


def _match_metrics(returned_codes: List[str], expected_codes: List[str], top_k: int) -> Dict[str, Any]:
    expected = {_normalize_code(c) for c in expected_codes}
    returned = [_normalize_code(c) for c in returned_codes]

    hit_at_5 = any(c in expected for c in returned[: min(5, len(returned))])
    hit_at_k = any(c in expected for c in returned[: min(top_k, len(returned))])

    rr = 0.0
    first_hit_rank = None
    for rank, code in enumerate(returned, start=1):
        if code in expected:
            rr = 1.0 / rank
            first_hit_rank = rank
            break

    found = set(returned[: min(top_k, len(returned))]) & expected
    recall_at_k = (len(found) / len(expected)) if expected else 0.0

    return {
        "expected": sorted(expected),
        "returned": returned[: min(top_k, len(returned))],
        "returned_top5": returned[:5],
        "hit@5": bool(hit_at_5),
        "hit@k": bool(hit_at_k),
        "mrr": float(rr),
        "recall@k": float(recall_at_k),
        "first_hit_rank": first_hit_rank,
    }


def run_match_eval(
    *,
    project_root: Path,
    top_k: int,
    cases: List[MatchCase],
    save_path: Optional[Path] = None,
) -> Dict[str, Any]:
    pricing = PricingLookup(project_root)
    matcher = Matcher(project_root, pricing)

    per_case: List[Dict[str, Any]] = []
    for c in cases:
        candidates = matcher.search(c.query, top_k=top_k) or []
        codes = []
        for cand in candidates:
            code = cand.get("code")
            if not code:
                continue
            code = _normalize_code(code)
            if code.startswith(ADMIN_CODE_PREFIXES):
                continue
            codes.append(code)

        m = _match_metrics(codes, c.expected, top_k)
        per_case.append({"query": c.query, **m})

    n = max(1, len(per_case))
    summary = {
        "hit@5": sum(1 for r in per_case if r["hit@5"]) / n,
        f"hit@{top_k}": sum(1 for r in per_case if r["hit@k"]) / n,
        "mrr": sum(r["mrr"] for r in per_case) / n,
        f"recall@{top_k}": sum(r["recall@k"] for r in per_case) / n,
        "n_cases": len(per_case),
        "top_k": top_k,
    }

    out = {"summary": summary, "results": per_case}

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    _print_match_report(out)
    return out


def _print_match_report(out: Dict[str, Any]) -> None:
    s = out["summary"]
    results = out["results"]
    top_k = s["top_k"]

    print("=" * 72)
    print("MATCH EVALUATION (production Matcher.search)")
    print("=" * 72)
    print(f"Cases      : {s['n_cases']}")
    print(f"Top-K      : {top_k}")
    print(f"Hit@5      : {s['hit@5']:.1%}")
    print(f"Hit@{top_k:<3}   : {s[f'hit@{top_k}']:.1%}")
    print(f"MRR        : {s['mrr']:.3f}")
    print(f"Recall@{top_k:<3}: {s[f'recall@{top_k}']:.1%}")
    print()

    for r in results:
        status = "HIT" if r["hit@k"] else "MISS"
        rank = f"rank {r['first_hit_rank']}" if r["first_hit_rank"] else "not found"
        print(f"[{status:>4}] {r['query']}")
        print(f"       expected: {r['expected']}")
        print(f"       top 5:    {r['returned_top5']}")
        print(f"       first hit: {rank} | recall@k: {r['recall@k']:.0%} | mrr: {r['mrr']:.3f}")
        print()


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="evaluation.py")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_match = sub.add_parser("match", help="Evaluate matching/searching logic (production Matcher)")
    p_match.add_argument("--top_k", type=int, default=10)
    p_match.add_argument("--file", type=str, default=None, help="JSON file with cases (list or {cases:[...]})")
    p_match.add_argument("--save", type=str, default=None, help="Write JSON results")

    args = parser.parse_args(argv)
    root = _project_root()

    if args.cmd == "match":
        cases = _load_match_cases(Path(args.file)) if args.file else DEFAULT_MATCH_CASES
        run_match_eval(
            project_root=root,
            top_k=int(args.top_k),
            cases=cases,
            save_path=Path(args.save) if args.save else None,
        )
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))