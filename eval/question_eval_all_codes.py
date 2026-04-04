from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import pandas as pd
import requests

from eval.ewc_code_rules import CODE_RULES, TOPIC_PATTERNS, expected_topics_for


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Make backend importable so we can reuse the tree-walking logic
sys.path.insert(0, str(PROJECT_ROOT / "backend"))
from src.eligibility_evaluator import _eval_node, _pick_next_question  # noqa: E402

IN_PATH = PROJECT_ROOT / "data" / "sample_data" / "eval" / "ewc_testset_all_codes.csv"
OUT_PATH = PROJECT_ROOT / "eval" / "results" / "question_eval_all_codes.csv"

API_BASE = "http://127.0.0.1:8000"
MATCH_ENDPOINT = f"{API_BASE}/match"
START_ENDPOINT = f"{API_BASE}/eligibility/start"
NEXT_ENDPOINT = f"{API_BASE}/eligibility/next"

MAMMO_CODES = {
    "77046", "77047", "77048", "77049",
    "77061", "77062", "77063", "77065", "77066", "77067",
}


# ------------------------------------------------------------
# API helpers
# ------------------------------------------------------------
def call_match(user_query: str, top_k: int = 10) -> List[Dict[str, Any]]:
    resp = requests.post(
        MATCH_ENDPOINT,
        json={"text": user_query, "top_k": top_k},
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    return data.get("candidates", [])


def choose_selected(
    candidates: List[Dict[str, Any]],
    expected_cpt: str,
) -> Tuple[Dict[str, Any] | None, bool]:
    expected_cpt = str(expected_cpt).strip()

    for cand in candidates:
        cand_code = str(cand.get("code", "")).strip()
        if cand_code == expected_cpt:
            selected = dict(cand)
            selected.setdefault("rerank_score", 0)
            return selected, True

    if candidates:
        selected = dict(candidates[0])
        selected.setdefault("rerank_score", 0)
        return selected, str(selected.get("code", "")).strip() == expected_cpt

    return None, False


def call_eligibility_start(
    user_query: str,
    selected: Dict[str, Any],
    top_k: int = 10,
) -> Dict[str, Any]:
    payload = {
        "user_text": user_query,
        "selected": selected,
        "top_k": top_k,
    }
    resp = requests.post(START_ENDPOINT, json=payload, timeout=60)
    resp.raise_for_status()
    return resp.json()


def call_eligibility_next(
    user_query: str,
    selected: Dict[str, Any],
    pathways: List[str],
    questions_per_pathway: List[List[str]],
    qa_so_far: List[Dict[str, Any]],
    top_k: int = 10,
    logic_tree: Any = None,
    logic_trees: Any = None,
    question_map: Any = None,
) -> Dict[str, Any]:
    payload = {
        "user_text": user_query,
        "selected": selected,
        "pathways": pathways if isinstance(pathways, list) else [],
        "questions_per_pathway": questions_per_pathway if isinstance(questions_per_pathway, list) else [],
        "qa_so_far": qa_so_far if isinstance(qa_so_far, list) else [],
        "top_k": top_k,
    }

    if logic_tree is not None:
        payload["logic_tree"] = logic_tree
    if logic_trees is not None:
        payload["logic_trees"] = logic_trees
    if question_map is not None:
        payload["question_map"] = question_map

    resp = requests.post(NEXT_ENDPOINT, json=payload, timeout=20)
    resp.raise_for_status()
    return resp.json()


# ------------------------------------------------------------
# Question/topic extraction
# ------------------------------------------------------------
def flatten_questions(step_data: Dict[str, Any]) -> List[str]:
    questions: List[str] = []

    q = step_data.get("question")
    if isinstance(q, str) and q.strip():
        questions.append(q.strip())

    next_q = step_data.get("next_question")
    if isinstance(next_q, str) and next_q.strip():
        questions.append(next_q.strip())

    qpp = step_data.get("questions_per_pathway", [])
    if isinstance(qpp, list):
        for item in qpp:
            if isinstance(item, list):
                for q_item in item:
                    if isinstance(q_item, str) and q_item.strip():
                        questions.append(q_item.strip())

    deduped: List[str] = []
    seen = set()
    for q_item in questions:
        if q_item not in seen:
            deduped.append(q_item)
            seen.add(q_item)

    return deduped


def detect_topics(questions: List[str]) -> Dict[str, bool]:
    text = "\n".join(questions)
    return {topic: bool(pattern.search(text)) for topic, pattern in TOPIC_PATTERNS.items()}


def extract_found_topics(topic_hits: Dict[str, bool]) -> List[str]:
    return [topic for topic, hit in topic_hits.items() if hit]


# ------------------------------------------------------------
# Coverage helpers
# ------------------------------------------------------------
def safe_load_attrs(attrs_json: str) -> Dict[str, Any]:
    if not attrs_json or str(attrs_json).strip() in {"", "nan", "None"}:
        return {}
    try:
        obj = json.loads(attrs_json)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def normalize_expected_coverage(value: Any) -> str:
    if pd.isna(value):
        return ""

    s = str(value).strip().lower()
    if not s or s in {"nan", "none"}:
        return ""

    mapping = {
        "covered": "covered",
        "cover": "covered",
        "yes": "covered",
        "approved": "covered",
        "approve": "covered",
        "eligible": "covered",
        "not_covered": "not_covered",
        "not covered": "not_covered",
        "not-covered": "not_covered",
        "no": "not_covered",
        "denied": "not_covered",
        "deny": "not_covered",
        "ineligible": "not_covered",
        "excluded": "not_covered",
    }
    return mapping.get(s, "")


def extract_coverage_decision(step_data: Dict[str, Any]) -> str:
    decision = step_data.get("decision", "")
    if isinstance(decision, str):
        d = decision.strip().lower()
        if d in {"covered", "not_covered"}:
            return d
    return ""


# ------------------------------------------------------------
# Question answering heuristics
# ------------------------------------------------------------
def answer_question(question: str, attrs: Dict[str, Any], expected_cpt: str) -> bool:
    q = question.lower().strip()

    if "routine screening" in q and "not a diagnostic" in q:
        return expected_cpt in {"77067", "77063"} or bool(attrs.get("screening", False))

    if "40 years or older" in q or "age 40" in q:
        age = attrs.get("age")
        return bool(age is not None and age >= 40)

    if "how old" in q or re.search(r"\bage\b", q):
        age = attrs.get("age")
        return bool(age is not None and age >= 40)

    if "365 days" in q or "last screening" in q or "last mammogram" in q or "at least a year" in q:
        days = attrs.get("days_since_last_screening")
        return bool(days is not None and days >= 365)

    if "brca" in q:
        return bool(attrs.get("brca_mutation", False))

    if "family history" in q:
        return bool(attrs.get("family_history", False))

    if "high-risk" in q or "high risk" in q or "lifetime risk" in q:
        return bool(attrs.get("high_risk", False))

    if "symptom" in q or "lump" in q or "pain" in q or "nipple discharge" in q or "skin changes" in q:
        return bool(attrs.get("symptoms_present", False))

    if "abnormal" in q and "screen" in q:
        return bool(attrs.get("abnormal_result", False) or attrs.get("abnormal_screening_result", False))

    if "left breast" in q:
        return attrs.get("laterality") == "left"
    if "right breast" in q:
        return attrs.get("laterality") == "right"
    if "both breasts" in q or "bilateral" in q:
        return attrs.get("laterality") == "bilateral"
    if "one breast" in q or "unilateral" in q:
        return attrs.get("laterality") in {"left", "right"}

    if "doctor" in q or "provider" in q or "ordered" in q or "referred" in q:
        return bool(attrs.get("provider_ordered", True))

    if "with screening" in q or "add-on" in q or "in addition" in q:
        return bool(attrs.get("includes_screening_context", False))

    if "ultrasound guidance" in q:
        return attrs.get("guidance_modality") == "ultrasound"

    if "mri guidance" in q:
        return attrs.get("guidance_modality") == "mri"

    if "stereotactic" in q:
        return attrs.get("guidance_modality") == "stereotactic"

    if "pregnan" in q or "pregnancy" in q:
        return bool(attrs.get("pregnant", False))

    if "bleeding" in q or "postmenopausal" in q or "spotting" in q:
        return bool(attrs.get("abnormal_bleeding", False) or attrs.get("postmenopausal", False))

    if "reflex" in q or "genotype" in q or "typing" in q:
        return bool(attrs.get("reflex_testing", False))

    if "new patient" in q:
        return bool(attrs.get("new_patient", False))
    if "established patient" in q:
        return not bool(attrs.get("new_patient", False))

    if "specimen" in q or "sample" in q or "tissue" in q:
        return bool(attrs.get("specimen_available", True))

    return False


def run_full_conversation(
    user_query: str,
    selected: Dict[str, Any],
    expected_cpt: str,
    attrs: Dict[str, Any],
    top_k: int = 10,
    max_turns: int = 10,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[str], Any, Any]:
    step = call_eligibility_start(user_query, selected, top_k=top_k)

    qa_so_far: List[Dict[str, Any]] = []
    all_questions: List[str] = flatten_questions(step)

    pathways = step.get("pathways", []) if isinstance(step.get("pathways", []), list) else []
    questions_per_pathway = (
        step.get("questions_per_pathway", [])
        if isinstance(step.get("questions_per_pathway", []), list)
        else []
    )
    logic_tree = step.get("logic_tree")
    logic_trees = step.get("logic_trees")
    question_map = step.get("question_map")

    current_question = (step.get("question") or "").strip()

    turns = 0
    while turns < max_turns:
        turns += 1

        decision = extract_coverage_decision(step)
        if decision:
            break

        if not current_question or current_question.lower() == "no question generated.":
            break

        answer = answer_question(current_question, attrs, expected_cpt)
        qa_so_far.append({"q": current_question, "a": bool(answer)})

        step = call_eligibility_next(
            user_query=user_query,
            selected=selected,
            pathways=pathways,
            questions_per_pathway=questions_per_pathway,
            qa_so_far=qa_so_far,
            top_k=top_k,
            logic_tree=logic_tree,
            logic_trees=logic_trees,
            question_map=question_map,
        )

        for q_item in flatten_questions(step):
            if q_item not in all_questions:
                all_questions.append(q_item)

        decision = extract_coverage_decision(step)
        if decision:
            break

        current_question = (step.get("next_question") or "").strip()

    return step, qa_so_far, all_questions, logic_tree, question_map


# ------------------------------------------------------------
# Metrics
# ------------------------------------------------------------
def compute_recall(expected_topics: List[str], found_topics: List[str]) -> float:
    if not expected_topics:
        return 1.0
    found = set(found_topics)
    hit_count = sum(1 for topic in expected_topics if topic in found)
    return hit_count / len(expected_topics)


def compute_precision(expected_topics: List[str], found_topics: List[str]) -> float:
    if not found_topics:
        return 0.0
    expected = set(expected_topics)
    found = set(found_topics)
    hit_count = sum(1 for topic in found if topic in expected)
    return hit_count / len(found)


def compute_f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ------------------------------------------------------------
# Tree-based metrics
# ------------------------------------------------------------
def get_ground_truth_qids(
    tree: Any,
    attrs: Dict[str, Any],
    question_map: Dict[str, str] | None,
    expected_cpt: str,
    max_turns: int = 20,
) -> Set[str]:
    """Simulate the correct tree walk using true patient attrs."""
    if not tree or not isinstance(tree, dict):
        return set()

    answers: Dict[str, bool] = {}
    necessary: Set[str] = set()

    for _ in range(max_turns):
        if _eval_node(tree, answers) is not None:
            break
        next_q_id = _pick_next_question(tree, answers)
        if not next_q_id:
            break
        q_text = (question_map or {}).get(next_q_id, next_q_id)
        answer = answer_question(q_text, attrs, expected_cpt)
        necessary.add(next_q_id)
        answers[next_q_id] = bool(answer)

    return necessary


def get_asked_qids(
    qa_so_far: List[Dict[str, Any]],
    question_map: Dict[str, str] | None,
) -> Set[str]:
    rev_map = {v: k for k, v in (question_map or {}).items()}
    return {rev_map.get(item["q"], item["q"]) for item in qa_so_far}


# ------------------------------------------------------------
# Main evaluation
# ------------------------------------------------------------
def _stratified_sample(df: pd.DataFrame, n_per_code: int, seed: int = 42) -> pd.DataFrame:
    """Sample n_per_code rows per CPT code, stratified by covered/not_covered and verbosity."""
    frames = []
    for code, group in df.groupby("expected_cpt"):
        sampled = (
            group.groupby(["expected_label", "verbosity"], group_keys=False)
            .apply(lambda x: x.sample(min(len(x), max(1, n_per_code // 8)), random_state=seed))
        )
        if len(sampled) < n_per_code:
            remaining = group.drop(sampled.index)
            extra_n = min(n_per_code - len(sampled), len(remaining))
            if extra_n > 0:
                sampled = pd.concat([sampled, remaining.sample(extra_n, random_state=seed)])
        if len(sampled) > n_per_code:
            sampled = sampled.sample(n_per_code, random_state=seed)
        frames.append(sampled)
    return pd.concat(frames).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mammo-only", action="store_true", help="Evaluate only mammography-related codes")
    parser.add_argument("--sample", type=int, default=None, help="Sample N rows per CPT code (stratified). Omit for full run.")
    args = parser.parse_args()

    if not IN_PATH.exists():
        raise FileNotFoundError(f"Input eval set not found: {IN_PATH}")

    df = pd.read_csv(IN_PATH)

    required_cols = {"expected_cpt", "user_query"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV missing required columns: {missing}")

    if args.mammo_only:
        df = df[df["expected_cpt"].astype(str).isin(MAMMO_CODES)].copy()

    if args.sample:
        df = _stratified_sample(df, args.sample)
        print(f"Sampled {len(df)} rows ({args.sample} per code, stratified)", flush=True)

    print(f"Testing on {len(df)} rows", flush=True)

    rows: List[Dict[str, Any]] = []
    total = len(df)

    for i, (_, r) in enumerate(df.iterrows(), start=1):
        start_time = time.time()

        print(f"[{i}/{total}] CPT={r['expected_cpt']} running...", flush=True)
        if i % 25 == 0:
            print(f"--- still running: {i}/{total} rows complete ---", flush=True)

        expected_cpt = str(r["expected_cpt"]).strip()
        user_query = str(r["user_query"]).strip()
        verbosity = str(r["verbosity"]).strip() if "verbosity" in df.columns and pd.notna(r.get("verbosity")) else ""
        attrs_json = str(r["attrs_json"]).strip() if "attrs_json" in df.columns and pd.notna(r.get("attrs_json")) else ""
        attrs = safe_load_attrs(attrs_json)

        expected_topics = expected_topics_for(expected_cpt) if expected_cpt in CODE_RULES else []
        raw_expected_label = r.get("expected_label", "")
        expected_coverage = normalize_expected_coverage(raw_expected_label)

        if expected_cpt not in CODE_RULES:
            rows.append({
                "expected_cpt": expected_cpt,
                "user_query": user_query,
                "verbosity": verbosity,
                "selected_code": "",
                "selected_title": "",
                "selected_matches_expected": False,
                "expected_topics_json": json.dumps([]),
                "found_topics_json": json.dumps([]),
                "first_question": "",
                "actual_questions_json": json.dumps([]),
                "qa_so_far_json": json.dumps([]),
                "pathways_json": json.dumps([]),
                "questions_per_pathway_json": json.dumps([]),
                "no_pathways": "",
                "no_pathways_reason": f"Code {expected_cpt} not present in CODE_RULES",
                "expected_coverage": expected_coverage,
                "coverage_decision": "",
                "coverage_correct": False,
                "topic_recall": 0.0,
                "topic_precision": 0.0,
                "topic_f1": 0.0,
                "ground_truth_qids_json": "[]",
                "asked_qids_json": "[]",
                "tree_recall": 0.0,
                "tree_precision": 0.0,
                "tree_f1": 0.0,
                "error": f"Missing CODE_RULES entry for {expected_cpt}",
            })
            elapsed = time.time() - start_time
            print(f"    finished row {i} in {elapsed:.2f}s", flush=True)
            continue

        try:
            candidates = call_match(user_query, top_k=10)
            selected, selected_matches_expected = choose_selected(candidates, expected_cpt)

            if selected is None:
                raise ValueError("No candidates returned from /match")

            final_step, qa_so_far, all_questions, logic_tree, question_map = run_full_conversation(
                user_query=user_query,
                selected=selected,
                expected_cpt=expected_cpt,
                attrs=attrs,
                top_k=10,
                max_turns=10,
            )

            # Keep topic metrics in CSV for debugging
            topic_hits = detect_topics(all_questions)
            found_topics = extract_found_topics(topic_hits)
            topic_recall = compute_recall(expected_topics, found_topics)
            topic_precision = compute_precision(expected_topics, found_topics)
            topic_f1 = compute_f1(topic_precision, topic_recall)

            # Tree metrics = primary reported metrics
            ground_truth_qids = get_ground_truth_qids(logic_tree, attrs, question_map, expected_cpt)
            asked_qids = get_asked_qids(qa_so_far, question_map)
            tree_recall = len(asked_qids & ground_truth_qids) / len(ground_truth_qids) if ground_truth_qids else 0.0
            tree_precision = len(asked_qids & ground_truth_qids) / len(asked_qids) if asked_qids else 0.0
            tree_f1 = compute_f1(tree_precision, tree_recall)

            coverage_decision = extract_coverage_decision(final_step)
            coverage_correct = bool(
                expected_coverage and coverage_decision and expected_coverage == coverage_decision
            )

            error = ""

        except Exception as e:
            selected = {}
            selected_matches_expected = False
            final_step = {}
            qa_so_far = []
            all_questions = []
            found_topics = []
            topic_recall = 0.0
            topic_precision = 0.0
            topic_f1 = 0.0
            ground_truth_qids = set()
            asked_qids = set()
            tree_recall = 0.0
            tree_precision = 0.0
            tree_f1 = 0.0
            coverage_decision = ""
            coverage_correct = False
            error = str(e)

        rows.append({
            "expected_cpt": expected_cpt,
            "user_query": user_query,
            "verbosity": verbosity,
            "selected_code": str(selected.get("code", "")).strip() if selected else "",
            "selected_title": str(selected.get("title", "")).strip() if selected else "",
            "selected_matches_expected": selected_matches_expected,
            "expected_topics_json": json.dumps(expected_topics),
            "found_topics_json": json.dumps(found_topics),
            "first_question": final_step.get("question", "") if final_step else "",
            "actual_questions_json": json.dumps(all_questions),
            "qa_so_far_json": json.dumps(qa_so_far),
            "pathways_json": json.dumps(final_step.get("pathways", [])) if final_step else "[]",
            "questions_per_pathway_json": json.dumps(final_step.get("questions_per_pathway", [])) if final_step else "[]",
            "no_pathways": final_step.get("no_pathways", "") if final_step else "",
            "no_pathways_reason": final_step.get("no_pathways_reason", "") if final_step else "",
            "expected_coverage": expected_coverage,
            "coverage_decision": coverage_decision,
            "coverage_correct": coverage_correct,
            "topic_recall": topic_recall,
            "topic_precision": topic_precision,
            "topic_f1": topic_f1,
            "ground_truth_qids_json": json.dumps(sorted(ground_truth_qids)),
            "asked_qids_json": json.dumps(sorted(asked_qids)),
            "tree_recall": tree_recall,
            "tree_precision": tree_precision,
            "tree_f1": tree_f1,
            "error": error,
        })

        elapsed = time.time() - start_time
        print(f"    finished row {i} in {elapsed:.2f}s", flush=True)

    out = pd.DataFrame(rows)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)

    print("\n---- SUMMARY ----")
    print(f"Rows evaluated: {len(out)}")
    print(f"Avg tree recall:     {out['tree_recall'].mean():.3f}")
    print(f"Avg tree precision:  {out['tree_precision'].mean():.3f}")
    print(f"Avg tree F1:         {out['tree_f1'].mean():.3f}")
    print(f"Selected CPT matches expected: {out['selected_matches_expected'].mean():.3%}")

    scored_cov = out[out["expected_coverage"].astype(str).str.len() > 0].copy()
    if len(scored_cov) > 0:
        print(f"Coverage decision accuracy: {scored_cov['coverage_correct'].mean():.3%}")
        print(f"Coverage rows scored: {len(scored_cov)}")
    else:
        print("Coverage decision accuracy: n/a (no rows with expected coverage values)")

    print("\n---- BY CATEGORY ----")
    tmp = out.copy()
    tmp["category"] = tmp["expected_cpt"].map(
        lambda c: CODE_RULES.get(str(c)).category if str(c) in CODE_RULES else "unknown"
    )

    summary_cols = ["tree_recall", "tree_precision", "tree_f1", "selected_matches_expected"]
    if len(scored_cov) > 0:
        summary_cols.append("coverage_correct")

    print(
        tmp.groupby("category")[summary_cols]
        .mean()
        .sort_values("tree_f1", ascending=False)
        .round(3)
    )

    print(f"\nSaved -> {OUT_PATH}")


if __name__ == "__main__":
    main()