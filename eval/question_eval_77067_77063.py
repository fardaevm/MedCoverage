#LEGACY

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List

import pandas as pd
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IN_PATH = PROJECT_ROOT / "data" / "sample_data" / "ewc_mammo_77067_77063_testset_v2.csv"
OUT_PATH = PROJECT_ROOT / "eval" / "results" / "question_eval_77067_77063.csv"

API_BASE = "http://127.0.0.1:8000"
ENDPOINT = f"{API_BASE}/eligibility/questions"


# -----------------------------
# Gate definitions (ONLY 67 + 63)
# -----------------------------
REQUIRED_GATES = {
    "77067": ["age", "frequency"],
    "77063": ["age", "frequency", "addon"],
}

GATE_PATTERNS = {
    "age": re.compile(r"\bage\b|\byears?\b|\bat least\s*\d{2}\b", re.I),
    "frequency": re.compile(r"\bdays?\b|\bmonths?\b|\byear\b|\bsince\b|\blast\b|\bannual\b", re.I),
    "addon": re.compile(r"\b77067\b|\badd[- ]?on\b|\bwith screening\b|\bin addition\b", re.I),
}


def call_questions(user_query: str, cpt: str) -> List[str]:
    # 1️⃣ call /match first
    match_resp = requests.post(
        f"{API_BASE}/match",
        json={"text": user_query, "top_k": 10},
        timeout=30,
    )
    match_resp.raise_for_status()
    matches = match_resp.json()

    if not matches:
        return []

    # 2️⃣ pick first match with same CPT if possible
    selected = None
    for m in matches:
        if str(m.get("code")) == str(cpt):
            selected = m
            break
    if selected is None:
        selected = matches[0]

    # 3️⃣ call eligibility/questions with correct schema
    payload = {
        "user_text": user_query,
        "selected": selected,
        "top_k": 10,
    }

    r = requests.post(ENDPOINT, json=payload, timeout=30)
    r.raise_for_status()
    data = r.json()

    if isinstance(data, list):
        return data

    if isinstance(data, dict) and "questions" in data:
        return data["questions"]

    return []

def detect_gates(questions: List[str]) -> Dict[str, bool]:
    joined = "\n".join(questions)
    return {
        gate: bool(pattern.search(joined))
        for gate, pattern in GATE_PATTERNS.items()
    }


def score(cpt: str, found: Dict[str, bool]) -> float:
    required = REQUIRED_GATES[cpt]
    present = sum(1 for g in required if found.get(g, False))
    return present / len(required)


def main():
    df = pd.read_csv(IN_PATH)
    df = df[df["expected_cpt"].astype(str).isin(["77067", "77063"])]

    rows = []

    for _, r in df.iterrows():
        cpt = str(r["expected_cpt"])
        user_query = str(r["user_query"])

        try:
            questions = call_questions(user_query, cpt)
            found = detect_gates(questions)
            recall = score(cpt, found)
            error = ""
        except Exception as e:
            questions = []
            found = {}
            recall = 0.0
            error = str(e)

        rows.append({
            "case_id": r["case_id"],
            "expected_cpt": cpt,
            "user_query": user_query,
            "actual_questions_json": json.dumps(questions),
            "gate_recall": recall,
            "gate_age": found.get("age", False),
            "gate_frequency": found.get("frequency", False),
            "gate_addon": found.get("addon", False),
            "error": error,
        })

    out = pd.DataFrame(rows)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)

    print("\n---- SUMMARY ----")
    print("Avg gate recall:", out["gate_recall"].mean())
    print("\nBy CPT:")
    print(out.groupby("expected_cpt")["gate_recall"].mean())
    print(f"\nSaved -> {OUT_PATH}")


if __name__ == "__main__":
    main()