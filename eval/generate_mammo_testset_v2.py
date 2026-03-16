#LEGACY

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))


import pandas as pd
from pypdf import PdfReader



# ---- repo imports (assumes you run from repo root with poetry) ----
# If these imports fail, run using: poetry run python eval/generate_mammo_testset_v2.py
from src.pricing import PricingLookup
from src.matcher import Matcher

PDF_PATH = PROJECT_ROOT / "data" / "raw" / "evwoman.pdf"
OUT_PATH = PROJECT_ROOT / "data" / "sample_data" / "ewc_mammo_77067_77063_testset_v2.csv"


# -----------------------------
# Text noise helpers
# -----------------------------
def inject_typos(text: str, prob: float = 0.35) -> str:
    typo_map = {
        "mammogram": ["mamogram", "mamagram", "mammoogram", "mamogrm", "mammogam"],
        "mammography": ["mamography", "mammo graphy", "mamogrphy"],
        "screening": ["screenign", "screeing", "screning", "screenin"],
        "routine": ["routin", "routne"],
        "tomosynthesis": ["tomosinthesis", "tomosynthisis", "tomosinthesys", "tomosynthis"],
        "annual": ["anual", "annnual"],
        "breast": ["breasst", "breest"],
    }

    words = text.split()
    out = []
    for w in words:
        stripped = w.strip(".,!?").lower()
        if stripped in typo_map and random.random() < prob:
            # preserve punctuation roughly
            punct = w[len(w.rstrip(".,!?")) :]
            w = random.choice(typo_map[stripped]) + punct
        out.append(w)

    # sometimes mess casing
    if random.random() < 0.2:
        out = [x.lower() for x in out]
    if random.random() < 0.08:
        out = [x.upper() for x in out]

    return " ".join(out)


def add_context_noise(base_text: str, attrs: Dict) -> str:
    age = attrs["age"]
    days = attrs["days_since_last_screening"]

    pieces = []

    # age variants
    if random.random() < 0.55:
        pieces.append(f"I am {age} years old")
    elif random.random() < 0.35:
        pieces.append(f"{age}yo")

    # frequency variants
    if days is not None:
        if random.random() < 0.55:
            pieces.append(f"last one was {days} days ago")
        elif random.random() < 0.35:
            pieces.append(f"it has been about {max(1, days // 30)} months")

    # light filler
    fillers = [
        "pls",
        "thanks",
        "can you help",
        "trying to schedule",
        "need an appointment",
        "quick question",
    ]
    if random.random() < 0.25:
        pieces.append(random.choice(fillers))

    random.shuffle(pieces)
    return base_text + (". " + ". ".join(pieces) if pieces else "")


# -----------------------------
# Template pools
# -----------------------------
SCREENING_BASE = [
    "I need a screening mammogram",
    "Schedule my routine screening mammogram",
    "Annual mammogram please",
    "Breast cancer screening",
    "Time for my yearly mammo",
    "Preventive mammography",
    "Breast screening appointment",
    "Boob scan",
    "Breast xray for screening",
    "Can I get a routine mammogram",
    "Need screening mammo",
]

DBT_BASE = [
    "I want a 3D screening mammogram",
    "Is 3D mammography covered for screening",
    "Need tomosynthesis screening",
    "DBT mammogram",
    "3D breast cancer screening",
    "Schedule tomosynthesis",
    "Do I qualify for tomo screening",
    "3D mammo appointment",
    "I want the 3D add-on for screening",
]


# -----------------------------
# Rules extraction (simple)
# -----------------------------
@dataclass
class Rules:
    screening_age_min: int = 40
    screening_freq_days: int = 365
    addon_rule_present: bool = True  # 77063 billed in addition to 77067


def extract_rules_from_pdf(pdf_path: Path) -> Rules:
    reader = PdfReader(str(pdf_path))
    pages = []
    for i, page in enumerate(reader.pages):
        txt = page.extract_text() or ""
        txt = re.sub(r"\s+", " ", txt).strip()
        if txt:
            pages.append(txt)

    full_text = "\n".join(pages)

    # age rule (best-effort regex)
    screening_age_min = 40
    m_age = re.search(r"average risk.*?age\s+(\d{2})", full_text, flags=re.IGNORECASE)
    if m_age:
        screening_age_min = int(m_age.group(1))

    # frequency rule (best-effort regex)
    screening_freq_days = 365
    m_freq = re.search(r"reimbursable once every\s+(\d{2,4})\s+days", full_text, flags=re.IGNORECASE)
    if m_freq:
        screening_freq_days = int(m_freq.group(1))

    addon_rule_present = bool(re.search(r"77063.*in addition to.*77067", full_text, flags=re.IGNORECASE))

    return Rules(
        screening_age_min=screening_age_min,
        screening_freq_days=screening_freq_days,
        addon_rule_present=addon_rule_present,
    )


# -----------------------------
# Case generator
# -----------------------------
SCOPE_CODES = ["77067", "77063"]


def build_checklist(code: str, rules: Rules) -> str:
    items = [
        f"Check age eligibility (>= {rules.screening_age_min} for average risk).",
        f"Check frequency limit (1 per {rules.screening_freq_days} days).",
    ]
    if code == "77063" and rules.addon_rule_present:
        items.append("Check that 77063 is billed in addition to 77067 (add-on).")
    return "\n- " + "\n- ".join(items)


def make_case(case_id: int, rules: Rules) -> Dict:
    code = random.choice(SCOPE_CODES)

    attrs = dict(
        age=random.choice([25, 30, 35, 39, 40, 45, 50, 55, 65]),
        days_since_last_screening=random.choice([30, 90, 200, 300, 364, 365, 400, 800, None]),
        avg_risk=True,
        includes_77067=True,  # relevant for 77063 add-on logic
    )

    # Sometimes violate the add-on rule for 77063
    if code == "77063" and rules.addon_rule_present and random.random() < 0.28:
        attrs["includes_77067"] = False

    # Choose base query
    base = random.choice(SCREENING_BASE if code == "77067" else DBT_BASE)

    # Add context noise + typos
    base = add_context_noise(base, attrs)
    user_query = inject_typos(base, prob=0.45)

    # Determine label
    label = "covered"

    if attrs["age"] < rules.screening_age_min:
        label = "not_covered"

    d = attrs["days_since_last_screening"]
    if d is not None and d < rules.screening_freq_days:
        label = "not_covered"

    if code == "77063" and rules.addon_rule_present and not attrs["includes_77067"]:
        label = "not_covered"

    return {
        "case_id": case_id,
        "user_query": user_query,
        "expected_cpt": code,
        "expected_label": label,
        "attrs_json": json.dumps(attrs),
        "expected_checklist": build_checklist(code, rules),
    }


# -----------------------------
# Optional matcher evaluation
# -----------------------------
def eval_matcher_top1(df: pd.DataFrame) -> float:
    pricing = PricingLookup(PROJECT_ROOT)
    matcher = Matcher(PROJECT_ROOT, pricing)

    correct = 0
    for _, r in df.iterrows():
        res = matcher.search(r["user_query"], top_k=10)
        top_code = str(res[0]["code"]) if res else None
        if top_code == str(r["expected_cpt"]):
            correct += 1

    return correct / max(1, len(df))


# -----------------------------
# Main
# -----------------------------
def main():
    random.seed(7)

    if not PDF_PATH.exists():
        raise FileNotFoundError(f"PDF not found at: {PDF_PATH}")

    rules = extract_rules_from_pdf(PDF_PATH)
    print("Rules extracted:", rules)

    # make dataset
    n = 2000  # bump this to 5000 if you want bigger
    rows = [make_case(i, rules) for i in range(1, n + 1)]
    df = pd.DataFrame(rows)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"Saved test set -> {OUT_PATH}")
    print(df.head(5))

    # optional quick match check
    try:
        acc = eval_matcher_top1(df.sample(min(200, len(df)), random_state=1))
        print(f"Matcher Top-1 accuracy on sample: {acc:.3f}")
    except Exception as e:
        print("Matcher eval skipped (error):", str(e))


if __name__ == "__main__":
    main()
