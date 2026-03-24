from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List

import lancedb
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "eval"))

from ewc_code_rules import CODE_RULES, expected_topics_for, category_for


OUT_PATH = PROJECT_ROOT / "data" / "sample_data" / "eval" / "ewc_testset_all_codes.csv"
META_PATH = PROJECT_ROOT / "data" / "processed" / "cpt_hcpcs_enriched.parquet"
LANCEDB_PATH = PROJECT_ROOT / "backend" / "data" / "sample-lancedb"

TARGET_CODES = [
    "77046", "77047", "77048", "77049",
    "77061", "77062", "77063", "77065", "77066", "77067",
]

N_PER_CODE = 100
N_COVERED = 50
N_NOT_COVERED = 50


# -------------------------------------------------------------------
# Checklist text builder
# -------------------------------------------------------------------
TOPIC_TO_CHECKLIST_TEXT = {
    "age": "Check age eligibility.",
    "screening_interval": "Check timing since prior screening or interval requirement.",
    "risk_factors": "Check relevant family history, BRCA status, or high-risk criteria.",
    "addon_screening_context": "Check that the procedure is tied to the required screening/add-on context.",
    "symptoms": "Check symptoms or abnormal clinical findings.",
    "abnormal_screening_result": "Check whether there was an abnormal prior screening or follow-up indication.",
    "medical_necessity": "Check provider order, referral, or medical necessity indication.",
    "laterality": "Check laterality (left, right, unilateral, bilateral, one/both breasts).",
    "contrast": "Check whether contrast is required or whether with/without contrast matters.",
    "image_guidance": "Check imaging guidance modality if relevant.",
}


# -------------------------------------------------------------------
# Utility loaders
# -------------------------------------------------------------------
def load_code_metadata(meta_path: Path) -> Dict[str, Dict]:
    if not meta_path.exists():
        return {}

    df = pd.read_parquet(meta_path)
    out: Dict[str, Dict] = {}

    for _, row in df.iterrows():
        code = str(row.get("code", "")).strip()
        if not code:
            continue

        clean_title = str(row.get("clean_title") or "").strip()
        if not clean_title or clean_title.startswith("["):
            clean_title = str(row.get("title") or "").strip().strip('"').rstrip(".")

        description = str(row.get("description") or "").strip().strip('"')

        out[code] = {
            "title": clean_title,
            "description": description,
        }

    return out


def load_policy_context() -> Dict[str, str]:
    db_path = LANCEDB_PATH
    if not db_path.exists():
        return {}

    db = lancedb.connect(str(db_path))
    if not db.table_names():
        return {}

    df = db.open_table(db.table_names()[0]).to_pandas()
    text_col = next((c for c in df.columns if c in ("content", "text")), None)
    if not text_col:
        return {}

    context_map: Dict[str, str] = {}

    for code in TARGET_CODES:
        matches = df[df[text_col].astype(str).str.contains(code, na=False)]
        if len(matches) == 0:
            context_map[code] = ""
            continue

        snippets = []
        for _, row in matches.head(2).iterrows():
            text = str(row[text_col]).strip()
            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if code in s]
            if sentences:
                snippets.append(" ".join(sentences[:2])[:400])
            else:
                snippets.append(text[:400])

        context_map[code] = " | ".join(snippets)

    return context_map


# -------------------------------------------------------------------
# Formatting helpers
# -------------------------------------------------------------------
def build_checklist(expected_topics: List[str]) -> str:
    items = [TOPIC_TO_CHECKLIST_TEXT.get(t, f"Check topic: {t}.") for t in expected_topics]
    return "\n".join(f"- {item}" for item in items)


# -------------------------------------------------------------------
# Easier deterministic cases
# 50 covered + 50 not_covered per CPT
# -------------------------------------------------------------------
def make_easy_case(code: str, label: str, idx: int) -> Dict:
    code = str(code).strip()
    variant = idx % 4

    # ------------------------------------------------------------
    # Screening mammography
    # ------------------------------------------------------------
    if code == "77067":
        covered_cases = [
            {
                "age": 45,
                "days_since_last_screening": 400,
                "high_risk": False,
                "brca_mutation": False,
                "family_history": False,
                "provider_ordered": True,
            },
            {
                "age": 50,
                "days_since_last_screening": 365,
                "high_risk": False,
                "brca_mutation": False,
                "family_history": False,
                "provider_ordered": True,
            },
            {
                "age": 39,
                "days_since_last_screening": 400,
                "high_risk": True,
                "brca_mutation": False,
                "family_history": True,
                "provider_ordered": True,
            },
            {
                "age": 35,
                "days_since_last_screening": 500,
                "high_risk": False,
                "brca_mutation": True,
                "family_history": False,
                "provider_ordered": True,
            },
        ]
        not_covered_cases = [
            {
                "age": 35,
                "days_since_last_screening": 180,
                "high_risk": False,
                "brca_mutation": False,
                "family_history": False,
                "provider_ordered": True,
            },
            {
                "age": 39,
                "days_since_last_screening": 200,
                "high_risk": False,
                "brca_mutation": False,
                "family_history": False,
                "provider_ordered": True,
            },
            {
                "age": 30,
                "days_since_last_screening": 364,
                "high_risk": False,
                "brca_mutation": False,
                "family_history": False,
                "provider_ordered": True,
            },
            {
                "age": 38,
                "days_since_last_screening": 300,
                "high_risk": False,
                "brca_mutation": False,
                "family_history": False,
                "provider_ordered": False,
            },
        ]
        return covered_cases[variant] if label == "covered" else not_covered_cases[variant]

    if code == "77063":
        covered_cases = [
            {
                "age": 45,
                "days_since_last_screening": 400,
                "high_risk": False,
                "brca_mutation": False,
                "family_history": False,
                "includes_screening_context": True,
                "provider_ordered": True,
                "laterality": "bilateral",
            },
            {
                "age": 40,
                "days_since_last_screening": 365,
                "high_risk": False,
                "brca_mutation": False,
                "family_history": False,
                "includes_screening_context": True,
                "provider_ordered": True,
                "laterality": "bilateral",
            },
            {
                "age": 39,
                "days_since_last_screening": 400,
                "high_risk": True,
                "brca_mutation": False,
                "family_history": True,
                "includes_screening_context": True,
                "provider_ordered": True,
                "laterality": "bilateral",
            },
            {
                "age": 35,
                "days_since_last_screening": 500,
                "high_risk": False,
                "brca_mutation": True,
                "family_history": False,
                "includes_screening_context": True,
                "provider_ordered": True,
                "laterality": "bilateral",
            },
        ]
        not_covered_cases = [
            {
                "age": 35,
                "days_since_last_screening": 180,
                "high_risk": False,
                "brca_mutation": False,
                "family_history": False,
                "includes_screening_context": False,
                "provider_ordered": True,
                "laterality": "bilateral",
            },
            {
                "age": 39,
                "days_since_last_screening": 200,
                "high_risk": False,
                "brca_mutation": False,
                "family_history": False,
                "includes_screening_context": True,
                "provider_ordered": True,
                "laterality": "bilateral",
            },
            {
                "age": 30,
                "days_since_last_screening": 364,
                "high_risk": False,
                "brca_mutation": False,
                "family_history": False,
                "includes_screening_context": True,
                "provider_ordered": True,
                "laterality": "bilateral",
            },
            {
                "age": 38,
                "days_since_last_screening": 300,
                "high_risk": False,
                "brca_mutation": False,
                "family_history": False,
                "includes_screening_context": False,
                "provider_ordered": False,
                "laterality": "bilateral",
            },
        ]
        return covered_cases[variant] if label == "covered" else not_covered_cases[variant]

    # ------------------------------------------------------------
    # Diagnostic imaging
    # ------------------------------------------------------------
    if code == "77065":
        covered_cases = [
            {
                "symptoms_present": True,
                "abnormal_result": False,
                "abnormal_screening_result": False,
                "laterality": "left",
                "provider_ordered": True,
                "includes_screening_context": False,
            },
            {
                "symptoms_present": False,
                "abnormal_result": True,
                "abnormal_screening_result": False,
                "laterality": "right",
                "provider_ordered": True,
                "includes_screening_context": False,
            },
            {
                "symptoms_present": False,
                "abnormal_result": False,
                "abnormal_screening_result": True,
                "laterality": "left",
                "provider_ordered": True,
                "includes_screening_context": False,
            },
            {
                "symptoms_present": True,
                "abnormal_result": True,
                "abnormal_screening_result": False,
                "laterality": "right",
                "provider_ordered": True,
                "includes_screening_context": False,
            },
        ]
        not_covered_cases = [
            {
                "symptoms_present": False,
                "abnormal_result": False,
                "abnormal_screening_result": False,
                "laterality": "left",
                "provider_ordered": False,
                "includes_screening_context": False,
            },
            {
                "symptoms_present": False,
                "abnormal_result": False,
                "abnormal_screening_result": False,
                "laterality": "right",
                "provider_ordered": True,
                "includes_screening_context": False,
            },
            {
                "symptoms_present": False,
                "abnormal_result": False,
                "abnormal_screening_result": False,
                "laterality": "unknown",
                "provider_ordered": False,
                "includes_screening_context": False,
            },
            {
                "symptoms_present": False,
                "abnormal_result": False,
                "abnormal_screening_result": False,
                "laterality": "left",
                "provider_ordered": False,
                "includes_screening_context": False,
            },
        ]
        return covered_cases[variant] if label == "covered" else not_covered_cases[variant]

    if code == "77066":
        covered_cases = [
            {
                "symptoms_present": True,
                "abnormal_result": False,
                "abnormal_screening_result": False,
                "laterality": "bilateral",
                "provider_ordered": True,
                "includes_screening_context": False,
            },
            {
                "symptoms_present": False,
                "abnormal_result": True,
                "abnormal_screening_result": False,
                "laterality": "bilateral",
                "provider_ordered": True,
                "includes_screening_context": False,
            },
            {
                "symptoms_present": False,
                "abnormal_result": False,
                "abnormal_screening_result": True,
                "laterality": "bilateral",
                "provider_ordered": True,
                "includes_screening_context": False,
            },
            {
                "symptoms_present": True,
                "abnormal_result": True,
                "abnormal_screening_result": False,
                "laterality": "bilateral",
                "provider_ordered": True,
                "includes_screening_context": False,
            },
        ]
        not_covered_cases = [
            {
                "symptoms_present": False,
                "abnormal_result": False,
                "abnormal_screening_result": False,
                "laterality": "bilateral",
                "provider_ordered": False,
                "includes_screening_context": False,
            },
            {
                "symptoms_present": False,
                "abnormal_result": False,
                "abnormal_screening_result": False,
                "laterality": "bilateral",
                "provider_ordered": True,
                "includes_screening_context": False,
            },
            {
                "symptoms_present": False,
                "abnormal_result": False,
                "abnormal_screening_result": False,
                "laterality": "unknown",
                "provider_ordered": False,
                "includes_screening_context": False,
            },
            {
                "symptoms_present": False,
                "abnormal_result": False,
                "abnormal_screening_result": False,
                "laterality": "bilateral",
                "provider_ordered": False,
                "includes_screening_context": False,
            },
        ]
        return covered_cases[variant] if label == "covered" else not_covered_cases[variant]

    if code == "77061":
        covered_cases = [
            {
                "symptoms_present": True,
                "abnormal_result": False,
                "abnormal_screening_result": False,
                "laterality": "left",
                "provider_ordered": True,
                "includes_screening_context": True,
            },
            {
                "symptoms_present": False,
                "abnormal_result": True,
                "abnormal_screening_result": False,
                "laterality": "right",
                "provider_ordered": True,
                "includes_screening_context": True,
            },
            {
                "symptoms_present": False,
                "abnormal_result": False,
                "abnormal_screening_result": True,
                "laterality": "left",
                "provider_ordered": True,
                "includes_screening_context": True,
            },
            {
                "symptoms_present": True,
                "abnormal_result": True,
                "abnormal_screening_result": False,
                "laterality": "right",
                "provider_ordered": True,
                "includes_screening_context": True,
            },
        ]
        not_covered_cases = [
            {
                "symptoms_present": False,
                "abnormal_result": False,
                "abnormal_screening_result": False,
                "laterality": "left",
                "provider_ordered": False,
                "includes_screening_context": False,
            },
            {
                "symptoms_present": False,
                "abnormal_result": False,
                "abnormal_screening_result": False,
                "laterality": "right",
                "provider_ordered": True,
                "includes_screening_context": False,
            },
            {
                "symptoms_present": False,
                "abnormal_result": False,
                "abnormal_screening_result": False,
                "laterality": "unknown",
                "provider_ordered": False,
                "includes_screening_context": False,
            },
            {
                "symptoms_present": False,
                "abnormal_result": False,
                "abnormal_screening_result": False,
                "laterality": "left",
                "provider_ordered": False,
                "includes_screening_context": True,
            },
        ]
        return covered_cases[variant] if label == "covered" else not_covered_cases[variant]

    if code == "77062":
        covered_cases = [
            {
                "symptoms_present": True,
                "abnormal_result": False,
                "abnormal_screening_result": False,
                "laterality": "bilateral",
                "provider_ordered": True,
                "includes_screening_context": True,
            },
            {
                "symptoms_present": False,
                "abnormal_result": True,
                "abnormal_screening_result": False,
                "laterality": "bilateral",
                "provider_ordered": True,
                "includes_screening_context": True,
            },
            {
                "symptoms_present": False,
                "abnormal_result": False,
                "abnormal_screening_result": True,
                "laterality": "bilateral",
                "provider_ordered": True,
                "includes_screening_context": True,
            },
            {
                "symptoms_present": True,
                "abnormal_result": True,
                "abnormal_screening_result": False,
                "laterality": "bilateral",
                "provider_ordered": True,
                "includes_screening_context": True,
            },
        ]
        not_covered_cases = [
            {
                "symptoms_present": False,
                "abnormal_result": False,
                "abnormal_screening_result": False,
                "laterality": "bilateral",
                "provider_ordered": False,
                "includes_screening_context": False,
            },
            {
                "symptoms_present": False,
                "abnormal_result": False,
                "abnormal_screening_result": False,
                "laterality": "bilateral",
                "provider_ordered": True,
                "includes_screening_context": False,
            },
            {
                "symptoms_present": False,
                "abnormal_result": False,
                "abnormal_screening_result": False,
                "laterality": "unknown",
                "provider_ordered": False,
                "includes_screening_context": False,
            },
            {
                "symptoms_present": False,
                "abnormal_result": False,
                "abnormal_screening_result": False,
                "laterality": "bilateral",
                "provider_ordered": False,
                "includes_screening_context": True,
            },
        ]
        return covered_cases[variant] if label == "covered" else not_covered_cases[variant]

    # ------------------------------------------------------------
    # Breast MRI
    # ------------------------------------------------------------
    if code == "77046":
        covered_cases = [
            {
                "high_risk": True,
                "family_history": True,
                "brca_mutation": False,
                "symptoms_present": False,
                "abnormal_result": False,
                "provider_ordered": True,
                "contrast_needed": False,
                "laterality": "left",
            },
            {
                "high_risk": False,
                "family_history": False,
                "brca_mutation": True,
                "symptoms_present": False,
                "abnormal_result": False,
                "provider_ordered": True,
                "contrast_needed": False,
                "laterality": "right",
            },
            {
                "high_risk": False,
                "family_history": False,
                "brca_mutation": False,
                "symptoms_present": True,
                "abnormal_result": False,
                "provider_ordered": True,
                "contrast_needed": False,
                "laterality": "left",
            },
            {
                "high_risk": False,
                "family_history": True,
                "brca_mutation": False,
                "symptoms_present": False,
                "abnormal_result": True,
                "provider_ordered": True,
                "contrast_needed": False,
                "laterality": "right",
            },
        ]
        not_covered_cases = [
            {
                "high_risk": False,
                "family_history": False,
                "brca_mutation": False,
                "symptoms_present": False,
                "abnormal_result": False,
                "provider_ordered": False,
                "contrast_needed": False,
                "laterality": "left",
            },
            {
                "high_risk": False,
                "family_history": False,
                "brca_mutation": False,
                "symptoms_present": False,
                "abnormal_result": False,
                "provider_ordered": True,
                "contrast_needed": False,
                "laterality": "left",
            },
            {
                "high_risk": False,
                "family_history": False,
                "brca_mutation": False,
                "symptoms_present": False,
                "abnormal_result": False,
                "provider_ordered": False,
                "contrast_needed": False,
                "laterality": "right",
            },
            {
                "high_risk": False,
                "family_history": False,
                "brca_mutation": False,
                "symptoms_present": False,
                "abnormal_result": False,
                "provider_ordered": False,
                "contrast_needed": False,
                "laterality": "left",
            },
        ]
        return covered_cases[variant] if label == "covered" else not_covered_cases[variant]

    if code == "77047":
        covered_cases = [
            {
                "high_risk": True,
                "family_history": True,
                "brca_mutation": False,
                "symptoms_present": False,
                "abnormal_result": False,
                "provider_ordered": True,
                "contrast_needed": False,
                "laterality": "bilateral",
            },
            {
                "high_risk": False,
                "family_history": False,
                "brca_mutation": True,
                "symptoms_present": False,
                "abnormal_result": False,
                "provider_ordered": True,
                "contrast_needed": False,
                "laterality": "bilateral",
            },
            {
                "high_risk": False,
                "family_history": False,
                "brca_mutation": False,
                "symptoms_present": True,
                "abnormal_result": False,
                "provider_ordered": True,
                "contrast_needed": False,
                "laterality": "bilateral",
            },
            {
                "high_risk": False,
                "family_history": True,
                "brca_mutation": False,
                "symptoms_present": False,
                "abnormal_result": True,
                "provider_ordered": True,
                "contrast_needed": False,
                "laterality": "bilateral",
            },
        ]
        not_covered_cases = [
            {
                "high_risk": False,
                "family_history": False,
                "brca_mutation": False,
                "symptoms_present": False,
                "abnormal_result": False,
                "provider_ordered": False,
                "contrast_needed": False,
                "laterality": "bilateral",
            },
            {
                "high_risk": False,
                "family_history": False,
                "brca_mutation": False,
                "symptoms_present": False,
                "abnormal_result": False,
                "provider_ordered": True,
                "contrast_needed": False,
                "laterality": "bilateral",
            },
            {
                "high_risk": False,
                "family_history": False,
                "brca_mutation": False,
                "symptoms_present": False,
                "abnormal_result": False,
                "provider_ordered": False,
                "contrast_needed": False,
                "laterality": "bilateral",
            },
            {
                "high_risk": False,
                "family_history": False,
                "brca_mutation": False,
                "symptoms_present": False,
                "abnormal_result": False,
                "provider_ordered": False,
                "contrast_needed": False,
                "laterality": "bilateral",
            },
        ]
        return covered_cases[variant] if label == "covered" else not_covered_cases[variant]

    if code == "77048":
        covered_cases = [
            {
                "high_risk": True,
                "family_history": True,
                "brca_mutation": False,
                "symptoms_present": False,
                "abnormal_result": False,
                "provider_ordered": True,
                "contrast_needed": True,
                "laterality": "left",
            },
            {
                "high_risk": False,
                "family_history": False,
                "brca_mutation": True,
                "symptoms_present": False,
                "abnormal_result": False,
                "provider_ordered": True,
                "contrast_needed": True,
                "laterality": "right",
            },
            {
                "high_risk": False,
                "family_history": False,
                "brca_mutation": False,
                "symptoms_present": True,
                "abnormal_result": False,
                "provider_ordered": True,
                "contrast_needed": True,
                "laterality": "left",
            },
            {
                "high_risk": False,
                "family_history": True,
                "brca_mutation": False,
                "symptoms_present": False,
                "abnormal_result": True,
                "provider_ordered": True,
                "contrast_needed": True,
                "laterality": "right",
            },
        ]
        not_covered_cases = [
            {
                "high_risk": False,
                "family_history": False,
                "brca_mutation": False,
                "symptoms_present": False,
                "abnormal_result": False,
                "provider_ordered": False,
                "contrast_needed": True,
                "laterality": "left",
            },
            {
                "high_risk": False,
                "family_history": False,
                "brca_mutation": False,
                "symptoms_present": False,
                "abnormal_result": False,
                "provider_ordered": True,
                "contrast_needed": True,
                "laterality": "left",
            },
            {
                "high_risk": False,
                "family_history": False,
                "brca_mutation": False,
                "symptoms_present": False,
                "abnormal_result": False,
                "provider_ordered": False,
                "contrast_needed": True,
                "laterality": "right",
            },
            {
                "high_risk": False,
                "family_history": False,
                "brca_mutation": False,
                "symptoms_present": False,
                "abnormal_result": False,
                "provider_ordered": False,
                "contrast_needed": True,
                "laterality": "left",
            },
        ]
        return covered_cases[variant] if label == "covered" else not_covered_cases[variant]

    if code == "77049":
        covered_cases = [
            {
                "high_risk": True,
                "family_history": True,
                "brca_mutation": False,
                "symptoms_present": False,
                "abnormal_result": False,
                "provider_ordered": True,
                "contrast_needed": True,
                "laterality": "bilateral",
            },
            {
                "high_risk": False,
                "family_history": False,
                "brca_mutation": True,
                "symptoms_present": False,
                "abnormal_result": False,
                "provider_ordered": True,
                "contrast_needed": True,
                "laterality": "bilateral",
            },
            {
                "high_risk": False,
                "family_history": False,
                "brca_mutation": False,
                "symptoms_present": True,
                "abnormal_result": False,
                "provider_ordered": True,
                "contrast_needed": True,
                "laterality": "bilateral",
            },
            {
                "high_risk": False,
                "family_history": True,
                "brca_mutation": False,
                "symptoms_present": False,
                "abnormal_result": True,
                "provider_ordered": True,
                "contrast_needed": True,
                "laterality": "bilateral",
            },
        ]
        not_covered_cases = [
            {
                "high_risk": False,
                "family_history": False,
                "brca_mutation": False,
                "symptoms_present": False,
                "abnormal_result": False,
                "provider_ordered": False,
                "contrast_needed": True,
                "laterality": "bilateral",
            },
            {
                "high_risk": False,
                "family_history": False,
                "brca_mutation": False,
                "symptoms_present": False,
                "abnormal_result": False,
                "provider_ordered": True,
                "contrast_needed": True,
                "laterality": "bilateral",
            },
            {
                "high_risk": False,
                "family_history": False,
                "brca_mutation": False,
                "symptoms_present": False,
                "abnormal_result": False,
                "provider_ordered": False,
                "contrast_needed": True,
                "laterality": "bilateral",
            },
            {
                "high_risk": False,
                "family_history": False,
                "brca_mutation": False,
                "symptoms_present": False,
                "abnormal_result": False,
                "provider_ordered": False,
                "contrast_needed": True,
                "laterality": "bilateral",
            },
        ]
        return covered_cases[variant] if label == "covered" else not_covered_cases[variant]

    raise ValueError(f"Unhandled code: {code}")


def make_easy_query(code: str, label: str, idx: int) -> str:
    query_templates = {
        "77067": {
            "covered": [
                "Does Medi-Cal cover a routine screening mammogram?",
                "Can I get a screening mammogram covered by Medi-Cal?",
                "Is a screening mammogram covered by Medi-Cal?",
                "Would Medi-Cal pay for a routine mammogram?",
            ],
            "not_covered": [
                "Would Medi-Cal cover a screening mammogram for me right now?",
                "Am I covered for a routine mammogram right now?",
                "Can Medi-Cal pay for my screening mammogram now?",
                "Is my screening mammogram covered right now?",
            ],
        },
        "77063": {
            "covered": [
                "Does Medi-Cal cover 3D screening mammography?",
                "Is screening digital breast tomosynthesis covered?",
                "Can I get 3D screening mammography covered?",
                "Would Medi-Cal pay for 3D screening breast imaging?",
            ],
            "not_covered": [
                "Would Medi-Cal cover 3D screening breast imaging for me now?",
                "Am I covered for 3D screening mammography right now?",
                "Can Medi-Cal pay for 3D screening mammography now?",
                "Is my 3D screening breast imaging covered right now?",
            ],
        },
        "77065": {
            "covered": [
                "Does Medi-Cal cover a diagnostic mammogram for one breast?",
                "Can I get a one-breast diagnostic mammogram covered?",
                "Is a unilateral diagnostic mammogram covered?",
                "Would Medi-Cal pay for a diagnostic mammogram of one breast?",
            ],
            "not_covered": [
                "Would Medi-Cal cover a diagnostic mammogram for one breast?",
                "Am I covered for a unilateral diagnostic mammogram?",
                "Can Medi-Cal pay for a one-breast diagnostic mammogram?",
                "Is my one-breast diagnostic mammogram covered?",
            ],
        },
        "77066": {
            "covered": [
                "Does Medi-Cal cover a diagnostic mammogram for both breasts?",
                "Can I get a bilateral diagnostic mammogram covered?",
                "Is a bilateral diagnostic mammogram covered?",
                "Would Medi-Cal pay for a diagnostic mammogram of both breasts?",
            ],
            "not_covered": [
                "Would Medi-Cal cover a bilateral diagnostic mammogram?",
                "Am I covered for a diagnostic mammogram of both breasts?",
                "Can Medi-Cal pay for both-breast diagnostic mammography?",
                "Is my bilateral diagnostic mammogram covered?",
            ],
        },
        "77061": {
            "covered": [
                "Does Medi-Cal cover 3D diagnostic breast imaging for one breast?",
                "Is unilateral diagnostic breast tomosynthesis covered?",
                "Can I get one-breast 3D diagnostic imaging covered?",
                "Would Medi-Cal pay for unilateral diagnostic tomosynthesis?",
            ],
            "not_covered": [
                "Would Medi-Cal cover unilateral 3D diagnostic breast imaging?",
                "Am I covered for one-breast diagnostic tomosynthesis?",
                "Can Medi-Cal pay for one-breast 3D diagnostic imaging?",
                "Is my unilateral diagnostic tomosynthesis covered?",
            ],
        },
        "77062": {
            "covered": [
                "Does Medi-Cal cover 3D diagnostic breast imaging for both breasts?",
                "Is bilateral diagnostic breast tomosynthesis covered?",
                "Can I get both-breast 3D diagnostic imaging covered?",
                "Would Medi-Cal pay for bilateral diagnostic tomosynthesis?",
            ],
            "not_covered": [
                "Would Medi-Cal cover bilateral 3D diagnostic breast imaging?",
                "Am I covered for both-breast diagnostic tomosynthesis?",
                "Can Medi-Cal pay for bilateral 3D diagnostic imaging?",
                "Is my bilateral diagnostic tomosynthesis covered?",
            ],
        },
        "77046": {
            "covered": [
                "Does Medi-Cal cover a breast MRI without contrast for one breast?",
                "Is unilateral breast MRI without contrast covered?",
                "Can I get a one-breast MRI without contrast covered?",
                "Would Medi-Cal pay for unilateral breast MRI without contrast?",
            ],
            "not_covered": [
                "Would Medi-Cal cover a unilateral breast MRI without contrast?",
                "Am I covered for one-breast MRI without contrast?",
                "Can Medi-Cal pay for unilateral breast MRI without contrast?",
                "Is my unilateral MRI without contrast covered?",
            ],
        },
        "77047": {
            "covered": [
                "Does Medi-Cal cover a breast MRI without contrast for both breasts?",
                "Is bilateral breast MRI without contrast covered?",
                "Can I get a both-breast MRI without contrast covered?",
                "Would Medi-Cal pay for bilateral breast MRI without contrast?",
            ],
            "not_covered": [
                "Would Medi-Cal cover a bilateral breast MRI without contrast?",
                "Am I covered for both-breast MRI without contrast?",
                "Can Medi-Cal pay for bilateral breast MRI without contrast?",
                "Is my bilateral MRI without contrast covered?",
            ],
        },
        "77048": {
            "covered": [
                "Does Medi-Cal cover a breast MRI with and without contrast for one breast?",
                "Is unilateral breast MRI with and without contrast covered?",
                "Can I get a one-breast MRI with and without contrast covered?",
                "Would Medi-Cal pay for unilateral breast MRI with and without contrast?",
            ],
            "not_covered": [
                "Would Medi-Cal cover unilateral breast MRI with and without contrast?",
                "Am I covered for one-breast MRI with and without contrast?",
                "Can Medi-Cal pay for unilateral MRI with and without contrast?",
                "Is my unilateral MRI with and without contrast covered?",
            ],
        },
        "77049": {
            "covered": [
                "Does Medi-Cal cover a breast MRI with and without contrast for both breasts?",
                "Is bilateral breast MRI with and without contrast covered?",
                "Can I get a both-breast MRI with and without contrast covered?",
                "Would Medi-Cal pay for bilateral breast MRI with and without contrast?",
            ],
            "not_covered": [
                "Would Medi-Cal cover bilateral breast MRI with and without contrast?",
                "Am I covered for both-breast MRI with and without contrast?",
                "Can Medi-Cal pay for bilateral MRI with and without contrast?",
                "Is my bilateral MRI with and without contrast covered?",
            ],
        },
    }

    pool = query_templates[code][label]
    return pool[idx % len(pool)]


# -------------------------------------------------------------------
# Main builder
# -------------------------------------------------------------------
def build_testset(
    metadata: Dict[str, Dict],
    policy_context: Dict[str, str],
) -> pd.DataFrame:
    rows = []
    case_id = 1

    for code in TARGET_CODES:
        if code not in CODE_RULES:
            continue

        category = category_for(code)
        expected_topics = expected_topics_for(code)

        meta = metadata.get(code, {})
        title = meta.get("title", "")
        description = meta.get("description", "")
        context = policy_context.get(code, "")

        for label, n_rows in [("covered", N_COVERED), ("not_covered", N_NOT_COVERED)]:
            for i in range(n_rows):
                attrs = make_easy_case(code, label, i)
                query = make_easy_query(code, label, i)

                rows.append({
                    "case_id": case_id,
                    "expected_cpt": code,
                    "category": category,
                    "user_query": query,
                    "verbosity": "direct",
                    "title": title,
                    "description": description,
                    "policy_context": context,
                    "expected_topics_json": json.dumps(expected_topics),
                    "expected_checklist": build_checklist(expected_topics),
                    "attrs_json": json.dumps(attrs),
                    "expected_label": label,
                })
                case_id += 1

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate narrowed-scope EWC breast imaging eval test set")
    parser.add_argument("--outfile", type=str, default=str(OUT_PATH))
    args = parser.parse_args()

    outfile = Path(args.outfile)

    metadata = load_code_metadata(META_PATH)
    policy_context = load_policy_context()

    out_df = build_testset(metadata=metadata, policy_context=policy_context)

    outfile.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(outfile, index=False)

    print("\n---- GENERATED TEST SET ----")
    print(f"Rows: {len(out_df)}")
    print(f"Codes: {out_df['expected_cpt'].nunique() if len(out_df) else 0}")
    print(f"Rows per code target: {N_PER_CODE}")
    print(f"Saved -> {outfile}")

    if len(out_df):
        print("\nExpected label counts:")
        print(out_df["expected_label"].value_counts(dropna=False).sort_index())

        print("\nRows by CPT:")
        print(out_df["expected_cpt"].value_counts().sort_index())

        print("\nSample:")
        print(
            out_df[
                ["expected_cpt", "category", "verbosity", "user_query", "expected_label"]
            ].head(20).to_string(index=False)
        )


if __name__ == "__main__":
    main()