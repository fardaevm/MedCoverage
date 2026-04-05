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

from ewc_code_rules import CODE_RULES, category_for


OUT_PATH = PROJECT_ROOT / "data" / "sample_data" / "eval" / "ewc_testset_mvp_4_codes.csv"
META_PATH = PROJECT_ROOT / "data" / "processed" / "cpt_hcpcs_enriched.parquet"
LANCEDB_PATH = PROJECT_ROOT / "backend" / "data" / "sample-lancedb"

# Narrowed MVP scope: easiest four breast imaging codes
TARGET_CODES = ["77067", "77063", "77065", "77066"]

# Smaller balanced set = fewer chances for noisy misses
N_COVERED = 40
N_NOT_COVERED = 40

# Minimal expected topics for easier evaluator scoring
MVP_TOPICS = {
    "77067": ["age", "screening_interval", "risk_factors"],
    "77063": ["age", "screening_interval", "addon_screening_context"],
    "77065": ["symptoms", "abnormal_screening_result", "laterality"],
    "77066": ["symptoms", "abnormal_screening_result", "laterality"],
}

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


def build_checklist(expected_topics: List[str]) -> str:
    items = [TOPIC_TO_CHECKLIST_TEXT.get(t, f"Check topic: {t}.") for t in expected_topics]
    return "\n".join(f"- {item}" for item in items)


def make_easy_case(code: str, label: str, idx: int) -> Dict:
    variant = idx % 2

    if code == "77067":
        if label == "covered":
            cases = [
                {
                    "age": 50,
                    "days_since_last_screening": 450,
                    "high_risk": False,
                    "brca_mutation": False,
                    "family_history": False,
                    "provider_ordered": True,
                },
                {
                    "age": 35,
                    "days_since_last_screening": 500,
                    "high_risk": True,
                    "brca_mutation": False,
                    "family_history": True,
                    "provider_ordered": True,
                },
            ]
        else:
            cases = [
                {
                    "age": 32,  # clear fail: too young / not high risk
                    "days_since_last_screening": 500,
                    "high_risk": False,
                    "brca_mutation": False,
                    "family_history": False,
                    "provider_ordered": True,
                },
                {
                    "age": 36,  # clear fail: interval too short
                    "days_since_last_screening": 200,
                    "high_risk": False,
                    "brca_mutation": False,
                    "family_history": False,
                    "provider_ordered": True,
                },
            ]
        return cases[variant]

    if code == "77063":
        if label == "covered":
            cases = [
                {
                    "age": 48,
                    "days_since_last_screening": 420,
                    "high_risk": False,
                    "brca_mutation": False,
                    "family_history": False,
                    "includes_screening_context": True,
                    "provider_ordered": True,
                    "laterality": "bilateral",
                },
                {
                    "age": 34,
                    "days_since_last_screening": 500,
                    "high_risk": False,
                    "brca_mutation": True,
                    "family_history": False,
                    "includes_screening_context": True,
                    "provider_ordered": True,
                    "laterality": "bilateral",
                },
            ]
        else:
            cases = [
                {
                    "age": 48,  # clear fail: no screening context for add-on
                    "days_since_last_screening": 420,
                    "high_risk": False,
                    "brca_mutation": False,
                    "family_history": False,
                    "includes_screening_context": False,
                    "provider_ordered": True,
                    "laterality": "bilateral",
                },
                {
                    "age": 31,  # clear fail: too young / not high risk
                    "days_since_last_screening": 500,
                    "high_risk": False,
                    "brca_mutation": False,
                    "family_history": False,
                    "includes_screening_context": True,
                    "provider_ordered": True,
                    "laterality": "bilateral",
                },
            ]
        return cases[variant]

    if code == "77065":
        if label == "covered":
            cases = [
                {
                    "symptoms_present": True,
                    "abnormal_result": False,
                    "abnormal_screening_result": False,
                    "laterality": "left",
                    "provider_ordered": True,
                },
                {
                    "symptoms_present": False,
                    "abnormal_result": False,
                    "abnormal_screening_result": True,
                    "laterality": "right",
                    "provider_ordered": True,
                },
            ]
        else:
            cases = [
                {
                    "symptoms_present": False,
                    "abnormal_result": False,
                    "abnormal_screening_result": False,  # clear fail: no diagnostic indication
                    "laterality": "left",
                    "provider_ordered": True,
                },
                {
                    "symptoms_present": False,
                    "abnormal_result": False,
                    "abnormal_screening_result": False,  # clear fail: no diagnostic indication
                    "laterality": "right",
                    "provider_ordered": True,
                },
            ]
        return cases[variant]

    if code == "77066":
        if label == "covered":
            cases = [
                {
                    "symptoms_present": True,
                    "abnormal_result": False,
                    "abnormal_screening_result": False,
                    "laterality": "bilateral",
                    "provider_ordered": True,
                },
                {
                    "symptoms_present": False,
                    "abnormal_result": True,
                    "abnormal_screening_result": False,
                    "laterality": "bilateral",
                    "provider_ordered": True,
                },
            ]
        else:
            cases = [
                {
                    "symptoms_present": False,
                    "abnormal_result": False,
                    "abnormal_screening_result": False,  # clear fail: no diagnostic indication
                    "laterality": "bilateral",
                    "provider_ordered": True,
                },
                {
                    "symptoms_present": False,
                    "abnormal_result": False,
                    "abnormal_screening_result": False,  # clear fail: no diagnostic indication
                    "laterality": "bilateral",
                    "provider_ordered": True,
                },
            ]
        return cases[variant]

    raise ValueError(f"Unhandled code: {code}")


def make_easy_query(code: str, label: str, idx: int) -> str:
    templates = {
        "77067": {
            "covered": "Medi-Cal coverage for routine screening mammogram CPT 77067",
            "not_covered": "Medi-Cal coverage for routine screening mammogram CPT 77067 not eligible case",
        },
        "77063": {
            "covered": "Medi-Cal coverage for 3D screening mammography CPT 77063",
            "not_covered": "Medi-Cal coverage for 3D screening mammography CPT 77063 not eligible case",
        },
        "77065": {
            "covered": "Medi-Cal coverage for unilateral diagnostic mammogram CPT 77065",
            "not_covered": "Medi-Cal coverage for unilateral diagnostic mammogram CPT 77065 not eligible case",
        },
        "77066": {
            "covered": "Medi-Cal coverage for bilateral diagnostic mammogram CPT 77066",
            "not_covered": "Medi-Cal coverage for bilateral diagnostic mammogram CPT 77066 not eligible case",
        },
    }
    return templates[code][label]


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
        expected_topics = MVP_TOPICS[code]

        meta = metadata.get(code, {})
        title = meta.get("title", "")
        description = meta.get("description", "")
        context = policy_context.get(code, "")

        for label, n_rows in [("covered", N_COVERED), ("not_covered", N_NOT_COVERED)]:
            for i in range(n_rows):
                attrs = make_easy_case(code, label, i)
                query = make_easy_query(code, label, i)

                rows.append(
                    {
                        "case_id": case_id,
                        "expected_cpt": code,
                        "category": category,
                        "user_query": query,
                        "verbosity": "easy",
                        "title": title,
                        "description": description,
                        "policy_context": context,
                        "expected_topics_json": json.dumps(expected_topics),
                        "expected_checklist": build_checklist(expected_topics),
                        "attrs_json": json.dumps(attrs),
                        "expected_label": label,
                    }
                )
                case_id += 1

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate easier MVP EWC test set for breast imaging scope"
    )
    parser.add_argument("--outfile", type=str, default=str(OUT_PATH))
    args = parser.parse_args()

    outfile = Path(args.outfile)

    metadata = load_code_metadata(META_PATH)
    policy_context = load_policy_context()

    out_df = build_testset(metadata=metadata, policy_context=policy_context)

    outfile.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(outfile, index=False)

    print("\n---- GENERATED MVP EASY TEST SET ----")
    print(f"Rows: {len(out_df)}")
    print(f"Codes: {out_df['expected_cpt'].nunique() if len(out_df) else 0}")
    print(
        f"Rows per code: "
        f"{len(out_df) // max(out_df['expected_cpt'].nunique(), 1) if len(out_df) else 0}"
    )
    print(f"Saved -> {outfile}")

    if len(out_df):
        print("\nExpected label counts:")
        print(out_df["expected_label"].value_counts(dropna=False))

        print("\nBy CPT / label:")
        print(
            out_df.groupby(["expected_cpt", "expected_label"])
            .size()
            .unstack(fill_value=0)
            .sort_index()
        )

        print("\nSample:")
        print(
            out_df[
                ["expected_cpt", "category", "user_query", "expected_label", "attrs_json"]
            ].head(12).to_string(index=False)
        )


if __name__ == "__main__":
    main()