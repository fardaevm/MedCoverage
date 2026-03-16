from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Dict, List

import lancedb
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "eval"))

from ewc_code_rules import CODE_RULES, expected_topics_for, category_for


IN_PATH = PROJECT_ROOT / "data" / "sample_data" / "eval" / "policy_eval_set.csv"
OUT_PATH = PROJECT_ROOT / "data" / "sample_data" / "eval" / "ewc_testset_all_codes.csv"
META_PATH = PROJECT_ROOT / "data" / "processed" / "cpt_hcpcs_enriched.parquet"
LANCEDB_PATH = PROJECT_ROOT / "backend" / "data" / "sample-lancedb"


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
    "lesion_target": "Check target lesion/site/localization details.",
    "pregnancy_status": "Check pregnancy status if clinically required.",
    "bleeding_history": "Check abnormal bleeding or related gynecologic history.",
    "specimen_context": "Check specimen/sample context.",
    "reflex_testing": "Check whether reflex/genotype/additional testing context applies.",
    "visit_reason": "Check reason for the visit or encounter.",
    "new_vs_established": "Check whether this is a new or established patient encounter.",
    "pathology_linkage": "Check linkage to a biopsy, specimen, or source procedure.",
    "functional_status": "Check functional limitation or exam reason.",
}


# -------------------------------------------------------------------
# Synthetic attrs for eval generation
# These support conversation simulation + gold label generation.
# -------------------------------------------------------------------
CATEGORY_ATTRS = {
    "breast_screening_mammo": {
        "age": [25, 30, 35, 39, 40, 45, 50, 55, 65],
        "days_since_last_screening": [30, 90, 200, 300, 364, 365, 400, 800, None],
        "high_risk": [True, False],
        "brca_mutation": [True, False],
        "family_history": [True, False],
        "provider_ordered": [True, False],
    },
    "breast_screening_addon": {
        "age": [25, 30, 35, 39, 40, 45, 50, 55, 65],
        "days_since_last_screening": [30, 90, 200, 300, 364, 365, 400, 800, None],
        "high_risk": [True, False],
        "brca_mutation": [True, False],
        "family_history": [True, False],
        "includes_screening_context": [True, False],
        "provider_ordered": [True, False],
        "laterality": ["left", "right", "bilateral"],
    },
    "breast_diagnostic_imaging": {
        "symptoms_present": [True, False],
        "abnormal_result": [True, False],
        "abnormal_screening_result": [True, False],
        "laterality": ["left", "right", "bilateral", "unknown"],
        "provider_ordered": [True, False],
        "includes_screening_context": [True, False],
    },
    "breast_mri": {
        "high_risk": [True, False],
        "family_history": [True, False],
        "brca_mutation": [True, False],
        "symptoms_present": [True, False],
        "abnormal_result": [True, False],
        "provider_ordered": [True, False],
        "contrast_needed": [True, False],
        "laterality": ["left", "right", "bilateral", "unknown"],
    },
    "breast_biopsy_localization": {
        "abnormal_result": [True, False],
        "provider_ordered": [True, False],
        "guidance_modality": ["ultrasound", "stereotactic", "mri", "unknown"],
        "laterality": ["left", "right", "bilateral", "unknown"],
    },
    "breast_pathology_support": {
        "linked_biopsy": [True, False],
        "specimen_available": [True, False],
    },
    "cervical_colposcopy_biopsy": {
        "abnormal_pap": [True, False],
        "abnormal_hpv": [True, False],
        "pregnant": [True, False],
        "provider_ordered": [True, False],
    },
    "endometrial_sampling": {
        "abnormal_bleeding": [True, False],
        "postmenopausal": [True, False],
        "pregnant": [True, False],
        "provider_ordered": [True, False],
    },
    "hpv_testing": {
        "abnormal_pap": [True, False],
        "age": [21, 25, 30, 35, 40, 50, 65],
        "reflex_testing": [True, False],
        "provider_ordered": [True, False],
    },
    "cytology_pathology_lab": {
        "specimen_available": [True, False],
        "linked_screening_abnormality": [True, False],
        "linked_biopsy": [True, False],
    },
    "pregnancy_test": {
        "pregnant": [True, False],
        "procedure_pending": [True, False],
    },
    "office_visit": {
        "new_patient": [True, False],
        "visit_reason": ["screening discussion", "follow-up", "abnormal result", "consult"],
    },
    "supply_misc": {
        "linked_procedure": [True, False],
    },
    "functional_exam": {
        "functional_limitation": [True, False],
    },
}


def sample_attrs(category: str) -> Dict:
    cfg = CATEGORY_ATTRS.get(category, {})
    out = {}
    for k, vals in cfg.items():
        out[k] = random.choice(vals)
    return out


# -------------------------------------------------------------------
# Gold label inference for synthetic eval rows
# -------------------------------------------------------------------
def infer_expected_label(code: str, category: str, attrs: Dict) -> str:
    code = str(code).strip()

    age = attrs.get("age")
    days = attrs.get("days_since_last_screening")
    high_risk = bool(attrs.get("high_risk", False))
    brca = bool(attrs.get("brca_mutation", False))
    family_history = bool(attrs.get("family_history", False))
    provider = bool(attrs.get("provider_ordered", False))
    includes = bool(attrs.get("includes_screening_context", False))
    symptoms = bool(attrs.get("symptoms_present", False))
    abnormal = bool(attrs.get("abnormal_result", False) or attrs.get("abnormal_screening_result", False))
    laterality = str(attrs.get("laterality", "")).strip().lower()

    # ------------------------------------------------------------
    # Mammography / breast imaging rules for current eval scope
    # ------------------------------------------------------------
    if code == "77067":
        # Screening mammogram, bilateral
        if age is None or days is None:
            return "not_covered"
        if age >= 40 and days >= 365:
            return "covered"
        if days >= 365 and (high_risk or brca or family_history):
            return "covered"
        return "not_covered"

    if code == "77063":
        # DBT add-on screening
        if age is None or days is None:
            return "not_covered"
        if not includes:
            return "not_covered"
        if age >= 40 and days >= 365:
            return "covered"
        if days >= 365 and (high_risk or brca or family_history):
            return "covered"
        return "not_covered"

    if code == "77065":
        # Diagnostic mammo unilateral
        if not provider:
            return "not_covered"
        if laterality not in {"left", "right", "unknown", ""}:
            return "not_covered"
        return "covered" if (symptoms or abnormal) else "not_covered"

    if code == "77066":
        # Diagnostic mammo bilateral
        if not provider:
            return "not_covered"
        if laterality not in {"bilateral", "unknown", ""}:
            return "not_covered"
        return "covered" if (symptoms or abnormal) else "not_covered"

    if code == "77061":
        # DBT add-on unilateral diagnostic
        if not provider or not includes:
            return "not_covered"
        if laterality not in {"left", "right", "unknown", ""}:
            return "not_covered"
        return "covered" if (symptoms or abnormal) else "not_covered"

    if code == "77062":
        # DBT add-on bilateral diagnostic
        if not provider or not includes:
            return "not_covered"
        if laterality not in {"bilateral", "unknown", ""}:
            return "not_covered"
        return "covered" if (symptoms or abnormal) else "not_covered"

    if code == "77046":
        # MRI without contrast, unilateral
        if not provider:
            return "not_covered"
        if laterality not in {"left", "right", "unilateral", "unknown", ""}:
            return "not_covered"
        return "covered" if (high_risk or brca or family_history or symptoms or abnormal) else "not_covered"

    if code == "77047":
        # MRI without contrast, bilateral
        if not provider:
            return "not_covered"
        if laterality not in {"bilateral", "unknown", ""}:
            return "not_covered"
        return "covered" if (high_risk or brca or family_history or symptoms or abnormal) else "not_covered"

    if code == "77048":
        # MRI with and without contrast, unilateral
        if not provider:
            return "not_covered"
        if laterality not in {"left", "right", "unilateral", "unknown", ""}:
            return "not_covered"
        return "covered" if (high_risk or brca or family_history or symptoms or abnormal) else "not_covered"

    if code == "77049":
        # MRI with and without contrast, bilateral
        if not provider:
            return "not_covered"
        if laterality not in {"bilateral", "unknown", ""}:
            return "not_covered"
        return "covered" if (high_risk or brca or family_history or symptoms or abnormal) else "not_covered"

    # ------------------------------------------------------------
    # Fallbacks for non-mammo categories
    # Keep simple for now so the column is populated.
    # ------------------------------------------------------------
    if category in {"breast_biopsy_localization", "cervical_colposcopy_biopsy", "endometrial_sampling", "hpv_testing"}:
        if provider:
            return "covered"

    if category in {"breast_pathology_support", "cytology_pathology_lab"}:
        if bool(attrs.get("specimen_available", False)) or bool(attrs.get("linked_biopsy", False)):
            return "covered"

    if category == "pregnancy_test":
        if bool(attrs.get("procedure_pending", False)):
            return "covered"

    if category == "office_visit":
        return "covered"

    if category == "supply_misc":
        return "covered" if bool(attrs.get("linked_procedure", False)) else "not_covered"

    if category == "functional_exam":
        return "covered" if bool(attrs.get("functional_limitation", False)) else "not_covered"

    return "not_covered"


# -------------------------------------------------------------------
# Utility loaders
# -------------------------------------------------------------------
def load_code_metadata(meta_path: Path) -> Dict[str, Dict]:
    if not meta_path.exists():
        return {}

    df = pd.read_parquet(meta_path)
    out = {}

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


def load_policy_context(project_root: Path) -> Dict[str, str]:
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

    for code in CODE_RULES.keys():
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


def normalize_query_text(query: str) -> str:
    query = str(query).strip()
    query = re.sub(r"\s+", " ", query)
    return query


# -------------------------------------------------------------------
# Main builder
# -------------------------------------------------------------------
def build_testset(
    query_df: pd.DataFrame,
    metadata: Dict[str, Dict],
    policy_context: Dict[str, str],
) -> pd.DataFrame:
    rows = []

    for i, row in enumerate(query_df.itertuples(index=False), start=1):
        code = str(row.code).strip()
        query = normalize_query_text(row.query)
        verbosity = getattr(row, "verbosity", "")

        if code not in CODE_RULES:
            continue

        category = category_for(code)
        expected_topics = expected_topics_for(code)
        attrs = sample_attrs(category)
        expected_label = infer_expected_label(code, category, attrs)

        meta = metadata.get(code, {})
        title = meta.get("title", "")
        description = meta.get("description", "")
        context = policy_context.get(code, "")

        rows.append({
            "case_id": i,
            "expected_cpt": code,
            "category": category,
            "user_query": query,
            "verbosity": verbosity,
            "title": title,
            "description": description,
            "policy_context": context,
            "expected_topics_json": json.dumps(expected_topics),
            "expected_checklist": build_checklist(expected_topics),
            "attrs_json": json.dumps(attrs),
            "expected_label": expected_label,
        })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Generate all-code EVwoman question-eval test set")
    parser.add_argument("--infile", type=str, default=str(IN_PATH))
    parser.add_argument("--outfile", type=str, default=str(OUT_PATH))
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    random.seed(args.seed)

    infile = Path(args.infile)
    outfile = Path(args.outfile)

    if not infile.exists():
        raise FileNotFoundError(
            f"Input query file not found: {infile}\n"
            f"Run your all-code query generator first."
        )

    query_df = pd.read_csv(infile)

    required_cols = {"code", "query"}
    missing = required_cols - set(query_df.columns)
    if missing:
        raise ValueError(f"Input file missing required columns: {missing}")

    metadata = load_code_metadata(META_PATH)
    policy_context = load_policy_context(PROJECT_ROOT)

    out_df = build_testset(query_df=query_df, metadata=metadata, policy_context=policy_context)

    outfile.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(outfile, index=False)

    print("\n---- GENERATED TEST SET ----")
    print(f"Rows: {len(out_df)}")
    print(f"Codes: {out_df['expected_cpt'].nunique() if len(out_df) else 0}")
    print(f"Saved -> {outfile}")

    if len(out_df):
        print("\nExpected label counts:")
        print(out_df["expected_label"].value_counts(dropna=False))

        print("\nSample:")
        print(
            out_df[
                ["expected_cpt", "category", "verbosity", "user_query", "expected_label"]
            ].head(10).to_string(index=False)
        )


if __name__ == "__main__":
    main()