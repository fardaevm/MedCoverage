from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Dict, List


# -------------------------------------------------------------------
# Policy-grounded CPT codes currently in scope
# Professor-approved narrowed scope
# -------------------------------------------------------------------
POLICY_CODES = [
    "77046", "77047", "77048", "77049",
    "77061", "77062", "77063", "77065", "77066", "77067",
]


# -------------------------------------------------------------------
# Rule object
# -------------------------------------------------------------------
@dataclass(frozen=True)
class CodeRule:
    category: str
    expected_topics: List[str]
    notes: str = ""


# -------------------------------------------------------------------
# Topic patterns for evaluator
# Used in question_eval_all_codes.py to detect whether the chatbot
# asked about the right policy / eligibility concepts.
# -------------------------------------------------------------------
TOPIC_PATTERNS = {
    "age": re.compile(
        r"\bage\b|\bhow old\b|\byears?\s+old\b|\b40 years?\b|\bolder than\b|\bunder\b|\bover\b",
        re.I,
    ),
    "screening_interval": re.compile(
        r"\blast\b|\bsince\b|\bhow long ago\b|\b365 days\b|\byear\b|\bannual\b|\bmonths?\b|\bdays?\b",
        re.I,
    ),
    "risk_factors": re.compile(
        r"\bbrca\b|\bfamily history\b|\blifetime risk\b|\bhigh risk\b|\brisk assessment\b",
        re.I,
    ),
    "addon_screening_context": re.compile(
        r"\bwith screening\b|\bin addition\b|\badd[- ]?on\b|\b2d mammogram\b|\broutine screening\b|\b77067\b|\b3d mammogram\b|\btomosynthesis\b",
        re.I,
    ),
    "symptoms": re.compile(
        r"\bsymptom\b|\blump\b|\bpain\b|\bnipple discharge\b|\bskin changes\b|\bmass\b|\babnormal\b|\bpalpable\b",
        re.I,
    ),
    "abnormal_screening_result": re.compile(
        r"\babnormal\b|\bfollow[- ]?up\b|\bpositive result\b|\bscreening result\b",
        re.I,
    ),
    "medical_necessity": re.compile(
        r"\bdoctor\b|\bprovider\b|\bordered\b|\breferred\b|\bmedical necessity\b|\bwhy\b|\bindication\b",
        re.I,
    ),
    "laterality": re.compile(
        r"\bleft\b|\bright\b|\bone breast\b|\bboth breasts\b|\bunilateral\b|\bbilateral\b",
        re.I,
    ),
    "contrast": re.compile(
        r"\bcontrast\b|\bwith contrast\b|\bwithout contrast\b",
        re.I,
    ),
    "image_guidance": re.compile(
        r"\bultrasound\b|\bmri\b|\bmammogram\b|\bimage guidance\b|\bstereotactic\b|\bneedle guidance\b",
        re.I,
    ),
}


# -------------------------------------------------------------------
# Category definitions
# Narrowed to the 4 categories relevant to the 10 target codes
# -------------------------------------------------------------------
CATEGORY_RULES: Dict[str, CodeRule] = {
    "breast_screening_mammo": CodeRule(
        category="breast_screening_mammo",
        expected_topics=["age", "screening_interval", "risk_factors"],
        notes="Routine screening mammography; age, interval, and elevated risk are central.",
    ),
    "breast_screening_addon": CodeRule(
        category="breast_screening_addon",
        expected_topics=["age", "screening_interval", "risk_factors", "addon_screening_context"],
        notes="3D/add-on screening context should be tied to screening mammography.",
    ),
    "breast_diagnostic_imaging": CodeRule(
        category="breast_diagnostic_imaging",
        expected_topics=["symptoms", "medical_necessity", "laterality"],
        notes="Diagnostic breast imaging usually hinges on symptoms/history and laterality.",
    ),
    "breast_mri": CodeRule(
        category="breast_mri",
        expected_topics=["risk_factors", "medical_necessity", "contrast", "laterality"],
        notes="MRI coverage often depends on risk/indication plus contrast/laterality context.",
    ),
}


# -------------------------------------------------------------------
# Code family mapping
# -------------------------------------------------------------------
BREAST_SCREENING_MAMMO_CODES = {
    "77067",
}

BREAST_SCREENING_ADDON_CODES = {
    "77063",
}

BREAST_DIAGNOSTIC_IMAGING_CODES = {
    "77061", "77062", "77065", "77066",
}

BREAST_MRI_CODES = {
    "77046", "77047", "77048", "77049",
}


def _category_for_code(code: str) -> str:
    code = str(code)

    if code in BREAST_SCREENING_MAMMO_CODES:
        return "breast_screening_mammo"
    if code in BREAST_SCREENING_ADDON_CODES:
        return "breast_screening_addon"
    if code in BREAST_DIAGNOSTIC_IMAGING_CODES:
        return "breast_diagnostic_imaging"
    if code in BREAST_MRI_CODES:
        return "breast_mri"

    raise KeyError(f"No category mapping defined for code: {code}")


def build_code_rules() -> Dict[str, CodeRule]:
    rules: Dict[str, CodeRule] = {}
    for code in POLICY_CODES:
        category = _category_for_code(code)
        rules[code] = CATEGORY_RULES[category]
    return rules


CODE_RULES: Dict[str, CodeRule] = build_code_rules()


def expected_topics_for(code: str) -> List[str]:
    return CODE_RULES[str(code)].expected_topics


def category_for(code: str) -> str:
    return CODE_RULES[str(code)].category