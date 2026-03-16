from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Dict, List


# -------------------------------------------------------------------
# Policy-grounded CPT/HCPCS codes currently in scope
# -------------------------------------------------------------------
POLICY_CODES = [
    "10004", "10005", "10006", "10007", "10008", "10011", "10012", "10021",
    "15600",
    "19000", "19001", "19081", "19082", "19083", "19084", "19085", "19086",
    "19100", "19101", "19120", "19125", "19126",
    "19281", "19282", "19283", "19284", "19285", "19286", "19287", "19288",
    "19300",
    "57452", "57454", "57455", "57456", "57500", "57505",
    "58100", "58110",
    "76098", "76641", "76642", "76942",
    "77046", "77047", "77048", "77049",
    "77053", "77061", "77062", "77063", "77065", "77066", "77067",
    "81025",
    "87624", "87625",
    "88141", "88142", "88143", "88164", "88172", "88173", "88174", "88175",
    "88177", "88305", "88307", "88331", "88332", "88341", "88342", "88360",
    "88364", "88365", "88366", "88367", "88368", "88369", "88373", "88374",
    "88377",
    "95852",
    "99070", "99202", "99203", "99204", "99211", "99212", "99213", "99214",
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
# These are used later in question_eval_all_codes.py to detect whether
# the chatbot asked about the right policy/eligibility topics.
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
        r"\babnormal\b|\bpap\b|\bhpv\b|\bbiopsy\b|\bfollow[- ]?up\b|\bpositive result\b|\bscreening result\b|\bcolposcopy\b",
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
    "lesion_target": re.compile(
        r"\bwhich lesion\b|\btarget\b|\bcalcification\b|\bmass\b|\bclip\b|\blocalization\b|\bsite\b",
        re.I,
    ),
    "pregnancy_status": re.compile(
        r"\bpregnan\b|\bpregnancy test\b|\b81025\b|\bLMP\b|\blast menstrual\b",
        re.I,
    ),
    "bleeding_history": re.compile(
        r"\bbleeding\b|\bpostmenopausal\b|\bmenopausal\b|\bspotting\b|\babnormal uterine bleeding\b",
        re.I,
    ),
    "specimen_context": re.compile(
        r"\bspecimen\b|\bsample\b|\btissue\b|\bcollected\b|\bcytology\b|\bpathology\b",
        re.I,
    ),
    "reflex_testing": re.compile(
        r"\breflex\b|\bhpv\b|\bgenotype\b|\btyping\b|\badditional testing\b",
        re.I,
    ),
    "visit_reason": re.compile(
        r"\bvisit\b|\bappointment\b|\bconsult\b|\bfollow[- ]?up\b|\bnew patient\b|\bestablished\b|\boffice visit\b",
        re.I,
    ),
    "new_vs_established": re.compile(
        r"\bnew patient\b|\bestablished patient\b|\bseen before\b|\bfirst visit\b",
        re.I,
    ),
    "pathology_linkage": re.compile(
        r"\bbiopsy\b|\bspecimen\b|\bfrom procedure\b|\bsource\b|\btissue\b|\bblock\b|\bslide\b",
        re.I,
    ),
    "functional_status": re.compile(
        r"\bmovement\b|\brange of motion\b|\bfunction\b|\bstrength\b|\bexam\b",
        re.I,
    ),
}


# -------------------------------------------------------------------
# Category definitions
# First-pass taxonomy: good enough to support all-code evaluation.
# Refine over time as you inspect real pathway outputs.
# -------------------------------------------------------------------
CATEGORY_RULES: Dict[str, CodeRule] = {
    "breast_screening_mammo": CodeRule(
        category="breast_screening_mammo",
        expected_topics=["age", "screening_interval", "risk_factors"],
        notes="Routine screening mammography; screening interval and age/risk are central.",
    ),
    "breast_screening_addon": CodeRule(
        category="breast_screening_addon",
        expected_topics=["age", "screening_interval", "risk_factors", "addon_screening_context"],
        notes="3D/add-on screening context should be tied back to screening mammography.",
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
    "breast_biopsy_localization": CodeRule(
        category="breast_biopsy_localization",
        expected_topics=["abnormal_screening_result", "medical_necessity", "image_guidance", "lesion_target", "laterality"],
        notes="Biopsy/localization workup should target lesion/site and guidance modality.",
    ),
    "breast_pathology_support": CodeRule(
        category="breast_pathology_support",
        expected_topics=["pathology_linkage", "specimen_context", "medical_necessity"],
        notes="Supportive breast procedural/pathology-linked services.",
    ),
    "cervical_colposcopy_biopsy": CodeRule(
        category="cervical_colposcopy_biopsy",
        expected_topics=["abnormal_screening_result", "medical_necessity", "pregnancy_status"],
        notes="Colposcopy/cervical biopsy usually follows abnormal screening or abnormal exam.",
    ),
    "endometrial_sampling": CodeRule(
        category="endometrial_sampling",
        expected_topics=["bleeding_history", "medical_necessity", "pregnancy_status"],
        notes="Endometrial biopsy/sampling usually requires bleeding history and pregnancy context.",
    ),
    "hpv_testing": CodeRule(
        category="hpv_testing",
        expected_topics=["abnormal_screening_result", "age", "reflex_testing"],
        notes="HPV testing is often linked to cervical screening pathways and reflex testing logic.",
    ),
    "cytology_pathology_lab": CodeRule(
        category="cytology_pathology_lab",
        expected_topics=["specimen_context", "abnormal_screening_result", "pathology_linkage"],
        notes="Pap/cytology/pathology work usually depends on specimen and linked screening/procedure context.",
    ),
    "pregnancy_test": CodeRule(
        category="pregnancy_test",
        expected_topics=["pregnancy_status", "medical_necessity"],
        notes="Pregnancy test often serves as prerequisite or safety check for another procedure.",
    ),
    "office_visit": CodeRule(
        category="office_visit",
        expected_topics=["visit_reason", "new_vs_established", "medical_necessity"],
        notes="E/M visits should be tied to reason for encounter and patient status.",
    ),
    "supply_misc": CodeRule(
        category="supply_misc",
        expected_topics=["medical_necessity", "pathology_linkage"],
        notes="Supplies/misc services should usually be linked to a covered procedure.",
    ),
    "functional_exam": CodeRule(
        category="functional_exam",
        expected_topics=["functional_status", "medical_necessity"],
        notes="Functional assessment-type services.",
    ),
}


# -------------------------------------------------------------------
# Code family mapping
# -------------------------------------------------------------------
BREAST_SCREENING_MAMMO_CODES = {
    "77067",
}

BREAST_SCREENING_ADDON_CODES = {
    "77053",
    "77063",
}

BREAST_DIAGNOSTIC_IMAGING_CODES = {
    "76641", "76642",
    "77061", "77062", "77065", "77066",
}

BREAST_MRI_CODES = {
    "77046", "77047", "77048", "77049",
}

BREAST_BIOPSY_LOCALIZATION_CODES = {
    "10004", "10005", "10006", "10007", "10008", "10011", "10012", "10021",
    "19000", "19001",
    "19081", "19082", "19083", "19084", "19085", "19086",
    "19100", "19101", "19120", "19125", "19126",
    "19281", "19282", "19283", "19284", "19285", "19286", "19287", "19288",
    "19300",
    "76098", "76942",
}

BREAST_PATHOLOGY_SUPPORT_CODES = {
    "15600",
}

CERVICAL_COLPOSCOPY_BIOPSY_CODES = {
    "57452", "57454", "57455", "57456", "57500", "57505",
}

ENDOMETRIAL_SAMPLING_CODES = {
    "58100", "58110",
}

PREGNANCY_TEST_CODES = {
    "81025",
}

HPV_TESTING_CODES = {
    "87624", "87625",
}

CYTOLOGY_PATHOLOGY_LAB_CODES = {
    "88141", "88142", "88143", "88164", "88172", "88173", "88174", "88175",
    "88177", "88305", "88307", "88331", "88332", "88341", "88342", "88360",
    "88364", "88365", "88366", "88367", "88368", "88369", "88373", "88374",
    "88377",
}

FUNCTIONAL_EXAM_CODES = {
    "95852",
}

SUPPLY_MISC_CODES = {
    "99070",
}

OFFICE_VISIT_CODES = {
    "99202", "99203", "99204",
    "99211", "99212", "99213", "99214",
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
    if code in BREAST_BIOPSY_LOCALIZATION_CODES:
        return "breast_biopsy_localization"
    if code in BREAST_PATHOLOGY_SUPPORT_CODES:
        return "breast_pathology_support"
    if code in CERVICAL_COLPOSCOPY_BIOPSY_CODES:
        return "cervical_colposcopy_biopsy"
    if code in ENDOMETRIAL_SAMPLING_CODES:
        return "endometrial_sampling"
    if code in PREGNANCY_TEST_CODES:
        return "pregnancy_test"
    if code in HPV_TESTING_CODES:
        return "hpv_testing"
    if code in CYTOLOGY_PATHOLOGY_LAB_CODES:
        return "cytology_pathology_lab"
    if code in FUNCTIONAL_EXAM_CODES:
        return "functional_exam"
    if code in SUPPLY_MISC_CODES:
        return "supply_misc"
    if code in OFFICE_VISIT_CODES:
        return "office_visit"

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