# backend/src/rag/response_generator.py
from __future__ import annotations
from typing import List, Optional, Dict, Any
import json
import re

from src.rag_interface.base_response_generator import BaseResponseGenerator
from src.invoke_ai import invoke_ai

# ── Prompts ────────────────────────────────────────────────────────

PATHWAYS_AND_QUESTIONS_SYSTEM_PROMPT = (
    "You are an insurance eligibility expert. Read the policy context and build a"
    " SINGLE decision tree for this procedure.\n\n"
    "TREE STRUCTURE:\n"
    "- The tree is ONE top-level OR node. Each child is a coverage pathway.\n"
    "- Each pathway child is an AND/OR/NOT subtree with a \"pathway\" label.\n"
    "- Leaf nodes: {\"q_id\": \"<id>\", \"value\": true}\n"
    "- Branch nodes: {\"operator\": \"AND\"|\"OR\"|\"NOT\", \"conditions\": [...]}\n"
    "- Pathway branches: same as branch but add \"pathway\": \"<description>\"\n"
    "- value=true means YES=meets requirement. value=false means YES=disqualifies.\n"
    "- Use OR when the policy lists ALTERNATIVE qualifying conditions.\n"
    "- Use AND when ALL conditions must be met simultaneously.\n"
    "- Use NOT sparingly, only when the policy explicitly excludes something.\n"
    "- Reuse the same q_id across pathway branches for shared conditions.\n\n"
    "QUESTION RULES:\n"
    "- Use ONLY information from the provided context. Do not invent requirements.\n"
    "- Write questions in second person (\"Do you...\", \"Are you...\", \"Have you...\").\n"
    "- Phrase so YES = patient meets the requirement (unless value=false).\n"
    "- One criterion per question. Plain English, short.\n"
    "- Do NOT include administrative questions (prior auth, referral, billing).\n"
    "- q_id should be a short snake_case identifier.\n\n"
    "OUTPUT FORMAT — strict JSON, no markdown:\n"
    "{\n"
    '  "title": "<plain-language title>",\n'
    '  "question_map": {\n'
    '    "age_40_plus": "Are you 40 years or older?",\n'
    '    "365_days_since_last": "Has it been at least 365 days since your last screening?",\n'
    '    "brca_mutation": "Do you have a BRCA gene mutation?",\n'
    '    "family_history": "Do you have a first-degree relative who is a BRCA carrier?"\n'
    '  },\n'
    '  "logic_tree": {\n'
    '    "operator": "OR",\n'
    '    "conditions": [\n'
    '      {\n'
    '        "pathway": "Routine screening for age 40+",\n'
    '        "operator": "AND",\n'
    '        "conditions": [\n'
    '          {"q_id": "age_40_plus", "value": true},\n'
    '          {"q_id": "365_days_since_last", "value": true}\n'
    '        ]\n'
    '      },\n'
    '      {\n'
    '        "pathway": "High-risk with BRCA or family history",\n'
    '        "operator": "AND",\n'
    '        "conditions": [\n'
    '          {"q_id": "365_days_since_last", "value": true},\n'
    '          {\n'
    '            "operator": "OR",\n'
    '            "conditions": [\n'
    '              {"q_id": "brca_mutation", "value": true},\n'
    '              {"q_id": "family_history", "value": true}\n'
    '            ]\n'
    '          }\n'
    '        ]\n'
    '      }\n'
    '    ]\n'
    '  }\n'
    '}\n\n'
    "IMPORTANT:\n"
    "- The top-level node MUST be OR with pathway-labeled children.\n"
    "- question_map MUST contain every q_id used in the tree.\n"
    "- Aim for 2-6 pathway branches. 2-4 conditions per branch.\n"
    "- Shared conditions (like time since last screening) should use the same q_id in multiple branches.\n"
)

PLANNER_SYSTEM_PROMPT = (
    "You are a coverage-question PLANNER.\n\n"
    "You will be given:\n"
    "- Policy context excerpts\n"
    "- Patient-facing coverage pathways\n"
    "- Q/A so far (YES/NO)\n\n"
    "Your job:\n"
    "Pick ONE YES/NO question that best distinguishes between surviving pathways.\n\n"
    "Rules:\n"
    "- Output EXACTLY one line: \"Q: <yes/no question>\"\n"
    "- Short, plain English, answerable YES/NO.\n"
    "- ONE criterion per question.\n"
    "- Do NOT ask administrative questions (prior auth, referral, billing).\n"
    "- Do NOT repeat a previously asked question.\n"
)

_LEGACY_SYSTEM_PROMPT = (
    "You are an insurance eligibility expert. Read the policy context and identify every"
    " distinct pathway by which this procedure is covered.\n\n"
    "Do not use outside the source information\n"
    "Only use the information inside the documents\n"
    "Write each pathway as a natural, flowing sentence a patient would use.\n\n"
    "Rules:\n"
    "- One pathway per line, starting with \"- \"\n"
    "- Plain English only, up to 25 words per pathway\n"
    "- Target 2-4 pathways maximum\n"
    "- Do NOT include administrative conditions\n\n"
    "Output format exactly:\n"
    "Title: <plain-language title>\n"
    "Pathways:\n"
    "- <pathway 1>\n"
    "- <pathway 2>\n"
)

# ── Helpers ────────────────────────────────────────────────────────

_ADMIN_PAT = re.compile(
    r"\b(prior auth|authorization|preauth|referral|ordering (doctor|physician)|provider|"
    r"billing|claim|claims|reimbursement|coverage policy number|npi|cpt|icd|"
    r"medical necessity form|paperwork)\b",
    re.IGNORECASE,
)


def _norm_q(s: str) -> str:
    s = re.sub(r"\s+", " ", (s or "").strip().lower())
    s = re.sub(r"[^\w\s\?]", "", s)
    return s


def _extract_planner_question(raw: str) -> str:
    raw = (raw or "").strip()
    m = re.search(r"^\s*Q:\s*(.+?)\s*$", raw, re.IGNORECASE | re.MULTILINE)
    return m.group(1).strip() if m else raw


def _parse_pathways_legacy(raw: str) -> Dict[str, Any]:
    lines = [l.strip() for l in (raw or "").split("\n") if l.strip()]
    title = "Coverage pathways"
    for line in lines:
        if line.lower().startswith("title:"):
            title = line.split(":", 1)[1].strip() or title
            break
    pathways: List[str] = []
    in_pathways = False
    for line in lines:
        if line.lower().rstrip(":") == "pathways":
            in_pathways = True
            continue
        if in_pathways and line.startswith("- "):
            txt = line[2:].strip()
            if txt:
                pathways.append(txt)
    return {"title": title, "pathways": pathways}


def _validate_tree(node: dict, question_map: dict) -> bool:
    if "q_id" in node:
        return node["q_id"] in question_map and "value" in node
    if "operator" not in node or "conditions" not in node:
        return False
    if node["operator"].upper() not in ("AND", "OR", "NOT"):
        return False
    if not isinstance(node["conditions"], list) or not node["conditions"]:
        return False
    if node["operator"].upper() == "NOT" and len(node["conditions"]) != 1:
        return False
    return all(_validate_tree(c, question_map) for c in node["conditions"])


def _extract_pathways_from_tree(tree: dict) -> List[str]:
    """Extract pathway labels from top-level OR children."""
    if tree.get("operator", "").upper() != "OR":
        return [tree.get("pathway", "Coverage pathway")]
    return [c.get("pathway", f"Pathway {i+1}") for i, c in enumerate(tree.get("conditions", []))]


def _parse_json(raw: str) -> Optional[Dict[str, Any]]:
    raw = (raw or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None

    question_map = data.get("question_map")
    if not isinstance(question_map, dict):
        return None

    # ── New single-tree format ──
    logic_tree = data.get("logic_tree")
    if isinstance(logic_tree, dict) and _validate_tree(logic_tree, question_map):
        pathways = _extract_pathways_from_tree(logic_tree)
        return {
            "title": str(data.get("title", "")).strip() or "Coverage pathways",
            "question_map": question_map,
            "logic_tree": logic_tree,
            "pathways": pathways,
            "questions_per_pathway": [],
        }

    # ── Old multi-tree format (auto-merge in evaluator) ──
    pathways = data.get("pathways")
    logic_trees = data.get("logic_trees")
    if (
        isinstance(pathways, list) and isinstance(logic_trees, list)
        and len(logic_trees) == len(pathways)
        and all(_validate_tree(t, question_map) for t in logic_trees)
    ):
        pathways = [str(p).strip() for p in pathways if str(p).strip()]
        return {
            "title": str(data.get("title", "")).strip() or "Coverage pathways",
            "question_map": question_map,
            "logic_trees": logic_trees,
            "pathways": pathways,
            "questions_per_pathway": [],
        }

    # ── Legacy flat format ──
    if isinstance(pathways, list):
        pathways = [str(p).strip() for p in pathways if str(p).strip()]
        qpp = data.get("questions_per_pathway")
        if isinstance(qpp, list) and len(qpp) == len(pathways):
            out_qs = []
            for qs in qpp:
                if isinstance(qs, list):
                    out_qs.append([str(q).strip() for q in qs if str(q).strip()])
                else:
                    out_qs.append([])
            return {
                "title": str(data.get("title", "")).strip() or "Coverage pathways",
                "pathways": pathways,
                "questions_per_pathway": out_qs,
            }

    return None


# ── Response Generator ─────────────────────────────────────────────

class ResponseGenerator(BaseResponseGenerator):

    def generate_response(self, query: str, context: List[str]) -> str:
        context_text = "\n".join(context)
        user_message = f"<context>\n{context_text}\n</context>\n<question>\n{query}\n</question>"
        return invoke_ai(system_message=_LEGACY_SYSTEM_PROMPT, user_message=user_message)

    def generate_pathways_and_questions(self, query: str, context: List[str]) -> Dict[str, Any]:
        context_text = "\n".join(context)
        user_message = (
            f"<context>\n{context_text}\n</context>\n"
            f"<procedure_query>\n{query}\n</procedure_query>"
        )
        raw = invoke_ai(
            system_message=PATHWAYS_AND_QUESTIONS_SYSTEM_PROMPT,
            user_message=user_message,
        )
        parsed = _parse_json(raw)
        if parsed:
            return parsed

        # Fallback: legacy text format
        raw_legacy = self.generate_response(query, context)
        legacy = _parse_pathways_legacy(raw_legacy)
        pathways = (legacy.get("pathways") or [])[:4]
        return {
            "title": legacy.get("title", "Coverage pathways") or "Coverage pathways",
            "pathways": pathways,
            "questions_per_pathway": [[] for _ in pathways],
        }

    def plan_next_question(
        self,
        *,
        query: str,
        context: List[str],
        pathways: List[str],
        qa_so_far: List[Dict[str, Any]],
        max_replans: int = 3,
    ) -> str:
        pathways_text = "\n".join(
            f"P{i+1}: {p.strip()}" for i, p in enumerate(pathways) if (p or "").strip()
        )
        asked = {_norm_q(x.get("q", "")) for x in (qa_so_far or []) if isinstance(x, dict)}
        qa_lines = []
        for i, x in enumerate(qa_so_far or [], start=1):
            if not isinstance(x, dict):
                continue
            q = str(x.get("q", "")).strip()
            a = x.get("a", None)
            qa_lines.append(f"- Q{i}: {q}\n  A: {'YES' if a is True else 'NO' if a is False else 'UNKNOWN'}")
        qa_text = "\n".join(qa_lines) if qa_lines else "(none)"
        context_text = "\n".join(context)
        base_user = (
            f"<context>\n{context_text}\n</context>\n"
            f"<procedure_query>\n{query}\n</procedure_query>\n"
            f"<pathways>\n{pathways_text}\n</pathways>\n"
            f"<qa_so_far>\n{qa_text}\n</qa_so_far>\n"
        )
        extra = ""
        for _ in range(max_replans):
            raw = invoke_ai(system_message=PLANNER_SYSTEM_PROMPT, user_message=base_user + extra)
            q = _extract_planner_question(raw)
            nq = _norm_q(q)
            if not bool(_ADMIN_PAT.search(q)) and nq and nq not in asked:
                return q
            extra = "\n<replan>Previous question rejected. Output a different YES/NO question.</replan>\n"
        fallback = "Is this being done as routine screening (not because of symptoms or a new problem)?"
        if _norm_q(fallback) in asked:
            fallback = "Has it been at least 12 months since your last similar procedure?"
        return fallback