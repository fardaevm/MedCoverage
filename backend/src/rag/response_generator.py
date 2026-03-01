# backend/src/rag/response_generator.py
from __future__ import annotations
from typing import List, Optional, Dict, Any
import json
import re

from src.rag_interface.base_response_generator import BaseResponseGenerator
from src.invoke_ai import invoke_ai

SYSTEM_PROMPT = (
    "You are an insurance eligibility expert. Read the policy context and identify every"
    " distinct pathway by which this procedure is covered.\n\n"
    "Do not use outside the source information\n"
    "Only use the information inside the documents\n"
    "Write each pathway as a natural, flowing sentence a patient would use to describe"
    " their own situation -- not a list of criteria.\n\n"
    "Rules:\n"
    "- One pathway per line, starting with \"- \"\n"
    "- Write in first person (I am..., I have..., I need...) or as a clear patient description\n"
    "- Each sentence must be COMPLETE: include every required condition for that pathway\n"
    "- Plain English only -- no CPT codes, procedure names, or billing jargon\n"
    "- Up to 25 words per pathway\n"
    "- Target 2-5 pathways; include every pathway present in the policy -- do not skip any\n"
    "- Do NOT include administrative conditions (prior auth, referrals, ordering physician)\n\n"
    "Output format exactly:\n"
    "Title: <plain-language title for this procedure>\n"
    "Pathways:\n"
    "- <pathway 1, flowing sentence, up to 25 words>\n"
    "- <pathway 2, flowing sentence, up to 25 words>\n"
)

PLANNER_SYSTEM_PROMPT = (
    "You are a coverage-question PLANNER.\n\n"
    "You will be given:\n"
    "- Policy context excerpts\n"
    "- Exactly 2 patient-facing coverage pathways (P1, P2)\n"
    "- Q/A so far (YES/NO)\n\n"
    "Your job:\n"
    "Pick ONE YES/NO question that best distinguishes P1 vs P2, or that completes the remaining"
    " required condition(s) of the surviving pathway.\n\n"
    "Rules:\n"
    "- Output EXACTLY one line: \"Q: <yes/no question>\"\n"
    "- The question must be short, plain English, and answerable YES/NO.\n"
    "- Ask exactly ONE criterion per question. Do NOT combine multiple options with \"or\" in a"
    " single question (e.g. do NOT ask \"Do you have A, or B, or C?\"). Ask about A, then later B,"
    " then C in separate questions.\n"
    "- Do NOT ask administrative questions (prior auth, referral, ordering doctor, claim, billing).\n"
    "- Do NOT repeat a previously asked question.\n"
    "- Prefer a gate question that eliminates an entire pathway when possible.\n"
)

PATHWAYS_AND_QUESTIONS_SYSTEM_PROMPT = (
    "You are an insurance eligibility expert. Read the policy context and identify every"
    " distinct pathway by which this procedure is covered.\n\n"
    "For each pathway you must also list the exact YES/NO questions that determine if the patient"
    " qualifies for that pathway. If the patient answers YES to every question for a pathway,"
    " that pathway is satisfied (covered).\n\n"
    "Rules:\n"
    "- Use ONLY the information inside the provided context. Do not invent requirements.\n"
    "- Each pathway is one short sentence (first person, plain English, up to 25 words).\n"
    "- For each pathway, list 2–5 YES/NO questions in order. Each question must be:\n"
    "  - Short, plain English, answerable YES or NO.\n"
    "  - One criterion per question (no \"A, or B, or C\" in one question).\n"
    "- Do NOT include administrative questions (prior auth, referral, ordering doctor, billing).\n"
    "- Output valid JSON only, no markdown or explanation.\n\n"
    "Return STRICT JSON only:\n"
    "{\n"
    '  "title": "<plain-language title for this procedure>",\n'
    '  "pathways": ["<pathway 1 sentence>", "<pathway 2 sentence>", ...],\n'
    '  "questions_per_pathway": [\n'
    '    ["<Q1 for pathway 1>", "<Q2 for pathway 1>", ...],\n'
    '    ["<Q1 for pathway 2>", "<Q2 for pathway 2>", ...]\n'
    "  ]\n"
    "}\n"
    "The length of questions_per_pathway must equal the length of pathways.\n"
)

DECISION_TREE_SYSTEM_PROMPT = (
    "You are an insurance eligibility EXPERT making a coverage determination.\n\n"
    "You will be given:\n"
    "- Policy context excerpts (authoritative source of truth)\n"
    "- A procedure query\n"
    "- Exactly 2 pathways written as patient sentences (P1, P2)\n"
    "- Q/A so far (YES/NO)\n\n"
    "Decide using strict rules:\n"
    "- \"covered\": at least one pathway is fully satisfied by the answers.\n"
    "- \"not_covered\": both pathways are definitively ruled out by explicit NO answers.\n"
    "- \"uncertain\": at least one pathway is not ruled out, but required info was not asked yet.\n\n"
    "CRITICAL:\n"
    "- Use ONLY provided context + pathways + answers.\n"
    "- If procedure is NOT mentioned in context -> \"uncertain\".\n"
    "- Do NOT invent requirements.\n\n"
    "REASON (plain language for the patient):\n"
    "- Write \"reason\" so a patient can understand it. No CPT/ICD/codes, no billing jargon.\n"
    "- If covered: explain in 1–3 short sentences why this is covered and why we believe so based on their answers.\n"
    "- If not covered: explain in 1–3 short sentences why it is not covered based on their answers.\n"
    "- Use simple, everyday words.\n\n"
    "Return STRICT JSON only:\n"
    "{\n"
    "  \"decision\": \"covered\" | \"not_covered\" | \"uncertain\",\n"
    "  \"reason\": \"<plain-language explanation for the patient>\",\n"
    "  \"missing_info\": [\"<short item>\", \"...\"]\n"
    "}\n"
)

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


def _is_compound_question(q: str) -> bool:
    """True if question lists multiple alternatives with 'or' (e.g. 'A, or B, or C')."""
    if not (q or "").strip():
        return False
    # Count " or " (with optional "a " before next phrase) — 2+ means compound
    or_count = len(re.findall(r"\s+or\s+(?:a\s+)?", (q or ""), re.IGNORECASE))
    return or_count >= 2

def _extract_planner_question(raw: str) -> str:
    raw = (raw or "").strip()
    m = re.search(r"^\s*Q:\s*(.+?)\s*$", raw, re.IGNORECASE | re.MULTILINE)
    if m:
        return m.group(1).strip()
    return raw

def _parse_pathways_legacy(raw: str) -> Dict[str, Any]:
    """Parse legacy 'Title: ... Pathways: \\n- ...' format (no JSON)."""
    lines = [l.strip() for l in (raw or "").split("\n") if l.strip()]
    title = "Coverage pathways"
    for line in lines:
        if line.lower().startswith("title:"):
            title = line.split(":", 1)[1].strip() or title
            break
    pathways: List[str] = []
    in_pathways = False
    for line in lines:
        low = line.lower().rstrip(":")
        if low == "pathways":
            in_pathways = True
            continue
        if in_pathways and line.startswith("- "):
            txt = line[2:].strip()
            if txt:
                pathways.append(txt)
    return {"title": title, "pathways": pathways}


def _parse_pathways_and_questions_json(raw: str) -> Optional[Dict[str, Any]]:
    """Parse JSON from LLM; return dict with title, pathways, questions_per_pathway or None."""
    raw = (raw or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    pathways = data.get("pathways")
    qpp = data.get("questions_per_pathway")
    if not isinstance(pathways, list) or not isinstance(qpp, list):
        return None
    pathways = [str(p).strip() for p in pathways if str(p).strip()]
    if len(pathways) != len(qpp):
        return None
    out_questions = []
    for i, qs in enumerate(qpp):
        if not isinstance(qs, list):
            return None
        out_questions.append([str(q).strip() for q in qs if str(q).strip()])
    return {
        "title": str(data.get("title", "")).strip() or "Coverage pathways",
        "pathways": pathways,
        "questions_per_pathway": out_questions,
    }


class ResponseGenerator(BaseResponseGenerator):
    def generate_response(self, query: str, context: List[str]) -> str:
        context_text = "\n".join(context)
        user_message = f"<context>\n{context_text}\n</context>\n<question>\n{query}\n</question>"
        return invoke_ai(system_message=SYSTEM_PROMPT, user_message=user_message)

    def generate_pathways_and_questions(self, query: str, context: List[str]) -> Dict[str, Any]:
        """
        One-call: return pathways and questions_per_pathway.
        questions_per_pathway[i] = list of yes/no questions; all YES => pathway i satisfied.
        On parse failure, falls back to pathways-only (questions_per_pathway = empty lists).
        """
        context_text = "\n".join(context)
        user_message = f"<context>\n{context_text}\n</context>\n<procedure_query>\n{query}\n</procedure_query>"
        raw = invoke_ai(
            system_message=PATHWAYS_AND_QUESTIONS_SYSTEM_PROMPT,
            user_message=user_message,
        )
        parsed = _parse_pathways_and_questions_json(raw)
        if parsed:
            return parsed
        # Fallback: pathways only (legacy text format)
        raw_legacy = self.generate_response(query, context)
        legacy = _parse_pathways_legacy(raw_legacy)
        pathways = (legacy.get("pathways") or [])[:2]
        while len(pathways) < 2:
            pathways.append("")
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
        p1 = (pathways[0] if len(pathways) > 0 else "").strip()
        p2 = (pathways[1] if len(pathways) > 1 else "").strip()

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
            f"<pathways>\nP1: {p1}\nP2: {p2}\n</pathways>\n"
            f"<qa_so_far>\n{qa_text}\n</qa_so_far>\n"
        )

        extra = ""
        for _ in range(max_replans):
            raw = invoke_ai(system_message=PLANNER_SYSTEM_PROMPT, user_message=base_user + extra)
            q = _extract_planner_question(raw)
            nq = _norm_q(q)

            bad_admin = bool(_ADMIN_PAT.search(q))
            dup = (nq in asked) or (nq == "")
            compound = _is_compound_question(q)

            if not bad_admin and not dup and not compound:
                return q

            why = []
            if bad_admin:
                why.append("administrative")
            if dup:
                why.append("duplicate/empty")
            if compound:
                why.append(
                    "compound (multiple options with 'or'). Ask exactly one criterion per question, "
                    "e.g. only BRCA mutation, or only first-degree relative BRCA carrier, or only "
                    "lifetime risk ≥20%, in a single short yes/no question."
                )
            extra = (
                "\n<replan>\n"
                f"The previous question was rejected: {', '.join(why)}\n"
                "Output a different YES/NO question.\n"
                "</replan>\n"
            )

        # last resort: generic discriminator
        fallback = "Are you getting this at the same time as a related imaging test (like an MRI)?"
        if _norm_q(fallback) in asked:
            fallback = "Is this being done as routine screening (not because of symptoms or a new problem)?"
        return fallback

    def decide_from_tree(
        self,
        *,
        query: str,
        context: List[str],
        pathways: List[str],
        qa_so_far: List[Dict[str, Any]],
    ) -> str:
        p1 = (pathways[0] if len(pathways) > 0 else "").strip()
        p2 = (pathways[1] if len(pathways) > 1 else "").strip()

        qa_lines = []
        for i, x in enumerate(qa_so_far or [], start=1):
            if not isinstance(x, dict):
                continue
            q = str(x.get("q", "")).strip()
            a = x.get("a", None)
            qa_lines.append(f"- {q}\n  A: {'YES' if a is True else 'NO' if a is False else 'UNKNOWN'}")
        qa_text = "\n".join(qa_lines) if qa_lines else "(none)"

        context_text = "\n".join(context)
        user_message = (
            f"<context>\n{context_text}\n</context>\n"
            f"<procedure_query>\n{query}\n</procedure_query>\n"
            f"<pathways>\nP1: {p1}\nP2: {p2}\n</pathways>\n"
            f"<qa_so_far>\n{qa_text}\n</qa_so_far>\n"
        )
        return invoke_ai(system_message=DECISION_TREE_SYSTEM_PROMPT, user_message=user_message)