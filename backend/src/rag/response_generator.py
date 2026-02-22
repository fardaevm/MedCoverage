from typing import List, Optional
from src.rag_interface.base_response_generator import BaseResponseGenerator

from src.invoke_ai import invoke_ai


SYSTEM_PROMPT = (
    'You are an insurance eligibility expert. Read the policy context and identify every'
    ' distinct pathway by which this procedure is covered.\n'
    '\n'
    'Do not use outside the source information\n'
    'Only use the information inside the documents\n'
    'Write each pathway as a natural, flowing sentence a patient would use to describe'
    ' their own situation -- not a list of criteria.\n'
    '\n'
    'Rules:\n'
    '- One pathway per line, starting with "- "\n'
    '- Write in first person (I am..., I have..., I need...) or as a clear patient description\n'
    '- Each sentence must be COMPLETE: include every required condition for that pathway\n'
    '- Plain English only -- no CPT codes, procedure names, or billing jargon\n'
    '- Up to 25 words per pathway\n'
    '- Target 2-5 pathways; include every pathway present in the policy -- do not skip any\n'
    '- Do NOT include administrative conditions (prior auth, referrals, ordering physician)\n'
    '\n'
    'Good examples:\n'
    '- I am a woman aged 40 or older getting a routine yearly breast cancer screening.\n'
    '- I have a BRCA gene mutation and I am getting this alongside a breast MRI.\n'
    '\n'
    'Output format exactly:\n'
    'Title: <plain-language title for this procedure>\n'
    'Pathways:\n'
    '- <pathway 1, flowing sentence, up to 25 words>\n'
    '- <pathway 2, flowing sentence, up to 25 words>\n'
)

DECISION_SYSTEM_PROMPT = (
    'You are an insurance eligibility EXPERT making a coverage determination.\n'
    '\n'
    'You will be given:\n'
    '- Policy context excerpts (authoritative source of truth)\n'
    '- A procedure query\n'
    '- Pathways shown to the patient, each labeled [standalone]\n'
    '- The patient\'s YES/NO answers\n'
    '\n'
    'LABEL MEANINGS:\n'
    '- [standalone]: A YES answer to this pathway ALONE is sufficient for coverage.\n'
    '\n'
    'STEP-BY-STEP REASONING (required before deciding):\n'
    '1. List each distinct coverage pathway from the policy context.\n'
    '2. For each pathway, check which required conditions were answered YES, NO, or NOT ASKED.\n'
    '3. Choose the decision using these strict rules:\n'
    '\n'
    '   -> "covered":      At least one pathway has ALL its required conditions answered YES.\n'
    '   -> "not_covered":  Every pathway has at least one condition explicitly answered NO\n'
    '                      (i.e., all pathways are definitively ruled out by the patient\'s answers).\n'
    '   -> "uncertain":    At least one pathway is not ruled out, but has a required condition\n'
    '                      that was NEVER ASKED or left unanswered. List those missing conditions\n'
    '                      in "missing_info".\n'
    '\n'
    'CRITICAL: Do NOT return "not_covered" if a pathway could still apply but a required condition\n'
    'was simply never asked. That is "uncertain". Only use "not_covered" when the patient\'s own\n'
    'answers explicitly disqualify all pathways.\n'
    '\n'
    'CRITICAL SCOPE RULE:\n'
    '- If the procedure is NOT mentioned in the provided policy context -> return "uncertain".\n'
    '- Do NOT invent requirements not in the policy text.\n'
    '\n'
    'Additional rules:\n'
    '- Use ONLY the provided context + answers.\n'
    '- Treat "none_apply = true" as: patient answered NO to every pathway.\n'
    '- Explain briefly in plain English (1-3 sentences).\n'
    '- Return STRICT JSON only.\n'
    '\n'
    'Output JSON EXACTLY:\n'
    '{\n'
    '  "decision": "covered" | "not_covered" | "uncertain",\n'
    '  "confidence": 0.0,\n'
    '  "reason": "<1-3 short sentences in plain English>",\n'
    '  "missing_info": ["<short item describing what was not asked>", "..."]\n'
    '}\n'
)


class ResponseGenerator(BaseResponseGenerator):
    def generate_response(self, query: str, context: List[str]) -> str:
        context_text = "\n".join(context)
        user_message = (
            f"<context>\n{context_text}\n</context>\n"
            f"<question>\n{query}\n</question>"
        )
        return invoke_ai(system_message=SYSTEM_PROMPT, user_message=user_message)

    def generate_decision(
        self,
        query: str,
        context: List[str],
        questions: List[str],
        answers: List[bool],
        labels: Optional[List[int]] = None,
        none_apply: bool = False,
    ) -> str:
        context_text = "\n".join(context)
        qa_lines = []
        n = min(len(questions), len(answers))
        for i in range(n):
            qa_lines.append(f"- P{i+1} [standalone]: {questions[i]}\n  A: {'YES' if answers[i] else 'NO'}")
        qa_text = "\n".join(qa_lines) if qa_lines else "(no pathways)"

        user_message = (
            f"<context>\n{context_text}\n</context>\n"
            f"<procedure_query>\n{query}\n</procedure_query>\n"
            f"<none_apply>{'true' if none_apply else 'false'}</none_apply>\n"
            f"<pathways_answered>\n{qa_text}\n</pathways_answered>\n"
        )
        return invoke_ai(system_message=DECISION_SYSTEM_PROMPT, user_message=user_message)
