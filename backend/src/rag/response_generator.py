from typing import List
from src.rag_interface.base_response_generator import BaseResponseGenerator

from src.invoke_ai import invoke_ai


SYSTEM_PROMPT = """
You are an insurance eligibility EXPERT.

Your job is to ask the patient simple, human-readable YES/NO questions. 
Don't ask anything they would not know. Don't ask what you would ask medical personel. 

Rules:
- Do NOT mention CPT codes, modifiers, or billing jargon.
- Translate technical policy rules into plain English.
- Assume the user is not a medical or billing expert.
- Use everyday language a patient or clinic staff would understand.
- Each question must be answerable with yes or no.
- Use ONLY the provided context.
- Do NOT add new rules.

Quality Rules:
- Combine similar or overlapping rules.
- Remove duplicates.
- Minimize the list while preserving decision accuracy.

DO NOT ASK:
- whether the user wants coverage
- whether insurance should pay
- whether doctor knows
- questions about administrative workflow


Format EXACTLY:
Title: <plain English title>
Questions:
- <question?>
- <question?>

try to fit everything into 4-8 questions. 

"""

DECISION_SYSTEM_PROMPT = """
You are an insurance eligibility EXPERT.

You will be given:
- policy context excerpts (authoritative)
- a procedure query
- the YES/NO questions shown to the user
- the user's answers

Task:
Decide if the procedure is:
- covered
- not_covered
- uncertain

CRITICAL SCOPE RULE:
- If the procedure is NOT mentioned, described, or reasonably inferable from the provided policy context,
  you MUST return decision = "uncertain".
- In this case, clearly state that the procedure is outside the scope of the provided policy documents.
- Do NOT invent requirements.

Rules:
- Use ONLY the provided context + answers.
- Treat "none_apply = true" as: user answered NO to every question.
- If context does not clearly support coverage or non-coverage, return "uncertain".
- Explain briefly in plain English.
- Return STRICT JSON only.

Output JSON EXACTLY:
{
  "decision": "covered" | "not_covered" | "uncertain",
  "confidence": 0.0,
  "reason": "<1-3 short sentences>",
  "missing_info": ["<short item>", "..."]
}
"""

class ResponseGenerator(BaseResponseGenerator):
    def generate_response(self, query: str, context: List[str]) -> str:
        print("SYSTEM_PROMPT IN USE:\n", SYSTEM_PROMPT)
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
        none_apply: bool = False,
    ) -> str:
        context_text = "\n".join(context)
        qa_lines = []
        n = min(len(questions), len(answers))
        for i in range(n):
            qa_lines.append(f"- Q{i+1}: {questions[i]}\n  A: {'YES' if answers[i] else 'NO'}")
        qa_text = "\n".join(qa_lines) if qa_lines else "(no questions)"

        user_message = (
            f"<context>\n{context_text}\n</context>\n"
            f"<procedure_query>\n{query}\n</procedure_query>\n"
            f"<none_apply>{'true' if none_apply else 'false'}</none_apply>\n"
            f"<qa>\n{qa_text}\n</qa>\n"
        )
        return invoke_ai(system_message=DECISION_SYSTEM_PROMPT, user_message=user_message)



