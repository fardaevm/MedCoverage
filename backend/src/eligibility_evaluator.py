# backend/src/eligibility_evaluator.py
"""
Deterministic eligibility evaluator.
Walks pre-generated questions per pathway — zero LLM calls per turn.
"""
from __future__ import annotations
from typing import Any


def evaluate_answers(
    pathways: list[str],
    questions_per_pathway: list[list[str]],
    qa_so_far: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Pure-logic decision engine.

    For each pathway, check if every question was answered YES.
    - All YES for any pathway        → covered
    - Any NO in every pathway         → not_covered
    - Unanswered questions remain     → uncertain + next_question

    Returns:
        {
            "decision": "covered" | "not_covered" | "uncertain",
            "reason": str,
            "pathway_index": int | None,
            "next_question": str | None,
            "completeness": float,        # 0-1, answered / total
        }
    """
    answers: dict[str, bool] = {item["q"]: item["a"] for item in qa_so_far}

    total_questions = sum(len(qs) for qs in questions_per_pathway)
    answered_count = sum(1 for qs in questions_per_pathway for q in qs if q in answers)
    completeness = (answered_count / total_questions) if total_questions > 0 else 0.0

    alive_pathways: list[int] = []  # indices of pathways not yet ruled out

    for i, questions in enumerate(questions_per_pathway):
        if not questions:
            continue
        results = [answers.get(q) for q in questions]

        # All YES → pathway satisfied → covered
        if all(r is True for r in results):
            return {
                "decision": "covered",
                "reason": _covered_reason(pathways[i]),
                "pathway_index": i,
                "next_question": None,
                "completeness": completeness,
            }

        # Any NO → pathway ruled out
        if any(r is False for r in results):
            continue

        # Pathway still alive (has unanswered questions, no NO yet)
        alive_pathways.append(i)

    # No pathway satisfied — check if any are still alive
    if alive_pathways:
        # Pick next unanswered question from the first alive pathway
        for i in alive_pathways:
            for q in questions_per_pathway[i]:
                if q not in answers:
                    return {
                        "decision": "uncertain",
                        "reason": "A few more questions to determine coverage.",
                        "pathway_index": None,
                        "next_question": q,
                        "completeness": completeness,
                    }

    # All pathways ruled out
    return {
        "decision": "not_covered",
        "reason": _not_covered_reason(pathways),
        "pathway_index": None,
        "next_question": None,
        "completeness": completeness,
    }


def _covered_reason(pathway: str) -> str:
    if pathway:
        return f"Based on your answers, this procedure is covered: {pathway}"
    return "Based on your answers, this procedure is covered under your plan."


def _not_covered_reason(pathways: list[str]) -> str:
    valid = [p for p in pathways if (p or "").strip()]
    if not valid:
        return "We couldn't find a coverage pathway that matches your situation."
    return (
        "Based on your answers, none of the coverage pathways apply to your situation."
    )