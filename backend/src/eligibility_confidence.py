# backend/src/eligibility_confidence.py

def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp x into [lo, hi] to keep confidence in a valid range."""
    return lo if x < lo else hi if x > hi else x


def rule_confidence(
    *,
    questions: list[str],
    answers: list[bool],
    none_apply: bool,
    llm_decision: str | None = None,
) -> float:
    """
    Confidence heuristic for STANDALONE pathways (OR logic).

    Assumptions:
      - Each question/pathway is independently sufficient for coverage.
      - Any YES => at least one coverage pathway is satisfied.
      - none_apply=True means the patient explicitly says NO to all pathways.

    Inputs:
      - questions: list of pathway texts (standalone)
      - answers: parallel list of booleans (True = checked/YES)
      - none_apply: patient checked "None of these apply to me"
      - llm_decision: optional final decision ("covered" | "not_covered" | "uncertain")

    Output:
      - confidence in [0, 1]
    """
    n_q = len(questions or [])
    n_a = min(len(answers or []), n_q)

    # If we have no pathways at all, we are not grounded -> keep confidence low.
    if n_q == 0:
        base = 0.15
        if llm_decision == "covered":
            base = 0.35  # slightly higher but still cautious without pathways
        return _clamp(base)

    # Completeness reflects how much of the questionnaire was actually answered.
    # - 1.0 means we have answers for every pathway
    # - lower values mean missing info, so reduce confidence.
    completeness = _clamp(n_a / n_q)

    # If the user explicitly says "none apply", treat that as a strong exclusion signal.
    # The more complete the answers, the higher the confidence for not-covered.
    if none_apply:
        conf = 0.55 + 0.35 * completeness  # 0.55..0.90
    else:
        # Count YES answers among those we actually received.
        yes = sum(1 for i in range(n_a) if answers[i] is True)

        # Standalone pathways => any YES implies a coverage condition is met.
        if yes > 0:
            conf = 0.75 + 0.20 * completeness  # 0.75..0.95
        else:
            # No YES selected, and user did not say "none apply".
            # This could mean "not covered" OR simply missing/unclear info.
            conf = 0.35 + 0.30 * completeness  # 0.35..0.65

    # If the LLM itself says "uncertain", cap confidence to reflect ambiguity
    # (even if the raw heuristic would be higher).
    if llm_decision == "uncertain":
        conf = min(conf, 0.60)

    return _clamp(conf)