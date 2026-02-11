# backend/src/eligibility_confidence.py
def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if x < lo else hi if x > hi else x

def rule_confidence(
    *,
    questions: list[str],
    answers: list[bool],
    none_apply: bool,
    llm_decision: str | None = None,
) -> float:
    n_q = len(questions or [])
    n_a = min(len(answers or []), n_q)
    yes = sum(1 for i in range(n_a) if answers[i] is True)

    if n_q == 0:
        ans_strength = 0.10
    elif none_apply:
        ans_strength = 0.80 if n_a >= n_q else 0.65
    else:
        # more YES triggers => stronger rule application signal (for exclusion-style questions)
        ans_strength = 0.55 + 0.45 * _clamp(yes / n_q)

    completeness = 0.35 + 0.65 * _clamp(n_a / max(1, n_q))

    contradiction = 0.35 if (none_apply and yes > 0) else 0.0

    conf = _clamp(
        0.20
        + 0.55 * ans_strength
        + 0.25 * completeness
        - contradiction
    )

    if llm_decision == "uncertain":
        conf = min(conf, 0.55)

    return conf
