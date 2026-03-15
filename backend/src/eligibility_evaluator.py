# backend/src/eligibility_evaluator.py
"""
Single-tree eligibility evaluator.

One big tree per procedure. Top-level OR branches = pathways.
Each branch can have an optional "pathway" label for user display.

{
  "operator": "OR",
  "conditions": [
    {
      "pathway": "Routine screening age 40+",
      "operator": "AND",
      "conditions": [
        {"q_id": "age_40_plus", "value": true},
        {"q_id": "365_days_since_last", "value": true}
      ]
    },
    {
      "pathway": "High-risk genetic/family",
      "operator": "AND",
      "conditions": [
        {"operator": "OR", "conditions": [
          {"q_id": "brca_mutation", "value": true},
          {"q_id": "family_history", "value": true}
        ]}
      ]
    }
  ]
}

Backward compatible with:
- logic_trees[] (per-pathway) → auto-merged into single OR
- questions_per_pathway (flat) → auto-converted to AND branches then merged
"""
from __future__ import annotations
from typing import Any


# ── Tree evaluation ────────────────────────────────────────────────

def _eval_node(node: dict, answers: dict[str, bool]) -> bool | None:
    if "q_id" in node:
        q_id = node["q_id"]
        expected = node.get("value", True)
        if q_id not in answers:
            return None
        return answers[q_id] == expected

    op = node.get("operator", "AND").upper()
    children = node.get("conditions", [])

    if op == "NOT":
        if not children:
            return None
        inner = _eval_node(children[0], answers)
        return None if inner is None else (not inner)

    if op == "AND":
        has_none = False
        for child in children:
            r = _eval_node(child, answers)
            if r is False:
                return False
            if r is None:
                has_none = True
        return None if has_none else True

    if op == "OR":
        has_none = False
        for child in children:
            r = _eval_node(child, answers)
            if r is True:
                return True
            if r is None:
                has_none = True
        return None if has_none else False

    return None


def _collect_question_ids(node: dict) -> set[str]:
    if "q_id" in node:
        return {node["q_id"]}
    ids = set()
    for child in node.get("conditions", []):
        ids |= _collect_question_ids(child)
    return ids


def _collect_unanswered(node: dict, answers: dict[str, bool]) -> set[str]:
    if "q_id" in node:
        return {node["q_id"]} if node["q_id"] not in answers else set()

    op = node.get("operator", "AND").upper()
    children = node.get("conditions", [])

    if op == "NOT":
        return _collect_unanswered(children[0], answers) if children else set()

    if op == "AND":
        unanswered = set()
        for child in children:
            if _eval_node(child, answers) is False:
                return set()
            unanswered |= _collect_unanswered(child, answers)
        return unanswered

    if op == "OR":
        unanswered = set()
        for child in children:
            if _eval_node(child, answers) is True:
                return set()
            unanswered |= _collect_unanswered(child, answers)
        return unanswered

    return set()


# ── Format conversion ─────────────────────────────────────────────

def _merge_trees(pathways: list[str], logic_trees: list[dict]) -> dict:
    branches = []
    for i, tree in enumerate(logic_trees):
        branch = dict(tree)
        if i < len(pathways):
            branch["pathway"] = pathways[i]
        branches.append(branch)
    return {"operator": "OR", "conditions": branches}


def _legacy_to_tree(pathways: list[str], questions_per_pathway: list[list[str]]) -> dict:
    branches = []
    for i, qs in enumerate(questions_per_pathway):
        branch = {
            "operator": "AND",
            "conditions": [{"q_id": q, "value": True} for q in qs],
        }
        if i < len(pathways):
            branch["pathway"] = pathways[i]
        branches.append(branch)
    return {"operator": "OR", "conditions": branches}


# ── Pathway tracking ──────────────────────────────────────────────

def _find_covered_pathway(tree: dict, answers: dict[str, bool]) -> str | None:
    if tree.get("operator", "").upper() != "OR":
        return tree.get("pathway")
    for child in tree.get("conditions", []):
        if _eval_node(child, answers) is True:
            return child.get("pathway")
    return None


def _get_alive_branches(tree: dict, answers: dict[str, bool]) -> list[dict]:
    if tree.get("operator", "").upper() != "OR":
        return [tree] if _eval_node(tree, answers) is not False else []
    return [c for c in tree.get("conditions", []) if _eval_node(c, answers) is not False]


# ── Question picker ───────────────────────────────────────────────

def _pick_next_question(
    tree: dict,
    answers: dict[str, bool],
    question_map: dict[str, str] | None = None,
) -> str | None:
    alive = _get_alive_branches(tree, answers)
    if not alive:
        return None

    q_count: dict[str, int] = {}
    for branch in alive:
        for q_id in _collect_unanswered(branch, answers):
            q_count[q_id] = q_count.get(q_id, 0) + 1

    if not q_count:
        return None

    best = max(q_count, key=q_count.get)
    if question_map and best in question_map:
        return question_map[best]
    return best


# ── Build single tree from any input format ───────────────────────

def _build_tree(
    pathways: list[str],
    questions_per_pathway: list[list[str]],
    logic_trees: list[dict] | None = None,
    logic_tree: dict | None = None,
) -> dict:
    if logic_tree and isinstance(logic_tree, dict):
        return logic_tree
    if logic_trees and len(logic_trees) == len(pathways):
        return _merge_trees(pathways, logic_trees)
    return _legacy_to_tree(pathways, questions_per_pathway)


# ── Main evaluator ────────────────────────────────────────────────

def evaluate_answers(
    pathways: list[str],
    questions_per_pathway: list[list[str]],
    qa_so_far: list[dict[str, Any]],
    logic_trees: list[dict] | None = None,
    logic_tree: dict | None = None,
    question_map: dict[str, str] | None = None,
) -> dict[str, Any]:
    # Build answers
    answers: dict[str, bool] = {}
    for item in qa_so_far:
        answers[item["q"]] = item["a"]
    if question_map:
        rev = {v: k for k, v in question_map.items()}
        for item in qa_so_far:
            t = item["q"]
            if t in rev:
                answers[rev[t]] = item["a"]
            if t in question_map:
                answers[question_map[t]] = item["a"]

    tree = _build_tree(pathways, questions_per_pathway, logic_trees, logic_tree)

    all_qids = _collect_question_ids(tree)
    answered = {q for q in all_qids if q in answers}
    completeness = (len(answered) / len(all_qids)) if all_qids else 0.0

    result = _eval_node(tree, answers)

    if result is True:
        pw = _find_covered_pathway(tree, answers)
        return {
            "decision": "covered",
            "reason": _covered_reason(pw),
            "pathway": pw,
            "next_question": None,
            "completeness": completeness,
        }

    if result is False:
        return {
            "decision": "not_covered",
            "reason": _not_covered_reason(),
            "pathway": None,
            "next_question": None,
            "completeness": completeness,
        }

    next_q = _pick_next_question(tree, answers, question_map)
    if next_q:
        return {
            "decision": "uncertain",
            "reason": "A few more questions to determine coverage.",
            "pathway": None,
            "next_question": next_q,
            "completeness": completeness,
        }

    return {
        "decision": "not_covered",
        "reason": _not_covered_reason(),
        "pathway": None,
        "next_question": None,
        "completeness": completeness,
    }


def _covered_reason(pathway: str | None = None) -> str:
    base = "Based on your answers, this procedure is covered under your Medi-Cal plan."
    if pathway:
        return f"{base} Pathway: {pathway}"
    return base


def _not_covered_reason() -> str:
    return "Based on your answers, this procedure does not appear to be covered under your Medi-Cal plan."