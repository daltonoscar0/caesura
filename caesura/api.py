"""Contract entry point for the Caesura stage.

``run(text, system=...)`` is the only function other stages should call. It
always strips punctuation from the incoming text before predicting anything:
the premise of the stage is that punctuation will not be there at inference
time, so letting it leak in would make the numbers a lie.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Sequence, Tuple

from . import emphasis as emphasis_mod
from . import rules as rules_mod
from .text import normalize
from .types import (
    B2,
    B3,
    NONE,
    Alternative,
    Decision,
    StageResult,
    Token,
    BREAK_TAGS,
    render,
)

STAGE = "caesura"
SYSTEMS = ("rules", "model", "both")

_STRENGTH = {NONE: 0, "b1": 1, B2: 2, B3: 3}


def _stronger(a: str, b: str) -> str:
    return a if _STRENGTH[a] >= _STRENGTH[b] else b


def predict_rules(tokens: Sequence[str], doc=None):
    labels, rule_by_index, vetoed = rules_mod.apply(tokens, doc=doc)
    scores = [1.0 if label != NONE else 1.0 for label in labels]
    return labels, scores, rule_by_index, vetoed


def predict_model(tokens: Sequence[str], model_path: Optional[str] = None):
    from .model import get_model

    labels, scores = get_model(model_path).predict(tokens)
    return labels, scores


def _merge(
    rule_labels: Sequence[str],
    rule_names: Dict[int, str],
    model_labels: Sequence[str],
    model_scores: Sequence[float],
) -> Tuple[List[str], List[str], List[float]]:
    """Recall-oriented union: take the stronger break at each boundary.

    Where the two systems agree the decision is labelled ``consensus``; where
    only one fires, that system owns the boundary and the other's choice is
    recorded as an alternative by the caller.
    """
    merged: List[str] = []
    owners: List[str] = []
    scores: List[float] = []
    for i, (r, m) in enumerate(zip(rule_labels, model_labels)):
        best = _stronger(r, m)
        merged.append(best)
        if best == NONE:
            owners.append("")
            scores.append(1.0)
        elif r == m:
            owners.append(f"consensus:{rule_names.get(i, 'model')}")
            scores.append(model_scores[i])
        elif best == r:
            owners.append(rule_names.get(i, "rules"))
            scores.append(1.0)
        else:
            owners.append("model")
            scores.append(model_scores[i])
    return merged, owners, scores


def run(
    text: str,
    system: str = "rules",
    model_path: Optional[str] = None,
    emphasis: bool = True,
) -> StageResult:
    """Predict prosodic breaks and emphasis for ``text``.

    Args:
        text: any string. Punctuation, casing and prior prosody markup are
            stripped before prediction.
        system: ``"rules"`` (System A), ``"model"`` (System B) or ``"both"``
            (union of the two, with each system's choice recorded).
        model_path: local directory or Hub id for System B.
        emphasis: set False to skip emphasis marking, which skips the parse on
            the model-only path. Used by the speed benchmark.

    Returns:
        A :class:`~caesura.types.StageResult`.
    """
    if system not in SYSTEMS:
        raise ValueError(f"system must be one of {SYSTEMS}, got {system!r}")

    started = time.perf_counter()
    tokens = normalize(text)
    normalized = " ".join(tokens)

    if not tokens:
        return StageResult(
            stage=STAGE,
            input_text=text,
            normalized_text="",
            text="",
            tokens=[],
            decisions=[],
            meta={"system": system, "n_tokens": 0, "seconds": 0.0},
        )

    needs_parse = system in ("rules", "both") or emphasis
    doc = rules_mod.parse(tokens) if needs_parse else None

    rule_labels: List[str] = []
    rule_names: Dict[int, str] = {}
    vetoed: Dict[int, str] = {}
    model_labels: List[str] = []
    model_scores: List[float] = []

    if system in ("rules", "both"):
        rule_labels, _, rule_names, vetoed = predict_rules(tokens, doc=doc)
    if system in ("model", "both"):
        model_labels, model_scores = predict_model(tokens, model_path)

    if system == "rules":
        labels = rule_labels
        owners = [rule_names.get(i, "") for i in range(len(tokens))]
        scores = [1.0] * len(tokens)
    elif system == "model":
        labels = model_labels
        owners = ["model" if label != NONE else "" for label in labels]
        scores = model_scores
    else:
        labels, owners, scores = _merge(
            rule_labels, rule_names, model_labels, model_scores
        )

    out_tokens = [Token(text=t, index=i, brk=labels[i]) for i, t in enumerate(tokens)]

    emphasis_marks: Dict[int, str] = {}
    if emphasis and doc is not None:
        emphasis_marks = emphasis_mod.mark(doc, labels)
        for i in emphasis_marks:
            out_tokens[i].emphasis = True

    decisions: List[Decision] = []
    for i, label in enumerate(labels):
        if label == NONE:
            continue
        nxt = tokens[i + 1] if i + 1 < len(tokens) else "</s>"
        alternatives: List[Alternative] = []
        if system == "both":
            if owners[i].startswith("consensus"):
                pass
            elif owners[i] == "model":
                alternatives.append(
                    Alternative("rules", BREAK_TAGS.get(rule_labels[i], "<none>"), 1.0,
                                rule_names.get(i, "no_rule_fired"))
                )
            else:
                alternatives.append(
                    Alternative("model", BREAK_TAGS.get(model_labels[i], "<none>"),
                                model_scores[i], "model")
                )
        decisions.append(
            Decision(
                index=i,
                boundary=f"{tokens[i]} | {nxt}",
                value=BREAK_TAGS[label],
                rule=owners[i] or system,
                score=round(float(scores[i]), 4),
                alternatives=alternatives,
            )
        )

    meta = {
        "system": system,
        "n_tokens": len(tokens),
        "seconds": round(time.perf_counter() - started, 4),
        "emphasis": {tokens[i]: rule for i, rule in sorted(emphasis_marks.items())},
        "vetoed": {str(i): reason for i, reason in sorted(vetoed.items())},
    }
    if system == "both":
        meta["rules_labels"] = rule_labels
        meta["model_labels"] = model_labels

    return StageResult(
        stage=STAGE,
        input_text=text,
        normalized_text=normalized,
        text=render(out_tokens),
        tokens=out_tokens,
        decisions=decisions,
        meta=meta,
    )
