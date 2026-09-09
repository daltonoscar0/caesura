"""Contract entry point for the Caesura stage.

``run(text, **opts)`` is the only function other stages should call. It always
strips punctuation, casing and any upstream markup before predicting anything:
the premise of the stage is that punctuation will not be there at inference
time, so letting it leak in would make every number a lie.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import emphasis as emphasis_mod
from . import rules as rules_mod
from .text import normalize_spans
from .types import (
    B2,
    B3,
    BREAK_TAGS,
    NONE,
    STAGE,
    Token,
    alternative,
    decision,
    stage_result,
    stronger,
)

SYSTEMS = ("rules", "model", "both")


def predict_rules(tokens: Sequence[str], doc=None):
    return rules_mod.apply(tokens, doc=doc)


def predict_model(tokens: Sequence[str], model_path: Optional[str] = None):
    from .model import get_model

    return get_model(model_path).predict(tokens)


def _merge(
    rule_labels: Sequence[str],
    rule_names: Dict[int, str],
    model_labels: Sequence[str],
    model_scores: Sequence[float],
) -> Tuple[List[str], List[str], List[float]]:
    """Recall-oriented union: take the stronger break at each boundary.

    Where the two systems agree the boundary is owned by ``consensus:<rule>``;
    where only one fires, that system owns it and the caller records the other's
    choice in ``alternatives``.
    """
    merged: List[str] = []
    owners: List[str] = []
    scores: List[float] = []
    for i, (r, m) in enumerate(zip(rule_labels, model_labels)):
        best = stronger(r, m)
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


def _note(rule: str, label: str) -> str:
    tag = BREAK_TAGS[label]
    if rule == "model":
        return f"sequence labeller put {tag} here"
    if rule.startswith("consensus:"):
        return f"both systems put {tag} here, rule {rule.split(':', 1)[1]}"
    return f"rule {rule} put {tag} here"


def run(
    text: str,
    system: str = "rules",
    model_path: Optional[str] = None,
    emphasis: bool = True,
) -> Dict[str, Any]:
    """Predict prosodic breaks and emphasis for ``text``.

    Args:
        text: any string. Punctuation, casing and prior prosody markup are
            stripped before prediction.
        system: ``"rules"`` (System A), ``"model"`` (System B) or ``"both"``
            (union of the two, with each system's choice recorded).
        model_path: local directory or Hub id for System B.
        emphasis: set False to skip emphasis marking, which also skips the parse
            on the model-only path. Used by the speed benchmark.

    Returns:
        A StageResult dict as documented in :mod:`caesura.types`.
    """
    if system not in SYSTEMS:
        raise ValueError(f"system must be one of {SYSTEMS}, got {system!r}")

    started = time.perf_counter()
    spans = normalize_spans(text)
    words = [w for w, _, _, _ in spans]

    if not words:
        return stage_result(
            text,
            [],
            [],
            {
                "system": system,
                "n_tokens": 0,
                "seconds": 0.0,
                "words": [],
                "breaks": [],
                "emphasis": {},
                "vetoed": {},
            },
        )

    tokens = [
        Token(text=w, index=i, start=start, end=end, surface=surface)
        for i, (w, start, end, surface) in enumerate(spans)
    ]

    needs_parse = system in ("rules", "both") or emphasis
    doc = rules_mod.parse(words) if needs_parse else None

    rule_labels: List[str] = []
    rule_names: Dict[int, str] = {}
    vetoed: Dict[int, str] = {}
    model_labels: List[str] = []
    model_scores: List[float] = []

    if system in ("rules", "both"):
        rule_labels, rule_names, vetoed = predict_rules(words, doc=doc)
    if system in ("model", "both"):
        model_labels, model_scores = predict_model(words, model_path)

    if system == "rules":
        labels = rule_labels
        owners = [rule_names.get(i, "") for i in range(len(words))]
        scores = [1.0] * len(words)
    elif system == "model":
        labels = model_labels
        owners = ["model" if label != NONE else "" for label in labels]
        scores = model_scores
    else:
        labels, owners, scores = _merge(rule_labels, rule_names, model_labels, model_scores)

    for token, label in zip(tokens, labels):
        token.brk = label

    emphasis_marks: Dict[int, str] = {}
    if emphasis and doc is not None:
        emphasis_marks = emphasis_mod.mark(doc, labels)
        for i in emphasis_marks:
            tokens[i].emphasis = True

    decisions: List[Dict[str, Any]] = []
    for i, label in enumerate(labels):
        if label == NONE:
            continue
        alternatives: List[Dict[str, Any]] = []
        if system == "both" and not owners[i].startswith("consensus"):
            if owners[i] == "model":
                other = rule_labels[i]
                alternatives.append(
                    alternative(
                        f"{words[i]} {BREAK_TAGS[other]}" if other != NONE else words[i],
                        1.0,
                        "rules",
                        rule_names.get(i, "no_rule_fired"),
                    )
                )
            else:
                other = model_labels[i]
                alternatives.append(
                    alternative(
                        f"{words[i]} {BREAK_TAGS[other]}" if other != NONE else words[i],
                        model_scores[i],
                        "model",
                        "model",
                    )
                )
        decisions.append(
            decision(
                tokens[i],
                label,
                rule=owners[i] or system,
                score=scores[i],
                alternatives=alternatives,
                note=_note(owners[i] or system, label),
            )
        )

    meta: Dict[str, Any] = {
        "system": system,
        "n_tokens": len(words),
        "seconds": round(time.perf_counter() - started, 4),
        "words": words,
        "breaks": labels,
        "emphasis": {words[i]: rule for i, rule in sorted(emphasis_marks.items())},
        "vetoed": {str(i): reason for i, reason in sorted(vetoed.items())},
    }
    if system == "both":
        meta["rules_breaks"] = rule_labels
        meta["model_breaks"] = model_labels

    return stage_result(text, tokens, decisions, meta)
