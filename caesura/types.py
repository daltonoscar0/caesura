"""The pipeline contract, and the small internal types used to build it.

A stage returns a ``StageResult``: a plain, JSON-serialisable dict, not a custom
class, because a sibling stage has to parse it without importing this package.

    {
      "stage":     "caesura",
      "input":     str,              # text this stage received
      "output":    str,              # text this stage produced, with markup
      "tokens":    [str, ...],       # whitespace tokens of `output`
      "decisions": [Decision, ...],  # only where the stage did something
      "meta":      { ... }           # stage-specific, free-form
    }

and each decision:

    {
      "span":         [start, end],  # char offsets into `input`
      "surface":      str,           # what was there
      "result":       str,           # what it became
      "kind":         "break",
      "rule":         str,           # rule function name, or "model"
      "alternatives": [{"result": str, "score": float, ...}, ...],
      "score":        float,
      "note":         str | None
    }

``caesura``'s ``output`` is the input tokens with break markers inserted,
``<b1>`` minor, ``<b2>`` intermediate, ``<b3>`` major or final, and ``*word*``
for emphasised tokens.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

STAGE = "caesura"

# Break inventory. <b1> is a minor juncture and is never predicted here: it is
# not derivable from punctuation, which is the only supervision available.
NONE = "none"
B1 = "b1"
B2 = "b2"
B3 = "b3"

BREAK_TAGS = {B1: "<b1>", B2: "<b2>", B3: "<b3>"}
LABELS = [NONE, B2, B3]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}

#: Strength ordering, used when two systems disagree about a boundary.
STRENGTH = {NONE: 0, B1: 1, B2: 2, B3: 3}


@dataclass
class Token:
    """Internal working type. Never appears in a ``StageResult``."""

    text: str
    index: int
    start: int
    end: int
    surface: str
    #: Break that follows this token: ``none``/``b1``/``b2``/``b3``.
    brk: str = NONE
    emphasis: bool = False

    def render(self) -> str:
        out = f"*{self.text}*" if self.emphasis else self.text
        if self.brk != NONE:
            out = f"{out} {BREAK_TAGS[self.brk]}"
        return out


def render(tokens: Sequence[Token]) -> str:
    """Join tokens into the marked-up ``output`` string."""
    return " ".join(t.render() for t in tokens)


def stronger(a: str, b: str) -> str:
    return a if STRENGTH[a] >= STRENGTH[b] else b


def alternative(result: str, score: float, system: str, rule: str) -> Dict[str, Any]:
    """One entry of a decision's ``alternatives`` list.

    ``result`` and ``score`` are what the contract requires; ``system`` and
    ``rule`` are extra and say which of the two systems wanted it.
    """
    return {
        "result": result,
        "score": round(float(score), 4),
        "system": system,
        "rule": rule,
    }


def decision(
    token: Token,
    label: str,
    rule: str,
    score: float,
    alternatives: Optional[List[Dict[str, Any]]] = None,
    note: Optional[str] = None,
) -> Dict[str, Any]:
    """One inserted break, expressed against the stage's own input."""
    return {
        "span": [token.start, token.end],
        "surface": token.surface,
        "result": f"{token.text} {BREAK_TAGS[label]}",
        "kind": "break",
        "rule": rule,
        "alternatives": alternatives or [],
        "score": round(float(score), 4),
        "note": note,
    }


def stage_result(
    input_text: str,
    tokens: Sequence[Token],
    decisions: List[Dict[str, Any]],
    meta: Dict[str, Any],
) -> Dict[str, Any]:
    """Assemble the contract dict."""
    output = render(tokens)
    return {
        "stage": STAGE,
        "input": input_text,
        "output": output,
        "tokens": output.split(),
        "decisions": decisions,
        "meta": meta,
    }


def breaks(result: Dict[str, Any]) -> List[str]:
    """Per-boundary label sequence, one entry per word, from a StageResult."""
    return list(result["meta"]["breaks"])


def words(result: Dict[str, Any]) -> List[str]:
    """The normalised words, without markup, from a StageResult."""
    return list(result["meta"]["words"])


def emphasised(result: Dict[str, Any]) -> List[str]:
    return list(result["meta"]["emphasis"])
