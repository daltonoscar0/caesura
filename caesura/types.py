"""Shared types for the Caesura stage of the TTS front end.

The pipeline contract is small on purpose: a stage takes a string and returns a
``StageResult`` that carries (a) the marked-up text, (b) a token-level view of
what was marked, and (c) one ``Decision`` per edit the stage made, so a later
stage or a human can audit why a break landed where it did.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

# Break inventory. <b1> is a word-internal / minor juncture and is never
# predicted here (see README: it is not derivable from punctuation).
NONE = "none"
B1 = "b1"
B2 = "b2"
B3 = "b3"

BREAK_TAGS = {B1: "<b1>", B2: "<b2>", B3: "<b3>"}
LABELS = [NONE, B2, B3]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}


@dataclass
class Token:
    """One output token plus the prosodic marks attached to it."""

    text: str
    index: int
    #: Break that follows this token, one of ``none``/``b1``/``b2``/``b3``.
    brk: str = NONE
    #: True when the token carries a pitch accent worth rendering.
    emphasis: bool = False

    def render(self) -> str:
        out = f"*{self.text}*" if self.emphasis else self.text
        if self.brk != NONE:
            out = f"{out} {BREAK_TAGS[self.brk]}"
        return out


@dataclass
class Alternative:
    """What the other system wanted at this boundary."""

    system: str
    value: str
    score: float
    rule: str


@dataclass
class Decision:
    """One inserted break, with the evidence behind it."""

    #: Index of the token the break follows.
    index: int
    #: Human-readable boundary, ``"band | on"``.
    boundary: str
    #: The break that was inserted, e.g. ``"<b2>"``.
    value: str
    #: Name of the firing rule, or ``"model"`` for the sequence labeller.
    rule: str
    #: Model probability, or 1.0 for a deterministic rule.
    score: float = 1.0
    #: The other system's choice at this boundary (only when system="both").
    alternatives: List[Alternative] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class StageResult:
    """Return value of every stage in the front end."""

    stage: str
    #: Text exactly as it arrived, before normalisation.
    input_text: str
    #: Normalised, punctuation-free text the stage actually operated on.
    normalized_text: str
    #: Marked-up output: tokens, ``*emphasis*``, ``<b2>``/``<b3>``.
    text: str
    tokens: List[Token]
    decisions: List[Decision]
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def emphasis(self) -> List[str]:
        return [t.text for t in self.tokens if t.emphasis]

    def breaks(self) -> List[str]:
        """Per-boundary label sequence, one entry per token."""
        return [t.brk for t in self.tokens]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage": self.stage,
            "input_text": self.input_text,
            "normalized_text": self.normalized_text,
            "text": self.text,
            "tokens": [asdict(t) for t in self.tokens],
            "decisions": [d.to_dict() for d in self.decisions],
            "meta": self.meta,
        }

    def to_json(self, indent: Optional[int] = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


def render(tokens: List[Token]) -> str:
    """Join tokens into the marked-up surface string."""
    return " ".join(t.render() for t in tokens)
