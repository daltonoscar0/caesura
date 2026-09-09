"""Emphasis marking, rule-based only.

There is no gold emphasis annotation anywhere in this project, so this stays
deliberately small: a default nuclear accent on the last content word of each
intonational phrase, plus a handful of explicitly contrastive configurations
that override the default. It is evaluated qualitatively on the adversarial set
and nowhere else, and the README says so.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from .rules import CONTENT_POS
from .types import NONE

CONTRAST_MARKERS = {"rather", "instead", "not", "n't", "never", "but"}


def _phrases(labels: Sequence[str]) -> List[Tuple[int, int]]:
    """Split boundary labels into (start, end) intonational phrases."""
    out: List[Tuple[int, int]] = []
    start = 0
    for i, label in enumerate(labels):
        if label != NONE:
            out.append((start, i))
            start = i + 1
    if start < len(labels):
        out.append((start, len(labels) - 1))
    return out


def emph_nuclear(doc, labels) -> Dict[int, str]:
    """Default: the last content word of each intonational phrase."""
    marks: Dict[int, str] = {}
    for start, end in _phrases(labels):
        for i in range(end, start - 1, -1):
            if doc[i].pos_ in CONTENT_POS:
                marks[i] = "emph_nuclear"
                break
    return marks


def emph_negation_focus(doc, labels) -> Dict[int, str]:
    """``not X`` puts focus on X."""
    marks: Dict[int, str] = {}
    for tok in doc:
        if tok.lower_ not in ("not", "n't", "never"):
            continue
        for nxt in doc[tok.i + 1 : min(tok.i + 4, len(doc))]:
            if nxt.pos_ in CONTENT_POS:
                marks[nxt.i] = "emph_negation_focus"
                break
    return marks


def emph_rather_than(doc, labels) -> Dict[int, str]:
    """``X rather than Y`` / ``X instead of Y`` accents both X and Y."""
    marks: Dict[int, str] = {}
    for tok in doc:
        if tok.lower_ not in ("rather", "instead"):
            continue
        nxt = doc[tok.i + 1] if tok.i + 1 < len(doc) else None
        if nxt is None or nxt.lower_ not in ("than", "of"):
            continue
        for prev in range(tok.i - 1, -1, -1):
            if doc[prev].pos_ in CONTENT_POS:
                marks[prev] = "emph_rather_than"
                break
        for after in range(tok.i + 2, len(doc)):
            if doc[after].pos_ in CONTENT_POS:
                marks[after] = "emph_rather_than"
                break
    return marks


def emph_contrastive_but(doc, labels) -> Dict[int, str]:
    """``not X but Y``: accent the head of each conjunct around ``but``."""
    marks: Dict[int, str] = {}
    for tok in doc:
        if tok.lower_ != "but" or tok.dep_ != "cc":
            continue
        head = tok.head
        conj = next(
            (c for c in head.children if c.dep_ == "conj" and c.i > tok.i), None
        )
        if conj is None:
            continue
        negated = any(
            t.lower_ in ("not", "n't", "never") for t in doc[: tok.i]
        )
        if not negated:
            continue
        marks[head.i] = "emph_contrastive_but"
        marks[conj.i] = "emph_contrastive_but"
    return marks


#: Later entries override earlier ones, so contrast beats the default accent.
EMPHASIS_RULES = [
    emph_nuclear,
    emph_negation_focus,
    emph_rather_than,
    emph_contrastive_but,
]


def mark(doc, labels: Sequence[str]) -> Dict[int, str]:
    """Return ``{token_index: rule_name}`` for every emphasised token."""
    marks: Dict[int, str] = {}
    for rule in EMPHASIS_RULES:
        marks.update(rule(doc, labels))
    return marks
