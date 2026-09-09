"""Normalisation and weak-label derivation.

Two directions:

``normalize`` goes text -> the punctuation-free lowercase token list the two
systems actually see. Everything downstream of this module is forbidden from
looking at punctuation, because the premise of the project is that punctuation
will not be there at inference time.

``labels_from_punctuation`` goes punctuated text -> (tokens, break labels). This
is the weak supervision described in the README: it is a proxy for prosody, not
prosody.
"""

from __future__ import annotations

import re
import unicodedata
from typing import List, Tuple

from .types import B2, B3, NONE

#: Punctuation that we read as a minor (phrase-internal) break.
B2_PUNCT = frozenset(",;:")
#: Punctuation that we read as a major (utterance-final) break.
B3_PUNCT = frozenset(".!?")

#: Dashes are treated as b2 when they separate material, which is their normal
#: use in the Gutenberg-derived LibriTTS text.
DASHES = ("—", "–", "--")

_MARKUP = re.compile(r"<b[123]>")
_EMPHASIS = re.compile(r"\*([^*]+)\*")
_APOSTROPHES = "’ʼ´`"
_WORD_CHARS = re.compile(r"[^a-z0-9']+")
_MULTISPACE = re.compile(r"\s+")


def strip_markup(text: str) -> str:
    """Remove prosody markup an upstream stage may have already inserted."""
    text = _MARKUP.sub(" ", text)
    return _EMPHASIS.sub(r"\1", text)


def normalize(text: str) -> List[str]:
    """Lowercase, strip punctuation and markup, return bare word tokens.

    Intra-word apostrophes survive ("don't" stays one token) because they are
    lexical rather than prosodic. Everything else goes.
    """
    text = strip_markup(text)
    text = unicodedata.normalize("NFKC", text)
    for dash in DASHES:
        text = text.replace(dash, " ")
    for apos in _APOSTROPHES:
        text = text.replace(apos, "'")
    text = text.lower()
    text = _WORD_CHARS.sub(" ", text)
    text = text.replace(" ' ", " ")
    text = _MULTISPACE.sub(" ", text).strip()
    if not text:
        return []
    tokens = [t.strip("'") for t in text.split(" ")]
    return [t for t in tokens if t]


def normalized_text(text: str) -> str:
    return " ".join(normalize(text))


def _trailing_break(chunk: str) -> str:
    """Classify the punctuation trailing a raw whitespace-delimited chunk.

    Reads right to left through closing quotes and brackets so that
    ``disappeared!"`` and ``(later),`` classify on the real punctuation mark.
    Sentence-final beats phrase-final when both are present.
    """
    saw_b2 = False
    for ch in reversed(chunk):
        if ch in B3_PUNCT:
            return B3
        if ch in B2_PUNCT:
            saw_b2 = True
            continue
        if ch in "\"')]}”’…":
            if ch == "…":
                return B3
            continue
        if ch == "-" or ch in DASHES:
            saw_b2 = True
            continue
        break
    return B2 if saw_b2 else NONE


def labels_from_punctuation(text: str) -> Tuple[List[str], List[str]]:
    """Derive ``(tokens, labels)`` from punctuated text.

    ``tokens`` is what ``normalize`` would produce; ``labels[i]`` is the break
    that follows ``tokens[i]``. Chunks that normalise to more than one token
    (rare: hyphen compounds) put the label on the last of them.
    """
    text = strip_markup(unicodedata.normalize("NFKC", text))
    # Brackets separate parenthetical material the same way a dash does. Quotes
    # deliberately do not: direct speech is pervasive in this corpus and every
    # quote mark would become a spurious break.
    for bracket in "()[]{}":
        text = text.replace(bracket, " — ")
    for dash in DASHES:
        text = text.replace(dash, " — ")

    tokens: List[str] = []
    labels: List[str] = []
    pending = NONE

    for chunk in text.split():
        if chunk == "—":
            # A free-standing dash separates what precedes from what follows.
            if labels:
                labels[-1] = _stronger(labels[-1], B2)
            continue
        words = normalize(chunk)
        brk = _trailing_break(chunk)
        if not words:
            # Punctuation-only chunk: attach its break to the previous token.
            if labels and brk != NONE:
                labels[-1] = _stronger(labels[-1], brk)
            continue
        if pending != NONE and labels:
            labels[-1] = _stronger(labels[-1], pending)
        pending = NONE
        for w in words[:-1]:
            tokens.append(w)
            labels.append(NONE)
        tokens.append(words[-1])
        labels.append(brk)

    return tokens, labels


_STRENGTH = {NONE: 0, B2: 1, B3: 2}


def _stronger(a: str, b: str) -> str:
    return a if _STRENGTH[a] >= _STRENGTH[b] else b
