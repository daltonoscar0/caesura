"""Build the training and evaluation sets.

Three sources, all reduced to the same shape: a list of
``{"tokens": [...], "labels": [...], "source": ..., "id": ...}`` records where
``labels[i]`` is the break following ``tokens[i]``.

1. LibriTTS-R transcripts (read speech, Gutenberg-derived prose). Consecutive
   utterances from the same chapter are stitched into passages so that ``b3``
   occurs mid-sequence and not only at the last token; a model trained on
   single sentences learns "the last token is b3" and nothing else.
2. Switchboard-derived conversational text, reused read-only from a sibling
   project. Spontaneous speech, much shorter, different break distribution.
3. A hand-built adversarial set (see ``data/adversarial.jsonl``), gold-annotated
   by hand rather than derived from punctuation.
"""

from __future__ import annotations

import glob
import json
import os
import random
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

from .text import labels_from_punctuation
from .types import NONE

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")
RAW = os.path.join(DATA, "raw")

SWDA_SOURCE = os.path.expanduser("~/mend/data/swda_all.jsonl")

#: Passages are grown until they reach this many tokens, then closed.
TARGET_PASSAGE_TOKENS = 45
#: Hard cap so nothing overflows the model's window.
MAX_PASSAGE_TOKENS = 110


@dataclass
class Example:
    tokens: List[str]
    labels: List[str]
    source: str
    id: str
    note: str = ""
    construction: str = ""

    def to_dict(self) -> Dict:
        d = {
            "id": self.id,
            "source": self.source,
            "tokens": self.tokens,
            "labels": self.labels,
        }
        if self.construction:
            d["construction"] = self.construction
        if self.note:
            d["note"] = self.note
        return d


# --------------------------------------------------------------------------
# LibriTTS-R
# --------------------------------------------------------------------------

_ID = re.compile(r"^(\d+)_(\d+)_(\d+)_(\d+)$")


def _sort_key(utt_id: str):
    m = _ID.match(utt_id)
    if not m:
        return (utt_id, 0, 0, 0)
    spk, chap, utt, seg = m.groups()
    return (int(spk), int(chap), int(utt), int(seg))


def load_libritts_split(split: str) -> List[Dict]:
    """Read one LibriTTS-R text parquet into ``[{"id", "text"}, ...]``."""
    import pyarrow.parquet as pq

    matches = sorted(glob.glob(os.path.join(RAW, f"{split}-*.parquet")))
    if not matches:
        raise FileNotFoundError(
            f"no parquet for split {split!r} under {RAW}. Run scripts/fetch_data.py first."
        )
    rows: List[Dict] = []
    for path in matches:
        table = pq.read_table(path, columns=["id", "text_original"])
        for rid, text in zip(
            table.column("id").to_pylist(), table.column("text_original").to_pylist()
        ):
            if text:
                rows.append({"id": rid, "text": text})
    rows.sort(key=lambda r: _sort_key(r["id"]))
    return rows


def build_passages(rows: Sequence[Dict], source: str) -> List[Example]:
    """Stitch consecutive same-chapter utterances into multi-sentence passages."""
    out: List[Example] = []
    cur_tokens: List[str] = []
    cur_labels: List[str] = []
    cur_key = None
    cur_id = None

    def flush():
        nonlocal cur_tokens, cur_labels, cur_id
        if len(cur_tokens) >= 4:
            # The passage ends where an utterance ended, so the final boundary
            # keeps whatever punctuation said; nothing is invented here.
            out.append(
                Example(list(cur_tokens), list(cur_labels), source, str(cur_id))
            )
        cur_tokens, cur_labels, cur_id = [], [], None

    for row in rows:
        m = _ID.match(row["id"])
        key = (m.group(1), m.group(2)) if m else (row["id"],)
        tokens, labels = labels_from_punctuation(row["text"])
        if not tokens:
            continue
        if key != cur_key or len(cur_tokens) + len(tokens) > MAX_PASSAGE_TOKENS:
            flush()
            cur_key = key
        if cur_id is None:
            cur_id = row["id"]
        cur_tokens.extend(tokens)
        cur_labels.extend(labels)
        if len(cur_tokens) >= TARGET_PASSAGE_TOKENS:
            flush()
    flush()
    return out


# --------------------------------------------------------------------------
# Switchboard
# --------------------------------------------------------------------------


def load_switchboard(limit: int = 500, seed: int = 13) -> List[Example]:
    """Sample punctuated conversational utterances from the sibling project.

    The ``clean`` field is the disfluency-free, punctuated version of each
    Switchboard turn; that punctuation is the gold signal here, exactly as for
    LibriTTS-R, and carries exactly the same proxy caveat.
    """
    if not os.path.exists(SWDA_SOURCE):
        raise FileNotFoundError(f"switchboard source not found at {SWDA_SOURCE}")
    pool: List[Example] = []
    with open(SWDA_SOURCE) as fh:
        for i, line in enumerate(fh):
            rec = json.loads(line)
            text = rec.get("clean") or ""
            tokens, labels = labels_from_punctuation(text)
            # Very short backchannels ("Uh-huh.") carry no internal structure.
            if len(tokens) < 6:
                continue
            pool.append(
                Example(tokens, labels, "switchboard", f"swda-{rec.get('conversation','?')}-{i}")
            )
    rng = random.Random(seed)
    rng.shuffle(pool)
    return pool[:limit]


# --------------------------------------------------------------------------
# Adversarial set
# --------------------------------------------------------------------------


def load_adversarial(path: Optional[str] = None) -> List[Example]:
    path = path or os.path.join(DATA, "adversarial.jsonl")
    out: List[Example] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            rec = json.loads(line)
            out.append(
                Example(
                    tokens=rec["tokens"],
                    labels=rec["labels"],
                    source="adversarial",
                    id=rec["id"],
                    note=rec.get("rationale", ""),
                    construction=rec.get("construction", ""),
                )
            )
    return out


# --------------------------------------------------------------------------
# IO
# --------------------------------------------------------------------------


def write_jsonl(examples: Iterable[Example], path: str) -> int:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    n = 0
    with open(path, "w") as fh:
        for ex in examples:
            fh.write(json.dumps(ex.to_dict()) + "\n")
            n += 1
    return n


def read_jsonl(path: str) -> List[Example]:
    out: List[Example] = []
    with open(path) as fh:
        for line in fh:
            rec = json.loads(line)
            out.append(
                Example(
                    rec["tokens"],
                    rec["labels"],
                    rec.get("source", "?"),
                    rec.get("id", "?"),
                    rec.get("note", ""),
                    rec.get("construction", ""),
                )
            )
    return out


def label_counts(examples: Sequence[Example]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for ex in examples:
        for label in ex.labels:
            counts[label] = counts.get(label, 0) + 1
    counts.setdefault(NONE, 0)
    return counts
