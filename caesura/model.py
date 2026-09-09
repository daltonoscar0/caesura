"""System B: a token-classification head over distilroberta-base.

Why this and not a BiLSTM over frozen embeddings: the decision at a boundary is
a function of the constituent that is being closed, which is a longer-range and
more structural signal than a frozen-embedding recurrent model recovers, and
distilroberta is small enough (82M parameters, 6 layers) to fine-tune end to end
inside the stated compute budget.

Labels sit on the *last* subword of each word, not the first, because the thing
being predicted is the boundary that follows the word.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Tuple

from .types import ID2LABEL, LABEL2ID, LABELS, NONE

BASE_MODEL = "distilroberta-base"
HUB_ID = "daltonoscar0/caesura-breaks"
DEFAULT_LOCAL = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "breaks"
)
MAX_LENGTH = 256


def encode(tokenizer, tokens: Sequence[str], labels: Optional[Sequence[str]] = None):
    """Tokenise a word list and align labels to last subwords."""
    enc = tokenizer(
        list(tokens),
        is_split_into_words=True,
        truncation=True,
        max_length=MAX_LENGTH,
    )
    word_ids = enc.word_ids()
    # -100 everywhere except the final subword of each word.
    aligned = [-100] * len(word_ids)
    last_pos: Dict[int, int] = {}
    for pos, wid in enumerate(word_ids):
        if wid is not None:
            last_pos[wid] = pos
    if labels is not None:
        for wid, pos in last_pos.items():
            aligned[pos] = LABEL2ID[labels[wid]]
        enc["labels"] = aligned
    return enc, last_pos


class BreakModel:
    """Thin inference wrapper. Loads lazily and caches on the instance."""

    def __init__(self, path: Optional[str] = None, device: Optional[str] = None):
        self.path = path or self._resolve_path()
        self.device = device
        self._model = None
        self._tokenizer = None

    @staticmethod
    def _resolve_path() -> str:
        env = os.environ.get("CAESURA_MODEL")
        if env:
            return env
        if os.path.isdir(DEFAULT_LOCAL) and os.path.exists(
            os.path.join(DEFAULT_LOCAL, "config.json")
        ):
            return DEFAULT_LOCAL
        return HUB_ID

    def load(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForTokenClassification, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.path, add_prefix_space=True)
        self._model = AutoModelForTokenClassification.from_pretrained(self.path)
        self._model.eval()
        if self.device is None:
            self.device = "cpu"
        self._model.to(self.device)
        self._torch = torch

    @property
    def tokenizer(self):
        self.load()
        return self._tokenizer

    def predict(self, tokens: Sequence[str]) -> Tuple[List[str], List[float]]:
        """Return ``(labels, scores)``, one per input token."""
        if not tokens:
            return [], []
        self.load()
        torch = self._torch
        enc, last_pos = encode(self._tokenizer, tokens)
        inputs = {
            k: torch.tensor([v], device=self.device)
            for k, v in enc.items()
            if k in ("input_ids", "attention_mask")
        }
        with torch.no_grad():
            logits = self._model(**inputs).logits[0]
        probs = torch.softmax(logits, dim=-1)

        labels: List[str] = []
        scores: List[float] = []
        for i in range(len(tokens)):
            pos = last_pos.get(i)
            if pos is None:
                # Word fell outside the truncated window.
                labels.append(NONE)
                scores.append(0.0)
                continue
            row = probs[pos]
            best = int(row.argmax())
            labels.append(ID2LABEL[best])
            scores.append(float(row[best]))
        return labels, scores

    def predict_batch(
        self, batch: Sequence[Sequence[str]], batch_size: int = 16
    ) -> List[Tuple[List[str], List[float]]]:
        """Padded batched inference; same output shape as ``predict`` per item."""
        if not batch:
            return []
        self.load()
        torch = self._torch
        out: List[Tuple[List[str], List[float]]] = []
        for start in range(0, len(batch), batch_size):
            chunk = list(batch[start : start + batch_size])
            encs = [encode(self._tokenizer, toks) for toks in chunk]
            width = max(len(e[0]["input_ids"]) for e in encs)
            pad_id = self._tokenizer.pad_token_id
            input_ids = torch.tensor(
                [e[0]["input_ids"] + [pad_id] * (width - len(e[0]["input_ids"])) for e in encs],
                device=self.device,
            )
            attn = torch.tensor(
                [
                    e[0]["attention_mask"] + [0] * (width - len(e[0]["attention_mask"]))
                    for e in encs
                ],
                device=self.device,
            )
            with torch.no_grad():
                logits = self._model(input_ids=input_ids, attention_mask=attn).logits
            probs = torch.softmax(logits, dim=-1)
            for row_idx, (toks, (_, last_pos)) in enumerate(zip(chunk, encs)):
                labels: List[str] = []
                scores: List[float] = []
                for i in range(len(toks)):
                    pos = last_pos.get(i)
                    if pos is None:
                        labels.append(NONE)
                        scores.append(0.0)
                        continue
                    row = probs[row_idx][pos]
                    best = int(row.argmax())
                    labels.append(ID2LABEL[best])
                    scores.append(float(row[best]))
                out.append((labels, scores))
        return out


_SHARED: Dict[str, BreakModel] = {}


def get_model(path: Optional[str] = None) -> BreakModel:
    key = path or "__default__"
    if key not in _SHARED:
        _SHARED[key] = BreakModel(path)
    return _SHARED[key]


def model_available(path: Optional[str] = None) -> bool:
    """True when System B can run without a network round trip."""
    target = path or BreakModel._resolve_path()
    return os.path.isdir(target) and os.path.exists(os.path.join(target, "config.json"))
