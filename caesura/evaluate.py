"""Evaluation: per-boundary scores, error taxonomy, garden paths, speed.

Everything is boundary-level. A boundary is the position after token *i*, and
there is one per token including the last, so an utterance of *n* tokens has
*n* boundaries and the final one is almost always ``b3``.

Three systems are scored plus one trivial baseline:

``rules``       System A, dependency rules
``model``       System B, the fine-tuned token classifier
``both``        the recall-oriented union that ``api.run(system="both")`` emits
``final_only``  b3 on the last token and nothing else, which is what a system
                gets for free and is the number the others have to beat
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

from .api import _merge
from .data import Example, load_adversarial, read_jsonl
from .rules import apply as apply_rules
from .rules import RULE_NAMES, get_nlp
from .types import B2, B3, NONE

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
OUTPUTS = os.path.join(ROOT, "outputs")

SYSTEMS = ["final_only", "rules", "model", "both"]

TEST_SETS = [
    ("libritts_test", os.path.join(DATA, "test_libritts.jsonl")),
    ("switchboard", os.path.join(DATA, "test_switchboard.jsonl")),
    ("adversarial", os.path.join(DATA, "adversarial.jsonl")),
]

#: Published punctuation-restoration results on IWSLT2011 reference
#: transcripts, for calibration only. Their COMMA class is the closest thing to
#: our b2 and their PERIOD+QUESTION classes to our b3. Reproduced from the
#: comparison table in Nagy et al., "Incorporating External POS Tagger for
#: Punctuation Restoration" (arXiv:2106.06731).
PUNCTUATION_BASELINES = [
    {
        "system": "Tilk and Alumae 2016, T-BRNN-pre",
        "test_set": "IWSLT2011 reference transcripts",
        "comma_f1": 54.8,
        "period_f1": 72.9,
        "question_f1": 66.7,
        "overall_micro_f1": 64.4,
    },
    {
        "system": "Alam et al. 2020, BiLSTM head on roberta-large",
        "test_set": "IWSLT2011 reference transcripts",
        "comma_f1": 76.3,
        "period_f1": 88.6,
        "question_f1": 81.9,
        "overall_micro_f1": 82.4,
    },
]


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------


def prf(tp: int, fp: int, fn: int) -> Dict[str, float]:
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {
        "p": round(100 * prec, 1),
        "r": round(100 * rec, 1),
        "f1": round(100 * f1, 1),
        "support": tp + fn,
        "predicted": tp + fp,
    }


def score(gold: Sequence[Sequence[str]], pred: Sequence[Sequence[str]]) -> Dict[str, Dict]:
    """Per-boundary precision/recall/F1 for b2, b3 and any-break."""
    out: Dict[str, Dict] = {}
    for cls in (B2, B3):
        tp = fp = fn = 0
        for g_seq, p_seq in zip(gold, pred):
            for g, p in zip(g_seq, p_seq):
                if g == cls and p == cls:
                    tp += 1
                elif g != cls and p == cls:
                    fp += 1
                elif g == cls and p != cls:
                    fn += 1
        out[cls] = prf(tp, fp, fn)

    tp = fp = fn = 0
    for g_seq, p_seq in zip(gold, pred):
        for g, p in zip(g_seq, p_seq):
            gb, pb = g != NONE, p != NONE
            if gb and pb:
                tp += 1
            elif pb and not gb:
                fp += 1
            elif gb and not pb:
                fn += 1
    out["any"] = prf(tp, fp, fn)
    out["macro_break_f1"] = round((out[B2]["f1"] + out[B3]["f1"]) / 2, 1)
    return out


# --------------------------------------------------------------------------
# Prediction
# --------------------------------------------------------------------------


def predict_rules_batch(examples: Sequence[Example]) -> List[Tuple[List[str], Dict[int, str]]]:
    """Rule predictions for a list of examples, parsing with ``nlp.pipe``."""
    from spacy.tokens import Doc

    nlp = get_nlp()
    docs = nlp.pipe((Doc(nlp.vocab, words=ex.tokens) for ex in examples), batch_size=64)
    out = []
    for ex, doc in zip(examples, docs):
        labels, rule_names, _ = apply_rules(ex.tokens, doc=doc)
        out.append((labels, rule_names))
    return out


_MODELS: Dict[Tuple[Optional[str], str], "object"] = {}


def _model(model_path: Optional[str], device: str):
    from .model import BreakModel

    key = (model_path, device)
    if key not in _MODELS:
        _MODELS[key] = BreakModel(model_path, device=device)
    return _MODELS[key]


def predict_model_batch(
    examples: Sequence[Example], model_path: Optional[str] = None, device: str = "cpu"
) -> List[Tuple[List[str], List[float]]]:
    return _model(model_path, device).predict_batch([ex.tokens for ex in examples])


def final_only(example: Example) -> List[str]:
    labels = [NONE] * len(example.tokens)
    if labels:
        labels[-1] = B3
    return labels


def predict_all(
    examples: Sequence[Example], model_path: Optional[str] = None, device: str = "cpu"
) -> Tuple[Dict[str, List[List[str]]], Counter]:
    """Run every system over ``examples`` once.

    Returns the label sequences per system and a count of how often each rule
    claimed a boundary, which comes free from the same pass.
    """
    rules_out = predict_rules_batch(examples)
    model_out = predict_model_batch(examples, model_path, device)

    preds: Dict[str, List[List[str]]] = {s: [] for s in SYSTEMS}
    firing: Counter = Counter()
    for ex, (r_labels, r_names), (m_labels, m_scores) in zip(examples, rules_out, model_out):
        preds["final_only"].append(final_only(ex))
        preds["rules"].append(r_labels)
        preds["model"].append(m_labels)
        merged, _, _ = _merge(r_labels, r_names, m_labels, m_scores)
        preds["both"].append(merged)
        firing.update(r_names.values())
    return preds, firing


# --------------------------------------------------------------------------
# Error taxonomy
# --------------------------------------------------------------------------


def error_taxonomy(
    examples: Sequence[Example], preds: Dict[str, List[List[str]]]
) -> Dict[str, Dict[str, Dict[str, int]]]:
    """Bucket every adversarial miss by construction type.

    ``missed`` is a gold break the system did not place at all, ``wrong_level``
    a gold break placed with the wrong strength, ``spurious`` a break placed
    where gold has none.
    """
    table: Dict[str, Dict[str, Dict[str, int]]] = {}
    for system, all_labels in preds.items():
        buckets: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {"gold_breaks": 0, "missed": 0, "wrong_level": 0, "spurious": 0, "exact_sentences": 0}
        )
        for ex, pred in zip(examples, all_labels):
            bucket = buckets[ex.construction or "unspecified"]
            if list(pred) == list(ex.labels):
                bucket["exact_sentences"] += 1
            for g, p in zip(ex.labels, pred):
                if g != NONE:
                    bucket["gold_breaks"] += 1
                    if p == NONE:
                        bucket["missed"] += 1
                    elif p != g:
                        bucket["wrong_level"] += 1
                elif p != NONE:
                    bucket["spurious"] += 1
        table[system] = {k: dict(v) for k, v in sorted(buckets.items())}
    return table


def rule_firing_counts(examples: Sequence[Example]) -> Counter:
    """How often each rule claims a boundary, for callers outside ``run_all``.

    A rule that never fires is as much a result as one that does, so every rule
    is reported with a zero rather than left out.
    """
    counts: Counter = Counter({name: 0 for name in RULE_NAMES})
    for _, rule_names in predict_rules_batch(examples):
        counts.update(rule_names.values())
    return counts


# --------------------------------------------------------------------------
# Garden paths
# --------------------------------------------------------------------------


def marked(tokens: Sequence[str], labels: Sequence[str]) -> str:
    parts = []
    for tok, label in zip(tokens, labels):
        parts.append(tok)
        if label == B2:
            parts.append("<b2>")
        elif label == B3:
            parts.append("<b3>")
    return " ".join(parts)


def garden_path_table(
    examples: Sequence[Example], preds: Dict[str, List[List[str]]]
) -> List[Dict]:
    rows = []
    for i, ex in enumerate(examples):
        if ex.construction not in ("reduced_relative", "np_z"):
            continue
        row = {
            "id": ex.id,
            "construction": ex.construction,
            "gold": marked(ex.tokens, ex.labels),
            "rationale": ex.note,
        }
        for system in ("rules", "model"):
            pred = preds[system][i]
            row[system] = marked(ex.tokens, pred)
            row[f"{system}_match"] = list(pred) == list(ex.labels)
        rows.append(row)
    return rows


# --------------------------------------------------------------------------
# Speed
# --------------------------------------------------------------------------


def speed(examples: Sequence[Example], model_path: Optional[str] = None, n: int = 300) -> Dict:
    """Tokens per second on CPU, one utterance at a time.

    Streaming, not batched, because a TTS front end sees one utterance at a
    time. The rules figure includes the spaCy parse, which dominates it. The
    model figure excludes the parse; the ``model+emphasis`` figure adds it back,
    because emphasis marking is rule-based and needs the parse either way.
    """
    from . import api

    sample = list(examples[:n])
    total_tokens = sum(len(ex.tokens) for ex in sample)
    out: Dict[str, Dict] = {}

    # System A, parse plus rules.
    get_nlp()
    started = time.perf_counter()
    for ex in sample:
        apply_rules(ex.tokens)
    elapsed = time.perf_counter() - started
    out["rules"] = {
        "tokens_per_second": round(total_tokens / elapsed),
        "ms_per_utterance": round(1000 * elapsed / len(sample), 2),
    }

    # System B, forward pass only.
    model = _model(model_path, "cpu")
    model.load()
    model.predict(sample[0].tokens)  # warm up
    started = time.perf_counter()
    for ex in sample:
        model.predict(ex.tokens)
    elapsed = time.perf_counter() - started
    out["model"] = {
        "tokens_per_second": round(total_tokens / elapsed),
        "ms_per_utterance": round(1000 * elapsed / len(sample), 2),
    }

    # System B end to end through the API, which parses for emphasis.
    started = time.perf_counter()
    for ex in sample:
        api.run(" ".join(ex.tokens), system="model", model_path=model_path)
    elapsed = time.perf_counter() - started
    out["model+emphasis"] = {
        "tokens_per_second": round(total_tokens / elapsed),
        "ms_per_utterance": round(1000 * elapsed / len(sample), 2),
    }

    out["_meta"] = {"utterances": len(sample), "tokens": total_tokens, "device": "cpu"}
    return out


# --------------------------------------------------------------------------
# Emphasis, qualitative only
# --------------------------------------------------------------------------


#: Probes for the three contrastive emphasis rules, which the adversarial set
#: does not exercise because it was written to test breaks, not accent.
CONTRAST_PROBES = [
    "i did not order the fish",
    "she took the train rather than the bus",
    "we went to the museum instead of the park",
    "he wanted not the money but the credit",
    "they never asked for permission",
]


def emphasis_table(examples: Sequence[Example], model_path: Optional[str] = None) -> List[Dict]:
    """What the emphasis rules mark. No gold exists, so this is a listing.

    Two per construction from the adversarial set, so every construction is
    represented, plus the contrastive probes.
    """
    from . import api

    def row(text: str, ident: str, construction: str) -> Dict:
        result = api.run(text, system="rules")
        return {
            "id": ident,
            "construction": construction,
            "text": result["output"],
            "emphasis": result["meta"]["emphasis"],
        }

    rows: List[Dict] = []
    seen: Dict[str, int] = defaultdict(int)
    for ex in examples:
        if seen[ex.construction] >= 2:
            continue
        seen[ex.construction] += 1
        rows.append(row(" ".join(ex.tokens), ex.id, ex.construction))
    for i, probe in enumerate(CONTRAST_PROBES, start=1):
        rows.append(row(probe, f"probe-{i:02d}", "contrastive probe"))
    return rows


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------


def run_all(
    model_path: Optional[str] = None, device: str = "cpu", limit: int = 0, speed_n: int = 300
) -> Dict:
    report: Dict = {
        "punctuation_restoration_reference": PUNCTUATION_BASELINES,
        "results": {},
    }

    adversarial: List[Example] = []
    adversarial_preds: Dict[str, List[List[str]]] = {}
    libritts: List[Example] = []
    firing_by_set: Dict[str, Counter] = {}

    for name, path in TEST_SETS:
        examples = load_adversarial(path) if name == "adversarial" else read_jsonl(path)
        if limit and name != "adversarial":
            examples = examples[:limit]
        preds, firing = predict_all(examples, model_path, device)
        firing_by_set[name] = firing
        report["results"][name] = {
            "n_utterances": len(examples),
            "n_boundaries": sum(len(ex.tokens) for ex in examples),
            "systems": {
                system: score([ex.labels for ex in examples], preds[system])
                for system in SYSTEMS
            },
        }
        if name == "adversarial":
            adversarial, adversarial_preds = examples, preds
        elif name == "libritts_test":
            libritts = examples

    report["error_taxonomy"] = error_taxonomy(adversarial, adversarial_preds)
    report["garden_paths"] = garden_path_table(adversarial, adversarial_preds)
    # Rules that never fire are reported as zeros rather than omitted.
    for key, source in (("libritts", "libritts_test"), ("adversarial", "adversarial")):
        counts = Counter({name: 0 for name in RULE_NAMES})
        counts.update(firing_by_set[source])
        report[f"rule_firing_{key}"] = dict(counts)
    report["speed"] = speed(libritts, model_path, n=speed_n)
    report["emphasis_adversarial"] = emphasis_table(adversarial)
    return report


def format_results(report: Dict) -> str:
    lines = []
    for name, block in report["results"].items():
        lines.append(
            f"\n{name}  ({block['n_utterances']} utterances, {block['n_boundaries']} boundaries)"
        )
        lines.append(
            f"  {'system':12s} {'b2 P':>6s} {'b2 R':>6s} {'b2 F1':>6s}   "
            f"{'b3 P':>6s} {'b3 R':>6s} {'b3 F1':>6s}   "
            f"{'any P':>6s} {'any R':>6s} {'any F1':>6s}"
        )
        for system, s in block["systems"].items():
            lines.append(
                f"  {system:12s} {s['b2']['p']:6.1f} {s['b2']['r']:6.1f} {s['b2']['f1']:6.1f}   "
                f"{s['b3']['p']:6.1f} {s['b3']['r']:6.1f} {s['b3']['f1']:6.1f}   "
                f"{s['any']['p']:6.1f} {s['any']['r']:6.1f} {s['any']['f1']:6.1f}"
            )
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Evaluate both Caesura systems")
    ap.add_argument("--model", default=None, help="local dir or Hub id for System B")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=os.path.join(OUTPUTS, "eval.json"))
    ap.add_argument("--limit", type=int, default=0,
                    help="cap the two corpus test sets, for a quick smoke run")
    ap.add_argument("--speed-n", type=int, default=300)
    args = ap.parse_args(argv)

    report = run_all(args.model, args.device, limit=args.limit, speed_n=args.speed_n)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=2)
    print(format_results(report))
    print(f"\nspeed (cpu, one utterance at a time)")
    for system, s in report["speed"].items():
        if system.startswith("_"):
            continue
        print(f"  {system:16s} {s['tokens_per_second']:>7d} tok/s  {s['ms_per_utterance']:>7.2f} ms/utt")
    print(f"\nfull report -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
