"""Push the trained break labeller and its card to the Hub."""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from caesura.model import HUB_ID  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCAL = os.path.join(ROOT, "outputs", "breaks")


def render_card(repo_id: str) -> str:
    summary = {}
    summary_path = os.path.join(LOCAL, "training_summary.json")
    if os.path.exists(summary_path):
        with open(summary_path) as fh:
            summary = json.load(fh)

    eval_path = os.path.join(ROOT, "outputs", "eval.json")
    results = {}
    if os.path.exists(eval_path):
        with open(eval_path) as fh:
            results = json.load(fh).get("results", {})

    def row(split: str) -> str:
        block = results.get(split, {}).get("systems", {}).get("model")
        if not block:
            return f"| {split} | - | - | - |"
        return (
            f"| {split} | {block['b2']['f1']} | {block['b3']['f1']} | {block['any']['f1']} |"
        )

    return f"""---
license: mit
language: en
library_name: transformers
pipeline_tag: token-classification
base_model: distilroberta-base
tags:
  - prosody
  - phrase-break
  - text-to-speech
  - front-end
---

# caesura-breaks

Predicts prosodic phrase breaks on lowercase, punctuation-free English text, one
label per word boundary: `none`, `b2` (minor break) or `b3` (major break). It is
the learned half of [caesura](https://github.com/daltonoscar0/caesura), a TTS
front-end stage that has to decide where the voice pauses when the input arrives
without reliable punctuation, which is the normal case for phone transcripts and
LLM output.

## Labels

| id | label | meaning |
|----|-------|---------|
| 0 | `none` | no break after this word |
| 1 | `b2` | minor break, roughly a comma-level juncture |
| 2 | `b3` | major break, roughly a sentence-level juncture |

`b1` exists in the break inventory of the wider project but is not predicted
here, because it cannot be derived from punctuation.

## The training labels are punctuation, not prosody

This is the most important thing to know before using the model. There is no
free ToBI-annotated corpus of usable size, so the labels come from the
punctuation in LibriTTS-R's original transcripts: comma, semicolon, colon, dash
and bracket become `b2`; full stop, question mark and exclamation mark become
`b3`; everything else is `none`. Punctuation and prosody agree often but not
always. Breaks that a reader takes at no punctuation site at all, which is where
the interesting front-end failures live, are systematically labelled `none` in
training. The model therefore inherits a bias against exactly the breaks that
matter most. The repository quantifies this on a hand-annotated adversarial set.

## Training data

LibriTTS-R `train-clean-100` transcripts, consecutive same-chapter utterances
stitched into passages of about 45 tokens so that `b3` occurs mid-sequence
rather than only on the final token. {summary.get('train_examples', '?')}
passages, roughly 576k boundaries, about 8.7% `b2` and 6.4% `b3`.

## Results

Per-boundary F1, from the repository's evaluation:

| test set | b2 F1 | b3 F1 | any-break F1 |
|---|---|---|---|
{row('libritts_test')}
{row('switchboard')}
{row('adversarial')}

`switchboard` is out-of-domain conversational text; `adversarial` is 60
hand-annotated sentences built to break syntactic break prediction, so the drop
there is the point of the set rather than a defect of the model.

## Usage

```python
from transformers import AutoModelForTokenClassification, AutoTokenizer
import torch

tok = AutoTokenizer.from_pretrained("{repo_id}", add_prefix_space=True)
model = AutoModelForTokenClassification.from_pretrained("{repo_id}")

words = "the man in the grey coat is my uncle".split()
enc = tok(words, is_split_into_words=True, return_tensors="pt")
with torch.no_grad():
    logits = model(**enc).logits[0]

# The label for a word sits on that word's LAST subword, not its first.
last = {{}}
for pos, wid in enumerate(enc.word_ids()):
    if wid is not None:
        last[wid] = pos
for i, word in enumerate(words):
    print(word, model.config.id2label[int(logits[last[i]].argmax())])
```

Or through the project, which adds the rule system and emphasis marking:

```bash
pip install git+https://github.com/daltonoscar0/caesura
python -m caesura --system both "the man in the grey coat is my uncle"
```

## Training details

| | |
|---|---|
| base model | `{summary.get('base_model', 'distilroberta-base')}` |
| epochs | {summary.get('epochs', '?')} |
| batch size | {summary.get('batch_size', '?')} |
| learning rate | {summary.get('lr', '?')} |
| schedule | one-cycle, 10% warmup |
| seed | {summary.get('seed', '?')} |
| selection | dev macro-F1 over `b2` and `b3` only |
| wall clock | {summary.get('wall_seconds', '?')} s on {summary.get('device', '?')} |

Labels are attached to the last subword of each word and every other subword
position is masked to `-100`.

## Limitations

- Trained on read speech from public-domain literature. Conversational input is
  out of domain and scores lower.
- The label set has no `b1`, so no minor juncture is available to the caller.
- Emphasis is not predicted by this model at all; the repository marks emphasis
  with rules over a dependency parse.
- On garden-path sentences the model places breaks consistent with the locally
  favoured reading rather than the globally correct one. The repository
  documents this case by case.

## Licence

MIT.
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=HUB_ID)
    ap.add_argument("--local", default=LOCAL)
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()

    from huggingface_hub import HfApi

    api = HfApi()
    who = api.whoami()["name"]
    print(f"authenticated as {who}")

    card_path = os.path.join(args.local, "README.md")
    with open(card_path, "w") as fh:
        fh.write(render_card(args.repo))
    print(f"wrote card -> {card_path}")

    api.create_repo(args.repo, repo_type="model", exist_ok=True, private=args.private)
    api.upload_folder(
        folder_path=args.local,
        repo_id=args.repo,
        repo_type="model",
        ignore_patterns=["*.pt", "optimizer*", "*.jsonl"],
    )
    print(f"pushed -> https://huggingface.co/{args.repo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
