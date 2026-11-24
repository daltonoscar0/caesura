# API, usage and reproducing

Part of the [Caesura](../README.md) documentation.

**Before reading any numbers here, read [the caveat](../README.md#read-this-before-the-results):** break labels for training and for two of the three test sets are derived from punctuation, and punctuation is not prosody.

## API

`caesura.run(text, system="rules" | "model" | "both") -> StageResult`.

Incoming punctuation, casing and any prosody markup from an upstream stage are
stripped before anything is predicted. That is the whole point of the stage, so
it is not optional and there is no flag to turn it off.

A `StageResult` is a plain JSON-serialisable dict, not a class, because a
sibling stage has to parse it without importing this package.

```python
import json
from caesura import run

result = run("the man in the grey coat is my uncle", system="both")

result["output"]            # "the man in the grey *coat* <b2> is my *uncle* <b3>"
result["tokens"]            # the whitespace tokens of output, markers included
result["meta"]["breaks"]    # one label per word: none / b1 / b2 / b3
result["meta"]["emphasis"]  # {"coat": "emph_nuclear", "uncle": "emph_nuclear"}
json.dumps(result)          # works as is
```

Top-level keys are `stage`, `input`, `output`, `tokens`, `decisions` and `meta`.
There is one entry in `decisions` per inserted break:

| field | meaning |
|---|---|
| `span` | `[start, end]` character offsets into `input` |
| `surface` | the text that was there, before normalisation |
| `result` | what it became, `"coat <b2>"` |
| `kind` | always `break` for this stage |
| `rule` | the rule function that fired, or `model`, or `consensus:<rule>` |
| `score` | model probability, or 1.0 for a rule |
| `alternatives` | the other system's choice at this boundary, when `system="both"` |
| `note` | one human-readable clause |

With `system="both"` the two systems are merged by taking the stronger break at
each boundary, which is recall-oriented on purpose, and every disagreement is
recorded in `alternatives` so the merge can be audited or replaced.

`meta` also carries `words` (the normalised words, no markup), `breaks` (one
label per word), `system`, `n_tokens`, `seconds`, `vetoed`, and for
`system="both"` the two systems' unmerged label sequences.

## Usage

```bash
pip install -e .
python -m spacy download en_core_web_sm

python -m caesura "the man in the grey coat is my uncle"          # StageResult as JSON
python -m caesura --pretty --system both "in the middle of the night the phone rang"
echo "some text" | python -m caesura --system model
python -m caesura --eval
```

`--system model` and `--system both` pull the weights from the Hub on first use,
or read a local directory given by `--model` or the `CAESURA_MODEL` environment
variable.

## Reproducing

```bash
pip install -e ".[train]"
python -m spacy download en_core_web_sm

python scripts/fetch_data.py          # LibriTTS-R transcripts, about 40 MB
python scripts/build_adversarial.py   # regenerates data/adversarial.jsonl
python scripts/build_data.py          # writes the four splits
python -m caesura.train               # System B
python -m caesura.evaluate            # writes outputs/eval.json
python scripts/make_tables.py         # the tables in this README
pytest
```

`scripts/build_data.py` needs the sibling `mend` checkout for the Switchboard
split; everything else is self-contained.
