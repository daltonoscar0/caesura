# Caesura

Phrase-break and emphasis prediction from syntax, on text that has no punctuation.

Phone transcripts and LLM output arrive at a TTS front end without reliable
punctuation, so the front end has to decide where the voice pauses and what it
stresses from the words alone. Where a break goes is a syntactic decision, about
constituent boundaries rather than word frequencies, which is the same machinery
that garden-path sentences exercise. This repository builds two systems that
predict breaks on punctuation-stripped text, compares them, and reports where
each one fails.

Any-break F1 per boundary:

<!--TABLE:headline-->

Before reading anything into those numbers, read [the caveat](#read-this-before-the-results):
the first two columns are scored against punctuation, which is not prosody. The
third column is scored against hand annotation, and it is the one that means
something.

```bash
pip install -e . && python -m spacy download en_core_web_sm
python -m caesura --pretty --system both "the man in the grey coat is my uncle"
```

```
the man in the grey *coat* <b2> is my *uncle* <b3>

  coat   coat <b2>   b2_long_subject               1.000  [model='coat' 0.934]
  uncle  uncle <b3>  consensus:b3_utterance_final  0.989

  emphasis: coat (emph_nuclear), uncle (emph_nuclear)
```

There is no comma after "coat" in written English. The break is there because
the subject is six tokens long and has to be closed before the predicate, which
is the kind of decision this stage exists to make and the kind that punctuation
never records.

## Read this before the results

The break labels used for training and for two of the three test sets are
derived from punctuation. Punctuation is not prosody. Everyone who works on TTS
knows this, so it is stated here rather than buried.

There is no free ToBI-annotated corpus of usable size, so the standard move is
to treat punctuation as a weak proxy: comma, semicolon, colon, dash and bracket
become `<b2>`; full stop, question mark and exclamation mark become `<b3>`;
everything else becomes no break. Three consequences follow, and all three show
up in the numbers below.

1. **The proxy is silent about the breaks that matter most.** A reader pauses
   after a long subject NP ("the man in the grey coat / is my uncle") where
   written English puts no comma at all. Every one of those boundaries is
   labelled "no break" in training, so both systems are trained and scored
   against a target that actively penalises getting them right. The adversarial
   set exists to measure exactly this.
2. **`<b1>` is not derivable at all.** Punctuation gives no signal for a minor
   juncture, so the break task here is three-way, `{none, b2, b3}`, over token
   boundaries. `<b1>` remains in the output vocabulary of the API and is never
   predicted.
3. **Punctuation conventions leak in as noise.** Serial commas, comma splices
   and a nineteenth-century preference for heavy comma use are all in the
   LibriTTS-R source text, and the model learns them as if they were prosody.

The honest summary: the LibriTTS-R and Switchboard numbers measure
punctuation prediction with prosodic labels attached. The adversarial numbers
measure prosody. They disagree, and the disagreement is the interesting part.

## Data

| split | source | utterances / passages | boundaries | none | b2 | b3 |
|---|---|---:|---:|---:|---:|---:|
| train | LibriTTS-R train-clean-100 | 10,196 | 575,700 | 488,618 | 50,034 | 37,048 |
| dev | LibriTTS-R dev-clean | 1,779 | 99,597 | 84,509 | 8,766 | 6,322 |
| test (a) | LibriTTS-R test-clean | 1,565 | 90,525 | 77,359 | 7,818 | 5,348 |
| test (b) | Switchboard conversational | 500 | 5,537 | 4,767 | 264 | 506 |
| test (c) | adversarial, hand annotated | 60 | 566 | 441 | 63 | 62 |

**LibriTTS-R.** The task description offered `librispeech_asr` text as a simpler
fallback. It is not usable: LibriSpeech transcripts are uppercased and already
stripped of punctuation, so there is nothing to derive labels from. What is
needed is LibriTTS-R's `text_original` field, which preserves the punctuation of
the underlying Project Gutenberg text. The full LibriTTS-R release is
multi-gigabyte audio, so `scripts/fetch_data.py` pulls a text-only mirror
(`ylacombe/libritts-r-text-tags-v2`, four parquet files, about 40 MB).

Consecutive utterances from the same chapter are stitched into passages of about
45 tokens. This matters more than it sounds: LibriTTS-R segments are mostly
single sentences, so a model trained on them one at a time learns "the last
token is `<b3>`" and nothing else, and every evaluation number becomes a
measurement of that shortcut. Stitching puts `<b3>` in the middle of the
sequence where it has to be predicted from syntax. The `final_only` baseline in
the results tables is there to show how much of the score that shortcut is worth.

**Switchboard.** 500 utterances of at least six tokens, sampled with a fixed
seed from the disfluency-cleaned, punctuated Switchboard text in the sibling
`mend` project, read only. Punctuation is the gold signal, with the same caveat
as above. This set is not redistributed here (Switchboard is licensed);
`scripts/build_data.py` rebuilds it from the sibling checkout.

**Adversarial set.** 60 sentences written by hand for this project, in
`data/adversarial.jsonl`, each with a gold break annotation and a one-clause
rationale. Seven construction types: reduced relative (10), NP/Z garden path
(8), long subject (8), coordination scope (8), appositive and parenthetical and
vocative (8), fronted PP or adverbial (8), and break at no punctuation site
(10). The last bucket is the one that no punctuation-derived corpus can contain.
Gold is my own judgement about where a competent reader pauses, and the
rationale is there so that a disagreement can be argued with rather than merely
noted.

## System A: parser rules

`en_core_web_sm` dependency parse, then seven rules consulted in priority order.
The first rule to claim a boundary owns it, so `decisions[i]["rule"]` always
names exactly one function. The parser is an off-the-shelf dependency, not the
contribution; the mapping from constituent structure to break placement is.

| priority | rule | what it does |
|---|---|---|
| 1 | `b3_utterance_final` | `<b3>` on the last boundary |
| 2 | `b3_clause_boundary` | `<b3>` at root-to-root transitions, and before a coordinating conjunction joining two finite clauses |
| 3 | `b2_appositive` | `<b2>` on both edges of an appositive, parenthetical or vocative |
| 4 | `b2_fronted_adverbial` | `<b2>` after a fronted adverbial or PP |
| 5 | `b2_relative_clause` | `<b2>` before a relative clause of at least four tokens |
| 6 | `b2_long_subject` | `<b2>` after a subject NP of at least five tokens |
| 7 | `b2_list_item` | `<b2>` after each non-final item of a list of three or more |

One veto runs afterwards and can delete a break but never create one:
`veto_tight_span` refuses any boundary whose left token is a determiner,
possessive, numeral, adjectival or nominal modifier, auxiliary, particle or
negation with its head to the right. No break inside a determiner-noun or
auxiliary-verb span, ever.

How often each rule actually fires:

<!--TABLE:firing-->

<!--FIRINGPROSE-->

### Emphasis

Rule-based only, and deliberately modest, because there is no gold emphasis
annotation anywhere in this project. The default is a nuclear accent on the last
content word of each intonational phrase, where phrases are the spans between
predicted breaks. Three contrastive rules override the default: `not X` accents
X, `X rather than Y` accents both X and Y, and `not X but Y` accents the head of
each conjunct. Emphasis is reported qualitatively on the adversarial set and is
not scored anywhere.

## System B: a small sequence labeller

A token-classification head on `distilroberta-base`, three labels per word
boundary. Chosen over a BiLSTM on frozen embeddings because the decision at a
boundary is a function of the constituent being closed, which is a longer-range
structural signal than a frozen-embedding recurrent model recovers, and because
82M parameters over six layers still fits the compute budget.

Labels sit on the **last** subword of each word rather than the first, which is
the usual convention, because the thing being predicted is the boundary that
follows the word. Every other subword position is masked to `-100`.

<!--TRAINING-->

Weights: [`daltonoscar0/caesura-breaks`](https://huggingface.co/daltonoscar0/caesura-breaks).

## Results

Per-boundary precision, recall and F1. A boundary is the position after token
*i*, so an utterance of *n* tokens has *n* boundaries and the last one is nearly
always `<b3>`. `any` collapses `b2` and `b3` and asks only whether a break was
placed at all.

<!--TABLE:results-->

### Calibration against published punctuation restoration

The closest published setting is punctuation restoration on unpunctuated
transcripts. It is not the same task (the label set is punctuation classes, the
input is cased, and the domain is TED talks) but it is the nearest anchor for
what a number in this range means.

<!--TABLE:baselines-->

Both rows are from IWSLT2011 reference transcripts, reproduced from the
comparison table in Nagy et al., *Incorporating External POS Tagger for
Punctuation Restoration* (arXiv:2106.06731). Their COMMA class is the closest
analogue of `b2` and PERIOD plus QUESTION of `b3`. The comparison flatters this
project in one direction (LibriTTS-R read speech is cleaner than TED talks) and
penalises it in another (their input keeps its casing, which is a strong cue for
sentence starts, and this input does not).

<!--CANONICAL-->

## The garden-path table

<!--TABLE:garden-->

<!--GARDENPROSE-->

## Error taxonomy

Every miss on the adversarial set, bucketed by construction. `missed` is a gold
break the system did not place, `wrong level` a gold break placed at the wrong
strength, `spurious` a break placed where gold has none.

<!--TABLE:taxonomy-->

Sentences reproduced exactly, out of the number of sentences in each bucket:

<!--TABLE:exact-->

<!--TAXONOMYPROSE-->

## Emphasis, qualitatively

No gold, so this is a table of what got marked and nothing more.

<!--TABLE:emphasis-->

## Speed

<!--TABLE:speed-->

<!--SPEEDPROSE-->

## Where it fails

<!--WHEREITFAILS-->

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

## Licence

MIT.
