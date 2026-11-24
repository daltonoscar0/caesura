# Caesura

[![tests](https://github.com/daltonoscar0/caesura/actions/workflows/tests.yml/badge.svg)](https://github.com/daltonoscar0/caesura/actions/workflows/tests.yml)

Phrase-break and emphasis prediction from syntax, on text that has no punctuation.

Phone transcripts and LLM output arrive at a TTS front end without reliable
punctuation, so the front end has to decide where the voice pauses and what it
stresses from the words alone. Where a break goes is a syntactic decision, about
constituent boundaries rather than word frequencies, which is the same machinery
that garden-path sentences exercise. This repository builds two systems that
predict breaks on punctuation-stripped text, compares them, and reports where
each one fails.

Any-break F1 per boundary:

| system | LibriTTS-R test-clean | Switchboard conversational | Adversarial set |
|---|---:|---:|---:|
| baseline: b3 on final token | 21.2 | 78.7 | 64.9 |
| A: parser rules | 46.1 | 74.0 | 78.8 |
| B: distilroberta | 83.1 | 81.4 | 75.0 |
| A+B union | 77.9 | 75.7 | 81.8 |

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

## Testing

```bash
pip install -e '.[dev]' && python -m spacy download en_core_web_sm
pytest -q
```

System B's tests skip themselves unless the model weights are present locally
(`caesura.model.model_available()`), so the suite runs from a clean checkout.

## Documentation

- [docs/DATA.md](docs/DATA.md) — the three test sets and how labels are derived
- [docs/SYSTEMS.md](docs/SYSTEMS.md) — System A (parser rules) and System B
  (distilroberta sequence labeller), including emphasis
- [docs/RESULTS.md](docs/RESULTS.md) — full results, calibration against
  published punctuation restoration, the canonical sentence, the garden-path
  table, and what the two systems do differently
- [docs/ERRORS.md](docs/ERRORS.md) — error taxonomy, emphasis qualitatively,
  and where it fails
- [docs/PERFORMANCE.md](docs/PERFORMANCE.md) — speed
- [docs/API.md](docs/API.md) — API, usage and reproducing the numbers

## Licence

MIT.
