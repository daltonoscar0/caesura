# The two systems

Part of the [Caesura](../README.md) documentation.

**Before reading any numbers here, read [the caveat](../README.md#read-this-before-the-results):** break labels for training and for two of the three test sets are derived from punctuation, and punctuation is not prosody.

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

| rule | fires on LibriTTS-R test | fires on adversarial |
|---|---:|---:|
| `b2_appositive` | 875 | 8 |
| `b2_fronted_adverbial` | 479 | 16 |
| `b2_list_item` | 695 | 4 |
| `b2_long_subject` | 918 | 19 |
| `b2_relative_clause` | 1710 | 8 |
| `b3_clause_boundary` | 1117 | 1 |
| `b3_utterance_final` | 1565 | 60 |

Every rule fires, which was not a foregone conclusion. `b2_list_item` was the
one I expected to be dead, on the reasoning that a list is marked in writing by
commas and commas are exactly what has been removed. It fires 695 times on the
test set, because the parser recovers `conj` chains from bare juxtaposition more
often than that reasoning allowed. `b3_clause_boundary` is the mirror image: 1117
firings on LibriTTS-R against 1 on the adversarial set, because the adversarial
sentences are single clauses by construction.

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

| | |
|---|---|
| base model | `distilroberta-base`, 82M parameters, 6 layers |
| training data | 10,196 LibriTTS-R passages, 575,700 boundaries |
| epochs | 1 |
| batch size | 16 |
| learning rate | 5e-5, one-cycle, 10% warmup |
| seed | 17 |
| selection | dev macro-F1 over `b2` and `b3` only, evaluated four times per epoch |
| best checkpoint | step 636 of 638 |
| dev macro-F1 | 0.720 (b2 0.658, b3 0.783) over all 1,779 dev passages |
| wall clock | 5,594 s on laptop CPU |

Overall accuracy is not reported anywhere, because 85% of boundaries are `none`
and a model that predicts `none` everywhere scores 85%.

One epoch rather than three: the run was budgeted at under an hour on a laptop
CPU, and the machine it ran on was heavily contended, so the schedule was cut to
a single pass over the full training set rather than several passes over a
subset. Dev macro-F1 was still climbing at the end (0.654 at step 159, 0.719 at
318, 0.724 at 636), so the numbers below are a floor rather than a ceiling.

Weights: [`daltonoscar0/caesura-breaks`](https://huggingface.co/daltonoscar0/caesura-breaks).
