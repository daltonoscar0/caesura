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

## Results

Per-boundary precision, recall and F1. A boundary is the position after token
*i*, so an utterance of *n* tokens has *n* boundaries and the last one is nearly
always `<b3>`. `any` collapses `b2` and `b3` and asks only whether a break was
placed at all.

**LibriTTS-R test-clean (held out, in domain)** (1565 utterances, 90525 boundaries)

| system | b2 P | b2 R | b2 F1 | b3 P | b3 R | b3 F1 | any P | any R | any F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline: b3 on final token | 0.0 | 0.0 | **0.0** | 98.0 | 28.7 | **44.4** | 99.6 | 11.8 | **21.2** |
| A: parser rules | 30.8 | 18.4 | **23.1** | 67.3 | 33.8 | **45.0** | 64.2 | 35.9 | **46.1** |
| B: distilroberta | 71.0 | 61.7 | **66.0** | 78.8 | 76.2 | **77.5** | 87.3 | 79.3 | **83.1** |
| A+B union | 52.9 | 57.6 | **55.2** | 68.9 | 78.5 | **73.4** | 74.1 | 82.2 | **77.9** |

**Switchboard conversational (out of domain)** (500 utterances, 5537 boundaries)

| system | b2 P | b2 R | b2 F1 | b3 P | b3 R | b3 F1 | any P | any R | any F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline: b3 on final token | 0.0 | 0.0 | **0.0** | 100.0 | 98.8 | **99.4** | 100.0 | 64.9 | **78.7** |
| A: parser rules | 21.1 | 15.2 | **17.6** | 97.5 | 98.8 | **98.1** | 77.5 | 70.8 | **74.0** |
| B: distilroberta | 48.3 | 26.5 | **34.2** | 91.9 | 99.0 | **95.3** | 86.1 | 77.1 | **81.4** |
| A+B union | 29.1 | 31.4 | **30.2** | 89.9 | 99.0 | **94.3** | 72.4 | 79.2 | **75.7** |

**Adversarial set (hand annotated)** (60 utterances, 566 boundaries)

| system | b2 P | b2 R | b2 F1 | b3 P | b3 R | b3 F1 | any P | any R | any F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline: b3 on final token | 0.0 | 0.0 | **0.0** | 100.0 | 96.8 | **98.4** | 100.0 | 48.0 | **64.9** |
| A: parser rules | 61.8 | 54.0 | **57.6** | 100.0 | 98.4 | **99.2** | 81.9 | 76.0 | **78.8** |
| B: distilroberta | 84.6 | 17.5 | **28.9** | 96.8 | 96.8 | **96.8** | 100.0 | 60.0 | **75.0** |
| A+B union | 62.7 | 58.7 | **60.7** | 96.8 | 98.4 | **97.6** | 82.8 | 80.8 | **81.8** |


### Calibration against published punctuation restoration

The closest published setting is punctuation restoration on unpunctuated
transcripts. It is not the same task (the label set is punctuation classes, the
input is cased, and the domain is TED talks) but it is the nearest anchor for
what a number in this range means.

| published system | test set | comma F1 | period F1 | question F1 | overall micro F1 |
|---|---|---:|---:|---:|---:|
| Tilk and Alumae 2016, T-BRNN-pre | IWSLT2011 reference transcripts | 54.8 | 72.9 | 66.7 | 64.4 |
| Alam et al. 2020, BiLSTM head on roberta-large | IWSLT2011 reference transcripts | 76.3 | 88.6 | 81.9 | 82.4 |

Both rows are from IWSLT2011 reference transcripts, reproduced from the
comparison table in Nagy et al., *Incorporating External POS Tagger for
Punctuation Restoration* (arXiv:2106.06731). Their COMMA class is the closest
analogue of `b2` and PERIOD plus QUESTION of `b3`. The comparison flatters this
project in one direction (LibriTTS-R read speech is cleaner than TED talks) and
penalises it in another (their input keeps its casing, which is a strong cue for
sentence starts, and this input does not).

### The canonical sentence

The demo sentence for the whole front end, as it reaches this stage after
normalisation:

    i read the second doctor lee lead a live band on reading road at ten thirty

The reading it is supposed to get is `<b2>` after "lee", `<b2>` after "band",
`<b3>` at the end, with emphasis on "band" and "thirty". What actually happens:

| system | output |
|---|---|
| A: parser rules | `i read the second doctor lee lead a live band on reading road at ten *thirty* <b3>` |
| B: distilroberta | `i read the second *doctor* <b3> lee lead a live band on reading road at ten *thirty* <b3>` |
| A+B union | `i read the second *doctor* <b3> lee lead a live band on reading road at ten *thirty* <b3>` |

All three get the final `<b3>` and the accent on "thirty". None of them gets
either internal break, and the accent on "band" is missing from all three.

System A produces one flat phrase because the parser reads "the second doctor
lee" as a single noun phrase, subject of "lead". No appositive is found, so
`b2_appositive` cannot fire, and the subject is four tokens, one below the
five-token floor, so `b2_long_subject` cannot fire either. The rules are not
wrong given the parse; the parse resolves the ambiguity in the compound-noun
direction and the rules faithfully follow it.

System B is worse than silent: it inserts `<b3>` between "doctor" and "lee",
splitting a name. That boundary is a plausible comma site in the training
distribution ("the second, Doctor Lee") but the break it produces is at the
wrong strength, in a position that would be audibly broken in speech. Because
the merge takes the stronger break at each boundary, A+B inherits it.

This one sentence contains most of the report: the rules are limited by the
parse, the model is limited by the punctuation it was trained on, and neither
limitation is fixed by combining them.

## The garden-path table

| id | type | gold | System A rules | System B model | A ok | B ok |
|---|---|---|---|---|:-:|:-:|
| adv-001 | reduced relative | `the horse raced past the barn <b2> fell <b3>` | `the horse raced past the barn fell <b3>` | `the horse raced past the barn fell <b3>` | no | no |
| adv-002 | reduced relative | `the man returned to his house <b2> was arrested <b3>` | `the man returned to his house <b2> was arrested <b3>` | `the man returned to his house was arrested <b3>` | yes | no |
| adv-003 | reduced relative | `the florist sent the flowers <b2> was pleased <b3>` | `the florist sent the flowers was pleased <b3>` | `the florist sent the flowers was pleased <b3>` | no | no |
| adv-004 | reduced relative | `the defendant examined by the lawyer <b2> turned out to be unreliable <b3>` | `the defendant <b2> examined by the lawyer <b2> turned out to be unreliable <b3>` | `the defendant examined by the lawyer <b2> turned out to be unreliable <b3>` | no | yes |
| adv-005 | reduced relative | `the boat floated down the river <b2> sank <b3>` | `the boat floated down the river sank <b3>` | `the boat floated down the river sank <b3>` | no | no |
| adv-006 | reduced relative | `the student given the answers <b2> cheated <b3>` | `the student given the answers cheated <b3>` | `the student given the answers cheated <b3>` | no | no |
| adv-007 | reduced relative | `the soldiers marched across the field <b2> were exhausted <b3>` | `the soldiers marched across the field were exhausted <b3>` | `the soldiers marched across the field were exhausted <b3>` | no | no |
| adv-008 | reduced relative | `the woman brought the sandwiches <b2> left early <b3>` | `the woman brought the sandwiches left early <b3>` | `the woman brought the sandwiches left early <b3>` | no | no |
| adv-009 | reduced relative | `the coach smiled at the player <b2> tossed the frisbee <b3>` | `the coach <b2> smiled at the player <b2> tossed the frisbee <b3>` | `the coach smiled at the player tossed the frisbee <b3>` | no | no |
| adv-010 | reduced relative | `the experienced soldiers warned about the dangers <b2> conducted the midnight raid <b3>` | `the experienced soldiers warned about the dangers conducted the midnight raid <b3>` | `the experienced soldiers warned about the dangers conducted the midnight raid <b3>` | no | no |
| adv-011 | NP/Z | `while the man hunted <b2> the deer ran into the woods <b3>` | `while the man hunted the deer <b2> ran into the woods <b3>` | `while the man hunted the deer ran into the woods <b3>` | no | no |
| adv-012 | NP/Z | `since jay always jogs <b2> a mile seems like a short distance <b3>` | `since jay always jogs a mile <b2> seems like a short distance <b3>` | `since jay always jogs <b2> a mile seems like a short distance <b3>` | no | yes |
| adv-013 | NP/Z | `after the child visited <b2> the doctor prescribed a course of antibiotics <b3>` | `after the child visited the doctor <b2> prescribed a course of antibiotics <b3>` | `after the child visited the doctor prescribed a course of antibiotics <b3>` | no | no |
| adv-014 | NP/Z | `before the police stopped <b2> the driver was getting nervous <b3>` | `before the police stopped the driver was getting nervous <b3>` | `before the police stopped <b2> the driver was getting nervous <b3>` | no | yes |
| adv-015 | NP/Z | `as the criminal shot <b2> the woman screamed <b3>` | `as the criminal shot the woman <b2> screamed <b3>` | `as the criminal shot the woman screamed <b3>` | no | no |
| adv-016 | NP/Z | `whenever the band played <b2> the crowd cheered loudly <b3>` | `whenever the band played the crowd <b2> cheered loudly <b3>` | `whenever the band played the crowd cheered loudly <b3>` | no | no |
| adv-017 | NP/Z | `though the boy read <b2> the letter remained on the table <b3>` | `though the boy read the letter <b2> remained on the table <b3>` | `though the boy read the letter remained on the table <b3>` | no | no |
| adv-018 | NP/Z | `when the dog scratched <b2> the vet gave it a sedative <b3>` | `when the dog scratched the vet <b2> gave it a sedative <b3>` | `when the dog scratched the vet gave it a sedative <b3>` | no | no |

### What the two systems do differently

The brief this project was written against asks whether the sequence labeller
shows the same signature as the finding in the garden-path work: structure
encoded but not consulted until something forces the issue. The answer is that
it does not show that signature, and what it does instead is worse.

**System A shows the signature clearly.** On the NP/Z sentences it does not fail
to place a break. It places one, in the wrong position, exactly where the locally
favoured reading says a constituent ends:

    gold      while the man hunted <b2> the deer ran into the woods <b3>
    System A  while the man hunted the deer <b2> ran into the woods <b3>

    gold      as the criminal shot <b2> the woman screamed <b3>
    System A  as the criminal shot the woman <b2> screamed <b3>

The parser attached "the deer" as the object of "hunted", `b2_long_subject` saw
a subject ending at "deer", and the rule closed it there. The system commits to
the garden path and then makes it audible. Six of the eight NP/Z sentences come
out this way, which is where most of System A's 7 spurious breaks in that bucket
come from. This is the failure mode the rule set was expected to have, and it is
the one that would be most damaging in a real voice: a listener hearing that
prosody is being actively steered into the wrong parse.

**System B does not show the signature. It stays silent.** Across the whole
60-sentence adversarial set the model produces **zero** spurious breaks, at 84.6%
`b2` precision, with `b2` recall of 17.5%. On ten reduced relatives it places one
internal break in total. It is not choosing the local reading over the global
one; it is declining to make a structural decision at all.

The contrast is sharpest in the model's own numbers. Its `b2` precision barely
moves between the in-domain test set and the adversarial set (71.0 to 84.6) while
its `b2` recall collapses (61.7 to 17.5). That is the profile of a system that
has learned where commas go, applies that knowledge confidently where commas
would go, and has nothing to say anywhere else. The training signal never once
rewarded placing a break at a boundary that punctuation does not mark, so it
never learned that such boundaries exist. Whatever syntax is encoded in
distilroberta's representations is not being consulted, and the objective gives
it no reason to be.

Both failures come from the same place, which is the proxy label. The rule
system inherits its errors from a parser trained on punctuated, cased text and
run on text that has neither. The model inherits its errors from a target that
defines "break" as "comma". Neither system is short of structure; both are short
of a reason to use it.

## Error taxonomy

Every miss on the adversarial set, bucketed by construction. `missed` is a gold
break the system did not place, `wrong level` a gold break placed at the wrong
strength, `spurious` a break placed where gold has none.

| construction | gold breaks | A: parser rules missed | A: parser rules wrong level | A: parser rules spurious | B: distilroberta missed | B: distilroberta wrong level | B: distilroberta spurious | A+B union missed | A+B union wrong level | A+B union spurious |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| reduced relative | 20 | 7 | 0 | 2 | 9 | 0 | 0 | 7 | 0 | 2 |
| NP/Z | 16 | 8 | 0 | 7 | 6 | 0 | 0 | 6 | 0 | 7 |
| long subject | 16 | 1 | 0 | 3 | 8 | 0 | 0 | 1 | 0 | 3 |
| coordination scope | 14 | 2 | 0 | 2 | 4 | 2 | 0 | 1 | 1 | 2 |
| appositive | 23 | 6 | 0 | 2 | 6 | 2 | 0 | 3 | 2 | 2 |
| fronted PP | 16 | 3 | 0 | 3 | 7 | 0 | 0 | 3 | 0 | 3 |
| break at no punctuation site | 20 | 3 | 0 | 2 | 10 | 0 | 0 | 3 | 0 | 2 |
| **all** | **125** | **30** | **0** | **21** | **50** | **4** | **0** | **24** | **3** | **21** |

Sentences reproduced exactly, out of the number of sentences in each bucket:

| construction | A: parser rules | B: distilroberta | A+B union |
|---|---:|---:|---:|
| reduced relative | 1 | 1 | 1 |
| NP/Z | 0 | 2 | 1 |
| long subject | 5 | 0 | 5 |
| coordination scope | 5 | 3 | 5 |
| appositive | 2 | 2 | 3 |
| fronted PP | 5 | 1 | 5 |
| break at no punctuation site | 5 | 0 | 5 |

Three things in that table are worth stating plainly.

**System B misses 50 of 125 gold breaks and invents none.** System A misses 30
and invents 21. Those are opposite error profiles, not different amounts of the
same error, and which one is preferable depends on the voice: a missing break
makes a phrase run long, a spurious break makes the wrong parse audible. For a
front end I would rather have the missing break, which is an argument for the
model that its F1 does not make.

**The `break at no punctuation site` bucket separates them completely.** System
A misses 3 of 20 and reproduces 5 of 10 sentences exactly. System B misses 10 of
20 and reproduces none. This bucket was built to be invisible to punctuation
supervision, and it is: the system trained on punctuation cannot find these
breaks, and the system that reasons from constituent structure mostly can.

**The union is the best of the three on this set** at 81.8 any-break F1 and 60.7
`b2` F1, but it buys that by inheriting all 21 of System A's spurious breaks. It
is a recall-oriented merge and the table shows the price.

## Emphasis, qualitatively

No gold, so this is a table of what got marked and nothing more.

| sentence | marked |
|---|---|
| `the horse raced past the barn *fell* <b3>` | `fell` (nuclear) |
| `the man returned to his *house* <b2> was *arrested* <b3>` | `house` (nuclear), `arrested` (nuclear) |
| `while the man hunted the *deer* <b2> ran into the *woods* <b3>` | `deer` (nuclear), `woods` (nuclear) |
| `since jay always jogs a *mile* <b2> seems like a short *distance* <b3>` | `mile` (nuclear), `distance` (nuclear) |
| `the *man* <b2> who came to dinner last *night* <b2> left his umbrella in the *hall* <b3>` | `man` (nuclear), `night` (nuclear), `hall` (nuclear) |
| `the very tall woman standing by the *door* <b2> *waved* at us <b3>` | `door` (nuclear), `waved` (nuclear) |
| `she went *home* <b3> and he stayed at the *office* <b3>` | `home` (nuclear), `office` (nuclear) |
| `i spoke to the manager and the assistant manager was there *too* <b3>` | `too` (nuclear) |
| `my *brother* <b2> the doctor said it was nothing *serious* <b3>` | `brother` (nuclear), `serious` (nuclear) |
| `doctor *lee* <b2> the head of *cardiology* <b2> will see you *now* <b3>` | `lee` (nuclear), `cardiology` (nuclear), `now` (nuclear) |
| `in the middle of the *night* <b2> the phone *rang* <b3>` | `night` (nuclear), `rang` (nuclear) |
| `after a long and difficult *negotiation* <b2> the two sides shook *hands* <b3>` | `negotiation` (nuclear), `hands` (nuclear) |
| `the man in the grey *coat* <b2> is my *uncle* <b3>` | `coat` (nuclear), `uncle` (nuclear) |
| `what i want to *know* <b2> is why nobody *told* me <b3>` | `know` (nuclear), `told` (nuclear) |
| `i did not *order* the *fish* <b3>` | `order` (negation_focus), `fish` (nuclear) |
| `she took the *train* rather than the *bus* <b3>` | `train` (rather_than), `bus` (rather_than) |
| `we went to the *museum* instead of the *park* <b3>` | `museum` (rather_than), `park` (rather_than) |
| `he wanted not the *money* but the *credit* <b3>` | `money` (contrastive_but), `credit` (contrastive_but) |
| `they never *asked* for *permission* <b3>` | `asked` (negation_focus), `permission` (nuclear) |

## Speed

Measured on CPU, one utterance at a time, over 300 LibriTTS-R test utterances (17014 tokens).

| system | tokens/second | ms per utterance |
|---|---:|---:|
| rules | 1299 | 43.65 |
| model | 461 | 123.08 |
| model+emphasis | 349 | 162.64 |

Streaming rather than batched, because a front end sees one utterance at a time.
The `rules` figure includes the spaCy parse, which is nearly all of it. The
`model` figure is the transformer forward pass alone; `model+emphasis` adds the
parse back, because emphasis marking is rule-based and needs a parse whichever
system placed the breaks. So the honest cost of the model path in a system that
also wants emphasis is 349 tokens per second, not 461.

Both are comfortably faster than real time (English speech runs at roughly 3
words per second) and neither is close to being the bottleneck in a TTS pipeline.
The interesting number is that they are within a factor of four of each other:
the parse is not cheap, and choosing System A does not buy much speed.

## Where it fails

**The ranking inverts between the two kinds of gold, and that is the headline.**
System B beats System A by 37 F1 on LibriTTS-R and loses to it on the adversarial
set, by 29 F1 on `b2` specifically. If this project reported only the corpus
numbers it would conclude that the sequence labeller is the answer. It is not,
and any paper that scores prosody against punctuation will reach the wrong
conclusion in the same way.

**Neither system recovers reduced relatives.** "the horse raced past the barn
fell" gets no internal break from either. spaCy makes "raced" the root and "fell"
a complement, so the subject `b2_long_subject` sees is two tokens and the rule
never fires. Seven of the twenty gold breaks in that bucket are missed by System
A and nine by System B. This is the case the rule set was written to make visible
rather than to solve, and it stayed visible.

**System A steers into the garden path on NP/Z.** Six of eight NP/Z sentences get
a break after the misattached object. See the section above; this is the worst
individual failure in the repository.

**System B produces almost no minor breaks off-distribution.** 17.5% `b2` recall
on the adversarial set against 61.7% in domain, with precision holding. It is
confident where commas would go and mute elsewhere.

**The parser degrades badly on lowercase, punctuation-free input.** This is a
compounding failure that is easy to miss. "the phone rang" has "rang" tagged
NOUN; "we bought apples oranges pears and bananas" has "oranges" tagged VBZ and
made the head of a clause. An early version of `b2_fronted_adverbial` required
the head of the fronted constituent to be a verb, and that requirement silently
disabled the rule on ordinary sentences until the POS tags were inspected
directly. It now also accepts a ROOT of any part of speech. Any rule written
against a dependency label is only as good as the parse of text the parser was
not trained on.

**Vocative detection is a heuristic and knows it.** The English spaCy model has
no `vocative` label, so `_is_vocative` looks for an address noun or bare proper
name hanging off the root without a grammatical function. On lowercased text,
proper-noun detection is itself unreliable, so this rule is the least trustworthy
of the seven.

**Emphasis is unmeasured.** There is no gold anywhere in this project. The
qualitative table above shows the rules behaving as designed, and that is the
entire evidential basis. Do not read it as an accuracy claim.

**Switchboard is the least informative of the three test sets.** Its utterances
are short and 98.8% of its gold breaks are utterance-final, so the trivial
baseline scores 78.7 and beats System A. The set is included because
conversational speech is the target domain, but almost all of the signal in it is
"does the utterance end here", which is not the question this stage is for.

**One epoch, one seed, one base model.** There is no variance estimate on any
number in this README. The training curve was still rising when the run stopped.

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
