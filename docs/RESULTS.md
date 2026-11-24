# Results

Part of the [Caesura](../README.md) documentation.

**Before reading any numbers here, read [the caveat](../README.md#read-this-before-the-results):** break labels for training and for two of the three test sets are derived from punctuation, and punctuation is not prosody.

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
the garden path and then makes it audible. Seven of the eight NP/Z sentences come
out this way, one spurious break each, which is all 7 of System A's spurious
breaks in that bucket. This is the failure mode the rule set was expected to
have, and it is
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
