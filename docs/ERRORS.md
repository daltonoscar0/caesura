# Error taxonomy, emphasis and failure modes

Part of the [Caesura](../README.md) documentation.

**Before reading any numbers here, read [the caveat](../README.md#read-this-before-the-results):** break labels for training and for two of the three test sets are derived from punctuation, and punctuation is not prosody.

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

No gold, so this is a listing of what got marked and nothing more. Two sentences
per construction from the adversarial set, then five probes at the end, because
the adversarial set was written to test breaks and contains no contrastive
constructions for the three non-default rules to fire on.

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
