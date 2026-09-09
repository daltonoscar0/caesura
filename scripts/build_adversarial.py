"""Build data/adversarial.jsonl from the compact notation below.

Notation: write the sentence lowercase and unpunctuated, with ``|`` where a
minor break (<b2>) belongs and ``||`` where a major break (<b3>) belongs. Every
sentence ends in ``||``. Writing the gold this way rather than as parallel
token/label arrays means the annotation cannot silently drift out of alignment
with the words.

Gold here is my own judgement about where a competent reader pauses, not
punctuation and not a ToBI transcription. The rationale on each item is the
one clause that justifies the break; if the rationale does not survive being
read aloud, the item is wrong and should be changed rather than defended.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from caesura.text import normalize  # noqa: E402
from caesura.types import B2, B3, NONE  # noqa: E402

OUT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "adversarial.jsonl"
)

# (construction, marked sentence, rationale)
ITEMS = [
    # ---------------------------------------------------------------- MV/RR
    ("reduced_relative",
     "the horse raced past the barn | fell ||",
     "raced heads a reduced relative, so the break closes the subject and blocks the main-verb reading"),
    ("reduced_relative",
     "the man returned to his house | was arrested ||",
     "returned is a participle modifying man, and the break marks where the subject ends"),
    ("reduced_relative",
     "the florist sent the flowers | was pleased ||",
     "sent is passive inside the subject, so the break has to precede the real main verb"),
    ("reduced_relative",
     "the defendant examined by the lawyer | turned out to be unreliable ||",
     "the by-phrase makes the reduced relative unambiguous, and the break closes it"),
    ("reduced_relative",
     "the boat floated down the river | sank ||",
     "floated is a participle, so the break separates the modified subject from sank"),
    ("reduced_relative",
     "the student given the answers | cheated ||",
     "given heads a reduced relative and the break closes the subject before cheated"),
    ("reduced_relative",
     "the soldiers marched across the field | were exhausted ||",
     "marched is a participle here, so the break precedes the finite verb"),
    ("reduced_relative",
     "the woman brought the sandwiches | left early ||",
     "brought modifies woman, and the break keeps left early as the matrix predicate"),
    ("reduced_relative",
     "the coach smiled at the player | tossed the frisbee ||",
     "smiled at the player is a reduced relative on coach, so the break precedes tossed"),
    ("reduced_relative",
     "the experienced soldiers warned about the dangers | conducted the midnight raid ||",
     "warned about the dangers modifies soldiers, and the break closes that long subject"),

    # ------------------------------------------------------------------ NP/Z
    ("np_z",
     "while the man hunted | the deer ran into the woods ||",
     "hunted is intransitive here, so the break stops the deer being taken as its object"),
    ("np_z",
     "since jay always jogs | a mile seems like a short distance ||",
     "jogs closes the subordinate clause, and the break keeps a mile as the matrix subject"),
    ("np_z",
     "after the child visited | the doctor prescribed a course of antibiotics ||",
     "visited ends the subordinate clause, so the break blocks the doctor as its object"),
    ("np_z",
     "before the police stopped | the driver was getting nervous ||",
     "stopped is intransitive here and the break prevents the driver being read as its object"),
    ("np_z",
     "as the criminal shot | the woman screamed ||",
     "shot ends the clause, and the break keeps the woman as subject of screamed"),
    ("np_z",
     "whenever the band played | the crowd cheered loudly ||",
     "played closes the when-clause, and the break stops the crowd being its object"),
    ("np_z",
     "though the boy read | the letter remained on the table ||",
     "read ends the concessive clause, so the break keeps the letter out of its object slot"),
    ("np_z",
     "when the dog scratched | the vet gave it a sedative ||",
     "scratched is intransitive here, and the break prevents the vet being taken as its object"),

    # --------------------------------------------------------- long subjects
    ("long_subject",
     "the man who came to dinner last night | left his umbrella in the hall ||",
     "a nine-token subject needs a break before the predicate for the listener to close it"),
    ("long_subject",
     "the very tall woman standing by the door | waved at us ||",
     "the participial modifier makes the subject long enough to require a closing break"),
    ("long_subject",
     "all of the people waiting in the long line outside the theatre | got in ||",
     "an eleven-token subject cannot be held open across the predicate without a break"),
    ("long_subject",
     "the report that the committee published last spring | was widely criticised ||",
     "the relative clause inside the subject forces a break before the matrix verb"),
    ("long_subject",
     "everything that could possibly go wrong on a trip like this | did ||",
     "the one-word predicate is only audible if the long subject is closed by a break"),
    ("long_subject",
     "the small brown dog with the torn left ear | belongs to my neighbour ||",
     "the postmodifying PP extends the subject past the point where a break is needed"),
    ("long_subject",
     "several of the older students in the advanced class | failed the exam ||",
     "the partitive subject runs eight tokens and closes with a break"),
    ("long_subject",
     "the letter my grandmother wrote during the war | arrived last week ||",
     "the bare relative clause inside the subject forces a break before arrived"),

    # ----------------------------------------------------------- coordination
    ("coordination",
     "she went home || and he stayed at the office ||",
     "and joins two finite clauses, which is a major break not a minor one"),
    ("coordination",
     "i spoke to the manager || and the assistant manager was there too ||",
     "the second conjunct has its own subject and verb, so the coordination is clausal"),
    ("coordination",
     "he ordered soup | and salad | and a glass of wine ||",
     "a three-item list takes a break after each non-final item"),
    ("coordination",
     "we invited the old men and women ||",
     "old scopes over the whole coordination on the default reading, so no internal break"),
    ("coordination",
     "old men and women were evacuated first ||",
     "the coordination is a single subject NP and takes no break inside it"),
    ("coordination",
     "you can have coffee or tea | and a biscuit ||",
     "the break groups coffee or tea against a biscuit rather than flattening the three"),
    ("coordination",
     "we studied french and german literature ||",
     "the two adjectives share literature, so breaking would impose the wrong scope"),
    ("coordination",
     "the professor said the students who cheated and their friends | would be expelled ||",
     "the break closes the coordinated subject of the embedded clause before its predicate"),

    # ------------------------------------------------------------ appositives
    ("appositive",
     "my brother | the doctor | said it was nothing serious ||",
     "the doctor is an appositive and is bracketed by breaks on both sides"),
    ("appositive",
     "doctor lee | the head of cardiology | will see you now ||",
     "the appositive is parenthetical to the subject and takes breaks on both sides"),
    ("appositive",
     "tell me | sir | what you saw ||",
     "sir is a vocative, which is prosodically parenthetical"),
    ("appositive",
     "the capital of france | paris | is a beautiful city ||",
     "paris renames the subject and is set off on both sides"),
    ("appositive",
     "she said | as far as i can tell | that the deal is off ||",
     "the hedge is a parenthetical inserted between the verb and its complement"),
    ("appositive",
     "his oldest friend | a man he had known since childhood | betrayed him ||",
     "the appositive is long enough that both its edges need marking"),
    ("appositive",
     "mister president | the delegation has arrived ||",
     "the vocative is closed off before the message begins"),
    ("appositive",
     "that idea | however | never gained much support ||",
     "however is a discourse adverb inserted parenthetically inside the clause"),

    # --------------------------------------------------------- fronted PP/adv
    ("fronted_pp",
     "in the middle of the night | the phone rang ||",
     "a six-token fronted PP closes with a break before the subject"),
    ("fronted_pp",
     "after a long and difficult negotiation | the two sides shook hands ||",
     "the fronted PP is closed before the matrix subject begins"),
    ("fronted_pp",
     "without any warning at all | the lights went out ||",
     "the fronted PP is long and takes a closing break"),
    ("fronted_pp",
     "having finished her work | she went for a walk ||",
     "the fronted participial clause is closed before the matrix subject"),
    ("fronted_pp",
     "on reading road at ten thirty | a live band was playing ||",
     "two stacked fronted PPs close together before the subject"),
    ("fronted_pp",
     "quite unexpectedly | the meeting was cancelled ||",
     "the sentence adverb is prosodically separate from the clause it modifies"),
    ("fronted_pp",
     "to everyone's surprise | the youngest candidate won ||",
     "the fronted PP is a stance adverbial and closes with a break"),
    ("fronted_pp",
     "if you leave now | you will still be late ||",
     "the fronted conditional clause closes before the matrix clause"),

    # ------------------------------------------- break with no punctuation site
    ("non_punctuation_break",
     "the man in the grey coat | is my uncle ||",
     "written English puts no comma here but the six-token subject still closes with a break"),
    ("non_punctuation_break",
     "what i want to know | is why nobody told me ||",
     "the pseudo-cleft closes its wh-subject before the copula with no comma in writing"),
    ("non_punctuation_break",
     "the reason he gave for being late | was completely absurd ||",
     "the subject contains a bare relative and closes before was with no written comma"),
    ("non_punctuation_break",
     "whether we go or stay | depends entirely on the weather ||",
     "the clausal subject closes before the verb although no comma is written"),
    ("non_punctuation_break",
     "that the experiment failed so completely | surprised everyone ||",
     "a that-clause subject needs a closing break that punctuation never marks"),
    ("non_punctuation_break",
     "running through the rain without an umbrella | is a bad idea ||",
     "the gerund subject runs seven tokens and closes before is with no comma"),
    ("non_punctuation_break",
     "john said | the professor is a fool ||",
     "the break marks the complement clause boundary that a comma would not be written at"),
    ("non_punctuation_break",
     "everyone who knew him well | agreed ||",
     "the relative clause inside the subject closes before the one-word predicate"),
    ("non_punctuation_break",
     "the number of people who applied this year | was much higher than expected ||",
     "an eight-token subject closes before the copula with no written comma"),
    ("non_punctuation_break",
     "the question of whether the treaty will be ratified | remains open ||",
     "the embedded interrogative extends the subject to nine tokens and forces a break"),
]


def parse_marked(marked: str):
    tokens = []
    labels = []
    for chunk in marked.split():
        if chunk == "||":
            if not labels:
                raise ValueError(f"break before any token in {marked!r}")
            labels[-1] = B3
            continue
        if chunk == "|":
            if not labels:
                raise ValueError(f"break before any token in {marked!r}")
            labels[-1] = B2
            continue
        words = normalize(chunk)
        if len(words) != 1:
            raise ValueError(f"chunk {chunk!r} normalises to {words!r} in {marked!r}")
        tokens.append(words[0])
        labels.append(NONE)
    if labels[-1] != B3:
        raise ValueError(f"sentence must end in || : {marked!r}")
    return tokens, labels


def main():
    seen = set()
    records = []
    for i, (construction, marked, rationale) in enumerate(ITEMS, start=1):
        tokens, labels = parse_marked(marked)
        text = " ".join(tokens)
        if text in seen:
            raise ValueError(f"duplicate sentence: {text!r}")
        seen.add(text)
        records.append(
            {
                "id": f"adv-{i:03d}",
                "source": "adversarial",
                "construction": construction,
                "text": text,
                "tokens": tokens,
                "labels": labels,
                "rationale": rationale,
            }
        )
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")

    by_construction = {}
    for rec in records:
        by_construction[rec["construction"]] = by_construction.get(rec["construction"], 0) + 1
    print(f"wrote {len(records)} items to {OUT}")
    for k, v in sorted(by_construction.items()):
        print(f"  {k:24s} {v}")


if __name__ == "__main__":
    main()
