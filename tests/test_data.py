"""Dataset construction and the hand-built adversarial set."""

import json
import os

import pytest

from caesura import data
from caesura.text import normalize
from caesura.types import B2, B3, NONE

ADVERSARIAL = os.path.join(data.DATA, "adversarial.jsonl")

CONSTRUCTIONS = {
    "reduced_relative",
    "np_z",
    "long_subject",
    "coordination",
    "appositive",
    "fronted_pp",
    "non_punctuation_break",
}


@pytest.fixture(scope="module")
def adversarial():
    return data.load_adversarial(ADVERSARIAL)


def test_adversarial_set_has_sixty_items(adversarial):
    assert len(adversarial) == 60


def test_every_adversarial_item_is_well_formed(adversarial):
    seen = set()
    for ex in adversarial:
        assert len(ex.tokens) == len(ex.labels)
        assert ex.tokens == normalize(" ".join(ex.tokens)), ex.id
        assert set(ex.labels) <= {NONE, B2, B3}
        assert ex.labels[-1] == B3, f"{ex.id} must end in b3"
        assert ex.construction in CONSTRUCTIONS, ex.id
        assert ex.note, f"{ex.id} has no rationale"
        assert ex.id not in seen
        seen.add(ex.id)


def test_adversarial_sentences_are_unique(adversarial):
    texts = [" ".join(ex.tokens) for ex in adversarial]
    assert len(set(texts)) == len(texts)


def test_every_construction_bucket_is_populated(adversarial):
    present = {ex.construction for ex in adversarial}
    assert present == CONSTRUCTIONS


def test_adversarial_text_carries_no_punctuation(adversarial):
    with open(ADVERSARIAL) as fh:
        for line in fh:
            rec = json.loads(line)
            assert rec["text"] == rec["text"].lower()
            assert not set(rec["text"]) & set(".,;:!?-()\"")


def test_build_passages_stitches_within_a_chapter():
    rows = [
        {"id": "1_1_000001_000000", "text": "First one."},
        {"id": "1_1_000002_000000", "text": "Second one, with a comma."},
        {"id": "2_9_000001_000000", "text": "A different chapter entirely, elsewhere."},
    ]
    passages = data.build_passages(rows, "test")
    assert len(passages) == 2
    assert passages[1].tokens[0] == "a"
    first = passages[0]
    assert first.tokens[:3] == ["first", "one", "second"]
    # The b3 from the first utterance survives inside the stitched passage.
    assert first.labels[1] == B3
    assert B2 in first.labels


def test_build_passages_drops_fragments_shorter_than_four_tokens():
    passages = data.build_passages([{"id": "1_1_000001_000000", "text": "Yes."}], "test")
    assert passages == []


def test_label_counts_always_reports_none():
    counts = data.label_counts([data.Example(["a"], [B3], "x", "1")])
    assert counts[NONE] == 0 and counts[B3] == 1


def test_round_trip_through_jsonl(tmp_path, adversarial):
    path = str(tmp_path / "round.jsonl")
    data.write_jsonl(adversarial, path)
    back = data.read_jsonl(path)
    assert [e.tokens for e in back] == [e.tokens for e in adversarial]
    assert [e.labels for e in back] == [e.labels for e in adversarial]
    assert [e.construction for e in back] == [e.construction for e in adversarial]
