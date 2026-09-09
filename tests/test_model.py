"""System B. Skipped unless the weights are present locally."""

import pytest

from caesura import run
from caesura.model import encode, model_available
from caesura.types import B2, B3, NONE

pytestmark = pytest.mark.skipif(
    not model_available(), reason="System B weights not present; train or fetch them first"
)


@pytest.fixture(scope="module")
def tokenizer():
    from caesura.model import get_model

    return get_model().tokenizer


def test_labels_land_on_the_last_subword(tokenizer):
    words = ["extraordinarily", "long", "vocabularies"]
    enc, last_pos = encode(tokenizer, words, [NONE, B2, B3])
    word_ids = enc.word_ids()
    for word_index, pos in last_pos.items():
        assert word_ids[pos] == word_index
        following = word_ids[pos + 1] if pos + 1 < len(word_ids) else None
        assert following != word_index, "not the last subword of this word"


def test_non_label_positions_are_masked(tokenizer):
    enc, last_pos = encode(tokenizer, ["extraordinarily", "long"], [NONE, B3])
    labelled = [i for i, v in enumerate(enc["labels"]) if v != -100]
    assert sorted(labelled) == sorted(last_pos.values())


def test_model_predicts_one_label_per_word():
    from caesura.model import get_model

    words = "the man in the grey coat is my uncle".split()
    labels, scores = get_model().predict(words)
    assert len(labels) == len(words) == len(scores)
    assert set(labels) <= {NONE, B2, B3}
    assert all(0.0 <= s <= 1.0 for s in scores)


def test_batched_prediction_matches_single_prediction():
    from caesura.model import get_model

    model = get_model()
    batch = ["the phone rang".split(), "the man in the grey coat is my uncle".split()]
    single = [model.predict(t)[0] for t in batch]
    batched = [labels for labels, _ in model.predict_batch(batch)]
    assert single == batched


def test_api_model_path_produces_scores_below_one():
    result = run("the man in the grey coat is my uncle", system="model")
    assert result.meta["system"] == "model"
    for d in result.decisions:
        assert d.rule == "model"
        assert 0.0 < d.score <= 1.0


def test_both_records_the_other_systems_choice():
    result = run("the man who came to dinner last night left", system="both")
    assert result.meta["system"] == "both"
    assert "rules_labels" in result.meta and "model_labels" in result.meta
    disagreements = [d for d in result.decisions if d.alternatives]
    for d in disagreements:
        alt = d.alternatives[0]
        assert alt.system in ("rules", "model")
        assert alt.value in ("<none>", "<b2>", "<b3>")
        assert alt.value != d.value


def test_both_is_the_union_of_the_two_systems():
    text = "the man who came to dinner last night left his umbrella in the hall"
    rules = run(text, system="rules").breaks()
    model = run(text, system="model").breaks()
    both = run(text, system="both").breaks()
    for r, m, b in zip(rules, model, both):
        assert (b != NONE) == (r != NONE or m != NONE)


def test_empty_input_on_the_model_path():
    result = run("!!!", system="model")
    assert result.tokens == [] and result.decisions == []
