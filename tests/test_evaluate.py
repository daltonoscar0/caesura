"""Scoring arithmetic and the error taxonomy."""

import pytest

from caesura import evaluate
from caesura.data import Example
from caesura.types import B2, B3, NONE


def test_prf_on_a_worked_example():
    #                gold: b2 at 1 and 3; pred: b2 at 1 and 2
    gold = [[NONE, B2, NONE, B2, B3]]
    pred = [[NONE, B2, B2, NONE, B3]]
    s = evaluate.score(gold, pred)
    assert s[B2] == {"p": 50.0, "r": 50.0, "f1": 50.0, "support": 2, "predicted": 2}
    assert s[B3]["f1"] == 100.0
    # any-break collapses b2 and b3: gold breaks at 1,3,4; pred at 1,2,4.
    assert s["any"]["p"] == pytest.approx(66.7, abs=0.1)
    assert s["any"]["r"] == pytest.approx(66.7, abs=0.1)


def test_prf_handles_a_class_with_no_predictions():
    s = evaluate.score([[NONE, B2, B3]], [[NONE, NONE, B3]])
    assert s[B2] == {"p": 0.0, "r": 0.0, "f1": 0.0, "support": 1, "predicted": 0}


def test_macro_break_f1_ignores_the_none_class():
    s = evaluate.score([[NONE, B2, B3]], [[NONE, B2, B3]])
    assert s["macro_break_f1"] == 100.0


def test_final_only_baseline_marks_exactly_the_last_boundary():
    ex = Example(["a", "b", "c"], [NONE, B2, B3], "x", "1")
    assert evaluate.final_only(ex) == [NONE, NONE, B3]


def test_marked_renders_breaks_inline():
    assert evaluate.marked(["a", "b"], [B2, B3]) == "a <b2> b <b3>"


def test_error_taxonomy_separates_the_three_failure_modes():
    examples = [
        Example(["a", "b", "c"], [NONE, B2, B3], "adv", "1", construction="long_subject"),
    ]
    preds = {"sys": [[B2, NONE, B2]]}
    table = evaluate.error_taxonomy(examples, preds)["sys"]["long_subject"]
    assert table["gold_breaks"] == 2
    assert table["missed"] == 1      # gold b2 at index 1 predicted none
    assert table["wrong_level"] == 1  # gold b3 at index 2 predicted b2
    assert table["spurious"] == 1     # predicted b2 at index 0 with gold none
    assert table["exact_sentences"] == 0


def test_error_taxonomy_counts_an_exact_match():
    examples = [Example(["a", "b"], [NONE, B3], "adv", "1", construction="appositive")]
    table = evaluate.error_taxonomy(examples, {"sys": [[NONE, B3]]})["sys"]["appositive"]
    assert table == {
        "gold_breaks": 1,
        "missed": 0,
        "wrong_level": 0,
        "spurious": 0,
        "exact_sentences": 1,
    }


def test_garden_path_table_selects_only_garden_path_items():
    examples = [
        Example(["a", "b"], [NONE, B3], "adv", "1", construction="reduced_relative"),
        Example(["c", "d"], [NONE, B3], "adv", "2", construction="appositive"),
        Example(["e", "f"], [NONE, B3], "adv", "3", construction="np_z"),
    ]
    preds = {"rules": [[NONE, B3]] * 3, "model": [[B2, B3]] * 3}
    rows = evaluate.garden_path_table(examples, preds)
    assert [r["id"] for r in rows] == ["1", "3"]
    assert rows[0]["rules_match"] is True
    assert rows[0]["model_match"] is False


def test_published_baselines_are_present_and_attributed():
    assert len(evaluate.PUNCTUATION_BASELINES) == 2
    for entry in evaluate.PUNCTUATION_BASELINES:
        assert entry["test_set"]
        assert 0 < entry["comma_f1"] < 100
        assert 0 < entry["period_f1"] < 100
