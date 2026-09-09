"""Shape of the object the pipeline contract requires, and the canonical case."""

import json

import pytest

from caesura import run
from caesura.api import SYSTEMS
from caesura.types import B2, B3, NONE, Decision, StageResult, Token

CANONICAL = (
    "i read the second doctor lee lead a live band on reading road at ten thirty"
)


def test_run_returns_a_stage_result():
    result = run("the phone rang", system="rules")
    assert isinstance(result, StageResult)
    assert result.stage == "caesura"
    assert all(isinstance(t, Token) for t in result.tokens)
    assert all(isinstance(d, Decision) for d in result.decisions)


def test_tokens_are_one_per_normalised_word():
    result = run("The phone, suddenly, rang!", system="rules")
    assert [t.text for t in result.tokens] == ["the", "phone", "suddenly", "rang"]
    assert [t.index for t in result.tokens] == [0, 1, 2, 3]


def test_output_uses_the_contract_markup():
    result = run("in the middle of the night the phone rang", system="rules")
    assert "<b2>" in result.text or "<b3>" in result.text
    assert result.text.endswith("<b3>")
    assert "*" in result.text  # at least one emphasised token


def test_break_values_come_from_the_declared_inventory():
    result = run(CANONICAL, system="rules")
    assert set(result.breaks()) <= {NONE, "b1", B2, B3}


def test_one_decision_per_inserted_break():
    result = run("in the middle of the night the phone rang", system="rules")
    breaks = [i for i, t in enumerate(result.tokens) if t.brk != NONE]
    assert [d.index for d in result.decisions] == breaks
    for d in result.decisions:
        assert d.value in ("<b1>", "<b2>", "<b3>")


def test_every_decision_names_a_rule_and_carries_a_score():
    result = run("the man who came to dinner last night left", system="rules")
    for d in result.decisions:
        assert d.rule
        assert d.rule != "rules"  # a named function, not the system name
        assert d.score == 1.0  # deterministic rule


def test_incoming_punctuation_is_always_stripped():
    with_punct = run("The phone, suddenly, rang!", system="rules")
    without = run("the phone suddenly rang", system="rules")
    assert with_punct.normalized_text == without.normalized_text
    assert with_punct.breaks() == without.breaks()


def test_upstream_markup_is_stripped():
    chained = run("the *phone* <b2> rang <b3>", system="rules")
    assert chained.normalized_text == "the phone rang"


def test_empty_input_is_a_well_formed_empty_result():
    result = run("   ...  ", system="rules")
    assert result.tokens == [] and result.decisions == [] and result.text == ""


def test_result_serialises_to_json():
    result = run(CANONICAL, system="rules")
    payload = json.loads(result.to_json())
    assert payload["stage"] == "caesura"
    assert len(payload["tokens"]) == len(result.tokens)
    for token in payload["tokens"]:
        assert set(token) == {"text", "index", "brk", "emphasis"}


def test_unknown_system_is_rejected():
    with pytest.raises(ValueError):
        run("the phone rang", system="magic")
    assert SYSTEMS == ("rules", "model", "both")


def test_canonical_sentence_runs_and_is_reported_faithfully():
    """The canonical Sayso sentence.

    The expected reading is <b2> after "lee", <b2> after "band" and <b3> at the
    end. This test does not assert that System A gets it right, because it does
    not; it pins the shape of the output and the fact that the sentence is
    processed at all. What each system actually produces is in the README.
    """
    result = run(CANONICAL, system="rules")
    tokens = [t.text for t in result.tokens]
    assert tokens[-1] == "thirty"
    assert result.tokens[-1].brk == B3
    assert len(result.tokens) == 16
    assert result.decisions[-1].rule == "b3_utterance_final"


def test_emphasis_can_be_switched_off():
    result = run(CANONICAL, system="rules", emphasis=False)
    assert not any(t.emphasis for t in result.tokens)
    assert result.meta["emphasis"] == {}


def test_meta_reports_the_system_and_size():
    result = run(CANONICAL, system="rules")
    assert result.meta["system"] == "rules"
    assert result.meta["n_tokens"] == 16
    assert result.meta["seconds"] >= 0
