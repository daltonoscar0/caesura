"""The pipeline contract, and the canonical demo sentence.

The contract says a StageResult is a plain JSON-serialisable dict with no custom
classes, because a sibling stage parses it without importing this package. These
tests hold that line.
"""

import json

import pytest

from caesura import run
from caesura.api import SYSTEMS
from caesura.types import B2, B3, NONE

# The canonical demo sentence as it arrives from upstream, and as it arrives
# after the normalisation stage has verbalised it.
CANONICAL_RAW = "I read the 2nd Dr. Lee lead a live band on Reading Rd at 10:30"
CANONICAL = (
    "i read the second doctor lee lead a live band on reading road at ten thirty"
)

DECISION_KEYS = {
    "span", "surface", "result", "kind", "rule", "alternatives", "score", "note",
}


def test_stage_result_is_a_plain_dict():
    result = run("the phone rang", system="rules")
    assert type(result) is dict
    assert set(result) == {"stage", "input", "output", "tokens", "decisions", "meta"}
    assert result["stage"] == "caesura"


def test_stage_result_is_json_serialisable_as_is():
    result = run(CANONICAL, system="rules")
    assert json.loads(json.dumps(result)) == result


def test_input_is_echoed_verbatim():
    result = run(CANONICAL_RAW, system="rules")
    assert result["input"] == CANONICAL_RAW


def test_tokens_are_the_whitespace_tokens_of_output():
    result = run("in the middle of the night the phone rang", system="rules")
    assert result["tokens"] == result["output"].split()


def test_output_uses_the_contract_markup():
    result = run("in the middle of the night the phone rang", system="rules")
    assert result["output"].endswith("<b3>")
    assert "<b2>" in result["output"]
    assert any(t.startswith("*") and t.endswith("*") for t in result["tokens"])


def test_break_markers_come_from_the_declared_inventory():
    result = run(CANONICAL, system="rules")
    markers = {t for t in result["tokens"] if t.startswith("<")}
    assert markers <= {"<b1>", "<b2>", "<b3>"}


def test_one_decision_per_inserted_break():
    result = run("in the middle of the night the phone rang", system="rules")
    n_breaks = sum(1 for t in result["tokens"] if t.startswith("<b"))
    assert len(result["decisions"]) == n_breaks


def test_decisions_have_exactly_the_contract_fields():
    result = run("the man who came to dinner last night left", system="rules")
    assert result["decisions"]
    for d in result["decisions"]:
        assert set(d) == DECISION_KEYS
        assert d["kind"] == "break"
        assert isinstance(d["span"], list) and len(d["span"]) == 2
        assert isinstance(d["rule"], str) and d["rule"]
        assert isinstance(d["alternatives"], list)
        assert isinstance(d["score"], float)


def test_decision_spans_are_char_offsets_into_the_input():
    text = "I read the 2nd Dr. Lee lead a live band on Reading Rd at 10:30"
    result = run(text, system="rules")
    for d in result["decisions"]:
        start, end = d["span"]
        assert text[start:end] == d["surface"]


def test_decisions_are_ordered_by_position():
    result = run("the man who came to dinner last night left his umbrella", system="rules")
    spans = [d["span"][0] for d in result["decisions"]]
    assert spans == sorted(spans)


def test_every_decision_names_a_rule_function_not_the_system():
    result = run("the man who came to dinner last night left", system="rules")
    for d in result["decisions"]:
        assert d["rule"] not in SYSTEMS
        assert d["score"] == 1.0  # deterministic rule
        assert d["alternatives"] == []
        assert d["note"]


def test_incoming_punctuation_is_always_stripped():
    with_punct = run("The phone, suddenly, rang!", system="rules")
    without = run("the phone suddenly rang", system="rules")
    assert with_punct["meta"]["words"] == without["meta"]["words"]
    assert with_punct["meta"]["breaks"] == without["meta"]["breaks"]


def test_upstream_markup_is_stripped():
    chained = run("the *phone* <b2> rang <b3>", system="rules")
    assert chained["meta"]["words"] == ["the", "phone", "rang"]


def test_empty_input_is_a_well_formed_empty_result():
    result = run("   ...  ", system="rules")
    assert result["output"] == ""
    assert result["tokens"] == []
    assert result["decisions"] == []
    assert result["meta"]["n_tokens"] == 0


def test_unknown_system_is_rejected():
    with pytest.raises(ValueError):
        run("the phone rang", system="magic")
    assert SYSTEMS == ("rules", "model", "both")


def test_canonical_sentence_runs_in_both_of_its_forms():
    """The canonical demo sentence.

    The contract expects a break after "Lee" and after "band" and nothing after
    "the". These assertions pin the shape and the fact that both the raw and the
    normalised form are processed; they do not assert that System A gets the
    breaks right, because it does not. What each system actually produces is in
    the README.
    """
    for text, last in ((CANONICAL_RAW, "30"), (CANONICAL, "thirty")):
        result = run(text, system="rules")
        words = result["meta"]["words"]
        assert words[-1] == last
        assert result["meta"]["breaks"][-1] == B3
        assert result["decisions"][-1]["rule"] == "b3_utterance_final"
        # Nothing after "the", which the contract calls out explicitly.
        assert result["meta"]["breaks"][words.index("the")] == NONE


def test_emphasis_can_be_switched_off():
    result = run(CANONICAL, system="rules", emphasis=False)
    assert "*" not in result["output"]
    assert result["meta"]["emphasis"] == {}


def test_meta_reports_the_system_and_size():
    result = run(CANONICAL, system="rules")
    assert result["meta"]["system"] == "rules"
    assert result["meta"]["n_tokens"] == 16
    assert result["meta"]["seconds"] >= 0
    assert len(result["meta"]["breaks"]) == len(result["meta"]["words"]) == 16


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def test_cli_prints_the_stage_result_as_json(capsys):
    from caesura.__main__ import main

    assert main(["the phone rang"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["stage"] == "caesura"
    assert payload["output"].endswith("<b3>")


def test_cli_pretty_prints_the_marked_up_text(capsys):
    from caesura.__main__ import main

    assert main(["--pretty", "the phone rang"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("the phone *rang* <b3>")
    assert "b3_utterance_final" in out


def test_cli_forwards_unknown_flags_to_the_evaluator():
    """``--eval`` must not let the positional text argument eat flag values."""
    from caesura.__main__ import main

    seen = {}

    def fake_eval(argv):
        seen["argv"] = list(argv)
        return 0

    import caesura.evaluate as evaluate_mod

    real = evaluate_mod.main
    evaluate_mod.main = fake_eval
    try:
        assert main(["--eval", "--limit", "20", "--speed-n", "10"]) == 0
    finally:
        evaluate_mod.main = real
    assert seen["argv"] == ["--limit", "20", "--speed-n", "10"]
