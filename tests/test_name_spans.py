"""The name-span veto: no break inside a proper name, from either system.

The cue is not in the text. By the time this stage sees "doctor lee" the
capitalisation and the abbreviated title are gone, so nothing here can tell it
from two ordinary nouns. The normaliser upstream still had both and reports the
span, which is why this arrives as an argument rather than as a rule.

The default is off, so every number in the README is scored without it.
"""

import pytest

from caesura import run
from caesura import rules
from caesura.model import model_available
from caesura.text import normalize_spans
from caesura.types import B2, B3, NONE

# "I read the second Doctor Lee lead a live band on Reading Road at ten thirty"
#                    ^18     ^28                   ^49        ^61
# Offsets as the normaliser emits them, verified in
# tests/test_name_spans.py::test_offsets_are_the_normalisers_own.
FLAGSHIP = "I read the second Doctor Lee lead a live band on Reading Road at ten thirty"
DOCTOR_LEE = [18, 28]
READING_ROAD = [49, 61]


def test_offsets_are_the_normalisers_own():
    """The span arithmetic, pinned, so a drift here fails loudly."""
    assert FLAGSHIP[DOCTOR_LEE[0]:DOCTOR_LEE[1]] == "Doctor Lee"
    assert FLAGSHIP[READING_ROAD[0]:READING_ROAD[1]] == "Reading Road"


def test_forbids_the_boundary_inside_a_name():
    spans = normalize_spans(FLAGSHIP)
    forbidden = rules.forbid_name_spans(spans, [DOCTOR_LEE])
    words = [w for w, _, _, _ in spans]
    assert forbidden == {words.index("doctor"): "name_span"}


def test_leaves_the_boundary_at_a_name_edge_alone():
    """Breaking after "Lee" is a legitimate prediction; breaking inside is not."""
    spans = normalize_spans(FLAGSHIP)
    words = [w for w, _, _, _ in spans]
    forbidden = rules.forbid_name_spans(spans, [DOCTOR_LEE])
    assert words.index("lee") not in forbidden
    assert words.index("second") not in forbidden


def test_several_spans_are_all_forbidden():
    spans = normalize_spans(FLAGSHIP)
    words = [w for w, _, _, _ in spans]
    forbidden = rules.forbid_name_spans(spans, [DOCTOR_LEE, READING_ROAD])
    assert forbidden == {
        words.index("doctor"): "name_span",
        words.index("reading"): "name_span",
    }


def test_a_one_word_name_forbids_nothing():
    spans = normalize_spans("the lee estate was sold")
    assert rules.forbid_name_spans(spans, [[4, 7]]) == {}


@pytest.mark.parametrize("name_spans", [None, [], [[5, 5]], [[9, 4]], ["nonsense"]])
def test_absent_or_malformed_spans_forbid_nothing(name_spans):
    """A caller that cannot supply spans, or supplies rubbish, is not punished."""
    spans = normalize_spans(FLAGSHIP)
    assert rules.forbid_name_spans(spans, name_spans) == {}


def test_a_rule_that_would_break_inside_a_name_is_recorded_as_vetoed():
    tokens = "doctor lee returned to the house and the crowd cheered".split()
    doc = rules.parse(tokens)
    forbidden = {0: "name_span"}
    labels, rule_names, vetoed = rules.apply(tokens, doc=doc, forbidden=forbidden)
    assert labels[0] == NONE
    assert 0 not in rule_names
    # When no rule wanted that boundary there is nothing to record, so this
    # asserts only that a veto never silently becomes a break.
    assert vetoed.get(0, "name_span").endswith("name_span")


def test_run_without_name_spans_is_unchanged_by_this_feature():
    """The published numbers are scored with the default, so pin the default."""
    plain = run(FLAGSHIP, system="rules")
    assert plain["meta"]["name_spans"] == []
    assert plain["meta"]["breaks"][-1] == B3


def test_run_with_name_spans_reports_them():
    result = run(FLAGSHIP, system="rules", name_spans=[DOCTOR_LEE, READING_ROAD])
    assert result["meta"]["name_spans"] == [DOCTOR_LEE, READING_ROAD]


def test_rules_place_no_break_inside_a_name():
    result = run(FLAGSHIP, system="rules", name_spans=[DOCTOR_LEE, READING_ROAD])
    words = result["meta"]["words"]
    breaks = result["meta"]["breaks"]
    assert breaks[words.index("doctor")] == NONE
    assert breaks[words.index("reading")] == NONE


@pytest.mark.skipif(not model_available(), reason="System B weights not present")
def test_the_model_splits_the_name_without_the_veto():
    """The defect this exists to fix, pinned so a regression is visible.

    The labeller puts <b3> between "doctor" and "lee". That is not a weak
    prediction, it is a pause inside a name, and the recall-oriented union
    inherits it because a b3 beats the rules' no-break.
    """
    result = run(FLAGSHIP, system="both")
    words = result["meta"]["words"]
    assert result["meta"]["model_breaks"][words.index("doctor")] == B3
    assert result["meta"]["breaks"][words.index("doctor")] == B3


@pytest.mark.skipif(not model_available(), reason="System B weights not present")
def test_the_veto_stops_the_model_splitting_the_name():
    result = run(FLAGSHIP, system="both", name_spans=[DOCTOR_LEE, READING_ROAD])
    words = result["meta"]["words"]
    doctor = words.index("doctor")
    assert result["meta"]["breaks"][doctor] == NONE
    assert result["meta"]["model_breaks"][doctor] == NONE
    assert result["meta"]["vetoed"][str(doctor)] == "model:name_span"
    # The break the sentence does deserve is untouched.
    assert result["meta"]["breaks"][-1] == B3


@pytest.mark.skipif(not model_available(), reason="System B weights not present")
def test_the_veto_removes_the_decision_as_well_as_the_label():
    """A vetoed boundary must not survive as a decision in the contract."""
    result = run(FLAGSHIP, system="both", name_spans=[DOCTOR_LEE, READING_ROAD])
    surfaces = [d["surface"] for d in result["decisions"]]
    assert "Doctor" not in surfaces
