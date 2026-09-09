"""Emphasis marking. There is no gold, so these pin behaviour, not accuracy."""

import pytest

from caesura import run
from caesura.emphasis import _phrases
from caesura.types import B2, B3, NONE


def emphasised(text, system="rules"):
    return list(rules_for(text, system))


def rules_for(text, system="rules"):
    return run(text, system=system)["meta"]["emphasis"]


def test_phrases_split_on_every_break():
    assert _phrases([NONE, B2, NONE, NONE, B3]) == [(0, 1), (2, 4)]


def test_phrases_handle_a_trailing_run_without_a_break():
    assert _phrases([NONE, B2, NONE]) == [(0, 1), (2, 2)]


def test_one_nuclear_accent_per_intonational_phrase():
    result = run("in the middle of the night the phone rang", system="rules")
    n_phrases = sum(1 for b in result["meta"]["breaks"] if b != NONE)
    assert len(result["meta"]["emphasis"]) >= n_phrases


def test_nuclear_accent_falls_on_a_content_word():
    assert emphasised("the phone rang") == ["rang"]


def test_function_words_are_not_accented_by_default():
    marked = emphasised("she looked at him")
    assert "at" not in marked and "she" not in marked


def test_negation_focus_is_marked():
    # The rule accents the first content word after the negation, which is the
    # verb here, and overrides the nuclear accent it would otherwise carry.
    assert rules_for("i did not order the fish")["order"] == "emph_negation_focus"


def test_rather_than_accents_both_sides():
    marks = rules_for("she took the train rather than the bus")
    assert marks.get("train") == "emph_rather_than"
    assert marks.get("bus") == "emph_rather_than"


def test_emphasis_appears_in_the_rendered_output():
    assert "*rang*" in run("the phone rang", system="rules")["output"]


def test_emphasis_marks_are_reported_with_their_rule():
    marks = rules_for("the phone rang")
    assert set(marks) == {"rang"}
    assert marks["rang"].startswith("emph_")
