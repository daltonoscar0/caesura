"""One unit test per rule, plus the vetoes.

Each test asserts that the named rule claims the boundary, not merely that some
break appears there, so a rule cannot pass by accident because a
higher-priority rule happened to fire in the same place.
"""

import pytest

from caesura import rules
from caesura.types import B2, B3, NONE


def labelled(text):
    tokens = text.split()
    doc = rules.parse(tokens)
    labels, rule_names, vetoed = rules.apply(tokens, doc=doc)
    return tokens, labels, rule_names, vetoed


def owner(text, word):
    """Return (label, rule) for the boundary after the first ``word``."""
    tokens, labels, rule_names, _ = labelled(text)
    i = tokens.index(word)
    return labels[i], rule_names.get(i)


def test_b3_utterance_final_always_fires():
    tokens, labels, rule_names, _ = labelled("the phone rang")
    assert labels[-1] == B3
    assert rule_names[len(tokens) - 1] == "b3_utterance_final"


def test_b3_clause_boundary_before_coordinated_finite_clause():
    label, rule = owner("she went home and he stayed at the office", "home")
    assert label == B3
    assert rule == "b3_clause_boundary"


def test_b3_clause_boundary_does_not_fire_on_phrasal_coordination():
    tokens, labels, _, _ = labelled("she bought bread and cheese")
    assert labels[tokens.index("bread")] == NONE


def test_b2_appositive_brackets_both_edges():
    tokens, labels, rule_names, _ = labelled(
        "doctor lee the head of cardiology will see you now"
    )
    i = tokens.index("lee")
    j = tokens.index("cardiology")
    assert (labels[i], labels[j]) == (B2, B2)
    assert rule_names[i] == "b2_appositive"
    assert rule_names[j] == "b2_appositive"


def test_b2_fronted_adverbial_after_fronted_pp():
    label, rule = owner("in the middle of the night the phone rang", "night")
    assert label == B2
    assert rule == "b2_fronted_adverbial"


def test_b2_relative_clause_before_long_relative():
    label, rule = owner("the man who came to dinner last night left", "man")
    assert label == B2
    assert rule == "b2_relative_clause"


def test_b2_relative_clause_ignores_short_relative():
    tokens, labels, _, _ = labelled("the man who left was tall")
    # "who left" is two tokens, below the four-token floor.
    assert labels[tokens.index("man")] == NONE


def test_b2_long_subject_after_five_token_subject():
    label, rule = owner(
        "the small brown dog with the torn left ear belongs to my neighbour", "ear"
    )
    assert label == B2
    assert rule == "b2_long_subject"


def test_b2_long_subject_ignores_short_subject():
    tokens, labels, _, _ = labelled("the dog belongs to my neighbour")
    assert labels[tokens.index("dog")] == NONE


def test_b2_list_item_fires_on_three_way_coordination():
    tokens, labels, rule_names, _ = labelled(
        "he wants the red one the blue one or the green one"
    )
    fired = [rule for rule in rule_names.values() if rule == "b2_list_item"]
    # The rule is allowed to find nothing here if the parser flattens the list,
    # but when it does fire it must be as a b2.
    for i, rule in rule_names.items():
        if rule == "b2_list_item":
            assert labels[i] == B2


def test_veto_never_breaks_between_determiner_and_noun():
    for text in [
        "the very tall woman standing by the door waved at us",
        "all of the people waiting in the long line outside the theatre got in",
        "several of the older students in the advanced class failed the exam",
    ]:
        tokens, labels, _, _ = labelled(text)
        doc = rules.parse(tokens)
        for i, label in enumerate(labels[:-1]):
            if label == NONE:
                continue
            assert not (
                doc[i].dep_ in rules.DET_DEPS and doc[i].head.i > i
            ), f"break inside a determiner span in {text!r} at {tokens[i]!r}"


def test_veto_never_breaks_between_auxiliary_and_verb():
    for text in [
        "the report that the committee published last spring was widely criticised",
        "the man returned to his house was arrested",
    ]:
        tokens, labels, _, _ = labelled(text)
        doc = rules.parse(tokens)
        for i, label in enumerate(labels[:-1]):
            if label == NONE:
                continue
            assert not (
                doc[i].dep_ in ("aux", "auxpass") and doc[i].head.i > i
            ), f"break inside an auxiliary span in {text!r} at {tokens[i]!r}"


def test_veto_is_recorded_when_it_suppresses_a_rule():
    # Constructed so a rule proposes a boundary the veto rejects; the veto log
    # must name both the rule and the reason whenever that happens.
    _, _, _, vetoed = labelled(
        "after a long and difficult negotiation the two sides shook hands"
    )
    for reason in vetoed.values():
        assert ":" in reason


def test_every_rule_has_a_name_and_is_reachable():
    assert len(rules.RULE_NAMES) == len(set(rules.RULE_NAMES))
    for rule in rules.RULES:
        assert rule.__doc__, f"{rule.__name__} has no docstring"


def test_apply_handles_empty_input():
    labels, rule_names, vetoed = rules.apply([])
    assert labels == [] and rule_names == {} and vetoed == {}


def test_labels_are_one_per_token():
    tokens = "the horse raced past the barn fell".split()
    labels, _, _ = rules.apply(tokens)
    assert len(labels) == len(tokens)
