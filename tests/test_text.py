"""Normalisation and weak-label derivation."""

import pytest

from caesura.text import (
    labels_from_punctuation,
    normalize,
    normalize_spans,
    strip_markup,
)
from caesura.types import B2, B3, NONE


def test_normalize_strips_punctuation_and_case():
    assert normalize("Doctor Lee, lead a live band!") == [
        "doctor", "lee", "lead", "a", "live", "band",
    ]


def test_normalize_keeps_intra_word_apostrophes():
    assert normalize("I don't know, it's Bill's.") == [
        "i", "don't", "know", "it's", "bill's",
    ]


def test_normalize_removes_upstream_markup():
    assert normalize("the *band* <b2> played <b3>") == ["the", "band", "played"]


def test_strip_markup_leaves_words():
    assert strip_markup("a *b* <b2> c") == "a b   c"


def test_normalize_handles_empty_and_punctuation_only():
    assert normalize("") == []
    assert normalize("... ?! --") == []


def test_labels_comma_is_b2_and_period_is_b3():
    tokens, labels = labels_from_punctuation("Yes, of course.")
    assert tokens == ["yes", "of", "course"]
    assert labels == [B2, NONE, B3]


def test_labels_read_through_closing_quote():
    tokens, labels = labels_from_punctuation('How quickly he disappeared!"')
    assert labels[-1] == B3


def test_labels_sentence_final_beats_phrase_final():
    _, labels = labels_from_punctuation("wait for it,.")
    assert labels[-1] == B3


def test_labels_dash_separates():
    tokens, labels = labels_from_punctuation("he said -- quietly -- that it was over.")
    assert tokens[:3] == ["he", "said", "quietly"]
    assert labels[1] == B2
    assert labels[2] == B2


def test_labels_brackets_separate_but_quotes_do_not():
    _, bracket = labels_from_punctuation("he said (quietly) that it was over.")
    assert bracket[1] == B2
    _, quoted = labels_from_punctuation('he said "quietly" that it was over.')
    assert quoted[1] == NONE


def test_labels_align_one_to_one_with_tokens():
    for text in [
        "A short one.",
        "Well--that is--I don't know; perhaps.",
        'She cried, "Stop!" and he stopped.',
        "1, 2, 3.",
    ]:
        tokens, labels = labels_from_punctuation(text)
        assert len(tokens) == len(labels)
        assert tokens == normalize(text)


def test_labels_empty_input():
    assert labels_from_punctuation("") == ([], [])


SPAN_CASES = [
    "I read the 2nd Dr. Lee lead a live band on Reading Rd at 10:30",
    "The phone, suddenly, rang!",
    "Well--that is--I don't know; perhaps.",
    'She cried, "Stop!" and he stopped.',
    "the *phone* <b2> rang <b3>",
    "  leading and trailing   space  ",
]


@pytest.mark.parametrize("text", SPAN_CASES)
def test_normalize_spans_agrees_with_normalize(text):
    assert [w for w, _, _, _ in normalize_spans(text)] == normalize(text)


@pytest.mark.parametrize("text", SPAN_CASES)
def test_normalize_spans_offsets_point_at_the_input(text):
    for _, start, end, surface in normalize_spans(text):
        assert text[start:end] == surface
        assert 0 <= start < end <= len(text)


@pytest.mark.parametrize("text", SPAN_CASES)
def test_normalize_spans_are_non_overlapping_and_ordered(text):
    spans = [(s, e) for _, s, e, _ in normalize_spans(text)]
    assert spans == sorted(spans)
    for (_, prev_end), (next_start, _) in zip(spans, spans[1:]):
        assert prev_end <= next_start


def test_normalize_spans_on_empty_input():
    assert normalize_spans("") == []
    assert normalize_spans("... ?!") == []
