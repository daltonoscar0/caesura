"""System A: break placement from a dependency parse.

Every rule is a named function with the signature ``(doc, tokens) -> [Hit]``.
Rules are consulted in the order given by ``RULES`` and the first one to claim a
boundary owns it, so ``decisions[i].rule`` always names exactly one function.
Two vetoes run afterwards and can delete a break but never create one.

The dependency parse is ``en_core_web_sm``. Using an off-the-shelf parser is
deliberate: the parse is an input, not the contribution. The contribution is the
mapping from constituent structure to break placement, and the finding is where
that mapping breaks down (see the garden-path table in the README).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .types import B2, B3, NONE

CONTENT_POS = frozenset({"NOUN", "PROPN", "VERB", "ADJ", "ADV", "NUM"})
FINITE_TAGS = frozenset({"VBD", "VBP", "VBZ", "MD"})
DISCOURSE_MARKERS = frozenset(
    {"well", "so", "now", "anyway", "however", "therefore", "meanwhile",
     "besides", "moreover", "furthermore", "actually", "okay", "yeah", "no",
     "yes", "oh", "why", "look", "listen"}
)
VOCATIVE_NOUNS = frozenset(
    {"sir", "madam", "ma'am", "doctor", "mister", "miss", "mrs", "mr", "dr",
     "father", "mother", "friend", "friends", "gentlemen", "ladies", "captain",
     "professor", "boss", "man", "dear"}
)

#: Minimum length, in tokens, for the length-gated rules.
MIN_RELCL_LEN = 4
MIN_SUBJECT_LEN = 5
MIN_FRONTED_LEN = 2
MIN_LIST_ITEMS = 3


@dataclass(frozen=True)
class Hit:
    """A rule claiming that a break follows token ``index``."""

    index: int
    label: str


_NLP = None


def get_nlp():
    """Load ``en_core_web_sm`` once per process."""
    global _NLP
    if _NLP is None:
        import spacy

        _NLP = spacy.load("en_core_web_sm")
    return _NLP


def parse(tokens: Sequence[str]):
    """Parse a pre-tokenised, lowercased, punctuation-free word list.

    The Doc is built from our tokens rather than from a string so that spaCy's
    token indices and ours are the same object, which the rules rely on.
    """
    from spacy.tokens import Doc

    nlp = get_nlp()
    doc = Doc(nlp.vocab, words=list(tokens))
    return nlp(doc)


def _span(tok) -> Tuple[int, int]:
    idx = [t.i for t in tok.subtree]
    return min(idx), max(idx)


def _left_edge(tok) -> int:
    return _span(tok)[0]


def _right_edge(tok) -> int:
    return _span(tok)[1]


# --------------------------------------------------------------------------
# Rules, in priority order
# --------------------------------------------------------------------------


def b3_utterance_final(doc, tokens) -> List[Hit]:
    """The last boundary of an utterance is always a major break."""
    if not tokens:
        return []
    return [Hit(len(tokens) - 1, B3)]


def b3_clause_boundary(doc, tokens) -> List[Hit]:
    """Major break between finite clauses.

    Two sources: a root-to-root transition (the parser has segmented the token
    stream into more than one sentence) and a coordinating conjunction that
    joins two finite clauses rather than two phrases.
    """
    hits: List[Hit] = []
    sents = list(doc.sents)
    for sent in sents[:-1]:
        end = sent.end - 1
        if end < len(tokens) - 1:
            hits.append(Hit(end, B3))
    for tok in doc:
        if tok.dep_ != "cc" or tok.i == 0:
            continue
        # Find the conjunct this cc introduces and check it is a finite clause.
        conjuncts = [c for c in tok.head.children if c.dep_ == "conj" and c.i > tok.i]
        for conj in conjuncts:
            finite = conj.tag_ in FINITE_TAGS or any(
                c.dep_ in ("aux", "auxpass") for c in conj.children
            )
            has_subject = any(
                c.dep_ in ("nsubj", "nsubjpass", "csubj", "expl") for c in conj.children
            )
            if finite and has_subject:
                hits.append(Hit(tok.i - 1, B3))
            break
    return hits


def b2_appositive(doc, tokens) -> List[Hit]:
    """Minor breaks bracketing appositives, parentheticals and vocatives."""
    hits: List[Hit] = []
    for tok in doc:
        if tok.dep_ in ("appos", "parataxis"):
            start, end = _span(tok)
            if start > 0:
                hits.append(Hit(start - 1, B2))
            hits.append(Hit(end, B2))
        elif _is_vocative(tok):
            start, end = _span(tok)
            if start > 0:
                hits.append(Hit(start - 1, B2))
            hits.append(Hit(end, B2))
    return hits


def _is_vocative(tok) -> bool:
    """Crude vocative detector.

    spaCy's English model has no ``vocative`` label, so this looks for an
    address noun or bare proper name hanging off the root without a
    grammatical function, which is what a vocative parses as in practice.
    """
    if tok.lower_ not in VOCATIVE_NOUNS and tok.pos_ != "PROPN":
        return False
    if tok.dep_ not in ("npadvmod", "dep", "intj", "nmod"):
        return False
    return tok.head.pos_ in ("VERB", "AUX")


def b2_fronted_adverbial(doc, tokens) -> List[Hit]:
    """Minor break after a fronted adverbial or prepositional phrase.

    "Fronted" means the constituent's right edge sits to the left of its head
    verb, and to the left of that verb's subject if there is one.
    """
    hits: List[Hit] = []
    for sent in doc.sents:
        for tok in sent:
            if tok.dep_ not in ("advcl", "prep", "advmod", "npadvmod", "intj"):
                continue
            head = tok.head
            # The ROOT test is not redundant: on lowercased, punctuation-free
            # input the tagger mislabels clause-final verbs as nouns often
            # enough ("the phone rang" -> rang/NOUN) that requiring a verbal
            # POS silently disables this rule on ordinary sentences.
            if head.pos_ not in ("VERB", "AUX") and head.dep_ != "ROOT":
                continue
            if head.i < tok.i:
                continue
            subj = next(
                (c for c in head.children if c.dep_ in ("nsubj", "nsubjpass", "expl")),
                None,
            )
            start, end = _span(tok)
            if start != sent.start:
                continue
            if subj is not None and subj.i < end:
                continue
            length = end - start + 1
            is_marker = tok.lower_ in DISCOURSE_MARKERS or tok.dep_ == "intj"
            if length >= MIN_FRONTED_LEN or is_marker:
                if end < len(tokens) - 1:
                    hits.append(Hit(end, B2))
    return hits


def b2_relative_clause(doc, tokens) -> List[Hit]:
    """Minor break before a relative clause of at least four tokens.

    ``acl`` is admitted only for past participles, which is where reduced
    relatives land when the parser recovers them. Admitting ``acl`` wholesale
    would fire on infinitival modifiers ("a way to do it") that take no break.
    """
    hits: List[Hit] = []
    for tok in doc:
        if not (tok.dep_ == "relcl" or (tok.dep_ == "acl" and tok.tag_ == "VBN")):
            continue
        start, end = _span(tok)
        if end - start + 1 < MIN_RELCL_LEN:
            continue
        if start > 0:
            hits.append(Hit(start - 1, B2))
    return hits


def b2_long_subject(doc, tokens) -> List[Hit]:
    """Minor break after a subject NP of at least five tokens.

    This is the rule that decides garden-path cases. It only fires when the
    parser has actually built the long subject, so for a reduced relative that
    the parser misanalyses ("the horse raced past the barn fell") it does not
    fire at all. That failure is reported rather than patched.
    """
    hits: List[Hit] = []
    for tok in doc:
        if tok.dep_ not in ("nsubj", "nsubjpass", "csubj", "csubjpass"):
            continue
        start, end = _span(tok)
        if end - start + 1 < MIN_SUBJECT_LEN:
            continue
        if end < len(tokens) - 1:
            hits.append(Hit(end, B2))
    return hits


def b2_list_item(doc, tokens) -> List[Hit]:
    """Minor break after each non-final item of a list of three or more."""
    hits: List[Hit] = []
    for tok in doc:
        conjuncts = [c for c in tok.children if c.dep_ == "conj"]
        if not conjuncts:
            continue
        members = [tok] + conjuncts
        if len(members) < MIN_LIST_ITEMS:
            continue
        for member in members[:-1]:
            end = _right_edge(member)
            # Do not break on the conjunction itself; break before it.
            following = [c for c in member.children if c.dep_ == "cc"]
            if following:
                end = min(end, min(c.i for c in following) - 1)
            if 0 <= end < len(tokens) - 1:
                hits.append(Hit(end, B2))
    return hits


#: Consulted in this order. First rule to claim a boundary owns it.
RULES: List[Callable] = [
    b3_utterance_final,
    b3_clause_boundary,
    b2_appositive,
    b2_fronted_adverbial,
    b2_relative_clause,
    b2_long_subject,
    b2_list_item,
]

RULE_NAMES = [r.__name__ for r in RULES]


# --------------------------------------------------------------------------
# Vetoes: may delete a break, never create one
# --------------------------------------------------------------------------


DET_DEPS = frozenset({"det", "poss", "predet", "nummod", "amod", "compound"})


def veto_tight_span(doc, tokens, index: int) -> Optional[str]:
    """Never break inside a determiner-noun or auxiliary-verb span.

    The test is whether the token on the left of the boundary is a dependent
    whose head lies on the right of it, for the dependency types that make a
    span phonologically indivisible.
    """
    if index >= len(tokens) - 1:
        return None
    left = doc[index]
    if left.dep_ in DET_DEPS and left.head.i > index:
        return "det_noun"
    if left.dep_ in ("aux", "auxpass", "neg") and left.head.i > index:
        return "aux_verb"
    if left.pos_ in ("AUX", "PART") and left.head.i > index:
        return "aux_verb"
    return None


VETOES: List[Callable] = [veto_tight_span]


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------


def apply(tokens: Sequence[str], doc=None) -> Tuple[List[str], Dict[int, str], Dict[int, str]]:
    """Run the rule set.

    Returns ``(labels, rule_by_index, vetoed_by_index)`` where ``labels[i]`` is
    the break after ``tokens[i]``.
    """
    n = len(tokens)
    labels = [NONE] * n
    rule_by_index: Dict[int, str] = {}
    vetoed: Dict[int, str] = {}
    if n == 0:
        return labels, rule_by_index, vetoed
    if doc is None:
        doc = parse(tokens)

    for rule in RULES:
        for hit in rule(doc, tokens):
            i = hit.index
            if not (0 <= i < n):
                continue
            if labels[i] != NONE:
                continue
            reason = None
            for veto in VETOES:
                reason = veto(doc, tokens, i)
                if reason:
                    break
            if reason:
                vetoed.setdefault(i, f"{rule.__name__}:{reason}")
                continue
            labels[i] = hit.label
            rule_by_index[i] = rule.__name__
    return labels, rule_by_index, vetoed
