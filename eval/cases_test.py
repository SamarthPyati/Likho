"""Consistency checks on the committed case files — catches annotation typos."""

from collections import Counter

import pytest

from eval.cases import load_cases, parse_compact_tokens
from likho.schemas import TokenKind
from likho.tokenize import split

# docs/EVALUATION.md §1.2, counted by each case's primary (first) tag
TEST_DISTRIBUTION = {
    "plain": 10,
    "loanword": 10,
    "named_entity": 8,
    "ambiguous": 8,
    "hinglish_verb": 6,
    "long": 4,
    "already_devanagari": 4,
}


def test_parse_compact_tokens():
    tokens = parse_compact_tokens("kal/h Rohit/ne 10:30/n ,/p")
    assert [(t.surface, t.kind) for t in tokens] == [
        ("kal", TokenKind.HINDI),
        ("Rohit", TokenKind.NAMED_ENTITY),
        ("10:30", TokenKind.NUMERIC),
        (",", TokenKind.PUNCT),
    ]


def test_bad_code_rejected():
    with pytest.raises(ValueError):
        parse_compact_tokens("kal/x")


def test_test_split_matches_spec_distribution():
    cases = load_cases("test")
    assert len(cases) == 50
    assert Counter(c.category for c in cases) == TEST_DISTRIBUTION


def test_dev_split_is_20_and_disjoint_from_test():
    dev, test = load_cases("dev"), load_cases("test")
    assert len(dev) == 20
    assert not {c.input for c in dev} & {c.input for c in test}


@pytest.mark.parametrize("split_name", ["test", "dev"])
def test_cases_are_internally_consistent(split_name):
    cases = load_cases(split_name)
    assert len({c.id for c in cases}) == len(cases)
    for case in cases:
        surfaces = [t.surface for t in case.expected_tokens]
        assert surfaces == split(case.input), case.id
        # Only HINDI tokens change script; everything else is copied through.
        native = split(case.expected_native)
        assert len(native) == len(surfaces), case.id
        for token, out in zip(case.expected_tokens, native, strict=True):
            if token.kind is not TokenKind.HINDI:
                assert out == token.surface, (case.id, token.surface)
        if "long" in case.tags:
            assert sum(t.kind not in {TokenKind.PUNCT} for t in case.expected_tokens) >= 40
