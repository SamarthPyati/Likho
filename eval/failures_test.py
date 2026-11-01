import pytest

from eval.failures import classify_token, classify_transliteration, diff_segments, primary
from likho.schemas import TokenKind

H, E, NE = TokenKind.HINDI, TokenKind.ENGLISH, TokenKind.NAMED_ENTITY


def test_diff_segments():
    assert diff_segments("दिन", "दीन") == [("ि", "ी")]
    assert diff_segments("कल", "कल") == []


@pytest.mark.parametrize(
    "gold, chosen, code",
    [
        ("दिन", "दीन", "F1"),  # i / ee
        ("काम", "कम", "F1"),  # aa / a
        ("कमरा", "कम्रा", "F2"),  # schwa
        ("टमाटर", "तमातर", "F3"),  # retroflex / dental
        ("ज़रूर", "जरूर", "F4"),  # nukta
        ("हैं", "हें", "other"),  # e / ai is vowel quality, not length
    ],
)
def test_transliteration_buckets(gold, chosen, code):
    assert classify_transliteration(gold, chosen, candidates=[chosen]) == code


def test_gold_in_candidates_is_f8_even_with_vowel_diff():
    assert classify_transliteration("कि", "की", ["की", "कि"]) == "F8"


@pytest.mark.parametrize(
    "gold_kind, predicted, gold_form, chosen, code",
    [
        (E, H, "meeting", "मीटिंग", "F5"),
        (H, E, "चाय", "chai", "F6"),
        (NE, H, "Rohit", "रोहित", "F7"),
        (NE, E, "Rohit", "Rohit", None),  # mistagged but output unchanged
        (H, H, "कल", "कल", None),
    ],
)
def test_token_buckets(gold_kind, predicted, gold_form, chosen, code):
    assert classify_token(gold_kind, predicted, gold_form, chosen, [chosen]) == code


def test_primary_is_earliest():
    assert primary(["F9", "F10"]) == "F9"
    assert primary([]) is None
