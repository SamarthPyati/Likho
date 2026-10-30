import pytest

from eval.metrics import cer, corpus_cer, levenshtein, normalize, top_k_accuracy


def test_levenshtein_worked_examples():
    assert levenshtein("kitten", "sitting") == 3  # k→s, e→i, +g
    assert levenshtein("", "abc") == 3
    assert levenshtein("abc", "abc") == 0


def test_cer_devanagari_worked_example():
    # कल = क + ल (2 code points). काल = क + ा + ल: one insertion → 1/2.
    assert cer("काल", "कल") == pytest.approx(0.5)
    assert cer("कल", "कल") == 0.0


def test_cer_empty_reference():
    assert cer("", "") == 0.0
    assert cer("क", "") == 1.0


def test_corpus_cer_pools_characters():
    # edits: 1 (काल vs कल) + 0; reference chars: कल (2) + है (ह + ै = 2) = 4
    assert corpus_cer(["काल", "है"], ["कल", "है"]) == pytest.approx(1 / 4)


def test_nukta_forms_compare_equal():
    precomposed = "ज़"  # ज़ as one code point
    decomposed = "ज़"  # ज + nukta
    assert normalize(precomposed) == normalize(decomposed)
    assert cer(precomposed, decomposed) == 0.0


def test_top_k_accuracy_worked_example():
    candidates = [["काल", "कल"], ["है"], ["मे", "में"]]
    references = ["कल", "है", "मैं"]
    assert top_k_accuracy(candidates, references, k=1) == pytest.approx(1 / 3)
    assert top_k_accuracy(candidates, references, k=5) == pytest.approx(2 / 3)


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        top_k_accuracy([["a"]], ["a", "b"], k=1)
