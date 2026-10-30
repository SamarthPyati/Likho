"""Evaluation metrics. See docs/EVALUATION.md §2.

All string comparisons are on NFC-normalised text, so precomposed and
decomposed forms of the same Devanagari (e.g. nukta letters) compare equal.
"""

import unicodedata
from collections.abc import Sequence


def normalize(text: str) -> str:
    return unicodedata.normalize("NFC", text.strip())


def levenshtein(a: str, b: str) -> int:
    """Edit distance over Unicode code points (insert, delete, substitute = 1)."""
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def cer(hypothesis: str, reference: str) -> float:
    """Character error rate for one pair: edit distance / reference length."""
    hyp, ref = normalize(hypothesis), normalize(reference)
    if not ref:
        return 0.0 if not hyp else 1.0
    return levenshtein(hyp, ref) / len(ref)


def corpus_cer(hypotheses: Sequence[str], references: Sequence[str]) -> float:
    """Total edits over total reference characters (not a mean of per-word CERs)."""
    _check_lengths(hypotheses, references)
    pairs = zip(hypotheses, references, strict=True)
    edits = sum(levenshtein(normalize(h), normalize(r)) for h, r in pairs)
    chars = sum(len(normalize(r)) for r in references)
    return edits / chars if chars else 0.0


def top_k_accuracy(candidates: Sequence[Sequence[str]], references: Sequence[str], k: int) -> float:
    """Fraction of items whose reference appears in the first k candidates."""
    _check_lengths(candidates, references)
    if not references:
        return 0.0
    hits = sum(
        normalize(ref) in {normalize(c) for c in cands[:k]}
        for cands, ref in zip(candidates, references, strict=True)
    )
    return hits / len(references)


def _check_lengths(a: Sequence, b: Sequence) -> None:
    if len(a) != len(b):
        raise ValueError(f"length mismatch: {len(a)} predictions vs {len(b)} references")
