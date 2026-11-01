"""F1–F12 failure taxonomy (docs/EVALUATION.md §4).

Every failing test case gets exactly one *primary* code: the earliest stage
that went wrong, in pipeline order (crash → route → token kind →
transliteration → rewrite → translation). Later failures in the same case are
listed too, so nothing is hidden, but counts per bucket use the primary code.

Transliteration errors on correctly tagged Hindi tokens are split by what the
character diff shows. F8 is checked first: if the gold word was among the
candidates, the model produced it and the reranker lost it — a different
problem from the model never producing it (EVALUATION §4). A diff that fits
none of F1–F4 is `other` and says so, rather than being forced into a bucket.
"""

import difflib

from eval.metrics import normalize
from likho.schemas import TokenKind

CODES = {
    "F1": "Wrong vowel length",
    "F2": "Schwa deletion error",
    "F3": "Retroflex/dental confusion",
    "F4": "Nukta mishandling",
    "F5": "Loanword over-transliteration",
    "F6": "Loanword under-transliteration",
    "F7": "Named entity mangled",
    "F8": "Context ambiguity (gold in top-k, not chosen)",
    "F9": "Router mis-stage",
    "F10": "Rewrite drift or rejection",
    "F11": "Translation error",
    "F12": "Crash / timeout",
    "other": "Transliteration error outside F1–F4 (e.g. nasalisation, e/ai)",
}

NUKTA = "़"
VIRAMA = "्"
# Unordered pairs whose swap is a vowel-length error. "" is the inherent
# vowel: अ/आ length shows up as ा present or absent.
VOWEL_LENGTH = [{"ि", "ी"}, {"ु", "ू"}, {"इ", "ई"}, {"उ", "ऊ"}, {"", "ा"}, {"अ", "आ"}]
RETROFLEX_DENTAL = [{"ट", "त"}, {"ठ", "थ"}, {"ड", "द"}, {"ढ", "ध"}, {"ण", "न"}]


def diff_segments(gold: str, chosen: str) -> list[tuple[str, str]]:
    """(gold part, chosen part) for every differing span, NFC characters."""
    a, b = normalize(gold), normalize(chosen)
    matcher = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    return [(a[i1:i2], b[j1:j2]) for op, i1, i2, j1, j2 in matcher.get_opcodes() if op != "equal"]


def classify_transliteration(gold: str, chosen: str, candidates: list[str]) -> str:
    """Bucket for a correctly tagged Hindi token whose chosen form is wrong."""
    if normalize(gold) in {normalize(c) for c in candidates}:
        return "F8"
    segments = diff_segments(gold, chosen)
    if any(NUKTA in g + c for g, c in segments):
        return "F4"
    if any({g, c} in VOWEL_LENGTH for g, c in segments):
        return "F1"
    if any({g, c} in RETROFLEX_DENTAL for g, c in segments):
        return "F3"
    if any(VIRAMA in g + c for g, c in segments):
        return "F2"
    return "other"


def classify_token(
    gold_kind: TokenKind,
    predicted_kind: TokenKind,
    gold_form: str,
    chosen: str,
    candidates: list[str],
) -> str | None:
    """Bucket for one token of a case, or None if its output form is right.

    Tagging mistakes that do not change the output (NE tagged ENGLISH: both
    kept as typed) are not failures here; token-classification metrics count
    them separately.
    """
    if normalize(chosen) == normalize(gold_form):
        return None
    if gold_kind is TokenKind.ENGLISH and predicted_kind is TokenKind.HINDI:
        return "F5"
    if gold_kind is TokenKind.HINDI and predicted_kind is not TokenKind.HINDI:
        return "F6"
    if gold_kind is TokenKind.NAMED_ENTITY and predicted_kind is TokenKind.HINDI:
        return "F7"
    if gold_kind is TokenKind.HINDI:
        return classify_transliteration(gold_form, chosen, candidates)
    return "other"


def primary(codes: list[str]) -> str | None:
    """The case's bucket: codes are collected in pipeline order, so the first."""
    return codes[0] if codes else None
