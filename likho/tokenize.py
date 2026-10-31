"""FR2: split text into tokens and label each one. See docs/ARCHITECTURE.md §4.

Roman tokens go through a cascade; each step only sees what earlier steps
left undecided, and `kind_confidence` records which step decided:

1. orthography  — mid-sentence Capitalised → NAMED_ENTITY, ALL CAPS → ENGLISH,
                  English-only suffixes (-tion, -ness, …) → ENGLISH
2. dictionary   — word is plausible in only one language
3. frequency    — Hindi vs English Zipf frequency differ by a clear margin
4. context      — homographs, where both readings are common (to/तो, me/में,
                  is/इस, the/थे), follow the sentence's majority language
5. llm          — one batched Param-1 call per sentence for what is left
6. fallback     — HINDI: a homograph's Hindi reading is a very common word

"Hindi frequency" is the frequency of IndicXlit's top-1 Devanagari candidate
(or the best-known lower one if top-1 is unknown), so `meeting` is scored as
मीटिंग and `kal` as कल.
"""

import re
from collections import Counter
from dataclasses import dataclass

from likho.config import settings
from likho.lexicon import english_zipf, hindi_zipf
from likho.logging import get_logger
from likho.parsing import extract_json_object
from likho.registry import LLMLike, XlitEngineLike, get_llm, get_xlit_engine
from likho.schemas import Token, TokenKind
from likho.transliterate import xlit_candidates

logger = get_logger(__name__)

TOKEN_RE = re.compile(
    r"""
    (?P<numeric>\d+(?:[.,:/]\d+)*(?:st|nd|rd|th|am|pm|k)?)  # \d includes ०-९
    |(?P<devanagari>[\u0900-\u0963\u0971-\u097f\u200c\u200d]+)  # excludes । ॥ and digits
    |(?P<latin>[A-Za-z]+(?:['’][A-Za-z]+)*)
    |(?P<punct>\S)
    """,
    re.VERBOSE,
)
SENTENCE_END = {".", "!", "?", "।", "॥"}
ENGLISH_SUFFIXES = ("tion", "sion", "ness", "ment", "ship", "ity", "ing", "ous", "ful", "less")

CONFIDENCE = {  # distinct per signal, so the deciding signal is recoverable from a Token
    "fixed": 1.0,
    "dictionary": 0.95,
    "frequency": 0.85,
    "orthography": 0.8,
    "llm": 0.7,
    "context": 0.65,
    "loanword": 0.6,
    "oov": 0.55,
    "fallback": 0.5,
}
SIGNAL_BY_CONFIDENCE = {v: k for k, v in CONFIDENCE.items()}

ENGLISH_MIN_ZIPF = 3.0  # below this, an "English" word is too rare to count as English
FREQUENCY_MARGIN = 0.75  # |hindi − english| Zipf gap that settles a word in both lists
HOMOGRAPH_HINDI_ZIPF = 6.0  # both readings at least this common → homograph,
HOMOGRAPH_ENGLISH_ZIPF = 5.0  # decided by context, never by frequency alone


@dataclass
class _Decision:
    kind: TokenKind | None  # None = homograph, still pending
    signal: str


def split(text: str) -> list[str]:
    return [m.group() for m in TOKEN_RE.finditer(text)]


def render(text: str, tokens: list[Token]) -> str:
    """Rebuild `text` with each token replaced by its `chosen` form, keeping the
    original spacing. `tokens` must come from `tag(text)` (same split)."""
    matches = list(TOKEN_RE.finditer(text))
    if len(matches) != len(tokens):
        raise ValueError(f"{len(tokens)} tokens for {len(matches)} spans")
    parts, last = [], 0
    for match, token in zip(matches, tokens, strict=True):
        parts.append(text[last : match.start()])
        parts.append(token.chosen if token.chosen is not None else token.surface)
        last = match.end()
    parts.append(text[last:])
    return "".join(parts)


def tag(
    text: str,
    *,
    top_k: int = 5,
    engine: XlitEngineLike | None = None,
    llm: LLMLike | None = None,
    use_llm: bool | None = None,
) -> list[Token]:
    """Split `text` and label every token with a `TokenKind` and confidence."""
    engine = engine or get_xlit_engine()
    use_llm = settings.llm_tiebreak if use_llm is None else use_llm

    surfaces: list[str] = []
    decisions: list[_Decision] = []
    sentence_start = True
    for match in TOKEN_RE.finditer(text):
        surface, group = match.group(), match.lastgroup
        surfaces.append(surface)
        if group == "latin":
            decisions.append(_lexical(surface, sentence_start, engine, top_k))
        else:
            decisions.append(_Decision(TokenKind(group), "fixed"))
        sentence_start = surface in SENTENCE_END

    _resolve_by_context(decisions)
    pending = [i for i, d in enumerate(decisions) if d.kind is None]
    if pending and use_llm:
        _resolve_by_llm(text, surfaces, decisions, pending, llm)
    for d in decisions:
        if d.kind is None:
            d.kind, d.signal = TokenKind.HINDI, "fallback"

    logger.debug("tag signals %s", dict(Counter(d.signal for d in decisions)))
    return [
        Token(surface=s, kind=d.kind, kind_confidence=CONFIDENCE[d.signal])
        for s, d in zip(surfaces, decisions, strict=True)
    ]


def _lexical(surface: str, sentence_start: bool, engine: XlitEngineLike, top_k: int) -> _Decision:
    if len(surface) >= 2 and surface.isupper():
        return _Decision(TokenKind.ENGLISH, "orthography")  # acronym: PDF, HR
    if surface[0].isupper() and not sentence_start:
        return _Decision(TokenKind.NAMED_ENTITY, "orthography")

    word = surface.lower()
    hindi = _hindi_reading_zipf(xlit_candidates(engine, word, top_k))
    english = english_zipf(word)
    in_english, in_hindi = english >= ENGLISH_MIN_ZIPF, hindi > 0.0
    gap = hindi - english

    if in_hindi and not in_english:
        return _Decision(TokenKind.HINDI, "dictionary")
    if in_english and not in_hindi:
        return _Decision(TokenKind.ENGLISH, "dictionary")
    if not in_english and not in_hindi:
        if word.endswith(ENGLISH_SUFFIXES):
            return _Decision(TokenKind.ENGLISH, "orthography")
        return _Decision(TokenKind.HINDI, "oov")  # chat spellings: kr, nhi, h
    if hindi >= HOMOGRAPH_HINDI_ZIPF and (
        english >= HOMOGRAPH_ENGLISH_ZIPF or abs(gap) < FREQUENCY_MARGIN
    ):
        return _Decision(None, "pending")
    if gap >= FREQUENCY_MARGIN:
        return _Decision(TokenKind.HINDI, "frequency")
    if gap <= -FREQUENCY_MARGIN:
        return _Decision(TokenKind.ENGLISH, "frequency")
    # Both readings plausible and the Hindi one is a transliterated loanword
    # (bank → बैंक). U5: English words stay English by default.
    return _Decision(TokenKind.ENGLISH, "loanword")


def _hindi_reading_zipf(candidates: tuple[str, ...]) -> float:
    """Frequency of the top-1 candidate if it is a known Hindi word; otherwise of
    the best-known lower candidate (kr → केआर is unknown, कर at rank 3 is not)."""
    if not candidates:
        return 0.0
    top = hindi_zipf(candidates[0])
    return top if top > 0.0 else max(hindi_zipf(c) for c in candidates)


CONTEXT_SIGNALS = {"dictionary", "frequency"}  # decisions trusted enough to vote


def _resolve_by_context(decisions: list[_Decision]) -> None:
    confident = Counter(
        d.kind
        for d in decisions
        if d.signal in CONTEXT_SIGNALS and d.kind in {TokenKind.HINDI, TokenKind.ENGLISH}
    )
    hindi, english = confident[TokenKind.HINDI], confident[TokenKind.ENGLISH]
    if hindi == english:
        return  # no majority: left for the LLM
    majority = TokenKind.HINDI if hindi > english else TokenKind.ENGLISH
    for d in decisions:
        if d.kind is None:
            d.kind, d.signal = majority, "context"


TAG_PROMPT = """TASK: tag
The sentence below is Hinglish: Hindi written in Roman letters, mixed with English.
For each numbered word, answer "hindi" if it is a Hindi word written in Roman letters
here, or "english" if it is an English word here.

Sentence: {sentence}
Words: <text>{words}</text>

Reply with only a JSON object mapping each numbered word to "hindi" or "english",
for example {{"3:to": "hindi"}}."""

_warned: set[str] = set()


def _resolve_by_llm(
    text: str,
    surfaces: list[str],
    decisions: list[_Decision],
    pending: list[int],
    llm: LLMLike | None,
) -> None:
    """One Param-1 call for every pending token in the sentence. Any failure
    leaves them pending for the fallback; this step is never fatal."""
    keys = {f"{i}:{surfaces[i]}": i for i in pending}
    prompt = TAG_PROMPT.format(sentence=text, words=" ".join(keys))
    try:
        reply = (llm or get_llm()).generate(prompt, max_new_tokens=128, temperature=0.0)
    except Exception as exc:  # noqa: BLE001 — model, parse or loader failure: fall back
        name = type(exc).__name__
        if name not in _warned:
            _warned.add(name)
            logger.warning("tag tiebreak failed (%s); using fallback", name)
        return
    labels = extract_json_object(reply)
    if labels is None:
        return
    for key, label in labels.items():
        i = keys.get(str(key))
        kind = {"hindi": TokenKind.HINDI, "english": TokenKind.ENGLISH}.get(str(label).lower())
        if i is not None and kind is not None:
            decisions[i].kind, decisions[i].signal = kind, "llm"
