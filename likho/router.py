"""FR1: decide which stages run for a request.

Param-1 classifies the input into one intent; a fixed table maps intents to
stages. The LLM's answer is used only if it parses, names a known intent,
and does not contradict hard facts about the text's script — otherwise the
rule-based router decides and `Route.source` says so.

Translation is not routed: it runs when the request asks for it (FR6), and
the pipeline appends it.
"""

from enum import Enum

from likho.logging import get_logger
from likho.parsing import extract_json_object
from likho.registry import LLMLike, get_llm
from likho.schemas import Route, Stage, TokenKind
from likho.tokenize import TOKEN_RE, tag

logger = get_logger(__name__)


class Intent(str, Enum):
    ROMAN_HINGLISH = "roman_hinglish"  # Hindi typed in Roman script, maybe with English
    MIXED_SCRIPT = "mixed_script"  # Devanagari and Roman words together
    DEVANAGARI = "devanagari"  # already in native script
    ENGLISH = "english"  # entirely English
    EMPTY = "empty"  # nothing to process


STAGES: dict[Intent, list[Stage]] = {
    Intent.ROMAN_HINGLISH: [
        Stage.ROUTE,
        Stage.TOKENIZE,
        Stage.TRANSLITERATE,
        Stage.RERANK,
        Stage.REWRITE,
    ],
    Intent.MIXED_SCRIPT: [
        Stage.ROUTE,
        Stage.TOKENIZE,
        Stage.TRANSLITERATE,
        Stage.RERANK,
        Stage.REWRITE,
    ],
    Intent.DEVANAGARI: [Stage.ROUTE, Stage.TOKENIZE, Stage.REWRITE],
    Intent.ENGLISH: [Stage.ROUTE, Stage.TOKENIZE, Stage.REWRITE],
    Intent.EMPTY: [Stage.ROUTE],
}

REASONS = {
    Intent.ROMAN_HINGLISH: "Roman-script Hindi: transliterate, rerank, then rewrite",
    Intent.MIXED_SCRIPT: "Devanagari and Roman mixed: transliterate the Roman Hindi words",
    Intent.DEVANAGARI: "Already in Devanagari: transliteration skipped",
    Intent.ENGLISH: "Entirely English: transliteration skipped, straight to rewrite",
    Intent.EMPTY: "No words to process",
}

ROUTE_PROMPT = """TASK: route
Classify the user's text into exactly one intent:
- roman_hinglish: Hindi in Roman letters, possibly with English words ("kal office jana hai")
- mixed_script: some words in Devanagari and some Hindi words in Roman letters ("मुझे kal जाना है")
- devanagari: Hindi already written in Devanagari ("मुझे कल जाना है")
- english: entirely English ("please call me tomorrow")
- empty: no words, only spaces, digits or punctuation

Text: <text>{text}</text>

Reply with only a JSON object: {{"intent": "<one of the five>", "reason": "<a few words>"}}"""


def route(text: str, *, llm: LLMLike | None = None, use_llm: bool = True) -> Route:
    """Pick the stages for `text`: LLM first, rule-based fallback on any doubt."""
    facts = _script_facts(text)
    if use_llm:
        intent, why_not = _llm_intent(text, llm)
        if intent is not None:
            contradiction = _contradicts(intent, facts)
            if contradiction is None:
                return Route(stages=STAGES[intent], reason=REASONS[intent], source="llm")
            why_not = contradiction
        logger.info("router fallback: %s", why_not)
    intent = rule_intent(text, facts)
    return Route(stages=STAGES[intent], reason=REASONS[intent], source="rule_fallback")


def rule_intent(text: str, facts: dict[str, int] | None = None) -> Intent:
    """Deterministic router: script counts, then the tagger for Roman-only text."""
    facts = facts or _script_facts(text)
    if facts["latin"] == 0 and facts["devanagari"] == 0:
        return Intent.EMPTY
    if facts["latin"] == 0:
        return Intent.DEVANAGARI
    if facts["devanagari"] > 0:
        return Intent.MIXED_SCRIPT
    tokens = tag(text, use_llm=False)
    if any(t.kind is TokenKind.HINDI for t in tokens):
        return Intent.ROMAN_HINGLISH
    return Intent.ENGLISH


def _script_facts(text: str) -> dict[str, int]:
    counts = {"latin": 0, "devanagari": 0}
    for match in TOKEN_RE.finditer(text):
        if match.lastgroup in counts:
            counts[match.lastgroup] += 1
    return counts


def _contradicts(intent: Intent, facts: dict[str, int]) -> str | None:
    """Why an LLM intent is impossible for this text, or None if it is possible."""
    latin, devanagari = facts["latin"] > 0, facts["devanagari"] > 0
    allowed = {
        Intent.EMPTY: not latin and not devanagari,
        Intent.DEVANAGARI: devanagari and not latin,
        Intent.MIXED_SCRIPT: devanagari and latin,
        Intent.ROMAN_HINGLISH: latin and not devanagari,
        Intent.ENGLISH: latin and not devanagari,
    }
    return None if allowed[intent] else f"llm intent {intent.value} contradicts script"


def _llm_intent(text: str, llm: LLMLike | None) -> tuple[Intent | None, str]:
    try:
        reply = (llm or get_llm()).generate(
            ROUTE_PROMPT.format(text=text), max_new_tokens=64, temperature=0.0
        )
    except Exception as exc:  # noqa: BLE001 — loader or generation failure: fall back
        return None, f"llm unavailable ({type(exc).__name__})"
    parsed = extract_json_object(reply)
    if parsed is None:
        return None, "llm reply not parseable as JSON"
    try:
        return Intent(str(parsed.get("intent", "")).strip().lower()), ""
    except ValueError:
        return None, "llm intent not one of the known intents"
