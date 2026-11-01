"""FR1: decide which stages run for a request.

Param-1 proposes one intent; a fixed table maps intents to stages. The
proposal is used only when the rule-based router agrees with it — otherwise
the rules decide and `Route.source` says so (see `route_with_proposal`).

Translation is not routed: it runs when the request asks for it (FR6), and
the pipeline appends it.
"""

import re
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


ROUTE_EXAMPLES = [  # few-shot turns; none of these sentences is in an eval set
    (ROUTE_PROMPT.format(text="aaj bahut thand hai"), '{"intent": "roman_hinglish"}'),
    (ROUTE_PROMPT.format(text="see you at the station"), '{"intent": "english"}'),
    (ROUTE_PROMPT.format(text="आज बहुत ठंड है"), '{"intent": "devanagari"}'),
]
ROUTE_PREFIX = '{"intent": "'  # forced start of the reply: the model only names the intent
_INTENT_AT_START = re.compile(
    "|".join(sorted((i.value for i in Intent), key=len, reverse=True)), re.IGNORECASE
)


def route(text: str, *, llm: LLMLike | None = None, use_llm: bool = True) -> Route:
    """Pick the stages for `text`. See `route_with_proposal`."""
    return route_with_proposal(text, llm=llm, use_llm=use_llm)[0]


def route_with_proposal(
    text: str, *, llm: LLMLike | None = None, use_llm: bool = True
) -> tuple[Route, Intent | None]:
    """The LLM proposes an intent; the rule router verifies it.

    Script counts plus the tagger determine exactly one intent for any text,
    so verification means the proposal is used only when it agrees with
    `rule_intent`. On Param-1 the proposal is unreliable (docs/ARCHITECTURE.md
    §5a), so an unverified answer would skip transliteration for Hinglish it
    mislabels as English. Returns the route and the raw proposal (None if
    there was none or it did not parse), so eval can report LLM accuracy.
    """
    rule = rule_intent(text)
    proposal = None
    if use_llm:
        proposal, problem = _llm_intent(text, llm)
        if proposal is rule:
            return Route(stages=STAGES[rule], reason=REASONS[rule], source="llm"), proposal
        logger.info("router fallback: %s", problem or "llm proposal disagrees with rules")
    route_ = Route(stages=STAGES[rule], reason=REASONS[rule], source="rule_fallback")
    return route_, proposal


def rule_intent(text: str) -> Intent:
    """Deterministic router: script counts, then the tagger for Roman-only text."""
    latin = devanagari = 0
    for match in TOKEN_RE.finditer(text):
        latin += match.lastgroup == "latin"
        devanagari += match.lastgroup == "devanagari"
    if latin == 0:
        return Intent.DEVANAGARI if devanagari else Intent.EMPTY
    if devanagari:
        return Intent.MIXED_SCRIPT
    tokens = tag(text, use_llm=False)
    if any(t.kind is TokenKind.HINDI for t in tokens):
        return Intent.ROMAN_HINGLISH
    return Intent.ENGLISH


def _llm_intent(text: str, llm: LLMLike | None) -> tuple[Intent | None, str]:
    try:
        reply = (llm or get_llm()).generate(
            ROUTE_PROMPT.format(text=text),
            max_new_tokens=12,
            temperature=0.0,
            examples=ROUTE_EXAMPLES,
            prefix=ROUTE_PREFIX,
        )
    except Exception as exc:  # noqa: BLE001 — loader or generation failure: fall back
        return None, f"llm unavailable ({type(exc).__name__})"
    match = _INTENT_AT_START.match(reply.strip())
    if match:
        return Intent(match.group().lower()), ""
    # A model that ignored the prefix may still have written a full JSON object.
    parsed = extract_json_object(reply)
    if parsed is None:
        return None, "llm reply not parseable"
    try:
        return Intent(str(parsed.get("intent", "")).strip().lower()), ""
    except ValueError:
        return None, "llm intent not one of the known intents"
