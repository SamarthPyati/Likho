"""FR5: register-controlled, meaning-preserving rewrite with Param-1.

Param-1 is constrained hard in the prompt and then checked, because it is an
early SFT checkpoint (docs/MODELS.md §3.4). A rewrite is rejected — raising
`StageFailure`, so the pipeline keeps the input — when it is empty, outside
0.5×–1.6× the input length, or has switched script (Devanagari in, mostly
Roman out). Rejection is about format, not quality; quality is the judge's
job in eval/judge.py.
"""

import re

from likho.errors import StageFailure
from likho.logging import get_logger
from likho.registry import LLMLike, get_llm
from likho.schemas import Register, Stage

logger = get_logger(__name__)

MIN_LENGTH_RATIO = 0.5
MAX_LENGTH_RATIO = 1.6
TEMPERATURE = 0.3
MAX_NEW_TOKENS = 512  # Param-1's context is 2048 tokens; the prompt must fit too

REGISTER_GUIDE = {
    Register.CASUAL: "casual and friendly, as you would text a friend",
    Register.NEUTRAL: "clear and neutral, as in an everyday message",
    Register.FORMAL: "formal and polite, as you would write to a teacher or a senior",
}

REWRITE_PROMPT = """TASK: rewrite
Rewrite the text below so that its tone is {guide}.

Rules:
- Keep exactly the same meaning. Do not add any new information.
- Do not drop any part of the text.
- Keep the same script and language mix. Keep English words and names as they are.
- Reply with only the rewritten text: no explanation, no quotes, no heading.

<text>{text}</text>"""

# Openers Param-1 tends to add despite the instruction.
_PREAMBLE = re.compile(
    r"^\s*(?:here(?:'s| is)[^:\n]*:|rewritten(?: text)?\s*:|output\s*:|पुनर्लिखित[^:\n]*:)\s*",
    re.IGNORECASE,
)
_DEVANAGARI = re.compile(r"[ऀ-ॿ]")
_LATIN = re.compile(r"[A-Za-z]")


def rewrite(text: str, register: Register, *, llm: LLMLike | None = None) -> str:
    """Return `text` rewritten in `register`. Raises StageFailure on a rejected rewrite."""
    if not text.strip():
        return text
    prompt = REWRITE_PROMPT.format(guide=REGISTER_GUIDE[register], text=text)
    try:
        reply = (llm or get_llm()).generate(
            prompt, max_new_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE
        )
    except Exception as exc:  # noqa: BLE001 — any model failure is a stage failure
        raise StageFailure(Stage.REWRITE, f"model error: {type(exc).__name__}") from exc

    candidate = clean_reply(reply)
    problem = validate(text, candidate)
    if problem:
        logger.info("rewrite rejected: %s (in=%d chars)", problem, len(text))
        raise StageFailure(Stage.REWRITE, f"rewrite rejected: {problem}")
    return candidate


def clean_reply(reply: str) -> str:
    """Strip code fences, a leading preamble, echoed tags and wrapping quotes."""
    out = reply.strip().strip("`").strip()
    out = _PREAMBLE.sub("", out, count=1)
    out = out.replace("<text>", "").replace("</text>", "").strip()
    if len(out) >= 2 and out[0] == out[-1] and out[0] in "\"'“”":
        out = out[1:-1].strip()
    if out.startswith("“") and out.endswith("”"):
        out = out[1:-1].strip()
    return out


def validate(original: str, rewritten: str) -> str | None:
    """Why `rewritten` is unacceptable as a rewrite of `original`, or None."""
    if not rewritten:
        return "empty output"
    ratio = len(rewritten) / len(original)
    if not MIN_LENGTH_RATIO <= ratio <= MAX_LENGTH_RATIO:
        return f"length ratio {ratio:.2f} outside {MIN_LENGTH_RATIO}–{MAX_LENGTH_RATIO}"
    if _mostly_devanagari(original) and not _mostly_devanagari(rewritten):
        return "script changed from Devanagari"
    return None


def _mostly_devanagari(text: str) -> bool:
    devanagari, latin = len(_DEVANAGARI.findall(text)), len(_LATIN.findall(text))
    return devanagari > latin
