"""LLM-as-judge for the rewrite stage (docs/EVALUATION.md §2.4).

Strict-JSON prompt, regex extraction, deterministic fallback score on a parse
failure; `parsed` records which happened so the fallback rate is reportable.

Caveat for the report: the judge is the same Param-1 checkpoint that wrote the
rewrite, so self-preference bias is possible. The human rubric on 20 cases
(M6) is the check on that.
"""

from pydantic import BaseModel

from likho.parsing import extract_json_object
from likho.registry import LLMLike, get_llm
from likho.schemas import Register

AXES = ("meaning", "register", "fluency")
FALLBACK_SCORE = 3  # neutral midpoint, used when the judge's reply cannot be parsed

JUDGE_PROMPT = """TASK: judge
You are grading a rewrite of a Hindi/Hinglish message. The target tone was: {register}.

Original: <original>{original}</original>
Rewrite: <rewrite>{rewritten}</rewrite>

Score each from 1 (very poor) to 5 (excellent):
- meaning: the rewrite says the same thing, adds no facts, drops nothing
- register: the rewrite matches the target tone
- fluency: the rewrite is natural, grammatical Hindi/Hinglish

Reply with only a JSON object: {{"meaning": <1-5>, "register": <1-5>, "fluency": <1-5>}}"""


class JudgeScore(BaseModel):
    meaning: int
    register: int
    fluency: int
    parsed: bool  # False → the fallback score was used


def judge(
    original: str, rewritten: str, register: Register, *, llm: LLMLike | None = None
) -> JudgeScore:
    prompt = JUDGE_PROMPT.format(register=register.value, original=original, rewritten=rewritten)
    try:
        reply = (llm or get_llm()).generate(prompt, max_new_tokens=64, temperature=0.0)
    except Exception:  # noqa: BLE001 — a failed judge call is scored like a parse failure
        reply = ""
    scores = _parse_scores(extract_json_object(reply))
    if scores is None:
        return JudgeScore(**dict.fromkeys(AXES, FALLBACK_SCORE), parsed=False)
    return JudgeScore(**scores, parsed=True)


def _parse_scores(obj: dict | None) -> dict[str, int] | None:
    if obj is None:
        return None
    scores = {}
    for axis in AXES:
        try:
            value = int(float(obj[axis]))
        except (KeyError, TypeError, ValueError):
            return None
        if not 1 <= value <= 5:
            return None
        scores[axis] = value
    return scores
