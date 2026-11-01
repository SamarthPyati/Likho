"""Orchestration: run the routed stages for one request (FR1–FR7).

Each stage runs inside `_run`, which records its time and turns any failure
into a `StageError` on the response. Later stages use whatever earlier ones
produced, so a failed rewrite never loses the transliteration (FR7):

    route → tokenize → transliterate → rerank → [render] → rewrite → translate

- rerank failed: the native script is rendered from IndicXlit's top-1
- transliterate failed: no native script; rewrite and translate are skipped
- rewrite failed: translation uses the native script instead

Logs record lengths and timings, never the user's text (NFR4).
"""

import re
import time
from collections.abc import Callable
from typing import TypeVar

from likho.config import settings
from likho.errors import StageFailure
from likho.logging import get_logger
from likho.rerank import rerank_tokens
from likho.rewrite import rewrite
from likho.router import route
from likho.schemas import AssistRequest, AssistResponse, Route, Stage, StageError, Token
from likho.tokenize import render, tag
from likho.translate import translate
from likho.transliterate import transliterate_tokens

logger = get_logger(__name__)

T = TypeVar("T")

_DEVANAGARI = re.compile(r"[ऀ-ॿ]")


def disabled_stages() -> set[Stage]:
    """Stages switched off with DISABLE_STAGES (comma-separated stage names)."""
    names = {n.strip().lower() for n in settings.disable_stages.split(",") if n.strip()}
    return {Stage(n) for n in names}


def assist(request: AssistRequest) -> AssistResponse:
    timings: dict[str, float] = {}
    errors: list[StageError] = []
    disabled = disabled_stages()
    text = request.text
    start = time.perf_counter()

    def run(stage: Stage, fn: Callable[[], T]) -> T | None:
        return _run(stage, fn, timings, errors, disabled)

    routed = run(Stage.ROUTE, lambda: route(text))
    if routed is None:  # route() falls back to rules itself; this is the tagger failing
        routed = Route(stages=[Stage.ROUTE], reason="Routing failed", source="rule_fallback")
    if request.translate and Stage.TOKENIZE in routed.stages:  # not for empty input
        routed = routed.model_copy(update={"stages": [*routed.stages, Stage.TRANSLATE]})
    stages = routed.stages

    tokens: list[Token] = []
    if Stage.TOKENIZE in stages:
        tokens = run(Stage.TOKENIZE, lambda: tag(text, top_k=request.top_k)) or []

    if Stage.TRANSLITERATE in stages:
        native = None
        if tokens:
            transliterated = run(
                Stage.TRANSLITERATE,
                lambda: transliterate_tokens(
                    tokens, request.top_k, transliterate_loanwords=request.transliterate_loanwords
                ),
            )
            if transliterated is not None:
                tokens = transliterated
                reranked = run(Stage.RERANK, lambda: rerank_tokens(transliterated))
                tokens = reranked if reranked is not None else transliterated
                native = render(text, tokens)
    else:
        # Devanagari or English input: nothing to transliterate.
        native = text if _DEVANAGARI.search(text) else None

    # English input is rewritten as English; otherwise only Devanagari is.
    base = native if Stage.TRANSLITERATE in stages else text
    rewritten = None
    if Stage.REWRITE in stages:
        if base is None:
            errors.append(_skipped(Stage.REWRITE))
        else:
            rewritten = run(Stage.REWRITE, lambda: rewrite(base, request.register))

    english = None
    if Stage.TRANSLATE in stages:
        source = rewritten or base
        if source is None:
            errors.append(_skipped(Stage.TRANSLATE))
        elif not _DEVANAGARI.search(source):
            english = source  # already English: nothing to translate
        else:
            english = run(Stage.TRANSLATE, lambda: translate(source))

    timings["total"] = _ms(start)
    logger.info(
        "assist chars=%d route=%s errors=%s total_ms=%.0f",
        len(text),
        routed.source,
        [e.stage.value for e in errors],
        timings["total"],
    )
    return AssistResponse(
        route=routed,
        tokens=tokens,
        native_script=native,
        rewritten=rewritten,
        english=english,
        errors=errors,
        timings_ms=timings,
    )


def _run(
    stage: Stage,
    fn: Callable[[], T],
    timings: dict[str, float],
    errors: list[StageError],
    disabled: set[Stage],
) -> T | None:
    """`fn()`, or None with a StageError recorded if the stage is disabled or fails."""
    if stage in disabled:
        errors.append(StageError(stage=stage, message="disabled by DISABLE_STAGES"))
        return None
    start = time.perf_counter()
    try:
        return fn()
    except StageFailure as exc:
        errors.append(exc.error)
    except Exception as exc:  # noqa: BLE001 — one stage's bug must not lose the others' output
        logger.error("stage %s raised %s", stage.value, type(exc).__name__)
        errors.append(StageError(stage=stage, message=f"unexpected error: {type(exc).__name__}"))
    finally:
        timings[stage.value] = _ms(start)
    return None


def _skipped(stage: Stage) -> StageError:
    return StageError(stage=stage, message="skipped: no Devanagari text from earlier stages")


def _ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 1)
