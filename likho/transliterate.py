"""FR3: word-level Roman → Devanagari transliteration with top-k candidates.

IndicXlit has no sentence context; this stage only produces candidates and a
provisional choice (candidate 0). Context-aware choice is `likho.rerank` (M3).
"""

from functools import lru_cache

from likho.config import settings
from likho.logging import get_logger
from likho.registry import XlitEngineLike, get_xlit_engine
from likho.schemas import Token, TokenKind

logger = get_logger(__name__)


def should_transliterate(token: Token, transliterate_loanwords: bool = False) -> bool:
    """FR2: only HINDI tokens, plus ENGLISH ones when the user opted in (U8)."""
    if not token.surface.strip():
        return False
    if token.kind is TokenKind.HINDI:
        return True
    return token.kind is TokenKind.ENGLISH and transliterate_loanwords


def transliterate_tokens(
    tokens: list[Token],
    top_k: int,
    *,
    transliterate_loanwords: bool = False,
    engine: XlitEngineLike | None = None,
) -> list[Token]:
    """Return copies of `tokens` with `candidates`, `chosen` and `chosen_rank` filled.

    Tokens that are not transliterated keep their surface form as `chosen`
    and get no candidates. The model is called once per distinct word.
    """
    engine = engine or get_xlit_engine()
    cache: dict[str, list[str]] = {}
    out: list[Token] = []
    for token in tokens:
        if not should_transliterate(token, transliterate_loanwords):
            out.append(token.model_copy(update={"candidates": [], "chosen": token.surface}))
            continue
        key = token.surface.lower()
        if key not in cache:
            cache[key] = list(xlit_candidates(engine, key, top_k))
        candidates = cache[key]
        if candidates:
            update = {"candidates": candidates, "chosen": candidates[0], "chosen_rank": 0}
        else:
            update = {"candidates": [], "chosen": token.surface, "chosen_rank": None}
        out.append(token.model_copy(update=update))
    logger.debug("transliterated %d distinct words for %d tokens", len(cache), len(tokens))
    return out


@lru_cache(maxsize=20_000)
def xlit_candidates(engine: XlitEngineLike, word: str, top_k: int) -> tuple[str, ...]:
    """Ranked, de-duplicated candidates for a lowercase word. Cached per process,
    because the tagger and this stage both need them for the same words."""
    result = engine.translit_word(word, topk=top_k)
    # The pip package returns {lang: [...]}; tolerate a bare list across versions.
    ranked = result.get(settings.language, []) if isinstance(result, dict) else result
    return tuple(dict.fromkeys(c for c in ranked if c))[:top_k]
