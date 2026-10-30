"""Model singletons. Each model loads once per process (NFR3), lazily.

With MOCK_MODELS=1 every getter returns the deterministic stub from
`likho.mocks`, and no model library is imported.
"""

from functools import lru_cache
from typing import Protocol

from likho.config import settings
from likho.logging import get_logger
from likho.mocks import MockLLM, MockXlitEngine

logger = get_logger(__name__)

XLIT_BEAM_WIDTH = 10


class XlitEngineLike(Protocol):
    def translit_word(self, word: str, topk: int = ...) -> dict[str, list[str]]: ...

    def translit_sentence(self, text: str) -> dict[str, str]: ...


class LLMLike(Protocol):
    def generate(
        self, prompt: str, *, max_new_tokens: int = ..., temperature: float = ...
    ) -> str: ...


@lru_cache(maxsize=2)  # one per rescore setting, for the rescore ablation
def get_xlit_engine(rescore: bool = True) -> XlitEngineLike:
    if settings.mock_models:
        return MockXlitEngine(settings.language)
    import argparse

    import torch
    from ai4bharat.transliteration import XlitEngine

    # torch>=2.6 loads checkpoints with weights_only=True; fairseq's checkpoint
    # pickles its training args as an argparse.Namespace. Allow just that class.
    torch.serialization.add_safe_globals([argparse.Namespace])
    logger.info("loading IndicXlit lang=%s rescore=%s", settings.language, rescore)
    return XlitEngine(settings.language, beam_width=XLIT_BEAM_WIDTH, rescore=rescore)


@lru_cache(maxsize=1)
def get_llm() -> LLMLike:
    """Param-1 (router, tagger tiebreak, rewrite). Callers must handle failure."""
    if settings.mock_models:
        return MockLLM()
    raise NotImplementedError("Param-1 loader lands in M4; callers fall back to rules")
