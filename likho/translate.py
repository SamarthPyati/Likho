"""FR6: Devanagari Hindi → English with IndicTrans2.

The model wrapper (`likho.registry.IndicTrans2Translator`) does the mandatory
IndicProcessor pre/post-processing; this stage adds the language codes and
turns any failure into a `StageFailure`.
"""

from likho.config import settings
from likho.errors import StageFailure
from likho.logging import get_logger
from likho.registry import TranslatorLike, get_translator
from likho.schemas import Stage

logger = get_logger(__name__)

FLORES_CODE = {"hi": "hin_Deva"}  # IndicTrans2 uses FLORES-style codes, not ISO 639-1
TARGET = "eng_Latn"


def translate(text: str, *, translator: TranslatorLike | None = None) -> str:
    """English translation of `text`. Raises StageFailure."""
    return translate_batch([text], translator=translator)[0]


def translate_batch(texts: list[str], *, translator: TranslatorLike | None = None) -> list[str]:
    """Translate several texts in one model call (seq2seq batching is a large win)."""
    todo = [i for i, t in enumerate(texts) if t.strip()]
    out = ["" for _ in texts]
    if not todo:
        return out
    try:
        results = (translator or get_translator()).translate_batch(
            [texts[i] for i in todo], src_lang=FLORES_CODE[settings.language], tgt_lang=TARGET
        )
    except Exception as exc:  # noqa: BLE001 — any model failure is a stage failure
        raise StageFailure(Stage.TRANSLATE, f"model error: {type(exc).__name__}") from exc
    if len(results) != len(todo):
        raise StageFailure(Stage.TRANSLATE, f"{len(results)} translations for {len(todo)} inputs")
    for i, result in zip(todo, results, strict=True):
        if not result.strip():
            raise StageFailure(Stage.TRANSLATE, "empty translation")
        out[i] = result.strip()
    return out
