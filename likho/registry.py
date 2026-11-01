"""Model singletons. Each model loads once per process (NFR3), lazily.

With MOCK_MODELS=1 every getter returns the deterministic stub from
`likho.mocks`, and no model library is imported.
"""

from collections.abc import Sequence
from functools import lru_cache
from typing import Protocol

from likho.config import settings
from likho.logging import get_logger
from likho.mocks import MockLLM, MockTranslator, MockXlitEngine

logger = get_logger(__name__)

XLIT_BEAM_WIDTH = 10


class XlitEngineLike(Protocol):
    def translit_word(self, word: str, topk: int = ...) -> dict[str, list[str]]: ...

    def translit_sentence(self, text: str) -> dict[str, str]: ...


class LLMLike(Protocol):
    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = ...,
        temperature: float = ...,
        examples: Sequence[tuple[str, str]] = ...,
        prefix: str = ...,
    ) -> str:
        """Reply to `prompt`. `examples` are earlier (user, assistant) turns for
        few-shot prompting; `prefix` is forced as the start of the reply, and
        only the text generated after it is returned."""
        ...


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


class TranslatorLike(Protocol):
    def translate_batch(
        self, texts: list[str], src_lang: str = ..., tgt_lang: str = ...
    ) -> list[str]: ...


def _device_and_dtype(torch):  # noqa: ANN001, ANN202 — torch is imported lazily
    """CUDA > Apple MPS > CPU. bf16 where CUDA supports it, else fp16 on a GPU
    (the Colab T4 has no native bf16), fp32 on CPU."""
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if torch.backends.mps.is_available():
        return "mps", torch.float16
    return "cpu", torch.float32


def _hf_token() -> str | None:
    return settings.hf_token.get_secret_value() if settings.hf_token else None


class Param1LLM:
    """Param-1 behind the `LLMLike` interface. See docs/MODELS.md §3."""

    def __init__(self, model_id: str) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device, dtype = _device_and_dtype(torch)
        logger.info("loading %s on %s (%s)", model_id, device, dtype)
        token = _hf_token()
        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=True, token=token
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, trust_remote_code=True, dtype=dtype, token=token
        ).to(device)
        self.model.eval()
        self.device = device
        eos = self.model.generation_config.eos_token_id
        self.eos_ids = set(eos if isinstance(eos, list) else [eos])

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        examples: Sequence[tuple[str, str]] = (),
        prefix: str = "",
    ) -> str:
        """Greedy (temperature 0) or sampled decoding with a hand-rolled loop.

        Param-1's remote modeling code predates transformers' Cache objects: it
        indexes `past_key_values[0][0]` as a tuple, and `model.generate()` in
        4.57 passes a DynamicCache instead, which crashes. Calling forward()
        directly with the tuple cache it returns sidesteps that.
        """
        torch = self._torch
        messages = []
        for user, assistant in examples:
            messages += [
                {"role": "user", "content": user},
                {"role": "assistant", "content": assistant},
            ]
        messages.append({"role": "user", "content": prompt})
        if getattr(self.tokenizer, "chat_template", None):
            text = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
        else:
            text = "\n\n".join(m["content"] for m in messages) + "\n\n"
        input_ids = self.tokenizer(
            text + prefix, return_tensors="pt", add_special_tokens=False
        ).input_ids
        next_input, past, generated = input_ids.to(self.device), None, []
        # Reseeded per call: the same prompt always samples the same reply, so
        # eval runs at temperature > 0 are reproducible on the same hardware.
        rng = torch.Generator().manual_seed(settings.llm_seed)
        with torch.no_grad():
            for _ in range(max_new_tokens):
                out = self.model(input_ids=next_input, past_key_values=past, use_cache=True)
                logits, past = out.logits[0, -1].float(), out.past_key_values
                if temperature > 0:
                    probs = torch.softmax(logits / temperature, dim=-1).cpu()
                    token = int(torch.multinomial(probs, num_samples=1, generator=rng))
                else:
                    token = int(torch.argmax(logits))
                if token in self.eos_ids:
                    break
                generated.append(token)
                next_input = torch.tensor([[token]], device=self.device)
        return self.tokenizer.decode(generated, skip_special_tokens=True)


@lru_cache(maxsize=1)
def get_llm() -> LLMLike:
    """Param-1 (router, tagger tiebreak, rewrite, judge). Callers must handle failure."""
    if settings.mock_models:
        return MockLLM()
    return Param1LLM(settings.llm_model_id)


class IndicTrans2Translator:
    """IndicTrans2 behind the `TranslatorLike` interface. See docs/MODELS.md §4.

    `IndicProcessor.preprocess_batch` before tokenizing and `postprocess_batch`
    after decoding are both mandatory: skipping either gives fluent-looking
    garbage, not an error.
    """

    MAX_LENGTH = 256
    NUM_BEAMS = 5

    def __init__(self, model_id: str) -> None:
        import torch
        from IndicTransToolkit import IndicProcessor
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        device, dtype = _device_and_dtype(torch)
        logger.info("loading %s on %s (%s)", model_id, device, dtype)
        token = _hf_token()
        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=True, token=token
        )
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_id, trust_remote_code=True, dtype=dtype, token=token
        ).to(device)
        self.model.eval()
        # The remote modeling code predates transformers' Cache objects: it reads
        # `past_key_values[0][0]` as a tuple, and generate() in 4.57 would hand it
        # an empty EncoderDecoderCache (crash). Declining the default cache makes
        # generate() start from None and keep the model's own tuple cache, which
        # beam search reorders through the model's `_reorder_cache`.
        self.model._supports_default_dynamic_cache = lambda: False
        self.device = device
        self.processor = IndicProcessor(inference=True)

    def translate_batch(
        self, texts: list[str], src_lang: str = "hin_Deva", tgt_lang: str = "eng_Latn"
    ) -> list[str]:
        batch = self.processor.preprocess_batch(texts, src_lang=src_lang, tgt_lang=tgt_lang)
        inputs = self.tokenizer(batch, padding="longest", truncation=True, return_tensors="pt").to(
            self.device
        )
        with self._torch.no_grad():
            generated = self.model.generate(
                **inputs,
                use_cache=True,
                min_length=0,
                max_length=self.MAX_LENGTH,
                num_beams=self.NUM_BEAMS,
                num_return_sequences=1,
            )
        decoded = self.tokenizer.batch_decode(
            generated.detach().cpu().tolist(),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        return self.processor.postprocess_batch(decoded, lang=tgt_lang)


@lru_cache(maxsize=1)
def get_translator() -> TranslatorLike:
    """IndicTrans2 Indic → English. Gated on Hugging Face: needs HF_TOKEN."""
    if settings.mock_models:
        return MockTranslator()
    return IndicTrans2Translator(settings.translate_model_id)


def warm_models() -> dict[str, bool]:
    """Load every model now (NFR3), so no user request pays the load cost.

    A model that fails to load is logged and reported, not raised: its stage
    then fails per request with a StageError and the other stages still run.
    """
    status = {}
    getters = {"xlit": get_xlit_engine, "llm": get_llm, "translate": get_translator}
    for name, getter in getters.items():
        try:
            getter()
            status[name] = True
        except Exception as exc:  # noqa: BLE001 — degrade, don't refuse to start
            logger.error("could not load %s: %s", name, type(exc).__name__)
            status[name] = False
    return status
