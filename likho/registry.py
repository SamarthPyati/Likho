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


class Param1LLM:
    """Param-1 behind the `LLMLike` interface. See docs/MODELS.md §3.

    Device: CUDA > Apple MPS > CPU. dtype: bf16 where CUDA supports it, else
    fp16 on a GPU (the Colab T4 has no native bf16), fp32 on CPU.
    """

    def __init__(self, model_id: str) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if torch.cuda.is_available():
            device = "cuda"
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        elif torch.backends.mps.is_available():
            device, dtype = "mps", torch.float16
        else:
            device, dtype = "cpu", torch.float32
        logger.info("loading %s on %s (%s)", model_id, device, dtype)
        token = settings.hf_token.get_secret_value() if settings.hf_token else None
        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=True, token=token
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, trust_remote_code=True, torch_dtype=dtype, token=token
        ).to(device)
        self.model.eval()
        self.device = device

    def generate(self, prompt: str, *, max_new_tokens: int = 256, temperature: float = 0.0) -> str:
        messages = [{"role": "user", "content": prompt}]
        if getattr(self.tokenizer, "chat_template", None):
            input_ids = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt"
            )
        else:
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        input_ids = input_ids.to(self.device)
        sampling = {"do_sample": True, "temperature": temperature} if temperature > 0 else {}
        with self._torch.no_grad():
            output = self.model.generate(
                input_ids,
                attention_mask=self._torch.ones_like(input_ids),
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                **sampling,
            )
        return self.tokenizer.decode(output[0, input_ids.shape[1] :], skip_special_tokens=True)


@lru_cache(maxsize=1)
def get_llm() -> LLMLike:
    """Param-1 (router, tagger tiebreak, rewrite, judge). Callers must handle failure."""
    if settings.mock_models:
        return MockLLM()
    return Param1LLM(settings.llm_model_id)
