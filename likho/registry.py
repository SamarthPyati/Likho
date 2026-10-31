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
            model_id, trust_remote_code=True, dtype=dtype, token=token
        ).to(device)
        self.model.eval()
        self.device = device
        eos = self.model.generation_config.eos_token_id
        self.eos_ids = set(eos if isinstance(eos, list) else [eos])

    def generate(self, prompt: str, *, max_new_tokens: int = 256, temperature: float = 0.0) -> str:
        """Greedy (temperature 0) or sampled decoding with a hand-rolled loop.

        Param-1's remote modeling code predates transformers' Cache objects: it
        indexes `past_key_values[0][0]` as a tuple, and `model.generate()` in
        4.57 passes a DynamicCache instead, which crashes. Calling forward()
        directly with the tuple cache it returns sidesteps that.
        """
        torch = self._torch
        messages = [{"role": "user", "content": prompt}]
        if getattr(self.tokenizer, "chat_template", None):
            input_ids = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt"
            )
        else:
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        next_input, past, generated = input_ids.to(self.device), None, []
        with torch.no_grad():
            for _ in range(max_new_tokens):
                out = self.model(input_ids=next_input, past_key_values=past, use_cache=True)
                logits, past = out.logits[0, -1].float(), out.past_key_values
                if temperature > 0:
                    probs = torch.softmax(logits / temperature, dim=-1)
                    token = int(torch.multinomial(probs, num_samples=1))
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
