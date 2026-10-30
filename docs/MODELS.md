# MODELS

Verified against model cards and package docs as of October 2026. Re-check
availability before relying on any of this — the brief says so explicitly,
and checkpoints do move.

## 1. Summary

| Role | Model | Size | Licence |
|---|---|---|---|
| Transliteration | IndicXlit via `ai4bharat-transliteration` | ~11M | MIT (code + model) |
| Rewrite + router | `bharatgenai/Param-1-2.9B-Instruct` | 2.9B | **Unresolved** — see note |
| Translation | `ai4bharat/indictrans2-indic-en-1B` | 1.1B | MIT, gated |

**Licence note.** As of October 2026 the Param-1 model card says apache-2.0
in its metadata and MIT in its text; this document originally said
"BharatGen non-commercial". Read the repo's LICENSE file and settle it before
the README states a licence. Either way the card calls it an early research
checkpoint not for production, so do not deploy it publicly as a product.

**Gated.** Both IndicTrans2 repos require logging in and accepting their
terms on Hugging Face; each team member does this once and sets `HF_TOKEN`.

## 2. IndicXlit — transliteration

### 2.1 Install

```bash
make setup-models   # installs requirements-models.txt
```

The pip package (1.1.3, released 2022) **does depend on fairseq** — it is
not a way around it. Verified working on Python 3.10 / macOS arm64,
October 2026, with the pins in `requirements-models.txt`:

- `fairseq==0.12.2` has no wheels past Python 3.8, so it is **built from
  source** — a C compiler is needed (Xcode CLT on macOS, `build-essential`
  on Linux). It pulls `hydra-core==1.0.7` / `omegaconf==2.0.6`.
- `numpy==2.2.6` and `torch==2.14.1` both work.
- **torch ≥ 2.6 refuses the checkpoint** (`weights_only=True` default; the
  fairseq checkpoint pickles an `argparse.Namespace`). `registry.py`
  allowlists exactly that class with `torch.serialization.add_safe_globals`
  rather than pinning an old torch, which newer `transformers` would reject.
- First use downloads ~930 MB (121 MB model + 812 MB rescoring dictionaries)
  into the package directory inside `.venv`, **not** the HF cache.
- Load ≈ 4 s on CPU; one 5-word sentence ≈ 0.1 s.

The project brief links `AI4Bharat/IndianNLP-Transliteration`, which is the
older repo — the current one is `AI4Bharat/IndicXlit`. Use the pip package.

### 2.2 Usage

```python
from ai4bharat.transliteration import XlitEngine

# roman → native is the default (src_script_type="roman")
e = XlitEngine("hi", beam_width=10, rescore=True)

out = e.translit_word("namaste", topk=5)
# {'hi': ['नमस्ते', 'नमसते', ...]}

sent = e.translit_sentence("mujhe kal office jana hai")
# {'hi': 'मुझे कल ऑफिस जाना है'}
```

- `translit_word(word, topk=k)` returns a dict keyed by language code, value
  is a ranked list. **This is the call we use** — we need the candidates.
- `translit_sentence` exists but returns only a single string with no
  alternatives, and it does not do what §4 of the architecture doc requires.
  Use it only as a baseline to compare against.
- `rescore=True` improves top-1 at some latency cost. Measure both; the
  comparison is a free result for the report.
- `beam_width=10` with `topk=5` is a sane default.

### 2.3 Pitfalls

- **Word-level, no sentence context.** The model cannot disambiguate using
  neighbouring words. Reranking is our job. This is the single most
  important fact about this model.
- Runs on CPU. ~11M parameters. No GPU needed, which is why this project is
  tractable.
- Engine construction is slow (model load). Construct once, in `registry.py`.
- Check the installed version's direction support before assuming
  native→roman works; it has varied across releases and we don't need it.

## 3. Param-1 — rewrite and routing

### 3.1 Model

`bharatgenai/Param-1-2.9B-Instruct` — a 2.9B decoder-only bilingual
(Hindi + English) model trained from scratch by BharatGen, instruction-tuned.

### 3.2 Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "bharatgenai/Param-1-2.9B-Instruct"
tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

messages = [{"role": "user", "content": prompt}]
inputs = tok.apply_chat_template(messages, tokenize=True, return_tensors="pt")
out = model.generate(inputs.to(model.device), max_new_tokens=256)
```

`trust_remote_code=True` is required — the architecture is not in stock
`transformers`.

### 3.3 Resource planning

- bf16 ≈ 6 GB VRAM. Fits a Colab T4 (16 GB) alongside IndicTrans2 at fp16
  (~2.5 GB) with room left. **But the T4 has no native bf16** — load in
  `torch.float16` (or 4-bit) there, and check outputs match bf16 on a few
  prompts.
- Context window is 2048 tokens (`max_position_embeddings`). Keep router and
  tiebreak prompts short.
- No GPU? Community GGUF quantizations exist (Q4_K_M ≈ 1.8 GB) and run under
  Ollama or `llama.cpp`. Useful for local development. Verify the quant
  against a few outputs before trusting it for evaluation numbers.
- If VRAM is tight, load Param-1 in 4-bit with `bitsandbytes` rather than
  dropping the stage.

### 3.4 Pitfalls

- This is an **early SFT checkpoint**. BharatGen's own model card says it has
  not been through full instruction tuning or RLHF. Expect it to sometimes
  ignore format instructions, pad with preamble, or misidentify itself.
- **Always parse defensively.** For the router, ask for a single JSON object,
  extract with `re.search(r'\{[\s\S]*\}', text)`, and fall back to the
  rule-based router on a parse failure. Same pattern as the Lab 9 judge.
- For the rewrite stage, constrain the task hard: give it the Devanagari
  text, the target register, and an explicit instruction not to add
  information. Then validate the output — if the rewrite is more than ~1.6×
  the input length or drops below ~0.5×, treat it as a failed rewrite and
  return the original with a `StageError`.
- Temperature 0.3 for rewriting, 0.0 for routing.

## 4. IndicTrans2 — translation

### 4.1 Model

`ai4bharat/indictrans2-indic-en-1B` for Hindi → English. Distilled variants
(~200M) exist in the same HF collection and are the right choice if you are
CPU-only: `ai4bharat/indictrans2-indic-en-dist-200M` (MIT, gated; confirmed
October 2026).

### 4.2 Usage

```python
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit import IndicProcessor

model_name = "ai4bharat/indictrans2-indic-en-1B"
tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForSeq2SeqLM.from_pretrained(
    model_name, trust_remote_code=True, torch_dtype=torch.float16
).to(DEVICE)
ip = IndicProcessor(inference=True)

batch = ip.preprocess_batch([text], src_lang="hin_Deva", tgt_lang="eng_Latn")
inputs = tok(batch, padding="longest", return_tensors="pt", truncation=True).to(DEVICE)

with torch.no_grad():
    generated = model.generate(**inputs, use_cache=True, min_length=0,
                               max_length=256, num_beams=5, num_return_sequences=1)

with tok.as_target_tokenizer():
    decoded = tok.batch_decode(generated.detach().cpu().tolist(),
                               skip_special_tokens=True,
                               clean_up_tokenization_spaces=True)

translations = ip.postprocess_batch(decoded, lang="eng_Latn")
```

### 4.3 Pitfalls

- **`preprocess_batch` before, `postprocess_batch` after. Both mandatory.**
  Skipping either produces plausible-looking garbage rather than an error,
  which is the worst kind of bug. If translations look subtly wrong, check
  this first.
- Language codes are FLORES-style: `hin_Deva`, `eng_Latn`. Not `hi`/`en`.
- `IndicTransToolkit` is a separate install from the IndicTrans2 repo. Older
  docs name it `IndicTransTokenizer` — same `IndicProcessor` class, different
  package name. Pin whichever you install.
- Skip `attn_implementation="flash_attention_2"` unless `flash_attn` is
  actually installed; it will not fall back gracefully.
- Batch sentences rather than calling per sentence. The model is seq2seq and
  batching is a large win at evaluation time.

## 5. Datasets

| Dataset | Use | Source |
|---|---|---|
| Aksharantar test set | Transliteration ground truth | AI4Bharat, HF + GitHub |
| Aksharantar train split (hi) | Lexicon + n-gram tables for reranking | same |
| Hand-built code-mixed set | Token classification + end-to-end | we write it |

Aksharantar holds ~26M pairs across 21 languages, and its test set has ~103k
word pairs across 19 languages. The Hindi test split is labelled by source:
Dakshina, AK-Freq, AK-NEI (Indian named entities), AK-NEF (foreign named
entities) — see `docs/EVALUATION.md` §1.1. **That split is the backbone of the failure analysis** — report
accuracy per category, not just overall.

Sample 50 Hindi pairs with a fixed seed, commit the sample as a TSV, and
never resample. Reproducibility matters more than sample quality here.

## 6. Mock implementations

Every model gets a stub in `likho/mocks.py`, selected by `MOCK_MODELS=1`.

- `MockXlitEngine` — a hand-written romanization table covering the ~200
  words in the test fixtures, returning deterministic top-k lists.
- `MockLLM` — returns fixed JSON for the router and an identity rewrite.
- `MockTranslator` — returns `"[EN] " + text`.

These are not for cutting corners. They are what makes `make test` run in
under ten seconds on a laptop, which is what makes anyone actually run it.
