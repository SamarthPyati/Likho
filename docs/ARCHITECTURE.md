# ARCHITECTURE

## 1. Shape of the system

```
                   ┌──────────────┐
   user text ─────▶│   Router     │  LLM intent classification
                   │  (Param-1)   │  + rule fallback
                   └──────┬───────┘
                          │ Route: which stages run
                          ▼
                   ┌──────────────┐
                   │  Tokenizer   │  split + classify each token
                   │  + Tagger    │  HINDI / ENGLISH / NE / NUM / PUNCT
                   └──────┬───────┘
                          ▼
                   ┌──────────────┐
                   │Transliterate │  IndicXlit, top-k per token
                   │  (IndicXlit) │
                   └──────┬───────┘
                          ▼
                   ┌──────────────┐
                   │  Reranker    │  sentence-context scoring of candidates
                   └──────┬───────┘
                          ▼
                   ┌──────────────┐
                   │   Rewrite    │  register control, meaning-preserving
                   │  (Param-1)   │
                   └──────┬───────┘
                          ▼
                   ┌──────────────┐
                   │  Translate   │  hin_Deva → eng_Latn
                   │ (IndicTrans2)│
                   └──────┬───────┘
                          ▼
                     AssistResponse
```

Stages are independent and individually testable. That is the point — it is
what makes the failure analysis tractable, because every error can be
attributed to exactly one stage.

## 2. Repository layout

```
likho/
├── CLAUDE.md
├── README.md
├── Makefile
├── requirements.txt
├── .env.example
├── likho/                      # core library — no UI, no FastAPI imports
│   ├── __init__.py
│   ├── config.py               # settings, env, MOCK_MODELS
│   ├── schemas.py              # all Pydantic models (§3)
│   ├── logging.py
│   ├── registry.py             # loads models once, holds singletons
│   ├── router.py               # FR1
│   ├── tokenize.py             # FR2: splitting + token classification
│   ├── transliterate.py        # FR3: IndicXlit wrapper
│   ├── rerank.py               # FR4: context reranking
│   ├── rewrite.py              # FR5: Param-1 rewrite
│   ├── translate.py            # FR6: IndicTrans2 wrapper
│   ├── pipeline.py             # orchestration, FR7 error handling
│   └── mocks.py                # deterministic stubs for every stage
├── api/
│   └── main.py                 # FastAPI, /assist /health /route
├── app/
│   └── gradio_app.py           # thin client over the API; never imports likho/
├── eval/
│   ├── data/
│   │   ├── aksharantar_sample.tsv    # 50 word pairs, sampled + committed
│   │   ├── codemixed_cases.yaml      # 50 hand-built sentences
│   │   └── router_cases.yaml         # 30 labelled intents
│   ├── metrics.py              # CER, WER, top-k accuracy, chrF++
│   ├── judge.py                # LLM-as-judge for the rewrite stage
│   ├── run_eval.py
│   ├── export_results.py       # one run → results/ for the report
│   └── results/                # committed JSON + CSV outputs
├── results/                    # generated report folder + FAILURE_ANALYSIS.md
├── docs/
└── tests/
```

**Dependency direction is one-way:** `app/` → `api/` → `likho/`, and
`eval/` → `likho/`. Nothing in `likho/` imports from `api/`, `app/` or
`eval/`. Enforce it with a test that walks the imports.

## 3. Data contracts

Every structure that crosses a module boundary is a Pydantic model in
`likho/schemas.py`. These are the contracts — agree on them before
writing stage code, because three people will be writing stages in parallel.

```python
from enum import Enum
from pydantic import BaseModel, Field

class TokenKind(str, Enum):
    HINDI = "hindi"              # Roman-script Hindi — transliterate
    ENGLISH = "english"          # English loanword — keep by default
    NAMED_ENTITY = "named_entity"  # person, place, brand — keep
    NUMERIC = "numeric"
    PUNCT = "punct"
    DEVANAGARI = "devanagari"    # already in native script

class Register(str, Enum):
    CASUAL = "casual"
    NEUTRAL = "neutral"
    FORMAL = "formal"

class Stage(str, Enum):
    ROUTE = "route"
    TOKENIZE = "tokenize"
    TRANSLITERATE = "transliterate"
    RERANK = "rerank"
    REWRITE = "rewrite"
    TRANSLATE = "translate"

class Token(BaseModel):
    surface: str                           # as the user typed it
    kind: TokenKind
    kind_confidence: float = 1.0
    candidates: list[str] = Field(default_factory=list)   # top-k, best first
    chosen: str | None = None              # after reranking
    chosen_rank: int | None = None         # which candidate won; 0 = model top-1

class Route(BaseModel):
    stages: list[Stage]
    reason: str                            # shown in the UI (U6)
    source: str                            # "llm" or "rule_fallback"

class StageError(BaseModel):
    stage: Stage
    message: str
    recoverable: bool = True

class AssistRequest(BaseModel):
    text: str
    register: Register = Register.NEUTRAL
    translate: bool = False
    transliterate_loanwords: bool = False  # U8
    top_k: int = 5

class AssistResponse(BaseModel):
    route: Route
    tokens: list[Token]
    native_script: str | None = None
    rewritten: str | None = None
    english: str | None = None
    errors: list[StageError] = Field(default_factory=list)
    timings_ms: dict[str, float] = Field(default_factory=dict)
```

`AssistResponse` always returns whatever succeeded. A failure in `rewrite`
populates `errors` and leaves `rewritten` as `None` — it never clears
`native_script`. That is FR7, and there is a test for it.

## 4. Token classification — the hard part

This is where the project is won or lost, so it gets its own section.

Given `"kal ke meeting ke liye Rohit ko bata dena"`, the correct behaviour is:

| Token | Kind | Output |
|---|---|---|
| kal | HINDI | कल |
| ke | HINDI | के |
| meeting | ENGLISH | meeting (or मीटिंग if opted in) |
| ke liye | HINDI | के लिए |
| Rohit | NAMED_ENTITY | Rohit |
| ko bata dena | HINDI | को बता देना |

Transliterating everything gives `कल के मीटिंग के लिए रोहित को बता देना` —
which is not wrong, exactly, but it is not what a Hindi writer would
produce, and `Rohit → रोहित` is a judgement call the user should own.

**Approach — three signals, combined:**

1. **Dictionary lookup.** An English wordlist (top ~20k) and a Hindi
   romanization lexicon built from the Aksharantar training split. A token in
   exactly one of them is settled cheaply. Most tokens are.
2. **Orthographic heuristics.** Capitalisation mid-sentence suggests a named
   entity. Character n-grams that never occur in romanized Hindi (`th` in
   `the`, trailing `-tion`, `-ness`) suggest English.
3. **LLM tiebreak.** Only for tokens that signals 1 and 2 disagree on or
   that neither resolves. Batch all ambiguous tokens from a sentence into
   **one** Param-1 call with the full sentence as context. Do not make one
   call per token.

Set `kind_confidence` from which signal decided, and log the distribution —
it is directly useful for the failure analysis.

**As built (M2), and why it differs.** The Aksharantar training split turned
out to be unusable as a Hindi lexicon: it lacks the commonest words (`kal`,
`hai`, `ke`, `nahi` are absent; they appear to be held out for the test
splits). And `th` is common in romanized Hindi (`tha`, `thoda`), so it is
not an English cue. `likho/tokenize.py` instead runs a cascade, each step
seeing only what earlier steps left undecided:

| Step | Signal | Rule |
|---|---|---|
| 1 | orthography | ALL CAPS → ENGLISH; Capitalised mid-sentence → NAMED_ENTITY; English-only suffix on an unknown word → ENGLISH |
| 2 | dictionary | `E` = English Zipf of the word, `H` = Hindi Zipf of IndicXlit's top candidate (both from wordfreq, committed in `likho/data/`). Plausible in only one language → that language |
| 3 | frequency | `H − E` ≥ 0.75 → HINDI, ≤ −0.75 → ENGLISH |
| — | loanword | close call where the Hindi reading is uncommon (`bank` → बैंक) → ENGLISH (U5) |
| 4 | context | homographs — both readings common (`H` ≥ 6 and `E` ≥ 5, or a close call with `H` ≥ 6: `to`, `me`, `is`, `the`) — take the sentence's majority language |
| 5 | llm | one Param-1 call per sentence for homographs with no majority |
| 6 | fallback | HINDI |

Thresholds were tuned on `eval/data/dev_cases.yaml` only. Each signal has a
distinct `kind_confidence` (`likho.tokenize.CONFIDENCE`), so the deciding
signal is recoverable from any `Token`; the eval prints the distribution.

Known limitation: a name at the start of a sentence (`Priya aur main…`) is
indistinguishable from a capitalised Hindi word and is tagged HINDI (F7).

**Known-hard cases.** Build the test set around these deliberately:

- Words that are both: `to` (English) vs `to` (Hindi तो), `me` (English) vs
  `me` (Hindi में), `is` (English) vs `is` (Hindi इस), `hum`, `bank`, `jai`.
- Transliterated English that users *want* in Devanagari: `scooter`, `train`.
- Names that collide with common words: `Anand`, `Prem`, `Asha`, `Dhanush`.
- Hinglish verb constructions on English stems: `manage karna`,
  `update kar dena`, `forward kar do`.

## 5. Reranking

IndicXlit scores each word in isolation. For `"mujhe kal bank jana hai"` it
has no way to know whether `bank` is the financial institution (बैंक) or a
river bank, and for genuinely ambiguous romanizations it returns several
plausible Devanagari strings with no sentence-level preference.

Rerank the top-k with a simple, explainable score:

```
score(candidate) = α · model_rank_score
                 + β · unigram_log_prob(candidate)
                 + γ · bigram_log_prob(prev_chosen, candidate)
```

Build the unigram and bigram tables from the Devanagari side of the
Aksharantar training split plus any Hindi monolingual corpus you can get.
Set α, β, γ on a 20-sentence dev split — do not tune them on the test set.

Report top-1 accuracy with and without reranking. The delta is a result
worth putting in the report, whichever direction it goes.

**As built (M3).** `likho/rerank.py`, greedy left to right.

- `model_rank_score` = −rank (IndicXlit's `translit_word` returns no scores).
- Unigram = Hindi Zipf frequency from wordfreq (`likho/data/hi_freq.tsv`);
  the Aksharantar Devanagari side is a word list without frequencies.
- Bigram = log10 P(c | previous Devanagari word) with stupid backoff, from a
  CC0 sample of AI4Bharat IndicCorpV2 Hindi (3.6M tokens,
  `likho/data/hi_bigrams.tsv`, built by `python -m scripts.build_ngrams`).
  A non-Devanagari token breaks the chain.
- Weights α=1, β=2, γ=0.5 from a grid search on the dev split. β=2 sits on
  a plateau where γ ∈ [0, 4] scores the same on dev, so γ=0.5 was picked as
  the midpoint before the test set was run.
- `python -m eval.run_eval --stage rerank` reports on/off top-1 for the
  Aksharantar words (no context) and for the gold-HINDI tokens of the
  code-mixed sentences, plus fixed/broken counts and every F8 case.

## 5a. Router and rewrite (as built, M4)

**Router** (`likho/router.py`). Param-1 *proposes* one intent:
`roman_hinglish`, `mixed_script`, `devanagari`, `english`, `empty`;
`STAGES[intent]` maps it to stages. The reply is forced to start with
`{"intent": "` (prefill) after three few-shot turns, so it can only name an
intent. Script counts plus the M2 tagger determine exactly one intent for
any text, and the proposal is used only when it agrees with that
(`route_with_proposal`); otherwise `Route.source = "rule_fallback"`.
Translation is not routed; the pipeline appends it when requested.

Why "LLM proposes, rules verify" (measured on Param-1, Apple M4 / MPS fp16,
2026-10-31, dev split):

| Router prompt | Parseable | LLM proposal correct |
|---|---|---|
| English instructions, free reply | 0/6 (prose answers) | — |
| + prefill | 20/20 | 0/20 (says `mixed_script` for everything) |
| + prefill + few-shot (shipped) | 20/20 | 10/20 (calls Roman Hinglish `devanagari`) |

An unverified `english` for Hinglish would skip transliteration, so the
proposal cannot be trusted alone. On the 30 router cases the rules alone
score 30/30. Report this as a finding about the early SFT checkpoint.

`LLM_ROUTER=0` skips the Param-1 call. Routes are unchanged (the rules
decide either way); each request saves one Param-1 call, which took 4–5.5 s
of an 8–10 s request on an Apple M4 (M5 run). Default stays on, since the
agentic component is part of the rubric; NFR1 numbers should be reported
both ways.

**Rewrite** (`likho/rewrite.py`). Instructions are in Hindi, with two
few-shot pairs per register (English words kept, तुम → आप for formal). The
reply is cleaned (preamble, fences, quotes, echoed tags) and rejected —
`StageFailure` → `StageError`, input preserved — if empty, outside
0.5×–1.6× the input length, or switched script (Devanagari ↔ mostly Roman).
Sampling (temperature 0.3) is reseeded per call (`LLM_SEED`), so the same
input gives the same rewrite and eval runs are reproducible.

Measured on the dev split (same setup as above):

| Rewrite prompt | Passed validation |
|---|---|
| English instructions | 0/9 — translated to English or wrote a full email |
| Hindi instructions | 14/18 |
| Hindi + 2 examples per register (shipped) | 21/24 |

Known weaknesses that pass validation, for the failure analysis (F10):
English words get translated despite the rule (cup → कप, flight → उड़ान);
meaning sometimes drifts (`कल match देखने चलें?` → `कल मैच देखने जाओ।`);
formal output is sometimes unchanged. The judge scored nearly everything
4/4/4, drift included — evidence for LLM-as-judge leniency that the human
rubric (M6) should test.

**Param-1 decoding.** `Param1LLM.generate` runs its own decode loop instead
of `model.generate()`: Param-1's remote code indexes `past_key_values` as a
legacy tuple and crashes on the `DynamicCache` that transformers 4.57 passes.

**Prompt convention.** Every Param-1 prompt starts `TASK: <name>` (route,
tag, rewrite, judge); `MockLLM` dispatches on it.

## 6. Model loading

`likho/registry.py` holds one lazily-initialised singleton per model.

```python
@lru_cache(maxsize=1)
def get_xlit_engine() -> "XlitEngine":
    if settings.mock_models:
        return MockXlitEngine()
    from ai4bharat.transliteration import XlitEngine
    return XlitEngine("hi", beam_width=10, rescore=True)
```

Same pattern for Param-1 and IndicTrans2. FastAPI warms all three in a
startup hook so the first user request is not the one that pays the load
cost.

## 7. Parallel work split

Three or four people, minimal collision:

- **Person A** — `tokenize.py` + `rerank.py` + the lexicons. The hardest and
  most interesting piece.
- **Person B** — `transliterate.py` + `translate.py` + `registry.py`.
  Model plumbing.
- **Person C** — `router.py` + `rewrite.py` + prompt design. The GenAI/agentic
  component the rubric asks for.
- **Person D** (or A) — `api/`, `app/`, `eval/`, CI.

`schemas.py` must be written and merged **first**, by whoever starts. Until
it exists everyone is blocked, and after it exists nobody is.
