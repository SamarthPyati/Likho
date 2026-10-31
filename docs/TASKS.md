# TASKS

Build order. Each milestone has acceptance criteria that can be checked
without a human looking at a screen. Do not start a milestone until the
previous one's criteria pass.

---

## M0 — Skeleton and contracts

**Goal:** everyone can work in parallel after this.

- [x] Repo, `.gitignore` (weights, `.env`, `__pycache__`, `eval/results/*.tmp`)
- [x] `requirements.txt`, every dep pinned with `==`
- [x] `Makefile` with `setup dev api test eval lint`
- [x] `likho/schemas.py` — all models from `docs/ARCHITECTURE.md` §3, verbatim
- [x] `likho/config.py` — pydantic-settings, reads `MOCK_MODELS`, `HF_TOKEN`
- [x] `likho/logging.py`
- [x] `likho/mocks.py` — stubs for all three models
- [x] CI: GitHub Actions running `make lint test` with `MOCK_MODELS=1`

**Accept when:** `make test` passes on a clean checkout with no GPU and no
model downloads. `python -c "from likho.schemas import AssistResponse"`
works. CI is green.

> Merge this before anyone writes a stage. It is the unblocking milestone.

---

## M1 — Transliteration stage

- [x] `likho/registry.py` — `get_xlit_engine()` singleton, mock-aware
- [x] `likho/transliterate.py` — `transliterate_tokens(tokens, top_k) -> list[Token]`
- [x] Populates `candidates`, sets `chosen` to candidate 0 for now
- [x] Handles the empty string, numerals, punctuation without calling the model
- [x] `eval/data/aksharantar_sample.tsv` — 50 pairs, seed 3235, committed
- [x] `eval/metrics.py` — CER, top-1, top-k; unit-tested against worked examples

**Accept when:** `python -m eval.run_eval --stage transliterate` prints top-1,
top-5 and CER overall and per Aksharantar category. Metric functions have
tests with hand-computed expected values.

---

## M2 — Tokenization and classification

The hard one. Budget the most time here.

- [x] `likho/tokenize.py` — splitter preserving punctuation and spacing
- [x] English wordlist + Hindi romanization lexicon, committed as data files
      (wordfreq-derived frequency lists; see `docs/ARCHITECTURE.md` §4 "As built")
- [x] Orthographic heuristics (capitalisation, n-grams)
- [x] LLM tiebreak — **one batched call per sentence**, never per token
      (falls back to rules until the Param-1 loader lands in M4)
- [x] `kind_confidence` set from the deciding signal
- [x] `eval/data/codemixed_cases.yaml` — 50 cases per `docs/EVALUATION.md` §1.2
- [ ] …double-labelled by two team members, disagreements recorded in the
      file header (currently a single AI-assisted draft)

**Accept when:** per-kind P/R/F1 and a confusion matrix print from the eval
harness. Token classification F1 ≥ 0.85 on the committed set, or a written
note in `eval/results/` explaining why not.

---

## M3 — Reranking

- [x] Unigram + bigram tables (wordfreq unigrams + IndicCorpV2 bigrams; see
      `docs/ARCHITECTURE.md` §5 "As built" for why not Aksharantar)
- [x] `likho/rerank.py` — the scoring function from `docs/ARCHITECTURE.md` §5
- [x] α, β, γ tuned on a 20-sentence dev split, **not** the test set
- [x] `chosen_rank` recorded so F8 failures are identifiable

**Accept when:** the eval harness reports top-1 accuracy with reranking on
and off, as separate numbers, from one command. Whether reranking helps is
a finding either way — do not tune until it does.

---

## M4 — Router and rewrite (the GenAI component)

- [x] `likho/registry.py` — `get_llm()` singleton for Param-1
      (`Param1LLM`; CUDA → MPS → CPU, fp16 where bf16 is unsupported)
- [ ] …loader verified against the real checkpoint (needs the ~6 GB download)
- [x] `likho/router.py` — LLM intent classification, strict-JSON prompt,
      regex extraction, rule-based fallback, `Route.source` records which fired
- [x] `likho/rewrite.py` — register-controlled rewrite
- [x] Length-ratio validation (0.5×–1.6×); out of range → `StageError`, original preserved
- [x] `eval/data/router_cases.yaml` — 30 labelled cases
- [x] `eval/judge.py` — LLM-as-judge for rewrite, Lab 9 pattern

**Accept when:** router accuracy and LLM-judge rewrite scores both appear in
the metrics JSON. Fallback rate is reported. A deliberately malformed LLM
response is covered by a test and does not crash the pipeline.

---

## M5 — Pipeline, API, frontend

- [ ] `likho/pipeline.py` — orchestrates stages per `Route`, fills `timings_ms`
- [ ] FR7: a failed stage populates `errors` and preserves earlier output
- [ ] `api/main.py` — `POST /assist`, `GET /health`, `POST /route`
- [ ] Startup hook warms all three models
- [ ] `app/gradio_app.py` — input box, register selector, translate toggle,
      loanword toggle, three output panes
- [ ] Per-word alternatives UI (U2) — clicking a word offers its candidates
- [ ] Route explanation shown (U6)

**Accept when:** `make api && make dev`, paste
`"kal ke meeting ke liye Rohit ko bata dena"`, get Devanagari with `meeting`
and `Rohit` preserved, switch register, see the rewrite change, toggle
translate, get English. Killing the rewrite stage via an env flag still
returns the transliteration.

---

## M6 — Full evaluation and failure analysis

- [ ] `eval/run_eval.py` runs every stage and writes both output files
- [ ] All four ablations from `docs/EVALUATION.md` §5
- [ ] Every failing case classified F1–F12 in the CSV
- [ ] `docs/FAILURE_ANALYSIS.md` — counts per bucket, 2–3 worked examples each,
      and what you would do about the top three
- [ ] A committed run in `eval/results/` that the report cites by name

**Accept when:** `make eval` reproduces the committed JSON on the same
hardware, and the failure analysis cites specific case IDs.

---

## M7 — Delivery

- [ ] `README.md` — what it is, a screenshot, install, run, results table
      quoting the named eval run, licences, team, course
- [ ] Demo recording, 3–5 minutes: problem, live demo, architecture, results,
      one honest failure
- [ ] `docs/REPORT.md` or the lab-format document
- [ ] Repo public, clean history, no weights, no tokens
- [ ] `pip install -r requirements.txt && make dev` verified on a machine
      that has never seen the project

**Accept when:** a teammate clones the repo on a fresh machine and has it
running from the README alone, without asking anyone a question.

---

## Sequencing

```
M0 ──┬── M1 ──┬── M3 ──┐
     ├── M2 ──┘        ├── M5 ── M6 ── M7
     └── M4 ───────────┘
```

M1, M2 and M4 are independent once M0 lands. M3 needs M1 and M2. M5 needs
everything.

## If you fall behind

Cut in this order. The first two cost almost nothing in marks:

1. The loanword toggle (U8) and the per-word alternatives UI (U2) — keep
   candidates in the API response so the data is there, drop the controls.
2. Ablations 3 and 4.
3. The LLM tiebreak in M2 — ship dictionary + heuristics only, and report
   the accuracy gap as future work.

**Never cut:** the committed test sets, `make eval`, or the failure
analysis. Those are four of the six rubric criteria, and a smaller app with
real measurement scores far better than a bigger one without.
