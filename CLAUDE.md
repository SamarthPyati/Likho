# CLAUDE.md

Context for Claude Code working in this repository. Read this first, then
`docs/PRD.md` for scope and `docs/TASKS.md` for what to build next.

## What this is

**Hinglish Smart Writing Assistant** — a web app that takes Roman-script
Hinglish, converts it to native Devanagari, improves the phrasing, and
optionally translates it to English.

CS3235 (Working with LLMs) mini-project, RV University. Team of 3–4
undergraduates. Academic, non-commercial use only.

## Document map

| File | What it holds |
|---|---|
| `docs/PRD.md` | Scope, user stories, what is explicitly out of scope |
| `docs/ARCHITECTURE.md` | Module boundaries, data contracts, repo layout |
| `docs/MODELS.md` | Exact model IDs, loading code, licences, known pitfalls |
| `docs/EVALUATION.md` | Metrics, test-set construction, failure taxonomy |
| `docs/TASKS.md` | Milestones M0–M7 with acceptance criteria |

When these documents disagree with each other, `docs/PRD.md` wins on scope
and `docs/ARCHITECTURE.md` wins on structure.

## Hard rules

1. **Never call a model from inside a Gradio callback.** All model access
   goes through `likho/` core functions. The UI layer is a thin client.
2. **`MOCK_MODELS=1` must always work.** Every stage needs a deterministic
   stub so the full app, the tests and CI run on a machine with no GPU and
   no model downloads. If you add a stage, you add its stub in the same
   commit.
3. **Never commit model weights, `.env`, or HF tokens.** Weights live in the
   HF cache. `.gitignore` covers `*.pt`, `*.bin`, `*.gguf`, `models/`.
4. **Evaluation code may not import the UI layer.** `eval/` depends on
   `likho/` only, so results are reproducible from the command line.
5. **Pin every dependency** in `requirements.txt` with `==`. The AI4Bharat
   stack is sensitive to `transformers` and `numpy` versions.
6. **Do not invent evaluation numbers.** Metrics in the README and the
   report come from a committed run of `eval/run_eval.py` whose output JSON
   is checked in under `eval/results/`.

## Commands

```bash
make setup          # venv + pinned deps
make dev            # Gradio app on :7860, talks to local API
make api            # FastAPI on :8000
make test           # pytest, runs with MOCK_MODELS=1
make eval           # full evaluation, writes eval/results/<timestamp>.json
make lint           # ruff + black --check
```

## Conventions

- Python 3.10. `ruff` + `black`, line length 100.
- Type hints on every public function in `likho/`.
- Pydantic models for every cross-module data structure — see
  `docs/ARCHITECTURE.md` §3. Do not pass bare dicts between stages.
- Tests next to the code they cover: `likho/transliterate_test.py`.
- Log through `likho.logging.get_logger(__name__)`, never `print`.
- Commit messages: `M3: add beam candidate reranking` — milestone prefix.

## Things that will bite you

- `ai4bharat-transliteration` is **word-level**. `translit_sentence` exists
  but has no sentence context; the model cannot disambiguate homographs from
  surrounding words. Reranking is our job, not the model's. See
  `docs/MODELS.md` §2.3.
- IndicTrans2 requires `IndicProcessor.preprocess_batch` *before*
  tokenization and `postprocess_batch` *after* decoding. Skipping either
  silently produces garbage rather than an error.
- Param-1 needs `trust_remote_code=True`. It is an early SFT checkpoint and
  will sometimes ignore output-format instructions — always parse its output
  defensively with a fallback, exactly as in `docs/MODELS.md` §3.4.
- Not every Roman token should be transliterated. Deciding which ones should
  be is the core difficulty of this project, not an edge case. See
  `docs/ARCHITECTURE.md` §4.

## Definition of done for the course

The rubric requires all six: a working end-to-end app, a usable frontend, a
GenAI/agentic component (the router and the rewrite stage — not just a
pipeline), 30–50 representative test cases, a written failure analysis, and
a public GitHub repository with a README and a demo recording.
