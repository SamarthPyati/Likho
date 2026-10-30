# Likho (लिखो)

Hinglish Smart Writing Assistant: paste Roman-script Hinglish, get Devanagari
(with per-word alternatives), a register-controlled rewrite, and an optional
English translation.

CS3235 (Working with LLMs) mini-project, RV University. Academic, non-commercial use.

> Work in progress — see `docs/TASKS.md` for milestone status.

## Quick start (no GPU needed)

```bash
make setup          # Python 3.10 venv + pinned deps (uses uv if installed)
make test           # runs with MOCK_MODELS=1
make lint
```

Copy `.env.example` to `.env` before running against real models.

## Docs

`docs/PRD.md` (scope) · `docs/ARCHITECTURE.md` (structure) · `docs/MODELS.md`
· `docs/EVALUATION.md` · `docs/TASKS.md` · `CLAUDE.md` (contributor rules)
