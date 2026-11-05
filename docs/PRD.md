# PRD — Hinglish Smart Writing Assistant

## 1. Problem

Most Indians type Hindi in Roman script. `"mujhe kal office jana hai"` is
how the sentence gets written in practice, but it is not what anyone wants
to publish, submit or send formally. Converting it by hand is slow, and
generic autocorrect makes it worse because it assumes English.

The gap: there is no single tool that takes casual Roman Hinglish and
returns something you can actually send — correct script, correct register,
and optionally English for an audience that needs it.

## 2. What we are building

A web app with one text box. The user pastes Roman-script Hinglish and gets
back, in one pass:

1. **Native script** — Devanagari, with alternatives offered for any word
   the model was unsure about.
2. **Improved phrasing** — the same meaning, cleaned up, in a register the
   user picks (casual / neutral / formal).
3. **English translation** — optional, on demand.

The user can accept, swap a word for one of its alternatives, or copy any of
the three outputs.

## 3. User stories

| # | As a… | I want to… | So that… | Priority |
|---|---|---|---|---|
| U1 | student | paste Roman Hinglish and get Devanagari | I can submit it in Hindi | **Must** |
| U2 | student | fix a wrongly transliterated word by picking from a list | I don't retype the whole line | **Must** |
| U3 | professional | rewrite a casual message as formal Hindi | I can send it to a senior | **Must** |
| U4 | professional | translate the result to English | I can send it to a non-Hindi colleague | **Must** |
| U5 | user | have my English words stay English | "meeting" doesn't become मीटिंग unless I want it to | **Must** |
| U6 | user | see which stages ran on my input | I understand what the tool did | Should |
| U7 | evaluator | run a scripted evaluation and get a metrics JSON | results are reproducible | **Must** |
| U8 | user | toggle whether loanwords get transliterated | I control the output style | Could |

## 4. Functional requirements

**FR1 — Intent routing.** The system decides, per request, which stages to
run. Input already in Devanagari skips transliteration. Input that is
entirely English skips transliteration and goes straight to rewrite. This
decision is made by an LLM-backed router with a deterministic rule-based
fallback, and the chosen route is returned to the UI.

**FR2 — Token classification.** Before transliteration, every token is
labelled `HINDI` (transliterate), `ENGLISH` (keep as-is or transliterate per
user setting), `NAMED_ENTITY` (keep), `NUMERIC`, or `PUNCT`. Only `HINDI`
tokens, plus `ENGLISH` tokens when the user has opted in, are sent to the
transliteration model.

**FR3 — Transliteration with alternatives.** Each transliterated token
carries its top-k candidates. The UI exposes these; the API returns them.

**FR4 — Context reranking.** The transliteration model is word-level and has
no sentence context. The system reranks the top-k candidates using sentence
context before choosing a default. See `docs/ARCHITECTURE.md` §5.

**FR5 — Register-controlled rewrite.** The rewrite stage takes a target
register (`casual`, `neutral`, `formal`) and must preserve meaning. It must
not add facts, and it must not silently drop clauses.

**FR6 — Translation.** Devanagari Hindi → English, on request.

**FR7 — Graceful degradation.** If any stage fails, the response carries the
output of the stages that succeeded plus an error marker for the one that
did not. A failed rewrite must never lose the user's transliteration.

**FR8 — Evaluation harness.** `eval/run_eval.py` runs the full test set
offline and writes a metrics JSON plus a per-case CSV.

## 5. Non-functional requirements

- **NFR1** — p95 latency under 4 s for a 30-word input on a Colab T4.
- **NFR2** — Runs end to end with `MOCK_MODELS=1` on a laptop, no GPU.
- **NFR3** — Models load once at process start, never per request.
- **NFR4** — No user text is persisted. Logs record lengths and timings, not
  content.
- **NFR5** — All model licences recorded in `docs/MODELS.md`. Param-1's card
  gives a permissive licence but restricts the release to research use; this
  project is academic and stays that way.

## 6. Out of scope

Stated so nobody builds them:

- Languages other than Hindi. The architecture is language-parameterised and
  IndicXlit covers 21 languages, but **Hindi only** ships. Adding Tamil is a
  config change, not a milestone.
- Speech input or output.
- User accounts, history, persistence of any kind.
- Native-script → Roman transliteration (the reverse direction).
- Fine-tuning any model. We use pretrained checkpoints as they are.
- Mobile app. Responsive web is enough.

## 7. Success criteria

| Criterion | Target |
|---|---|
| Top-1 word accuracy, transliteration, Aksharantar sample | ≥ 70% |
| Top-5 word accuracy, transliteration | ≥ 90% |
| Character Error Rate, transliteration | ≤ 0.15 |
| Token classification F1 (HINDI vs ENGLISH vs NE) | ≥ 0.85 |
| Router intent accuracy | ≥ 0.90 |
| Rewrite meaning preservation (LLM-judge, 1–5) | ≥ 4.0 |
| Translation chrF++ vs references | ≥ 45 |
| Test cases, hand-built code-mixed set | 50 |

Targets are the pass bar for the report, not a promise. If a number lands
below target, that is a finding for the failure analysis, not a reason to
change the metric. Record it and explain it.
