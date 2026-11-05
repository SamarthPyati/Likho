# Likho — evaluation results

Results of evaluation run **`20261105-m4`**, for the project report. Everything in
this folder is generated from that run by `python -m eval.export_results 20261105-m4`;
no number here was typed by hand. Cite the run ID when quoting a number.

**Failure analysis:** [`FAILURE_ANALYSIS.md`](FAILURE_ANALYSIS.md) — what went wrong,
by failure type, with worked examples and fixes.

## Files

| File | What it is |
|---|---|
| [`metrics.json`](metrics.json) | Every metric from the run (the source for all tables here) |
| [`cases.csv`](cases.csv) | One row per test case: every stage's output, pass/fail, failure codes |
| [`examples.md`](examples.md) | Selected real outputs per category, for the report |
| [`tables/`](tables/) | Each table below as a CSV, for charts |
| [`human_rubric.csv`](human_rubric.csv) | Blank sheet for the two human raters (see below) |
| [`test_sets/`](test_sets/) | The exact test data this run used |
| [`SHA256SUMS`](SHA256SUMS) | Checksums tying these files to the raw run in `eval/results/` |

## Run details

| | |
|---|---|
| run ID | `20261105-m4` |
| date (UTC) | 2026-11-05T10:53:50+00:00 |
| code version | `576027c-dirty` |
| real models | yes |
| machine | macOS-27.0.1-arm64-arm-64bit, Python 3.10.19, torch 2.14.1 |
| LLM device | mps |
| transliteration | IndicXlit (`ai4bharat-transliteration`), top-5 |
| rewrite / router / judge LLM | `bharatgenai/Param-1-2.9B-Instruct` |
| translation | `ai4bharat/indictrans2-indic-en-1B` |
| settings | LLM router True, LLM tie-break True, seed 3235, end-to-end register neutral |

### Reproduce

```bash
make setup-models
MOCK_MODELS=0 .venv/bin/python -m eval.run_eval --run-id <new-id>
.venv/bin/python -m eval.run_eval --compare 20261105-m4 <new-id>
```

`--compare` diffs two runs' metrics, ignoring timestamps and latency.

## Test sets

| set | size | used for |
|---|---|---|
| `codemixed_cases.yaml` | 50 sentences | tagging, rerank, rewrite, translation, end to end |
| `router_cases.yaml` | 30 inputs | router accuracy |
| `aksharantar_sample.tsv` | 50 words | word-level transliteration |
| `dev_cases.yaml` | — | prompt and weight tuning only; **not** used for these numbers |

Copies are in [`test_sets/`](test_sets/). How the code-mixed set was built, and the
labelling protocol, are in `docs/EVALUATION.md` §2.

## Headline

### PRD §7 targets

The success criteria from `docs/PRD.md` §7, checked against this run.

| criterion | value | target | met |
|---|---|---|---|
| transliteration top-1 (Aksharantar) | 0.580 | >= 0.7 | **no** |
| transliteration top-5 (Aksharantar) | 0.780 | >= 0.9 | **no** |
| transliteration CER (Aksharantar) | 0.095 | <= 0.15 | yes |
| token classification macro-F1 (HINDI/ENGLISH/NE) | 0.944 | >= 0.85 | yes |
| router accuracy | 1.000 | >= 0.9 | yes |
| rewrite meaning (LLM judge, 1-5) | 4.284 | >= 4.0 | yes |
| translation chrF++ | 75.000 | >= 45.0 | yes |
| hand-built code-mixed cases | 50 | >= 50 | yes |

CSV: [`tables/prd_targets.csv`](tables/prd_targets.csv)

### End to end on the 50 code-mixed test cases (register: neutral)

A case *passes* only if every stage is right: route, native script, rewrite accepted and judged meaning-preserving, and translation chrF++ ≥ 45.

| measure | cases |
|---|---|
| passed every stage | 19/50 (38.0%) |
| route correct | 50/50 (100.0%) |
| native script exactly right | 35/50 (70.0%) |
| native script right (lenient) | 37/50 (74.0%) |
| native script CER | 0.029 |
| rewrite accepted and meaning kept | 33/50 (66.0%) |
| translation evaluated | 50/50 (100.0%) |
| translation chrF++ ≥ 45 | 43/50 (86.0%) |

CSV: [`tables/end_to_end.csv`](tables/end_to_end.csv)

## Per stage

### Transliteration (IndicXlit, word level, Aksharantar + Dakshina sample)

Ablation 2: IndicXlit's built-in dictionary rescoring on vs off. CER is character error rate of the top-1 (lower is better).

| subset | n | top-1 rescore on | top-5 rescore on | CER rescore on | top-1 rescore off | top-5 rescore off | CER rescore off |
|---|---|---|---|---|---|---|---|
| overall | 50 | 58.0% | 78.0% | 0.095 | 46.0% | 78.0% | 0.128 |
| ak_freq | 13 | 69.2% | 84.6% | 0.086 | 46.2% | 84.6% | 0.129 |
| ak_nef | 13 | 46.2% | 84.6% | 0.137 | 46.2% | 84.6% | 0.137 |
| ak_nei | 12 | 58.3% | 58.3% | 0.082 | 50.0% | 58.3% | 0.137 |
| dakshina | 12 | 58.3% | 83.3% | 0.077 | 41.7% | 83.3% | 0.108 |

CSV: [`tables/transliteration.csv`](tables/transliteration.csv)

### Token classification (which Roman words to transliterate)

Ablation 3: Param-1 as a tie-breaker for low-confidence tokens, on vs off.

| kind | precision (LLM tie-break on) | recall (on) | F1 (on) | support | F1 (LLM tie-break off) |
|---|---|---|---|---|---|
| hindi | 99.2% | 98.7% | 98.9% | 380 | 98.9% |
| english | 92.8% | 100.0% | 96.2% | 64 | 96.2% |
| named_entity | 100.0% | 78.6% | 88.0% | 14 | 88.0% |
| devanagari | 100.0% | 100.0% | 100.0% | 18 | 100.0% |
| punct | 100.0% | 100.0% | 100.0% | 20 | 100.0% |
| **accuracy (all tokens)** |  |  | 98.4% | 496 | 98.4% |
| **macro-F1 (hindi, english, NE)** |  |  | 94.4% |  | 94.4% |

CSV: [`tables/token_classification.csv`](tables/token_classification.csv)

### Token confusion matrix (LLM tie-break on; rows = gold, columns = predicted)

| gold \ predicted | hindi | english | named_entity | devanagari | punct | numeric |
|---|---|---|---|---|---|---|
| hindi | 375 | 5 | 0 | 0 | 0 | 0 |
| english | 0 | 64 | 0 | 0 | 0 | 0 |
| named_entity | 3 | 0 | 11 | 0 | 0 | 0 |
| devanagari | 0 | 0 | 0 | 18 | 0 | 0 |
| punct | 0 | 0 | 0 | 0 | 20 | 0 |

CSV: [`tables/token_confusion.csv`](tables/token_confusion.csv)

### What decided each token's kind (LLM tie-break on)

| signal | tokens |
|---|---|
| dictionary | 242 |
| frequency | 142 |
| fixed | 38 |
| loanword | 36 |
| context | 25 |
| orthography | 11 |
| oov | 2 |

CSV: [`tables/token_signals.csv`](tables/token_signals.csv)

### Candidate reranking

Ablation 1: reranking IndicXlit's top-k with the lexicon and context, on vs off. Weights α=1.0, β=2.0, γ=0.5.

| measure | value |
|---|---|
| Hindi tokens in test sentences | 380 |
| top-1 accuracy, rerank off | 89.7% |
| top-1 accuracy, rerank on | 93.4% |
| top-1 accuracy (lenient), rerank off | 91.1% |
| top-1 accuracy (lenient), rerank on | 95.3% |
| ceiling: gold anywhere in top-k | 96.8% |
| tokens fixed by rerank | 16 |
| tokens broken by rerank | 2 |
| gold in top-k but not chosen (F8) | 13 |
| gold not in top-k at all | 12 |
| Aksharantar words (n=50) top-1 off → on | 58.0% → 58.0% |

CSV: [`tables/rerank.csv`](tables/rerank.csv)

### Router (agentic component)

The LLM proposes an intent; rules verify it and override on disagreement (`docs/ARCHITECTURE.md` §5a).

| measure | LLM router on | rules only |
|---|---|---|
| cases | 30 | 30 |
| final stage-plan accuracy | 100.0% | 100.0% |
| LLM proposal accuracy | 46.7% | — |
| LLM output unparseable | 13.3% | — |
| rules decided (fallback rate) | 53.3% | 100.0% |
| routing errors | 0 | 0 |

CSV: [`tables/routing.csv`](tables/routing.csv)

### Rewrite by register (Param-1 on gold Devanagari, n=50 per register)

Judge scores are 1–5 from Param-1 acting as judge — the same model family that wrote the rewrite, so treat them as indicative only; the human rubric is the real measure. When a rewrite is rejected the app shows no rewrite and keeps the Devanagari (FR7).

| register | accepted | rejected | length ratio | judge meaning | judge register | judge fluency | judge unparsed |
|---|---|---|---|---|---|---|---|
| casual | 37 | 13 | 1.15 | 3.97 | 4.49 | 4.49 | 0.0% |
| neutral | 40 | 10 | 1.00 | 4.35 | 4.62 | 4.62 | 0.0% |
| formal | 39 | 11 | 1.08 | 4.51 | 4.72 | 4.72 | 0.0% |

CSV: [`tables/rewrite.csv`](tables/rewrite.csv)

### Why rewrites were rejected

Reasons from the rule check in `likho/rewrite.py` (`validate`), grouped: the exact length ratios are in `metrics.json`.

| register | reason | count |
|---|---|---|
| casual | script changed from Roman | 7 |
| casual | length ratio outside 0.5–1.6 | 6 |
| neutral | script changed from Roman | 10 |
| formal | script changed from Roman | 9 |
| formal | length ratio outside 0.5–1.6 | 2 |

CSV: [`tables/rewrite_rejections.csv`](tables/rewrite_rejections.csv)

### Does the rewrite keep English words and names in Roman? (PRD U5)

No failure code counts this. Computed from `cases.csv`: the Roman words of each case's gold Devanagari, looked up in its accepted neutral rewrite.

| measure | value |
|---|---|
| accepted rewrites of sentences with English words or names | 19 |
| rewrites that changed at least one of them | 19/19 (100.0%) |
| …of which the case still passed every stage | 6 |
| English words and names kept, overall | 7/50 (14.0%) |

CSV: [`tables/rewrite_keeps_english.csv`](tables/rewrite_keeps_english.csv)

### Translation (IndicTrans2 indic→en 1B)

chrF++ and BLEU from sacrebleu (0–100). *Gold Devanagari* isolates the translation model; *end to end* translates the pipeline's own rewrite.

| input | n | chrF++ | BLEU |
|---|---|---|---|
| gold Devanagari | 50 | 75.0 | 53.0 |
| end to end | 50 | 70.0 | 47.0 |

CSV: [`tables/translation.csv`](tables/translation.csv)

## Ablations

The four ablations from `docs/EVALUATION.md` §5:

1. Rerank on vs off — table *Candidate reranking*
2. IndicXlit rescoring on vs off — table *Transliteration*
3. LLM tie-break on vs off — table *Token classification*
4. Whole-sentence baseline vs pipeline — below

The router's LLM-on vs rules-only comparison is in table *Router*.

### Ablation 4: naive whole-sentence transliteration vs the Likho pipeline

The baseline transliterates every word, which is what the IndicXlit library offers out of the box. The gap on English words and names is the reason the token classifier exists.

| measure | translit_sentence baseline | Likho pipeline |
|---|---|---|
| sentences exactly right | 11/50 (22.0%) | 35/50 (70.0%) |
| sentences right (lenient) | 11/50 (22.0%) | 37/50 (74.0%) |
| character error rate | 0.234 | 0.029 |
| tokens right: all | 378/496 (76.2%) | 467/496 (94.2%) |
| tokens right: hindi | 340/380 (89.5%) | 354/380 (93.2%) |
| tokens right: english | 0/64 (0.0%) | 64/64 (100.0%) |
| tokens right: named_entity | 0/14 (0.0%) | 11/14 (78.6%) |

CSV: [`tables/ablation_sentence_baseline.csv`](tables/ablation_sentence_baseline.csv)

## Failures

### Failure taxonomy F1–F12 (`docs/EVALUATION.md` §4)

*Primary* is the earliest stage that failed in a case — each failing case is counted once. *Any* counts every occurrence, including after an earlier failure (a case with two F8 words adds two).

| code | meaning | cases (primary) | occurrences (any) |
|---|---|---|---|
| F1 | Wrong vowel length | 0 | 0 |
| F2 | Schwa deletion error | 2 | 2 |
| F3 | Retroflex/dental confusion | 0 | 0 |
| F4 | Nukta mishandling | 0 | 0 |
| F5 | Loanword over-transliteration | 0 | 0 |
| F6 | Loanword under-transliteration | 3 | 5 |
| F7 | Named entity mangled | 3 | 3 |
| F8 | Context ambiguity (gold in top-k, not chosen) | 5 | 13 |
| F9 | Router mis-stage | 0 | 0 |
| F10 | Rewrite drift or rejection | 14 | 17 |
| F11 | Translation error | 2 | 7 |
| F12 | Crash / timeout | 0 | 0 |
| other | Transliteration error outside F1–F4 (e.g. nasalisation, e/ai) | 2 | 6 |

CSV: [`tables/failure_buckets.csv`](tables/failure_buckets.csv)

### End to end by test category

| category | n | native script right | passed every stage |
|---|---|---|---|
| plain | 10 | 6/10 (60.0%) | 5/10 (50.0%) |
| loanword | 10 | 9/10 (90.0%) | 3/10 (30.0%) |
| named_entity | 8 | 5/8 (62.5%) | 2/8 (25.0%) |
| ambiguous | 8 | 6/8 (75.0%) | 4/8 (50.0%) |
| hinglish_verb | 6 | 5/6 (83.3%) | 1/6 (16.7%) |
| long | 4 | 0/4 (0.0%) | 0/4 (0.0%) |
| already_devanagari | 4 | 4/4 (100.0%) | 4/4 (100.0%) |

CSV: [`tables/end_to_end_by_category.csv`](tables/end_to_end_by_category.csv)

## Latency

### Latency per stage (end-to-end run)

Measured on macOS-27.0.1-arm64-arm-64bit, LLM on `mps`. PRD NFR1 targets < 4 s per request on a Colab T4; this machine is not a T4.

| stage | requests | p50 ms | p95 ms |
|---|---|---|---|
| route | 50 | 3444 | 6750 |
| tokenize | 50 | 0 | 2 |
| transliterate | 46 | 0 | 2 |
| rerank | 46 | 3 | 16 |
| rewrite | 50 | 2759 | 9989 |
| translate | 49 | 375 | 2426 |
| total | 50 | 6626 | 16240 |

CSV: [`tables/latency.csv`](tables/latency.csv)

## Human evaluation

**Not scored yet.** Two team members should fill in `human_rubric.csv` independently (one copy each), then run `python -m eval.human <sheet.csv>` for each; it writes `eval/results/<run>_human.json` with means, inter-rater agreement and the judge-vs-human gap. Re-run this export afterwards.

## Caveats for the report

- The code-mixed set has 50 cases: enough to find failure patterns,
  too few for fine-grained percentages. Quote counts alongside percentages.
- The rewrite judge is Param-1 scoring Param-1; its scores cluster and should not
  be the main evidence for rewrite quality. Use the human rubric.
- Latency was measured on macOS-27.0.1-arm64-arm-64bit, not the Colab T4 the PRD targets.
