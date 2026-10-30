# EVALUATION

The rubric asks for 30–50 representative test cases, quantitative testing
and a failure analysis. This document specifies all three so that the
numbers in the report come from a committed script rather than a screenshot.

## 1. Test sets

### 1.1 `aksharantar_sample.tsv` — 50 word pairs

Sampled from the Aksharantar Hindi test split with `random.seed(3235)`,
committed to the repo, never resampled. Columns:

```
roman	devanagari	category
```

`category` is Aksharantar's own `source` label for the Hindi test split
(10,112 pairs; checked October 2026):

| `category` | Aksharantar `source` | Pairs in test split | What it is |
|---|---|---|---|
| `dakshina` | Dakshina | 4451 | General vocabulary |
| `ak_freq` | AK-Freq | 3647 | Most frequent words |
| `ak_nei` | AK-NEI | 1188 | Indian named entities |
| `ak_nef` | AK-NEF | 826 | Foreign named entities |

There is no native/foreign/frequent/rare split in the released data; these
four labels are what exists. The sample is stratified 13/13/12/12 so the
per-category breakdown has enough support to mean anything. Regenerate it
only with `python -m eval.sample_aksharantar path/to/hin_test.json`
(script committed; seed and per-category counts are fixed in it).

Some references are themselves debatable (nukta present or absent, e.g.
`firozpur` → फ़िरोज़पुर). That is reference noise, not model error, and the
failure analysis should say so rather than hide it.

This set measures the transliteration stage in isolation, against published
ground truth we did not write ourselves. That independence is the reason it
exists.

### 1.2 `codemixed_cases.yaml` — 50 sentences

Hand-built, because no public set covers what this app actually does. Each
case:

```yaml
- id: cm_017
  input: "kal ke meeting ke liye Rohit ko bata dena"
  expected_native: "कल के meeting के लिए Rohit को बता देना"
  expected_tokens:
    - {surface: kal, kind: hindi}
    - {surface: ke, kind: hindi}
    - {surface: meeting, kind: english}
    - {surface: ke, kind: hindi}
    - {surface: liye, kind: hindi}
    - {surface: Rohit, kind: named_entity}
    - {surface: ko, kind: hindi}
    - {surface: bata, kind: hindi}
    - {surface: dena, kind: hindi}
  expected_english: "Tell Rohit about tomorrow's meeting"
  difficulty: medium
  tags: [loanword, named_entity]
```

**Compose the set deliberately**, not by typing fifty sentences off the top
of your head. Target distribution:

| Tag | Count | What it probes |
|---|---|---|
| `plain` | 10 | Baseline — pure Hindi, no code-mixing |
| `loanword` | 10 | English nouns inside Hindi syntax |
| `named_entity` | 8 | People, places, brands |
| `ambiguous` | 8 | `to`/तो, `me`/में, `is`/इस, `bank`, `hum` |
| `hinglish_verb` | 6 | `manage karna`, `update kar dena` |
| `long` | 4 | 40+ words, tests context and latency |
| `already_devanagari` | 4 | Router must skip transliteration |

Have two team members label each case independently and reconcile
disagreements. Report the disagreement count — an honest inter-annotator
number is worth more marks than a suspiciously clean set.

**As built.** `expected_tokens` is written compactly for reviewability —
`"kal/h meeting/e Rohit/ne ,/p"` (codes: h e ne n p d) — and expanded to the
`{surface, kind}` list above by `eval/cases.py`. `eval/cases_test.py` checks
every case: token surfaces match `likho.tokenize.split`, and
`expected_native` differs from the input only at HINDI tokens. Each case
counts toward the distribution table by its first tag.

`eval/data/dev_cases.yaml` holds 20 further sentences in the same format for
tuning (tagger thresholds, reranker α/β/γ). It shares no sentence with the
test set.

### 1.3 `router_cases.yaml` — 30 labelled intents

Inputs paired with the stage list the router should pick. Include English-only
input, Devanagari-only input, mixed input, and a few single words.

## 2. Metrics

### 2.1 Transliteration

- **Top-1 word accuracy** — exact string match against the reference.
- **Top-5 accuracy** — reference appears anywhere in the top-5 candidates.
  The gap between top-1 and top-5 is the headroom that reranking can recover,
  and quoting both is more informative than either alone.
- **CER** — Levenshtein distance over characters, normalised by reference
  length. Use `jiwer` or implement it; either way, unit-test it against a
  worked example.
- Report all three **per Aksharantar category** and overall.
- Report **with and without reranking**, and **with and without `rescore=True`**.

### 2.2 Token classification

Precision, recall and F1 per `TokenKind`, plus a confusion matrix. The
HINDI↔ENGLISH confusion cell is the interesting one and should be discussed
explicitly in the report.

### 2.3 Router

Exact-match accuracy of the predicted stage list. Also report how often the
LLM router failed to parse and the rule fallback fired — a high fallback
rate is a legitimate finding about early-checkpoint models.

### 2.4 Rewrite

No ground truth exists, so use two independent judgements:

1. **LLM-as-judge**, 1–5 on meaning preservation, register match, fluency.
   Reuse the pattern from Lab 9: strict JSON prompt, regex extraction,
   deterministic fallback score on parse failure.
2. **Human rubric**, same three axes, on 20 of the 50 cases, scored by two
   team members.

Report both, and report the correlation between them. If the judge is
systematically more generous than the humans, say so — that is a finding
about LLM-as-judge leniency, and it is a real one.

### 2.5 Translation

chrF++ and BLEU via `sacrebleu` against the `expected_english` references.
Note in the report that single-reference BLEU on 50 short sentences is a
weak signal; it is reported for completeness, and chrF++ is the number to
lead with.

### 2.6 Latency

p50 and p95 per stage, measured from `AssistResponse.timings_ms` over the
full test set. Report the hardware.

## 3. Output format

`make eval` writes two files to `eval/results/`:

- `<timestamp>_metrics.json` — every number above, nested by stage.
- `<timestamp>_cases.csv` — one row per test case with input, expected,
  actual at each stage, and a boolean pass flag. **This CSV is the raw
  material for the failure analysis.** Do not skip it.

Both are committed. The README quotes numbers from a named run.

## 4. Failure taxonomy

Classify every failing case in the CSV into exactly one of these. Counts per
bucket, with two or three worked examples each, is the failure analysis
section of the report.

| Code | Failure | Expected cause |
|---|---|---|
| F1 | Wrong vowel length | `i`/`ee`, `u`/`oo` ambiguity in Roman input |
| F2 | Schwa deletion error | Devanagari inherent-vowel rules; romanization drops the cue |
| F3 | Retroflex/dental confusion | `t`/`ṭ`, `d`/`ḍ` not distinguished in Roman |
| F4 | Nukta mishandling | क़, ज़, फ़ — loanword phonemes |
| F5 | Loanword over-transliteration | English token classified HINDI |
| F6 | Loanword under-transliteration | Hindi token classified ENGLISH |
| F7 | Named entity mangled | NE classified HINDI and transliterated |
| F8 | Context ambiguity | Correct candidate present but ranked below top-1 |
| F9 | Router mis-stage | Wrong stage list chosen |
| F10 | Rewrite drift | Meaning changed, clause dropped, or facts added |
| F11 | Translation error | Faithful Devanagari, wrong English |
| F12 | Crash / timeout | Infrastructure, not linguistic |

F8 is worth isolating carefully. A case where the right answer was in the
top-5 but lost the rerank is a *different* problem from one where the model
never produced it, and conflating the two hides where the real headroom is.

## 5. Ablations

Cheap to run once the harness exists, and they turn a project into an
evaluation:

1. **Reranking on vs off** — isolates §5 of the architecture.
2. **`rescore=True` vs `False`** — accuracy against latency.
3. **LLM tiebreak on vs off** in token classification — is the Param-1 call
   earning its latency?
4. **`translit_sentence` baseline vs our word-level pipeline** — does the
   extra machinery beat the library's own one-liner? If it does not, that is
   still a result, and reporting it honestly is better than hiding it.

## 6. Reporting rules

- Every number in the report traces to a committed JSON.
- State the hardware and the date of the run.
- Quote the sample size next to every percentage. 50 cases means a single
  case is two percentage points; do not report to one decimal place and
  imply precision you do not have.
- If a target in `docs/PRD.md` §7 is missed, report the miss and explain it.
  A missed target with a good explanation scores better than a hit nobody
  believes.
