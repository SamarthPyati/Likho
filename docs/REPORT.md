# Likho: a Hinglish Smart Writing Assistant

CS3235 Working with LLMs — mini-project report, RV University

*Team: Samarth Sanjay Pyati, (add the other members).*

> **Draft.** Every number in this report comes from evaluation run
> `20261105-m4` (see [`results/`](../results/README.md)); cite that run when
> quoting one. Sections marked *(team)* need input only the team can give.

---

## Abstract

Most Indians type Hindi in Roman letters, mixed with English ("kal ke meeting
ke liye Rohit ko bata dena"). Likho converts such text to Devanagari, rewrites
it in a chosen tone, and optionally translates it to English. The central
difficulty is deciding which Roman words are Hindi and should be converted,
and which are English words or names that should stay as typed. Likho combines
a word-level transliteration model (IndicXlit), a rule-and-lexicon token
tagger, a context reranker, an LLM agent that routes each input (Param-1 2.9B,
verified by rules), an LLM rewrite with a rule-based output check, and
IndicTrans2 for translation. On 50 hand-built code-mixed sentences, Likho gets
35 sentences exactly right in Devanagari, against 11 for transliterating every
word, and keeps all 64 English words as typed. Routing is correct for all 30
router test inputs and translation reaches chrF++ 75.0. The main weakness is
the rewrite: the LLM converts the user's English words in every rewrite that
contains them.

## 1. Problem

The PRD (`docs/PRD.md` §1) frames the gap: there is no single tool that takes
casual Roman Hinglish and returns something sendable — correct script, chosen
register, and optionally English. Generic transliteration converts every word,
which mangles the English words and names that code-mixed text is full of;
generic autocorrect assumes English.

User requirements (PRD §3), the ones the design is built around:

- **U1** Roman Hinglish → Devanagari. **U2** fix a word by picking an
  alternative. **U3** rewrite casual → formal. **U4** translate to English.
- **U5** English words stay English ("meeting" must not become मीटिंग unless
  the user asks). **U6** show which stages ran.
- **U7** a reproducible, scripted evaluation.

## 2. System design

### 2.1 Pipeline

```
text → Router → Tokenizer/Tagger → Transliterate → Rerank → Rewrite → Translate
```

Each stage is a separate module in `likho/` with typed (Pydantic) inputs and
outputs, so each can be tested and evaluated on its own — which is what makes
the failure analysis in §5 possible. The Gradio UI is a thin HTTP client of a
FastAPI service; no model is ever called from the UI. A deterministic mock of
every model (`MOCK_MODELS=1`) lets the whole app and its 200+ tests run on a
laptop without a GPU or downloads.

| Stage | Method | Model |
|---|---|---|
| Router | LLM proposes one of five intents; rules verify | Param-1 2.9B Instruct |
| Tagger | Cascade: orthography → dictionary → frequency → context → LLM tie-break | lexicons, Param-1 |
| Transliterate | Top-5 candidates per Hindi word | IndicXlit (`ai4bharat-transliteration`) |
| Rerank | α·rank + β·unigram frequency + γ·bigram with previous word | Hindi word frequencies, IndicCorp bigrams |
| Rewrite | Hindi few-shot prompt per register; rule check on output | Param-1 2.9B Instruct |
| Translate | IndicProcessor pre/post-processing + beam search (5) | IndicTrans2 indic-en 1B |

### 2.2 The GenAI / agentic component

**Router (FR1).** For each request Param-1 reads the text and proposes an
intent — Roman Hinglish, mixed script, Devanagari, English or empty — which a
fixed table maps to the stages that run. Because Param-1 is an early
instruction-tuned checkpoint, its proposal is *verified*: a rule-based router
classifies the same text from the tagger's output, and Param-1's intent is used
only when the two agree. On disagreement the rules decide and the response
says so (`Route.source`), and the UI shows the route, the reason and per-stage
timings (U6). The design principle is "the LLM proposes, rules verify".

**Rewrite (FR5).** Param-1 rewrites the Devanagari in the chosen register from
a Hindi-language few-shot prompt (English instructions made it translate). Its
output is parsed defensively and checked before use: it is rejected if empty,
outside 0.5×–1.6× the input length, or if its script changed. A rejected
rewrite is reported as a stage error and the user keeps the Devanagari (FR7).

**Tie-break.** When the tagger's lexical signals cannot decide a word's
language, one batched Param-1 call per sentence decides the rest.

### 2.3 Which words to transliterate

The tagger (`likho/tokenize.py`) labels each Roman word HINDI, ENGLISH or
NAMED_ENTITY (plus NUMERIC, PUNCT, DEVANAGARI) through a cascade where each
step sees only what earlier steps left undecided:

1. **orthography** — a capitalised word mid-sentence is a name; ALL CAPS and
   English-only suffixes (-tion, -ness) are English;
2. **dictionary** — the word is plausible in only one language;
3. **frequency** — Hindi and English word frequencies differ clearly (the Hindi
   frequency is that of IndicXlit's Devanagari candidate, so *meeting* is
   scored as मीटिंग);
4. **context** — common homographs (*to*/तो, *me*/में) follow the sentence's
   majority language;
5. **LLM tie-break**, then a fallback to HINDI.

Only HINDI words (and ENGLISH words if the user opts in, U8) go to IndicXlit.

### 2.4 Reranking

IndicXlit is word-level, so it cannot use the sentence to choose between, say,
कि and की. The reranker (`likho/rerank.py`) rescores the top-5 candidates
with IndicXlit's rank, the candidate's Hindi word frequency and a bigram score
with the previous chosen word, using weights tuned on a separate development
set (α = 1.0, β = 2.0, γ = 0.5).

## 3. Implementation

*(team: add who built what, and anything about the development process the
course asks for.)*

- Python 3.10; FastAPI service and Gradio 6 UI; models load once at start-up
  and a model that fails to load disables only its stage.
- Every dependency pinned; the AI4Bharat stack needs specific `transformers`
  and `torch` versions (`docs/MODELS.md` §2.1).
- Two model-integration problems had to be solved: IndicXlit's checkpoint
  needs `argparse.Namespace` allowlisted under torch ≥ 2.6, and both Param-1's
  and IndicTrans2's remote code predate the current `transformers` cache API
  (Param-1 uses a hand-written decoding loop; IndicTrans2 keeps its own tuple
  cache) — `docs/MODELS.md`.
- Hardware: an Apple M4 with 16 GB RAM. Holding Param-1 and IndicTrans2 in
  memory together made the evaluation swap, so it runs all Param-1 work first
  and translates afterwards, with identical outputs (`docs/EVALUATION.md` §7).
- The UI has a Write tab and a Demo tab of one-click showcase examples
  (screenshots in `docs/images/`).

## 4. Evaluation

### 4.1 Test sets

| Set | Size | Used for |
|---|---|---|
| Aksharantar + Dakshina sample | 50 words | word-level transliteration |
| Hand-built code-mixed sentences | 50 sentences, 7 categories | tagging, reranking, rewrite, translation, end to end |
| Router inputs | 30 | routing |
| Development set | separate | prompt and weight tuning only |

The 50 sentences cover plain Hinglish, loanwords, named entities, ambiguous
words, Hinglish verbs (*manage karna*), long sentences and Devanagari input,
each labelled with gold Devanagari, per-word kinds and an English reference
(`docs/EVALUATION.md` §1).

### 4.2 Metrics

Transliteration top-1/top-5 accuracy and character error rate (CER); token
classification precision/recall/F1; router accuracy; rewrite meaning, register
and fluency scored 1–5 by an LLM judge; translation chrF++ and BLEU
(sacrebleu); per-stage latency. Every failing sentence gets one failure code
F1–F12 for the earliest stage that went wrong (`docs/EVALUATION.md` §4).

### 4.3 Results against the PRD targets

| What | Result | Target |
|---|---|---|
| Token classification macro-F1 (Hindi / English / name) | 94.4% | ≥ 85% ✅ |
| Router accuracy (30 inputs) | 100% | ≥ 90% ✅ |
| Rewrite meaning, LLM judge (1–5) | 4.28 | ≥ 4.0 ✅ |
| Translation chrF++ (gold Devanagari) | 75.0 | ≥ 45 ✅ |
| Transliteration CER, single words | 0.095 | ≤ 0.15 ✅ |
| Transliteration top-1 / top-5, single words | 58% / 78% | ≥ 70% / 90% ❌ |

### 4.4 Per stage

- **Tagging.** 98.4% of 496 tokens correct. All 64 English words were
  recognised (recall 100%); names had recall 78.6% (11/14). Most decisions
  came from the dictionary (242 tokens) and frequency (142) steps.
- **Reranking.** On the 380 Hindi words in the test sentences, top-1 accuracy
  rose from 89.7% to 93.4% (16 words fixed, 2 broken); the ceiling — gold
  anywhere in the top 5 — is 96.8%.
- **Routing.** Final routes were 100% correct with or without the LLM.
  Param-1's own proposal was right 46.7% of the time and unparseable 13.3% of
  the time; the rules overrode it in 53.3% of cases. Verification, not the LLM,
  is what makes routing reliable.
- **Rewrite.** 37/50 casual, 40/50 neutral and 39/50 formal rewrites were
  accepted; mean judge meaning 3.97, 4.35 and 4.51.
- **Translation.** chrF++ 75.0 / BLEU 53.0 from gold Devanagari; chrF++ 70.0 /
  BLEU 47.0 end to end.

### 4.5 Ablations

| Ablation | Without | With |
|---|---|---|
| 1. Reranking (Hindi words in sentences, top-1) | 89.7% | 93.4% |
| 2. IndicXlit dictionary rescoring (single words, top-1) | 46% | 58% |
| 3. LLM tie-break in tagging (macro-F1) | 94.4% | 94.4% |
| 4. Transliterate every word vs Likho (sentences exactly right) | 11/50 | 35/50 |

The LLM tie-break made no difference: on this test set the lexical steps
decide every word before it is reached. Ablation 4 is the project's central
result: transliterating every word — what the library offers out of the box —
keeps 0 of 64 English words and 0 of 14 names; Likho keeps 64 and 11.

### 4.6 End to end

19 of 50 sentences pass every stage (route, Devanagari, rewrite accepted and
meaning ≥ 4, translation chrF++ ≥ 45). Routing was right for all 50, the
Devanagari for 35 (37 counting equivalent spellings), and 33 rewrites were
accepted with meaning kept. All 4 Devanagari-input sentences passed; none of
the 4 long sentences got the Devanagari exactly right.

### 4.7 Latency

On the Apple M4 the median request took 6.6 s and the 95th percentile 16.2 s,
almost all of it in Param-1: the router (median 3.4 s) and the rewrite
(median 2.8 s). The PRD target is under 4 s at the 95th percentile on a Colab
T4, which this run does not measure. Disabling the router's LLM call
(`LLM_ROUTER=0`) removes the largest cost without changing any route on the
test set.

## 5. Failure analysis

Full analysis with case IDs: [`results/FAILURE_ANALYSIS.md`](../results/FAILURE_ANALYSIS.md).
Of the 31 failing sentences:

| Primary failure | Cases |
|---|---|
| F10 rewrite rejected or meaning changed | 14 |
| F8 right spelling in top 5, not chosen | 5 |
| F6 Hindi word left in Roman | 3 |
| F7 name transliterated | 3 |
| F2 schwa deletion | 2 |
| F11 translation below chrF++ 45 | 2 |
| other spelling | 2 |

Nothing crashed, no route was wrong, and no English word was wrongly
transliterated.

**1. The rewrite does not keep English words.** In all 19 accepted rewrites of
sentences containing English words or names, Param-1 converted at least one
(office → कार्यालय, phone → फोन); only 7 of 50 such words survived. This breaks
U5, a Must requirement, and the LLM judge does not penalise it, so 6 sentences
that "pass" have it. The rewrite's script check catches the most extreme
cases — those are 12 of the F10 rejections (e.g. cm_014). *Fix:* replace English
words and names with placeholders before rewriting and restore them after, so
the model cannot convert them.

**2. Homographs and spelling variants (F8).** The word-level model plus
reranker confuses की/कि (cm_046), कहा/खा (cm_005, which turns the translation
into *"Did you ask for food?"*) and वाहन/वहां (cm_043). Two cases are equally
valid spellings (हूँ/हूं, cm_035). *Fix:* context rules for frequent
homographs, e.g. *ki* after a verb of saying or knowing → कि.

**3. Words that exist in both languages (F6, F7).** *maine* (Hindi "I did")
matches the US state; *use* (Hindi "to him") is an English word (cm_040,
cm_045). Names that are Hindi words — *Priya* ("dear"), *Asha* ("hope") — are
transliterated when they start the sentence (cm_021, cm_026). *Fix:* a
Hinglish-first list for frequent collisions, and a names list for capitalised
sentence-initial words.

Two primary F11 failures are correct translations scored against a single
reference (cm_002, cm_031); IndicTrans2 also lowercases some outputs when the
input still contains Roman words (cm_022).

## 6. Limitations

- **Small test set.** 50 sentences show the failure patterns but cannot rank
  close categories reliably.
- **Self-judging.** The rewrite is judged by the same model that wrote it. A
  20-case human rubric for two raters is prepared
  (`results/human_rubric.csv`) *(team: score it and report agreement)*.
- **Single annotation.** The gold Devanagari was not double-labelled; some
  "failures" are annotation choices (cm_010, cm_024, cm_035).
- **Latency** misses the PRD target on the hardware tested, and was not
  measured on a T4.
- **Licence.** Param-1's card restricts it to research use, so Likho is not
  deployable as a product.

## 7. Future work

The three fixes in §5, in that order — the placeholder rewrite alone addresses
the largest failure and a Must requirement. Then: a second reference
translation per test sentence, the human rubric, a smaller or quantised LLM
for latency, and measuring on a T4.

## 8. Reproducibility

```bash
make setup-models && cp .env.example .env   # MOCK_MODELS=0, HF_TOKEN set
MOCK_MODELS=0 make eval
.venv/bin/python -m eval.run_eval --compare eval/results/20261105-m4_metrics.json
```

The committed run reproduced exactly from commit `4553cc6` (every metric
identical; only timings differ). `make results RUN=<id>` regenerates the
report folder from a run.

## References

*(team: format these in the course's citation style.)*

- AI4Bharat. IndicXlit / `ai4bharat-transliteration` — Aksharantar.
- AI4Bharat. IndicTrans2, `ai4bharat/indictrans2-indic-en-1B`.
- BharatGen. Param-1 2.9B Instruct, `bharatgenai/Param-1-2.9B-Instruct`.
- Dakshina dataset (Google Research).
- sacrebleu (chrF++, BLEU); wordfreq (word frequencies); IndicCorp V2.
