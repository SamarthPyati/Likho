# Failure analysis — run `20261105-m4`

Every number here comes from this run's [`metrics.json`](metrics.json) and
[`cases.csv`](cases.csv); every example cites its case ID so it can be checked
there. Failure codes are defined in `docs/EVALUATION.md` §4 and summarised in
the failure table of [`README.md`](README.md).

## Summary

19 of the 50 code-mixed test cases pass every stage. The other 31 are counted
once each, under the earliest stage that went wrong (*primary* code). *Any*
counts every occurrence of a code, including after an earlier failure — a case
with four F8 words (cm_046) adds four.

| code | failure | primary | any |
|---|---|---|---|
| F10 | Rewrite rejected, or judged to change the meaning | **14** | 17 |
| F8 | Right word was in the top 5, reranker chose another | **5** | 13 |
| F6 | Hindi word left in Roman (tagged English) | 3 | 5 |
| F7 | Name transliterated | 3 | 3 |
| F2 | Schwa deletion error | 2 | 2 |
| F11 | Translation chrF++ below 45 | 2 | 7 |
| other | Spelling outside F1–F4 (e.g. तोह, क्यूंकि) | 2 | 6 |
| F1, F3, F4, F5, F9, F12 | vowel length, retroflex, nukta, loanword over-transliteration, routing, crash | 0 | 0 |

What did not fail is worth stating too: routing was right for all 50 cases,
no English word was ever transliterated by mistake (F5 = 0, English tokens
64/64 kept), and nothing crashed (F12 = 0).

The three largest problems, in order: **the rewrite does not keep English
words (F10, and more cases than F10 counts)**, **reranking between valid
spellings and homographs (F8)**, and **Hindi words or names that are also
dictionary words in the other language (F6 + F7)**.

---

## F10 — Rewrite rejected or meaning changed (14 primary, 17 any)

### The underlying problem: Param-1 converts English words

User story U5 in `docs/PRD.md` is a *Must*: "have my English words stay
English". The rewrite breaks it almost every time. Of the 19 accepted rewrites
of sentences containing English words or names, **all 19 changed at least one
of them, and only 7 of 50 such words survived** (table *Does the rewrite keep
English words and names in Roman?*). Param-1 either writes them in Devanagari
(phone → फोन, scooter → स्कूटर) or translates them (message → संदेश,
office → कार्यालय).

No failure code measures this, and the judge does not penalise it, so
**6 cases that pass every stage still have it**: cm_011 (office →
कार्यालय), cm_012, cm_017, cm_023, cm_027, cm_041.

The rest of F10 is this same behaviour, caught in two different ways.

### Caught by the script check: 12 rejections

`likho/rewrite.py` `validate()` rejects a rewrite whose script differs from
the input's, decided by `_mostly_devanagari`: more Devanagari letters than
Latin letters. In a sentence with several English words, Latin letters
outnumber Devanagari, so the input counts as "Roman"; once Param-1 converts
the English words, the output is "mostly Devanagari" and is rejected.

12 of the 50 end-to-end rewrites were rejected this way
(`rewrite_error` = "script changed from Roman"); 10 are primary F10. In the
rewrite-only evaluation on gold Devanagari it was the largest rejection reason
in every register: 7 casual, 10 neutral, 9 formal (table *Why rewrites were
rejected*).

- **cm_014** `train दो घंटे late है` — 9 Latin letters vs 8 Devanagari, so
  "Roman". Rejected; the user sees no rewrite.
- **cm_037** `report को manage करना पड़ेगा` — the same with a Hinglish verb.
- **cm_034** `please call me when you reach` — pure English input; Param-1
  answered in Hindi. The check worked as designed here.

So the script check is acting as an accidental, partial U5 guard: it catches
the conversion only when English words outnumber Hindi ones. Loosening it
alone would let these 11 code-mixed rewrites through *with* the U5 violation.
In the app the rejection fails safely (FR7): the Devanagari is kept and still
translated — all 12 got a translation.

### Caught by the judge: 5 cases with meaning below 4

- **cm_020** `hostel का खाना बिल्कुल अच्छा नहीं है` → `खाना अच्छा नहीं है।`
  (judge 1). Real drift: *hostel* and *bilkul* were dropped. The length check
  did not catch it: the ratio is 0.53, just inside the 0.5 floor.
- **cm_042** `उसने मेरा message ignore कर दिया` →
  `उसने मेरा संदेश नजरअंदाज कर दिया।` (judge 3). The meaning is kept; the English
  words were translated.
- **cm_013** `मेरा phone charge नहीं हो रहा` → `मेरा फोन चार्ज नहीं हो रहा है।`
  (judge 3). A faithful rewrite apart from U5; the low score looks like judge
  noise (Param-1 judging its own output, see caveats). **cm_018** is similar.

### Fix

1. **Enforce U5 in code, not just in the prompt.** Before rewriting, replace
   each Roman token tagged ENGLISH or NAMED_ENTITY with a placeholder
   (`<E1>`, `<E2>`…), rewrite, then put the original words back; reject the
   rewrite if a placeholder is missing. The model then cannot convert what it
   cannot see. The token tags are already on the response.
2. **Then replace the letter-count script check** with one based on the route:
   reject Roman → Devanagari only for `Intent.ENGLISH` input (keeps cm_034
   rejected). With placeholders in place this no longer lets U5 violations
   through.
3. **Catch dropped content** with a word-overlap floor alongside the length
   ratio (cm_020).

---

## F8 — Right word in the top 5, not chosen (5 primary, 13 any)

The transliteration model produced the gold spelling, but the reranker ranked
another candidate first (13 tokens in 7 cases). The reranker still helps
overall: top-1 accuracy on Hindi tokens rises from 89.7% to 93.4% with it, fixing 16 tokens and breaking
2 (table *Candidate reranking*). The remaining F8 tokens are of two kinds.

### Two valid spellings (not really errors)

- **cm_010** `maa` → मां, gold माँ.
- **cm_035** `hoon` → हूँ, gold हूं.

Both forms are standard Hindi. The *lenient* score, which treats chandrabindu
and anusvara as equal, counts both cases correct — the reason native-script
accuracy is 35/50 strict but 37/50 lenient. For the report, these two are
annotation choices, not system failures.

### Homographs that change the meaning

- **cm_005** `kya aapne khana kha liya?` — `kha` → कहा ("said") instead of खा
  ("eat"). The translation becomes *"Did you ask for food?"* (chrF++ 13.2),
  so one wrong word ruins the output.
- **cm_043** `wahan` → वाहन ("vehicle") instead of वहां ("there").
- **ki** → की instead of कि (the conjunction "that") in **cm_043**, **cm_045**
  and twice in **cm_046**. The word-level model cannot tell them apart; it is
  the canonical F8 case described in `docs/MODELS.md` §2.3.

### Fix

Add **context rules for frequent homographs** to the reranker: `ki` after a
verb of saying/knowing (*pata chala ki*, *kehna hai ki*, *batana chahta hoon
ki*) → कि; `kha` before *liya / lo / raha* → खा. The γ context weight already
exists in `likho/rerank.py`; these are bigram entries. A small LLM tie-break
for the remaining low-margin cases is the general version, at a latency cost.

---

## F6 + F7 — Words that exist in both languages (6 primary, 8 any)

The token classifier is 98.4% accurate overall (macro-F1 94.4%), but most of
its remaining errors share one cause: the word is valid in both languages.

### F6: Hindi word tagged as English, left in Roman (3 primary, 5 any)

- **cm_006**, **cm_040** `maine` ("I did") — matches the US state *Maine* in
  the English lexicon. In cm_040 the translation then reads
  *"The maine form has been submitted"*.
- **cm_045** `use` ("to him/her" = उसे) — an English dictionary word; and
  `chai` (चाय) — tagged English.
- **cm_036** `jai` ("victory", जय) — kept in Roman.

### F7: Name transliterated (3 primary)

- **cm_021** `Priya` → प्रिय — also a Hindi word ("dear"). The translation
  then drops Priya: *"Tomorrow I am going to Delhi."*
- **cm_026** `Asha` → आशा — also a Hindi word ("hope").
- **cm_024** `Bengaluru` → बेंगलुरु — this is the correct Devanagari spelling;
  it counts as a failure only because the annotation keeps names in Roman.

### Fix

1. **Names at the start of a sentence.** The tagger already treats a
   capitalised word *mid-sentence* as a name (`likho/tokenize.py`, step 1).
   All three F7 cases are the **first word** of their sentence — *Priya*,
   *Bengaluru*, *Asha* — where capitalisation says nothing, so the dictionary
   step decides and finds a Hindi word. Add a list of common Indian given names
   and place names, checked for capitalised sentence-initial words before the
   dictionary; or send those words to the Param-1 tie-break.
2. **A Hinglish-first override list** for high-frequency collisions where the
   Hindi reading is overwhelmingly likely in a Hindi sentence: *maine, use,
   jai, chai, main*. The context signal exists; these words need a stronger
   prior.

---

## Smaller buckets

### F2 — Schwa deletion (2)

**cm_007**, **cm_016** `khatam` → खतम, gold खत्म. The model writes the
inherent vowel the speaker drops. One lexicon entry would fix both; the
general fix is preferring candidates found in the Hindi word-frequency list,
which the reranker's β weight already does when the gold form is in the top 5
— here it was not.

### other — Common Hinglish spellings (2 primary, 6 any)

All in the 4 long cases (native script right in 0 of 4): `toh` → तोह (should
be तो), `kyunki` → क्यूंकि (क्योंकि), `tumhe` → तुम्हे (तुम्हें). IndicXlit maps
the letters faithfully; the Hinglish spelling just does not match the standard
one. Long sentences have more words, so one such word fails the whole case.
A short **chat-spelling lexicon** (toh, kyunki, tumhe, nahi, hai…) applied
before the model would fix all six.

### F11 — Translation below chrF++ 45 (2 primary, 7 any)

Both primary cases are **correct translations scored against a single
reference**:

- **cm_002** *"When are you coming home?"* vs reference *"When will you come
  home?"* — chrF++ 42.8.
- **cm_031** *"No one is at home."* vs *"There is no one at home"* — 39.6.

The other five are knock-on effects of earlier failures (cm_005 from F8,
cm_040 from F6, cm_006 where the rewrite dropped the subject, cm_038 from a
rejected rewrite) or reference choices
(**cm_018** *"His scooter"* vs *"Their scooter"*: उसका has no gender).
Corpus-level translation quality is good: chrF++ 75.0 on gold Devanagari and
70.0 end to end. **Fix:** a second reference translation per case.

---

## Problems the taxonomy does not count

- **IndicTrans2 lowercases** some outputs whose source still contains Roman
  words: cm_022 *"tomorrow is rahul's birthday"*, cm_025 *"i need anand's
  number"*, cm_039 *"forward this mail to all"*. All three are cases whose
  rewrite was rejected, so the translation input was code-mixed. chrF++ is
  case-sensitive, so this also lowers scores. A post-processing step
  (capitalise the first letter; restore names from the input's tokens) fixes it.
- **Latency.** Median request time is 6.6 s, 95th percentile 16.2 s
  (table *Latency*). The LLM router (p50 3.4 s) and the rewrite (p50 2.8 s)
  account for nearly all of it. The PRD target is < 4 s on a Colab T4; this run
  was on an Apple M4, so it does not measure the target directly. Routing is
  100% correct with rules alone (table *Router*), so `LLM_ROUTER=0` removes
  ~3.4 s at no accuracy cost on this test set.
- **Transliteration on isolated words** misses the PRD target: top-1 58%
  (target 70%), top-5 78% (target 90%) on the 50-word Aksharantar/Dakshina
  sample. In sentences, with tagging and reranking, Hindi tokens reach 93.4%
  top-1 — so the pipeline compensates, but the model alone does not meet the
  target.

## Caveats

- **Small test set.** 50 cases: enough to find these patterns, not to rank
  close buckets. F6 and F7 tie at 3 primary cases each.
- **The judge is Param-1 judging Param-1.** Its scores cluster near 4–5 and
  include at least one doubtful low score (cm_013). The human rubric
  (`human_rubric.csv`, two raters) is the real measure of rewrite quality and
  has not been scored yet.
- **Latency varies between runs** on the same machine (other load, thermal
  state); a rerun for the reproducibility check matched every metric but not
  the timings. Treat the latency table as indicative.
- **Single annotation.** The gold Devanagari has not yet been double-labelled
  (`docs/TASKS.md` M2); cm_010, cm_035 and cm_024 show annotation choices
  counting as failures.
