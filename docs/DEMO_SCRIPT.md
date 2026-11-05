# Demo recording script (3–5 minutes)

The course asks for: the problem, a live demo, the architecture, the results,
and one honest failure. This script fits all five into about 4½ minutes.
Every number quoted is from run `20261105-m4` (`results/README.md`).

## Before recording

1. In `.env`: `MOCK_MODELS=0`, `LLM_ROUTER=0` (saves ~6–7 s per request with
   identical routes on the test set), `HF_TOKEN` set.
2. Start `make api` and wait for `/health` to say `"status":"ok"`
   (about a minute); then `make dev` and open http://127.0.0.1:7860.
3. **Warm up:** run one sentence and every Demo button once before recording.
   The first request after start-up is much slower (≈30 s or more on an M4).
4. Close other heavy apps (the models need ~8 GB of RAM).
5. Record with QuickTime (File → New Screen Recording) or OBS at 1080p;
   zoom the browser to 125% so Devanagari is readable.
6. Each request still takes several seconds — keep talking over the wait, or
   cut the waits in editing.

## Script

### 1. The problem — 0:00–0:35 (no app yet, or the empty Write tab)

> "Most of us type Hindi in Roman letters, mixed with English — like *kal ke
> meeting ke liye Rohit ko bata dena*. That's fine for chat, but not for
> anything formal. Converting it is harder than it looks: *kal* is Hindi and
> should become कल, but *meeting* is English and *Rohit* is a name — they
> should stay as typed. A plain transliterator converts everything. Likho
> decides word by word."

### 2. Live demo — 0:35–2:20 (Write tab, then Demo tab)

1. **Write tab.** Click the first example (*kal ke meeting…*), tick
   *Translate to English*, click **Likho**.
   > "Likho returns the Devanagari, a rewrite and the English. The coloured
   > words show its decision — orange is Hindi and was converted, blue English
   > and green names were kept."
2. **Route line.**
   > "Here it shows how it routed the text and how long each stage took."
3. **Click the word कल**, pick **काल**.
   > "If a word is wrong I click it and pick another spelling — the model's
   > top five. The Devanagari updates instantly; *Update rewrite* redoes the
   > rest."
4. **Demo tab → *Names and English words kept*** then ***English words in
   Devanagari***.
   > "The same sentence with the loanword switch on: now *meeting* is written
   > मीटिंग, but *Rohit* is still a name."
5. **Demo tab → *Plain English*.**
   > "English input skips transliteration altogether — the route says why."
6. **Compare tones** (default sentence).
   > "And one sentence in three tones, casual to formal."

### 3. Architecture — 2:20–3:05 (show the diagram in README.md)

> "Six stages, each its own module. A router — Param-1, a 2.9-billion
> parameter Indian LLM — proposes how to handle the input, and rules verify it.
> A tagger decides which words are Hindi; IndicXlit gives five candidates per
> Hindi word; a reranker picks one using word frequency and the previous word;
> Param-1 rewrites in the chosen tone, checked by rules; IndicTrans2
> translates. The UI only talks to an API, and every stage has a mock, so the
> whole thing runs and tests without a GPU."

### 4. Results — 3:05–3:45 (`results/README.md`, *PRD targets* and *Ablation 4*)

> "We built 50 code-mixed test sentences. Transliterating every word gets 11
> exactly right; Likho gets 35, and keeps all 64 English words where the
> naive approach keeps none. Routing was right for all 30 router inputs,
> translation scores chrF++ 75, and the tagger's macro-F1 is 94%. We miss one
> target: single-word transliteration top-1 is 58% against 70%."

### 5. One honest failure — 3:45–4:30 (`results/FAILURE_ANALYSIS.md`, F10)

> "Our biggest failure is in the rewrite. Whenever the sentence has English
> words, Param-1 converts some of them — *office* becomes कार्यालय — in all 19
> such rewrites, which breaks our own requirement to keep English words.
> The rule check catches only some of these. The fix we propose is to hide
> the English words behind placeholders before the rewrite and put them back
> afterwards, so the model can't touch them."
>
> "Everything here — the numbers, the failure analysis, the code — is in the
> repository, and the evaluation reproduces exactly. Thanks."

## Checklist

- [ ] All five parts covered, total 3–5 minutes
- [ ] Devanagari readable in the recording
- [ ] One failure shown honestly (part 5)
- [ ] Uploaded and linked from `README.md`
