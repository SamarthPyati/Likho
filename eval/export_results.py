"""Build the top-level `results/` folder from one committed evaluation run.

    python -m eval.export_results 20261105-m4

`eval/run_eval.py` writes raw output under `eval/results/` and `eval/human/`.
This turns one run into a folder someone writing the report can use without
running anything: readable tables, one CSV per table (for charts), example
outputs, and the raw files themselves with checksums. Every number here is
read from the run's metrics JSON or cases CSV — nothing is typed by hand
(CLAUDE.md rule 6) — so re-running this command reproduces the folder exactly.
"""

import argparse
import csv
import hashlib
import json
import re
import shutil
import sys
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

from eval.failures import CODES
from eval.full import RESULTS

ROOT = Path(__file__).resolve().parent.parent
HUMAN = Path(__file__).parent / "human"
DATA = Path(__file__).parent / "data"
OUT = ROOT / "results"

KINDS = ["hindi", "english", "named_entity", "devanagari", "punct", "numeric"]
STAGE_ORDER = ["route", "tokenize", "transliterate", "rerank", "rewrite", "translate", "total"]
EXAMPLES_PER_CATEGORY = 2


def pct(x: float | None) -> str:
    return "—" if x is None else f"{100 * x:.1f}%"


def num(x: float | None, digits: int = 2) -> str:
    return "—" if x is None else f"{x:.{digits}f}"


def ratio(correct: int, n: int) -> str:
    return f"{correct}/{n} ({pct(correct / n)})" if n else "—"


class Table:
    """One results table, written both as Markdown (for reading) and CSV (for charts)."""

    def __init__(self, name: str, title: str, header: list[str], note: str = "") -> None:
        self.name, self.title, self.header, self.note = name, title, header, note
        self.rows: list[list] = []

    def add(self, *cells: object) -> None:
        self.rows.append(list(cells))

    def markdown(self) -> str:
        lines = [f"### {self.title}", ""]
        if self.note:
            lines += [self.note, ""]
        lines.append("| " + " | ".join(self.header) + " |")
        lines.append("|" + "---|" * len(self.header))
        lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in self.rows]
        lines += ["", f"CSV: [`tables/{self.name}.csv`](tables/{self.name}.csv)", ""]
        return "\n".join(lines)

    def write_csv(self, folder: Path) -> None:
        with (folder / f"{self.name}.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, lineterminator="\n")
            writer.writerow(self.header)
            writer.writerows(self.rows)


def build_tables(m: dict) -> list[Table]:
    tables = []

    t = Table(
        "prd_targets",
        "PRD §7 targets",
        ["criterion", "value", "target", "met"],
        "The success criteria from `docs/PRD.md` §7, checked against this run.",
    )
    for row in m["prd_targets"]:
        value = row["value"]
        shown = num(value, 3) if isinstance(value, float) else value
        t.add(row["criterion"], shown, row["target"], "yes" if row["met"] else "**no**")
    tables.append(t)

    t = Table(
        "transliteration",
        "Transliteration (IndicXlit, word level, Aksharantar + Dakshina sample)",
        [
            "subset",
            "n",
            "top-1 rescore on",
            "top-5 rescore on",
            "CER rescore on",
            "top-1 rescore off",
            "top-5 rescore off",
            "CER rescore off",
        ],
        "Ablation 2: IndicXlit's built-in dictionary rescoring on vs off. "
        "CER is character error rate of the top-1 (lower is better).",
    )
    on, off = m["transliterate"]["rescore_on"], m["transliterate"]["rescore_off"]
    for subset in on:
        a, b = on[subset], off[subset]
        t.add(
            subset,
            a["n"],
            pct(a["top1"]),
            pct(a["top5"]),
            num(a["cer"], 3),
            pct(b["top1"]),
            pct(b["top5"]),
            num(b["cer"], 3),
        )
    tables.append(t)

    t = Table(
        "token_classification",
        "Token classification (which Roman words to transliterate)",
        [
            "kind",
            "precision (LLM tie-break on)",
            "recall (on)",
            "F1 (on)",
            "support",
            "F1 (LLM tie-break off)",
        ],
        "Ablation 3: Param-1 as a tie-breaker for low-confidence tokens, on vs off.",
    )
    on, off = m["tokenize"]["llm_tiebreak_on"], m["tokenize"]["llm_tiebreak_off"]
    for kind in KINDS:
        a, b = on["per_kind"][kind], off["per_kind"][kind]
        if a["support"]:
            t.add(
                kind,
                pct(a["precision"]),
                pct(a["recall"]),
                pct(a["f1"]),
                a["support"],
                pct(b["f1"]),
            )
    t.add(
        "**accuracy (all tokens)**",
        "",
        "",
        pct(on["accuracy"]),
        on["n_tokens"],
        pct(off["accuracy"]),
    )
    t.add(
        "**macro-F1 (hindi, english, NE)**",
        "",
        "",
        pct(on["macro_f1_hindi_english_ne"]),
        "",
        pct(off["macro_f1_hindi_english_ne"]),
    )
    tables.append(t)

    t = Table(
        "token_confusion",
        "Token confusion matrix (LLM tie-break on; rows = gold, columns = predicted)",
        ["gold \\ predicted", *KINDS],
    )
    confusion = on["confusion"]
    for gold in KINDS:
        if sum(confusion[gold].values()):
            t.add(gold, *(confusion[gold][p] for p in KINDS))
    tables.append(t)

    t = Table(
        "token_signals",
        "What decided each token's kind (LLM tie-break on)",
        ["signal", "tokens"],
    )
    for signal, count in sorted(on["signals"].items(), key=lambda kv: -kv[1]):
        t.add(signal, count)
    tables.append(t)

    r = m["rerank"]
    t = Table(
        "rerank",
        "Candidate reranking",
        ["measure", "value"],
        "Ablation 1: reranking IndicXlit's top-k with the lexicon and context, on vs off. "
        f"Weights α={r['weights']['alpha']}, β={r['weights']['beta']}, "
        f"γ={r['weights']['gamma']}.",
    )
    s = r["test_sentences"]
    t.add("Hindi tokens in test sentences", s["n_hindi_tokens"])
    t.add("top-1 accuracy, rerank off", pct(s["top1_off"]))
    t.add("top-1 accuracy, rerank on", pct(s["top1_on"]))
    t.add("top-1 accuracy (lenient), rerank off", pct(s["top1_off_lenient"]))
    t.add("top-1 accuracy (lenient), rerank on", pct(s["top1_on_lenient"]))
    t.add("ceiling: gold anywhere in top-k", pct(s["topk_ceiling"]))
    t.add("tokens fixed by rerank", s["fixed_by_rerank"])
    t.add("tokens broken by rerank", s["broken_by_rerank"])
    t.add("gold in top-k but not chosen (F8)", s["f8_in_topk_not_chosen"])
    t.add("gold not in top-k at all", s["not_in_topk"])
    w = r["aksharantar_words"]
    t.add(
        f"Aksharantar words (n={w['n']}) top-1 off → on",
        f"{pct(w['top1_off'])} → {pct(w['top1_on'])}",
    )
    tables.append(t)

    t = Table(
        "routing",
        "Router (agentic component)",
        ["measure", "LLM router on", "rules only"],
        "The LLM proposes an intent; rules verify it and override on disagreement "
        "(`docs/ARCHITECTURE.md` §5a).",
    )
    a, b = m["route"]["llm_on"], m["route"]["llm_off"]
    t.add("cases", a["n"], b["n"])
    t.add("final stage-plan accuracy", pct(a["accuracy"]), pct(b["accuracy"]))
    t.add("LLM proposal accuracy", pct(a["llm_proposal_accuracy"]), "—")
    t.add("LLM output unparseable", pct(a["llm_unparsed_rate"]), "—")
    t.add("rules decided (fallback rate)", pct(a["fallback_rate"]), pct(b["fallback_rate"]))
    t.add("routing errors", len(a["errors"]), len(b["errors"]))
    tables.append(t)

    w = m["rewrite"]
    t = Table(
        "rewrite",
        f"Rewrite by register (Param-1 on gold Devanagari, n={w['n_cases']} per register)",
        [
            "register",
            "accepted",
            "rejected",
            "length ratio",
            "judge meaning",
            "judge register",
            "judge fluency",
            "judge unparsed",
        ],
        "Judge scores are 1–5 from Param-1 acting as judge — the same model family that "
        "wrote the rewrite, so treat them as indicative only; the human rubric is the "
        "real measure. When a rewrite is rejected the app shows no rewrite and keeps "
        "the Devanagari (FR7).",
    )
    for reg, d in w["registers"].items():
        j = d["judge_mean"]
        t.add(
            reg,
            d["accepted"],
            d["rejected"],
            num(d["mean_length_ratio"]),
            num(j["meaning"]),
            num(j["register"]),
            num(j["fluency"]),
            pct(d["judge_fallback_rate"]),
        )
    tables.append(t)

    t = Table(
        "rewrite_rejections",
        "Why rewrites were rejected",
        ["register", "reason", "count"],
        "Reasons from the rule check in `likho/rewrite.py` (`validate`), grouped: the "
        "exact length ratios are in `metrics.json`.",
    )
    for reg, d in w["registers"].items():
        grouped: Counter = Counter()
        for reason, count in d["rejected_by_reason"].items():
            reason = reason.removeprefix("rewrite rejected: ")
            grouped[re.sub(r"length ratio [\d.]+ ", "length ratio ", reason)] += count
        for reason, count in grouped.most_common():
            t.add(reg, reason, count)
    tables.append(t)

    tr, e2e_tr = m["translate"], m["end_to_end"].get("translation", {})
    t = Table(
        "translation",
        "Translation (IndicTrans2 indic→en 1B)",
        ["input", "n", "chrF++", "BLEU"],
        "chrF++ and BLEU from sacrebleu (0–100). *Gold Devanagari* isolates the "
        "translation model; *end to end* translates the pipeline's own rewrite.",
    )
    if tr.get("available"):
        t.add("gold Devanagari", tr["n"], num(tr["chrf_pp"], 1), num(tr["bleu"], 1))
    else:
        t.add("gold Devanagari", "not available in this run", "", "")
    if e2e_tr:
        t.add("end to end", e2e_tr["n"], num(e2e_tr["chrf_pp"], 1), num(e2e_tr["bleu"], 1))
    tables.append(t)

    a = m["ablation_translit_sentence"]
    t = Table(
        "ablation_sentence_baseline",
        "Ablation 4: naive whole-sentence transliteration vs the Likho pipeline",
        ["measure", "translit_sentence baseline", "Likho pipeline"],
        "The baseline transliterates every word, which is what the IndicXlit library "
        "offers out of the box. The gap on English words and names is the reason the "
        "token classifier exists.",
    )
    base, pipe = a["translit_sentence"], a["pipeline"]
    t.add(
        "sentences exactly right",
        ratio(base["sentence_exact"], a["n"]),
        ratio(pipe["sentence_exact"], a["n"]),
    )
    t.add(
        "sentences right (lenient)",
        ratio(base["sentence_exact_lenient"], a["n"]),
        ratio(pipe["sentence_exact_lenient"], a["n"]),
    )
    t.add("character error rate", num(base["cer"], 3), num(pipe["cer"], 3))
    for group in ["all", "hindi", "english", "named_entity"]:
        x, y = base[f"tokens_{group}"], pipe[f"tokens_{group}"]
        t.add(f"tokens right: {group}", ratio(x["correct"], x["n"]), ratio(y["correct"], y["n"]))
    tables.append(t)

    e = m["end_to_end"]
    t = Table(
        "end_to_end",
        f"End to end on the {e['n']} code-mixed test cases (register: {e['register']})",
        ["measure", "cases"],
        "A case *passes* only if every stage is right: route, native script, rewrite "
        "accepted and judged meaning-preserving, and translation chrF++ ≥ 45.",
    )
    t.add("passed every stage", ratio(e["passed"], e["n"]))
    t.add("route correct", ratio(e["route_ok"], e["n"]))
    t.add("native script exactly right", ratio(e["native_ok"], e["n"]))
    t.add("native script right (lenient)", ratio(e["native_ok_lenient"], e["n"]))
    t.add("native script CER", num(e["native_cer"], 3))
    t.add("rewrite accepted and meaning kept", ratio(e["rewrite_ok"], e["n"]))
    t.add("translation evaluated", ratio(e["translation_evaluated"], e["n"]))
    t.add("translation chrF++ ≥ 45", ratio(e["translation_ok"], e["translation_evaluated"]))
    tables.append(t)

    t = Table(
        "end_to_end_by_category",
        "End to end by test category",
        ["category", "n", "native script right", "passed every stage"],
    )
    for cat, d in e["by_category"].items():
        t.add(cat, d["n"], ratio(d["native_ok"], d["n"]), ratio(d["passed"], d["n"]))
    tables.append(t)

    t = Table(
        "failure_buckets",
        "Failure taxonomy F1–F12 (`docs/EVALUATION.md` §4)",
        ["code", "meaning", "cases (primary)", "occurrences (any)"],
        "*Primary* is the earliest stage that failed in a case — each failing case is "
        "counted once. *Any* counts every occurrence, including after an earlier failure "
        "(a case with two F8 words adds two).",
    )
    for code, meaning in CODES.items():
        t.add(code, meaning, e["failures_primary"].get(code, 0), e["failures_any"].get(code, 0))
    tables.append(t)

    lat = m["latency"]["e2e_per_stage"]
    hardware = m["meta"]["hardware"]
    t = Table(
        "latency",
        "Latency per stage (end-to-end run)",
        ["stage", "requests", "p50 ms", "p95 ms"],
        f"Measured on {hardware['platform']}, LLM on `{hardware['llm_device']}`. "
        "PRD NFR1 targets < 4 s per request on a Colab T4; this machine is not a T4.",
    )
    for stage in STAGE_ORDER:
        if stage in lat:
            d = lat[stage]
            t.add(stage, d["n"], num(d["p50_ms"], 0), num(d["p95_ms"], 0))
    tables.append(t)

    return tables


_LATIN_WORD = re.compile(r"[A-Za-z]+")


def english_kept_table(rows: list[dict]) -> Table:
    """U5 ("my English words stay English"), which no failure code measures.

    For each accepted end-to-end rewrite, the Roman words of the gold
    Devanagari — the sentence's real English words and names — are looked up
    in the rewrite. English-only cases (gold equals input) are excluded: there
    the whole rewrite is English by design.
    """
    cases = words = kept = changed = changed_passed = 0
    for row in rows:
        gold = [w.lower() for w in _LATIN_WORD.findall(row["expected_native"])]
        if not row["rewritten"] or not gold or row["expected_native"] == row["input"]:
            continue
        out = {w.lower() for w in _LATIN_WORD.findall(row["rewritten"])}
        cases += 1
        words += len(gold)
        kept += sum(w in out for w in gold)
        if not set(gold) <= out:
            changed += 1
            changed_passed += row["passed"] == "True"
    t = Table(
        "rewrite_keeps_english",
        "Does the rewrite keep English words and names in Roman? (PRD U5)",
        ["measure", "value"],
        "No failure code counts this. Computed from `cases.csv`: the Roman words of "
        "each case's gold Devanagari, looked up in its accepted neutral rewrite.",
    )
    t.add("accepted rewrites of sentences with English words or names", cases)
    t.add("rewrites that changed at least one of them", ratio(changed, cases))
    t.add("…of which the case still passed every stage", changed_passed)
    t.add("English words and names kept, overall", ratio(kept, words))
    return t


def examples_markdown(rows: list[dict]) -> str:
    """Real outputs, a passing and a failing case per category where they exist."""
    by_cat: dict[str, list[dict]] = {}
    for row in rows:
        by_cat.setdefault(row["category"], []).append(row)
    out = [
        "# Example outputs",
        "",
        "Real pipeline output from this run, copied from [`cases.csv`](cases.csv). "
        "For each test category: up to one case that passed every stage and one that did "
        "not, with its failure code (meanings in the failure table of [`README.md`](README.md)). "
        "Every case is in `cases.csv`.",
        "",
    ]
    for cat, cases in by_cat.items():
        out += [f"## {cat}", ""]
        picks = [c for c in cases if c["passed"] == "True"][:1]
        picks += [c for c in cases if c["passed"] != "True"][: EXAMPLES_PER_CATEGORY - len(picks)]
        for c in picks:
            verdict = "passed" if c["passed"] == "True" else f"failed — **{c['failure']}**"
            out += [
                f"**{c['id']}** ({c['difficulty']}, {verdict})",
                "",
                "| | |",
                "|---|---|",
                f"| input | {c['input']} |",
                f"| expected Devanagari | {c['expected_native']} |",
                f"| Likho Devanagari | {c['native_script']} |",
                f"| rewrite (neutral) | {c['rewritten'] or '—'}"
                + (f" *(rejected: {c['rewrite_error']})*" if c["rewrite_error"] else "")
                + " |",
                f"| expected English | {c['expected_english']} |",
                f"| Likho English | {c['english'] or '—'} |",
                "",
            ]
            if c["detail"]:
                out += [f"Detail: `{c['detail']}`", ""]
    return "\n".join(out)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def readme(run_id: str, m: dict, tables: list[Table], files: dict[str, Path]) -> str:
    meta = m["meta"]
    hw, models, settings = meta["hardware"], meta["models"], meta["settings"]
    by_name = {t.name: t for t in tables}

    def section(*names: str) -> str:
        return "\n".join(by_name[n].markdown() for n in names)

    file_rows = "\n".join(
        f"| [`{name}`]({name}) | {desc} |"
        for name, desc in [
            ("metrics.json", "Every metric from the run (the source for all tables here)"),
            ("cases.csv", "One row per test case: every stage's output, pass/fail, failure codes"),
            ("examples.md", "Selected real outputs per category, for the report"),
            ("tables/", "Each table below as a CSV, for charts"),
            ("human_rubric.csv", "Blank sheet for the two human raters (see below)"),
            ("test_sets/", "The exact test data this run used"),
            ("SHA256SUMS", "Checksums tying these files to the raw run in `eval/results/`"),
        ]
    )
    human_line = (
        "Scored: see [`human.json`](human.json)."
        if "human.json" in files
        else "**Not scored yet.** Two team members should fill in `human_rubric.csv` "
        "independently (one copy each), then run `python -m eval.human <sheet.csv>` "
        "for each; it writes `eval/results/<run>_human.json` with means, inter-rater "
        "agreement and the judge-vs-human gap. Re-run this export afterwards."
    )
    settings_line = (
        f"LLM router {settings['llm_router']}, LLM tie-break {settings['llm_tiebreak']}, "
        f"seed {settings['llm_seed']}, end-to-end register {settings['e2e_register']}"
    )
    n_cases, n_router = m["end_to_end"]["n"], m["route"]["llm_on"]["n"]
    n_words = m["transliterate"]["rescore_on"]["overall"]["n"]
    return f"""# Likho — evaluation results

Results of evaluation run **`{run_id}`**, for the project report. Everything in
this folder is generated from that run by `python -m eval.export_results {run_id}`;
no number here was typed by hand. Cite the run ID when quoting a number.

**Failure analysis:** [`FAILURE_ANALYSIS.md`](FAILURE_ANALYSIS.md) — what went wrong,
by failure type, with worked examples and fixes.

## Files

| File | What it is |
|---|---|
{file_rows}

## Run details

| | |
|---|---|
| run ID | `{run_id}` |
| date (UTC) | {meta['date_utc']} |
| code version | `{meta['git_commit']}` |
| real models | {'no — MOCK RUN, numbers are meaningless' if meta['mock_models'] else 'yes'} |
| machine | {hw['platform']}, Python {hw['python']}, torch {hw['torch']} |
| LLM device | {hw['llm_device']} |
| transliteration | IndicXlit (`ai4bharat-transliteration`), top-5 |
| rewrite / router / judge LLM | `{models['llm']}` |
| translation | `{models['translate']}` |
| settings | {settings_line} |

### Reproduce

```bash
make setup-models
MOCK_MODELS=0 .venv/bin/python -m eval.run_eval --run-id <new-id>
.venv/bin/python -m eval.run_eval --compare {run_id} <new-id>
```

`--compare` diffs two runs' metrics, ignoring timestamps and latency.

## Test sets

| set | size | used for |
|---|---|---|
| `codemixed_cases.yaml` | {n_cases} sentences | tagging, rerank, rewrite, translation, end to end |
| `router_cases.yaml` | {n_router} inputs | router accuracy |
| `aksharantar_sample.tsv` | {n_words} words | word-level transliteration |
| `dev_cases.yaml` | — | prompt and weight tuning only; **not** used for these numbers |

Copies are in [`test_sets/`](test_sets/). How the code-mixed set was built, and the
labelling protocol, are in `docs/EVALUATION.md` §2.

## Headline

{section('prd_targets', 'end_to_end')}
## Per stage

{section('transliteration', 'token_classification', 'token_confusion', 'token_signals',
         'rerank', 'routing', 'rewrite', 'rewrite_rejections', 'rewrite_keeps_english',
         'translation')}
## Ablations

The four ablations from `docs/EVALUATION.md` §5:

1. Rerank on vs off — table *Candidate reranking*
2. IndicXlit rescoring on vs off — table *Transliteration*
3. LLM tie-break on vs off — table *Token classification*
4. Whole-sentence baseline vs pipeline — below

The router's LLM-on vs rules-only comparison is in table *Router*.

{section('ablation_sentence_baseline')}
## Failures

{section('failure_buckets', 'end_to_end_by_category')}
## Latency

{section('latency')}
## Human evaluation

{human_line}

## Caveats for the report

- The code-mixed set has {m['end_to_end']['n']} cases: enough to find failure patterns,
  too few for fine-grained percentages. Quote counts alongside percentages.
- The rewrite judge is Param-1 scoring Param-1; its scores cluster and should not
  be the main evidence for rewrite quality. Use the human rubric.
- Latency was measured on {hw['platform']}, not the Colab T4 the PRD targets.
"""


def export(run_id: str, out: Path = OUT, results: Path = RESULTS, human: Path = HUMAN) -> Path:
    metrics_path = results / f"{run_id}_metrics.json"
    cases_path = results / f"{run_id}_cases.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"no run {run_id!r}: {metrics_path} is missing")
    m = json.loads(metrics_path.read_text(encoding="utf-8"))
    with cases_path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    analysis = out / "FAILURE_ANALYSIS.md"
    kept = analysis.read_text(encoding="utf-8") if analysis.exists() else None
    if out.exists():
        shutil.rmtree(out)
    (out / "tables").mkdir(parents=True)
    (out / "test_sets").mkdir()

    files = {"metrics.json": metrics_path, "cases.csv": cases_path}
    rubric = human / f"{run_id}_rubric.csv"
    if rubric.exists():
        files["human_rubric.csv"] = rubric
    human_json = results / f"{run_id}_human.json"
    if human_json.exists():
        files["human.json"] = human_json
    for name, src in files.items():
        shutil.copyfile(src, out / name)
    for src in sorted(DATA.iterdir()):
        if src.suffix in {".yaml", ".tsv"}:
            shutil.copyfile(src, out / "test_sets" / src.name)

    tables = build_tables(m)
    tables.append(english_kept_table(rows))
    for table in tables:
        table.write_csv(out / "tables")
    (out / "examples.md").write_text(examples_markdown(rows), encoding="utf-8")
    (out / "README.md").write_text(readme(run_id, m, tables, files), encoding="utf-8")
    if kept is not None:  # hand-written analysis survives a re-export
        analysis.write_text(kept, encoding="utf-8")

    sums = sorted(
        f"{sha256(src)}  eval/{src.relative_to(src.parents[1])}" for src in files.values()
    )
    (out / "SHA256SUMS").write_text("\n".join(sums) + "\n", encoding="utf-8")
    return out


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run_id", help="run ID, e.g. 20261105-m4")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(list(argv) if argv is not None else None)
    out = export(args.run_id, args.out)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
