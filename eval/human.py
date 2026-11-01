"""Score the human rubric against the LLM judge (docs/EVALUATION.md §2.4).

    python -m eval.human eval/human/<run>_rubric.csv

The sheet comes from `python -m eval.run_eval` and is filled in by two team
members independently, 1–5 per axis, without seeing the judge's scores. This
joins the judge's scores back in from eval/results/<run>_metrics.json and
writes eval/results/<run>_human.json: per-axis means, inter-rater agreement,
and judge-vs-human correlation and leniency (judge mean − human mean).
"""

import argparse
import csv
import json
import statistics
from pathlib import Path

from eval.judge import AXES

RESULTS = Path(__file__).parent / "results"
RATERS = ("rater_a", "rater_b")


def load_sheet(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    missing = [
        f"{r['case']}:{rater}_{axis}"
        for r in rows
        for rater in RATERS
        for axis in AXES
        if not r[f"{rater}_{axis}"].strip()
    ]
    if missing:
        raise ValueError(f"{len(missing)} empty score cells, e.g. {missing[:3]}")
    return rows


def judge_scores(metrics: dict) -> dict[tuple[str, str], dict]:
    return {(c["case"], c["register"]): c for c in metrics["rewrite"]["cases"] if c["rewritten"]}


def _correlation(xs: list[float], ys: list[float]) -> float | None:
    if len(set(xs)) < 2 or len(set(ys)) < 2:
        return None  # undefined for a constant series, which a lenient judge produces
    return round(statistics.correlation(xs, ys), 3)


def score(rows: list[dict], judged: dict[tuple[str, str], dict]) -> dict:
    out: dict = {"n": len(rows), "axes": {}}
    for axis in AXES:
        a = [float(r[f"rater_a_{axis}"]) for r in rows]
        b = [float(r[f"rater_b_{axis}"]) for r in rows]
        human = [(x + y) / 2 for x, y in zip(a, b, strict=True)]
        llm = [float(judged[(r["case"], r["register"])][f"judge_{axis}"]) for r in rows]
        out["axes"][axis] = {
            "rater_a_mean": round(statistics.fmean(a), 2),
            "rater_b_mean": round(statistics.fmean(b), 2),
            "human_mean": round(statistics.fmean(human), 2),
            "judge_mean": round(statistics.fmean(llm), 2),
            "judge_minus_human": round(statistics.fmean(llm) - statistics.fmean(human), 2),
            "raters_exact_agreement": sum(x == y for x, y in zip(a, b, strict=True)) / len(rows),
            "raters_within_one": sum(abs(x - y) <= 1 for x, y in zip(a, b, strict=True))
            / len(rows),
            "raters_pearson": _correlation(a, b),
            "judge_human_pearson": _correlation(llm, human),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Score the human rewrite rubric")
    parser.add_argument("sheet", type=Path)
    args = parser.parse_args()
    rows = load_sheet(args.sheet)
    run_id = rows[0]["run"]
    metrics = json.loads((RESULTS / f"{run_id}_metrics.json").read_text(encoding="utf-8"))
    result = score(rows, judge_scores(metrics))
    out = RESULTS / f"{run_id}_human.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for axis, r in result["axes"].items():
        print(  # CLI report output, not logging
            f"{axis:<9} human={r['human_mean']:.2f} judge={r['judge_mean']:.2f} "
            f"(judge−human {r['judge_minus_human']:+.2f})  "
            f"raters agree {r['raters_exact_agreement']:.0%} exact, "
            f"{r['raters_within_one']:.0%} ±1  r(judge,human)={r['judge_human_pearson']}"
        )
    print(f"→ {out}")


if __name__ == "__main__":
    main()
