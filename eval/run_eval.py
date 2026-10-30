"""Offline evaluation harness. Depends on `likho/` only (CLAUDE.md rule 4).

    python -m eval.run_eval --stage transliterate [--no-rescore] [--top-k 5]

Stages are added milestone by milestone; M6 adds the full run and the
metrics JSON + per-case CSV in eval/results/.
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from eval.metrics import corpus_cer, top_k_accuracy
from likho.config import settings
from likho.logging import get_logger
from likho.registry import get_xlit_engine
from likho.schemas import Token, TokenKind
from likho.transliterate import transliterate_tokens

logger = get_logger("eval.run_eval")

DATA = Path(__file__).parent / "data"


def load_aksharantar_sample(path: Path = DATA / "aksharantar_sample.tsv") -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def eval_transliterate(top_k: int, rescore: bool) -> dict[str, dict[str, float]]:
    """Top-1, top-k and CER on the Aksharantar sample, overall and per category."""
    rows = load_aksharantar_sample()
    tokens = [Token(surface=r["roman"], kind=TokenKind.HINDI) for r in rows]
    out = transliterate_tokens(tokens, top_k=top_k, engine=get_xlit_engine(rescore=rescore))

    groups: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        groups["overall"].append(i)
        groups[row["category"]].append(i)

    results: dict[str, dict[str, float]] = {}
    for name, idx in sorted(groups.items(), key=lambda kv: (kv[0] != "overall", kv[0])):
        candidates = [out[i].candidates for i in idx]
        references = [rows[i]["devanagari"] for i in idx]
        results[name] = {
            "n": len(idx),
            "top1": top_k_accuracy(candidates, references, k=1),
            f"top{top_k}": top_k_accuracy(candidates, references, k=top_k),
            "cer": corpus_cer([out[i].chosen or "" for i in idx], references),
        }
    return results


def print_table(title: str, results: dict[str, dict[str, float]]) -> None:
    metrics = [m for m in next(iter(results.values())) if m != "n"]
    logger.info(title)
    header = f"{'category':<10} {'n':>3}  " + "  ".join(f"{m:>6}" for m in metrics)
    lines = [header, "-" * len(header)]
    for name, row in results.items():
        cells = [f"{row[m]:6.3f}" if m == "cer" else f"{row[m]:6.0%}" for m in metrics]
        lines.append(f"{name:<10} {row['n']:>3}  " + "  ".join(cells))
    print("\n".join(lines))  # CLI report output, not logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Likho evaluation harness")
    parser.add_argument("--stage", choices=["transliterate"], required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--rescore", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    if settings.mock_models:
        logger.warning("MOCK_MODELS=1: numbers below come from stubs and must not be reported")

    if args.stage == "transliterate":
        results = eval_transliterate(args.top_k, args.rescore)
        print_table(f"transliterate (IndicXlit, rescore={args.rescore})", results)


if __name__ == "__main__":
    main()
