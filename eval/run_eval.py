"""Offline evaluation harness. Depends on `likho/` only (CLAUDE.md rule 4).

    python -m eval.run_eval --stage transliterate [--no-rescore] [--top-k 5]
    python -m eval.run_eval --stage tokenize [--split dev] [--no-llm]

Stages are added milestone by milestone; M6 adds the full run and the
metrics JSON + per-case CSV in eval/results/.
"""

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

from eval.cases import load_cases
from eval.metrics import (
    classification_report,
    confusion_matrix,
    corpus_cer,
    macro_f1,
    top_k_accuracy,
)
from likho.config import settings
from likho.logging import get_logger
from likho.registry import get_xlit_engine
from likho.schemas import Token, TokenKind
from likho.tokenize import SIGNAL_BY_CONFIDENCE, tag
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


KINDS = [k.value for k in TokenKind]
HEADLINE_KINDS = [TokenKind.HINDI.value, TokenKind.ENGLISH.value, TokenKind.NAMED_ENTITY.value]


def eval_tokenize(split: str, use_llm: bool) -> dict:
    """Per-kind P/R/F1, confusion matrix, deciding-signal counts and every error."""
    gold, predicted, signals, errors = [], [], Counter(), []
    for case in load_cases(split):
        tokens = tag(case.input, use_llm=use_llm)
        if [t.surface for t in tokens] != [t.surface for t in case.expected_tokens]:
            raise ValueError(f"{case.id}: tokenizer split differs from annotation")
        for expected, token in zip(case.expected_tokens, tokens, strict=True):
            signal = SIGNAL_BY_CONFIDENCE[token.kind_confidence]
            gold.append(expected.kind.value)
            predicted.append(token.kind.value)
            signals[signal] += 1
            if expected.kind is not token.kind:
                errors.append(
                    {
                        "case": case.id,
                        "surface": token.surface,
                        "gold": expected.kind.value,
                        "predicted": token.kind.value,
                        "signal": signal,
                    }
                )
    report = classification_report(gold, predicted, KINDS)
    return {
        "n_tokens": len(gold),
        "accuracy": sum(g == p for g, p in zip(gold, predicted, strict=True)) / len(gold),
        "macro_f1_hindi_english_ne": macro_f1(report, HEADLINE_KINDS),
        "per_kind": report,
        "confusion": confusion_matrix(gold, predicted, KINDS),
        "signals": dict(signals.most_common()),
        "errors": errors,
    }


def print_tokenize(title: str, result: dict) -> None:
    logger.info(title)
    short = {k: k[:5] for k in KINDS}
    lines = [
        f"tokens={result['n_tokens']}  accuracy={result['accuracy']:.0%}  "
        f"macro-F1(hindi/english/NE)={result['macro_f1_hindi_english_ne']:.2f}",
        "",
        f"{'kind':<13} {'P':>5} {'R':>5} {'F1':>5} {'n':>4}",
    ]
    for kind, row in result["per_kind"].items():
        lines.append(
            f"{kind:<13} {row['precision']:5.2f} {row['recall']:5.2f} {row['f1']:5.2f} "
            f"{row['support']:4d}"
        )
    lines += ["", "confusion (rows = gold, cols = predicted)"]
    lines.append(" " * 13 + " ".join(f"{short[k]:>6}" for k in KINDS))
    for g, row in result["confusion"].items():
        lines.append(f"{g:<13}" + " ".join(f"{row[p]:6d}" for p in KINDS))
    lines += ["", "deciding signal: " + ", ".join(f"{s}={n}" for s, n in result["signals"].items())]
    lines += ["", f"errors ({len(result['errors'])}):"]
    lines += [
        f"  {e['case']:<7} {e['surface']:<12} gold={e['gold']:<13} "
        f"pred={e['predicted']:<13} via {e['signal']}"
        for e in result["errors"]
    ]
    print("\n".join(lines))  # CLI report output, not logging


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
    parser.add_argument("--stage", choices=["transliterate", "tokenize"], required=True)
    parser.add_argument("--split", choices=["test", "dev"], default="test")
    parser.add_argument("--llm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--rescore", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    if settings.mock_models:
        logger.warning("MOCK_MODELS=1: numbers below come from stubs and must not be reported")

    if args.stage == "transliterate":
        results = eval_transliterate(args.top_k, args.rescore)
        print_table(f"transliterate (IndicXlit, rescore={args.rescore})", results)
    elif args.stage == "tokenize":
        result = eval_tokenize(args.split, args.llm)
        print_tokenize(f"tokenize split={args.split} llm_tiebreak={args.llm}", result)


if __name__ == "__main__":
    main()
