"""Offline evaluation harness. Depends on `likho/` only (CLAUDE.md rule 4).

    python -m eval.run_eval --stage transliterate [--no-rescore] [--top-k 5]
    python -m eval.run_eval --stage tokenize [--split dev] [--no-llm]
    python -m eval.run_eval --stage rerank [--split dev [--tune]]

Stages are added milestone by milestone; M6 adds the full run and the
metrics JSON + per-case CSV in eval/results/.
"""

import argparse
import csv
import itertools
from collections import Counter, defaultdict
from pathlib import Path

from eval.cases import load_cases
from eval.metrics import (
    classification_report,
    confusion_matrix,
    corpus_cer,
    fold_variants,
    macro_f1,
    normalize,
    top_k_accuracy,
)
from likho.config import settings
from likho.logging import get_logger
from likho.registry import get_xlit_engine
from likho.rerank import DEFAULT_WEIGHTS, OFF, RerankWeights, rerank_tokens
from likho.schemas import Token, TokenKind
from likho.tokenize import SIGNAL_BY_CONFIDENCE, split, tag
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


def _gold_hindi_tokens(split_name: str, top_k: int) -> list[tuple[str, Token, str]]:
    """(case id, transliterated token, gold Devanagari) for every gold-HINDI token,
    with sentence order kept so the reranker sees real context. Uses gold kinds
    so tagger errors do not leak into this stage's numbers."""
    out = []
    for case in load_cases(split_name):
        tokens = [Token(surface=t.surface, kind=t.kind) for t in case.expected_tokens]
        tokens = transliterate_tokens(tokens, top_k=top_k)
        for token, gold in zip(tokens, split(case.expected_native), strict=True):
            out.append((case.id, token, gold))
    return out


def _rerank_sentences(
    rows: list[tuple[str, Token, str]], weights: RerankWeights
) -> list[tuple[str, Token, str]]:
    out = []
    for case_id, group in itertools.groupby(rows, key=lambda r: r[0]):
        group = list(group)
        reranked = rerank_tokens([t for _, t, _ in group], weights)
        out += [(case_id, t, gold) for (_, _, gold), t in zip(group, reranked, strict=True)]
    return [r for r in out if r[1].kind is TokenKind.HINDI]


def _sentence_scores(rows: list[tuple[str, Token, str]], weights: RerankWeights) -> dict:
    on = _rerank_sentences(rows, weights)
    off = _rerank_sentences(rows, OFF)
    strict = [normalize(t.chosen or "") == normalize(g) for _, t, g in on]
    strict_off = [normalize(t.chosen or "") == normalize(g) for _, t, g in off]
    lenient = [fold_variants(t.chosen or "") == fold_variants(g) for _, t, g in on]
    lenient_off = [fold_variants(t.chosen or "") == fold_variants(g) for _, t, g in off]
    in_topk = [normalize(g) in {normalize(c) for c in t.candidates} for _, t, g in on]
    n = len(on)
    return {
        "n_hindi_tokens": n,
        "top1_off": sum(strict_off) / n,
        "top1_on": sum(strict) / n,
        "top1_off_lenient": sum(lenient_off) / n,
        "top1_on_lenient": sum(lenient) / n,
        "topk_ceiling": sum(in_topk) / n,
        "fixed_by_rerank": sum(s and not o for s, o in zip(strict, strict_off, strict=True)),
        "broken_by_rerank": sum(o and not s for s, o in zip(strict, strict_off, strict=True)),
        "f8_in_topk_not_chosen": sum(k and not s for k, s in zip(in_topk, strict, strict=True)),
        "not_in_topk": sum(not k for k in in_topk),
        "f8_cases": [
            {
                "case": cid,
                "surface": t.surface,
                "gold": g,
                "chosen": t.chosen,
                "rank": t.chosen_rank,
            }
            for (cid, t, g), s, k in zip(on, strict, in_topk, strict=True)
            if k and not s
        ],
    }


def eval_rerank(split_name: str, top_k: int, weights: RerankWeights) -> dict:
    rows = load_aksharantar_sample()
    words = transliterate_tokens(
        [Token(surface=r["roman"], kind=TokenKind.HINDI) for r in rows], top_k=top_k
    )
    refs = [r["devanagari"] for r in rows]

    def word_top1(w: RerankWeights) -> float:
        chosen = [[t.chosen or ""] for t in rerank_tokens(words, w)]  # each word alone
        return top_k_accuracy(chosen, refs, k=1)

    return {
        "weights": vars(weights),
        "aksharantar_words": {
            "n": len(rows),
            "top1_off": word_top1(OFF),
            "top1_on": word_top1(weights),
        },
        f"{split_name}_sentences": _sentence_scores(_gold_hindi_tokens(split_name, top_k), weights),
    }


TUNE_GRID = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]


def tune_rerank(top_k: int) -> list[tuple[float, float, RerankWeights]]:
    """Grid over β, γ (α fixed at 1 sets the scale) on the dev split only."""
    rows = _gold_hindi_tokens("dev", top_k)
    results = []
    for beta, gamma in itertools.product(TUNE_GRID, TUNE_GRID):
        weights = RerankWeights(alpha=1.0, beta=beta, gamma=gamma)
        s = _sentence_scores(rows, weights)
        results.append((s["top1_on"], s["top1_on_lenient"], weights))
    # best strict, then lenient, then the smallest weights (simplest model)
    results.sort(key=lambda r: (-r[0], -r[1], r[2].beta + r[2].gamma))
    return results


def print_rerank(result: dict) -> None:
    w = result["weights"]
    lines = [f"weights α={w['alpha']} β={w['beta']} γ={w['gamma']}", ""]
    a = result["aksharantar_words"]
    lines.append(
        f"aksharantar words (no context)  n={a['n']}  "
        f"top1 off={a['top1_off']:.0%} on={a['top1_on']:.0%}"
    )
    for key, s in result.items():
        if not key.endswith("_sentences"):
            continue
        lines += [
            f"{key} (gold HINDI tokens, n={s['n_hindi_tokens']})",
            f"  top1 strict   off={s['top1_off']:.0%}  on={s['top1_on']:.0%}",
            f"  top1 lenient  off={s['top1_off_lenient']:.0%}  on={s['top1_on_lenient']:.0%}",
            f"  top-k ceiling {s['topk_ceiling']:.0%}",
            f"  fixed by rerank={s['fixed_by_rerank']}  broken by rerank={s['broken_by_rerank']}",
            f"  F8 (in top-k, not chosen)={s['f8_in_topk_not_chosen']}  "
            f"not in top-k={s['not_in_topk']}",
        ]
        lines += [
            f"    {c['case']:<7} {c['surface']:<10} gold={c['gold']:<10} chose={c['chosen']}"
            f" (rank {c['rank']})"
            for c in s["f8_cases"]
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
    parser.add_argument("--stage", choices=["transliterate", "tokenize", "rerank"], required=True)
    parser.add_argument("--tune", action="store_true", help="rerank: grid-search on dev")
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
    elif args.stage == "rerank" and args.tune:
        if args.split != "dev":
            parser.error("--tune runs on --split dev only")
        for strict, lenient, w in tune_rerank(args.top_k)[:8]:
            print(f"β={w.beta:<5} γ={w.gamma:<5} strict={strict:.1%} lenient={lenient:.1%}")
    elif args.stage == "rerank":
        print_rerank(eval_rerank(args.split, args.top_k, DEFAULT_WEIGHTS))


if __name__ == "__main__":
    main()
