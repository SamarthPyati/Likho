"""The full evaluation run behind `make eval` (docs/EVALUATION.md §2–§5).

Writes `eval/results/<run>_metrics.json` (every number, nested by stage) and
`eval/results/<run>_cases.csv` (one row per test case, with its F-code).

Everything except the `meta` and `latency` sections is deterministic on the
same hardware (greedy decoding, or sampling reseeded per call), so
`--compare` checks a new run against a committed one on those sections only.
"""

import csv
import json
import platform
import statistics
import subprocess
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from sacrebleu.metrics import BLEU, CHRF

from eval.cases import Case
from eval.failures import CODES, classify_token, primary
from eval.judge import AXES, JudgeScore, judge
from eval.metrics import corpus_cer, fold_variants, normalize
from likho.config import settings
from likho.errors import StageFailure
from likho.logging import get_logger
from likho.pipeline import add_translation, assist_untranslated
from likho.registry import get_llm, get_translator, get_xlit_engine, release_llm
from likho.router import STAGES, Intent
from likho.schemas import AssistRequest, AssistResponse, Register, Stage, TokenKind
from likho.tokenize import split
from likho.translate import translate_batch

logger = get_logger("eval.full")

RESULTS = Path(__file__).parent / "results"
E2E_REGISTER = Register.NEUTRAL
MEANING_PASS = 4  # judge meaning score at or above which a rewrite passes
CHRF_PASS = 45.0  # sentence chrF++ at or above which a translation passes (PRD §7 target)
NONDETERMINISTIC = ("meta", "latency")


def gold_intent(case: Case) -> Intent:
    """The intent the router should pick, from the gold token labels."""
    kinds = {t.kind for t in case.expected_tokens}
    roman = kinds & {TokenKind.HINDI, TokenKind.ENGLISH, TokenKind.NAMED_ENTITY}
    if TokenKind.DEVANAGARI in kinds:
        return Intent.MIXED_SCRIPT if roman else Intent.DEVANAGARI
    if TokenKind.HINDI in kinds:
        return Intent.ROMAN_HINGLISH
    return Intent.ENGLISH if roman else Intent.EMPTY


def run_case(case: Case) -> tuple[AssistResponse | None, str | None]:
    """Every stage but translation (see `end_to_end`); a crash is returned, not raised."""
    request = AssistRequest(text=case.input, register=E2E_REGISTER, translate=True)
    try:
        return assist_untranslated(request), None
    except Exception as exc:  # noqa: BLE001 — a crash is an F12 result, not a harness crash
        return None, type(exc).__name__


def judge_rewrite(case: Case, response: AssistResponse | None) -> JudgeScore | None:
    if response is None or response.rewritten is None:
        return None
    return judge(response.native_script or case.input, response.rewritten, E2E_REGISTER)


def evaluate_case(
    case: Case,
    response: AssistResponse | None,
    crash: str | None,
    score: JudgeScore | None = None,
) -> dict:
    """One CSV row: what each stage produced, whether it passed, and F-codes.
    `score` is the rewrite's judge score, judged here if not given."""
    row: dict = {
        "id": case.id,
        "category": case.category,
        "difficulty": case.difficulty,
        "input": case.input,
        "expected_native": case.expected_native,
        "expected_english": case.expected_english,
        "expected_route": gold_intent(case).value,
    }
    codes: list[str] = []
    details: list[str] = []
    if response is None:
        row.update(failures="F12", failure="F12", detail=f"crash: {crash}", passed=False)
        return row

    stage_errors = {e.stage: e.message for e in response.errors}
    row["stage_errors"] = "; ".join(f"{s.value}: {m}" for s, m in stage_errors.items())
    row["route_stages"] = " ".join(s.value for s in response.route.stages)
    row["route_source"] = response.route.source
    expected_stages = [*STAGES[gold_intent(case)], Stage.TRANSLATE]
    row["route_ok"] = response.route.stages == expected_stages
    if not row["route_ok"]:
        codes.append("F9")
        details.append(f"route {row['route_stages']}")

    gold_forms = split(case.expected_native)
    native = response.native_script if response.native_script is not None else case.input
    row["native_script"] = response.native_script
    row["native_ok"] = normalize(native) == normalize(case.expected_native)
    row["native_ok_lenient"] = fold_variants(native) == fold_variants(case.expected_native)
    kind_errors, token_codes = [], []
    if len(response.tokens) == len(case.expected_tokens) == len(gold_forms):
        for gold, token, form in zip(
            case.expected_tokens, response.tokens, gold_forms, strict=True
        ):
            if gold.kind is not token.kind:
                kind_errors.append(f"{token.surface}:{gold.kind.value}→{token.kind.value}")
            chosen = token.chosen if token.chosen is not None else token.surface
            code = classify_token(gold.kind, token.kind, form, chosen, token.candidates)
            if code:
                token_codes.append(code)
                details.append(f"{code} {token.surface}: gold {form}, got {chosen}")
    elif not row["native_ok"]:
        token_codes.append("F12")
        details.append("token count differs from annotation")
    row["kind_errors"] = "; ".join(kind_errors)
    codes += token_codes

    row["rewritten"] = response.rewritten
    row["rewrite_error"] = stage_errors.get(Stage.REWRITE, "")
    if response.rewritten is not None:
        score = score or judge_rewrite(case, response)
        row.update({f"judge_{a}": getattr(score, a) for a in AXES})
        row["judge_parsed"] = score.parsed
        row["rewrite_ok"] = score.meaning >= MEANING_PASS
    else:
        row["rewrite_ok"] = False
    if not row["rewrite_ok"]:
        codes.append("F10")
        details.append(f"F10 {row['rewrite_error'] or 'judge meaning < 4'}")

    row["english"] = response.english
    translate_error = stage_errors.get(Stage.TRANSLATE, "")
    if response.english is not None:
        chrf = CHRF(word_order=2).sentence_score(response.english, [case.expected_english])
        row["chrf"] = round(chrf.score, 1)
        row["translation_ok"] = chrf.score >= CHRF_PASS
        if not row["translation_ok"]:
            codes.append("F11")
            details.append(f"F11 chrF++ {row['chrf']}")
    elif not translate_error or any(
        reason in translate_error for reason in ("OSError", "disabled", "skipped")
    ):
        # Not run, model unavailable (OSError: gated or missing weights), switched
        # off, or skipped after an earlier failure that is already coded.
        row["translation_ok"] = None
    else:
        row["translation_ok"] = False
        codes.append("F12")  # the model ran and raised: infrastructure
        details.append(f"F12 translate: {translate_error}")

    row["total_ms"] = response.timings_ms.get("total")
    row["failures"] = " ".join(codes)
    row["failure"] = primary(codes)
    row["detail"] = " | ".join(details)
    row["passed"] = not codes
    return row


def end_to_end(cases: list[Case]) -> tuple[list[dict], list[AssistResponse]]:
    """The full pipeline on every case, in two passes over the cases.

    Pass 1 runs every stage except translation and judges the rewrite (all
    Param-1 work). Param-1 is then released and pass 2 translates (IndicTrans2).
    Each case's output is identical to one `assist()` call — `assist` is the
    same two functions back to back — but only one large model is in memory at
    a time. With both loaded on a 16 GB machine, a case took ~15 min of swap.
    Translate latency is still measured per request.
    """
    get_llm()  # load before timing: latency should not include model loading
    first = []
    for i, case in enumerate(cases, 1):
        response, crash = run_case(case)
        first.append((case, response, crash, judge_rewrite(case, response)))
        logger.info("e2e pass 1 %d/%d %s", i, len(cases), case.id)
    release_llm()
    try:
        get_translator()  # load before timing, as above
    except Exception as exc:  # noqa: BLE001 — unavailable: each case records it per request
        logger.warning("translator failed to load: %s", type(exc).__name__)

    rows, responses = [], []
    for i, (case, response, crash, score) in enumerate(first, 1):
        if response is not None:
            try:
                response = add_translation(case.input, response)
            except Exception as exc:  # noqa: BLE001 — same rule as run_case
                response, crash = None, type(exc).__name__
        rows.append(evaluate_case(case, response, crash, score))
        if response is not None:
            responses.append(response)
        logger.info("e2e %d/%d %s %s", i, len(cases), case.id, rows[-1]["failure"] or "pass")
    return rows, responses


def summarise_e2e(rows: list[dict]) -> dict:
    n = len(rows)
    by_category: dict[str, dict] = {}
    for category in dict.fromkeys(r["category"] for r in rows):
        group = [r for r in rows if r["category"] == category]
        by_category[category] = {
            "n": len(group),
            "native_ok": sum(r.get("native_ok", False) for r in group),
            "passed": sum(r["passed"] for r in group),
        }
    buckets = Counter(r["failure"] for r in rows if r["failure"])
    all_codes = Counter(c for r in rows for c in r["failures"].split())
    translated = [r for r in rows if r.get("translation_ok") is not None]
    return {
        "n": n,
        "register": E2E_REGISTER.value,
        "passed": sum(r["passed"] for r in rows),
        "route_ok": sum(r.get("route_ok", False) for r in rows),
        "native_ok": sum(r.get("native_ok", False) for r in rows),
        "native_ok_lenient": sum(r.get("native_ok_lenient", False) for r in rows),
        "native_cer": corpus_cer(
            [r.get("native_script") or r["input"] for r in rows],
            [r["expected_native"] for r in rows],
        ),
        "rewrite_ok": sum(r.get("rewrite_ok", False) for r in rows),
        "translation_evaluated": len(translated),
        "translation_ok": sum(bool(r["translation_ok"]) for r in translated),
        "by_category": by_category,
        "failures_primary": {c: buckets.get(c, 0) for c in CODES},
        "failures_any": {c: all_codes.get(c, 0) for c in CODES},
    }


def translit_sentence_baseline(cases: list[Case], rows: list[dict]) -> dict:
    """Ablation 4: the library's `translit_sentence` one-liner vs our pipeline."""
    engine = get_xlit_engine()
    outputs = [engine.translit_sentence(c.input).get(settings.language, "") for c in cases]
    gold = [c.expected_native for c in cases]
    ours = [r.get("native_script") or r["input"] for r in rows]

    def token_accuracy(texts: list[str], kind: TokenKind | None) -> tuple[int, int]:
        hits = total = 0
        for case, text in zip(cases, texts, strict=True):
            produced, forms = split(text), split(case.expected_native)
            if len(produced) != len(forms):
                continue
            for gold_token, p, f in zip(case.expected_tokens, produced, forms, strict=True):
                if kind is None or gold_token.kind is kind:
                    total += 1
                    hits += normalize(p) == normalize(f)
        return hits, total

    def scores(texts: list[str]) -> dict:
        pairs = list(zip(texts, gold, strict=True))
        out = {
            "sentence_exact": sum(normalize(t) == normalize(g) for t, g in pairs),
            "sentence_exact_lenient": sum(fold_variants(t) == fold_variants(g) for t, g in pairs),
            "cer": corpus_cer(texts, gold),
        }
        for name, kind in [
            ("all", None),
            ("hindi", TokenKind.HINDI),
            ("english", TokenKind.ENGLISH),
            ("named_entity", TokenKind.NAMED_ENTITY),
        ]:
            hits, total = token_accuracy(texts, kind)
            out[f"tokens_{name}"] = {"correct": hits, "n": total}
        return out

    for row, text in zip(rows, outputs, strict=True):
        row["baseline_native"] = text
        row["baseline_ok"] = normalize(text) == normalize(row["expected_native"])
    return {"n": len(cases), "translit_sentence": scores(outputs), "pipeline": scores(ours)}


def translation_stage(cases: list[Case]) -> dict:
    """Translation in isolation: gold Devanagari in, chrF++ and BLEU out (§2.5)."""
    sources = [c.expected_native for c in cases]
    refs = [c.expected_english for c in cases]
    try:
        hyps = translate_batch(sources)
    except StageFailure as failure:
        return {"available": False, "error": failure.error.message, "n": len(cases)}
    return {
        "available": True,
        "n": len(cases),
        "chrf_pp": round(CHRF(word_order=2).corpus_score(hyps, [refs]).score, 1),
        "bleu": round(BLEU().corpus_score(hyps, [refs]).score, 1),
        "e2e_note": "end-to-end translations are scored per case in the CSV",
    }


def e2e_translation_scores(rows: list[dict]) -> dict | None:
    scored = [r for r in rows if r.get("english") and r.get("translation_ok") is not None]
    if not scored:
        return None
    hyps, refs = [r["english"] for r in scored], [r["expected_english"] for r in scored]
    return {
        "n": len(scored),
        "chrf_pp": round(CHRF(word_order=2).corpus_score(hyps, [refs]).score, 1),
        "bleu": round(BLEU().corpus_score(hyps, [refs]).score, 1),
    }


def latency(responses: list[AssistResponse]) -> dict:
    """p50 / p95 per stage over the end-to-end run (§2.6)."""
    per_stage: dict[str, list[float]] = {}
    for r in responses:
        for stage, ms in r.timings_ms.items():
            per_stage.setdefault(stage, []).append(ms)
    return {stage: _percentiles(values) for stage, values in per_stage.items()}


def _percentiles(values: list[float]) -> dict:
    ordered = sorted(values)
    p95_index = min(len(ordered) - 1, round(0.95 * (len(ordered) - 1)))
    return {
        "n": len(ordered),
        "p50_ms": round(statistics.median(ordered), 1),
        "p95_ms": round(ordered[p95_index], 1),
    }


def timed(fn, *args, **kwargs):  # noqa: ANN001, ANN201
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, round(time.perf_counter() - start, 2)


def hardware() -> dict:
    llm = get_llm()
    info = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "llm_device": getattr(llm, "device", "mock"),
    }
    try:
        import torch

        info["torch"] = torch.__version__
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return info


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        )
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return out.stdout.strip() + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def meta(run_id: str) -> dict:
    return {
        "run": run_id,
        "date_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "mock_models": settings.mock_models,
        "hardware": hardware(),
        "models": {"llm": settings.llm_model_id, "translate": settings.translate_model_id},
        "settings": {
            "llm_router": settings.llm_router,
            "llm_tiebreak": settings.llm_tiebreak,
            "llm_seed": settings.llm_seed,
            "e2e_register": E2E_REGISTER.value,
        },
    }


CSV_FIELDS = [
    "id", "category", "difficulty", "passed", "failure", "failures", "detail",
    "input", "expected_native", "native_script", "native_ok", "native_ok_lenient",
    "baseline_native", "baseline_ok", "kind_errors",
    "expected_route", "route_stages", "route_source", "route_ok",
    "rewritten", "rewrite_error", "judge_meaning", "judge_register", "judge_fluency",
    "judge_parsed", "rewrite_ok",
    "expected_english", "english", "chrf", "translation_ok",
    "stage_errors", "total_ms",
]  # fmt: skip


def write_results(run_id: str, metrics: dict, rows: list[dict], out_dir: Path = RESULTS) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{run_id}_metrics.json"
    json_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (out_dir / f"{run_id}_cases.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=CSV_FIELDS, extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    return json_path


def compare(new: dict, committed: dict) -> list[str]:
    """Differences between two metrics files, ignoring meta and latency."""
    diffs: list[str] = []

    def walk(a: object, b: object, path: str) -> None:
        if isinstance(a, dict) and isinstance(b, dict):
            for key in sorted(set(a) | set(b)):
                if not path and key in NONDETERMINISTIC:
                    continue
                walk(a.get(key), b.get(key), f"{path}.{key}" if path else key)
        elif isinstance(a, float) and isinstance(b, float):
            if abs(a - b) > 1e-9:
                diffs.append(f"{path}: {b} → {a}")
        elif a != b:
            diffs.append(
                f"{path}: {json.dumps(b, ensure_ascii=False)[:80]} → "
                f"{json.dumps(a, ensure_ascii=False)[:80]}"
            )

    walk(new, committed, "")
    return diffs
