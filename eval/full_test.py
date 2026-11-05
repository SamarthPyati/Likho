import csv
import json

import pytest

from eval import full
from eval.cases import load_cases
from eval.run_eval import prd_targets, run_all, write_human_rubric
from likho.pipeline import assist
from likho.schemas import AssistRequest, Route, Stage, StageError

CASES = {c.id: c for c in load_cases("test")}


@pytest.fixture(scope="module")
def mock_run():
    return run_all("unit")


def test_full_run_has_every_section(mock_run):
    metrics, rows = mock_run
    assert {
        "meta",
        "transliterate",
        "tokenize",
        "rerank",
        "route",
        "rewrite",
        "translate",
        "end_to_end",
        "ablation_translit_sentence",
        "latency",
        "prd_targets",
    } <= set(metrics)
    assert set(metrics["transliterate"]) == {"rescore_on", "rescore_off"}
    assert set(metrics["tokenize"]) == {"llm_tiebreak_on", "llm_tiebreak_off"}
    assert len(rows) == metrics["end_to_end"]["n"] == 50
    assert all(r["passed"] == (r["failure"] is None) for r in rows)
    assert sum(metrics["end_to_end"]["failures_primary"].values()) == sum(
        not r["passed"] for r in rows
    )


def test_results_files_and_rubric(mock_run, tmp_path):
    metrics, rows = mock_run
    json_path = full.write_results("unit", metrics, rows, out_dir=tmp_path)
    assert json.loads(json_path.read_text(encoding="utf-8"))["end_to_end"]["n"] == 50
    with (tmp_path / "unit_cases.csv").open(encoding="utf-8") as f:
        assert len(list(csv.DictReader(f))) == 50
    rubric = write_human_rubric("unit", metrics["rewrite"]["cases"], out_dir=tmp_path)
    with rubric.open(encoding="utf-8") as f:
        sheet = list(csv.DictReader(f))
    assert len(sheet) == 20
    assert "judge_meaning" not in sheet[0]  # raters must be blind to the judge
    assert {r["register"] for r in sheet} == {"casual", "neutral", "formal"}


def test_compare_ignores_meta_and_latency():
    a = {"meta": {"run": "a"}, "latency": {"x": 1.0}, "score": 0.5, "n": 3}
    b = {"meta": {"run": "b"}, "latency": {"x": 2.0}, "score": 0.5, "n": 3}
    assert full.compare(a, b) == []
    assert full.compare({**a, "score": 0.6}, b) == ["score: 0.5 → 0.6"]


def test_prd_targets_handle_missing_values():
    targets = {t["criterion"]: t for t in prd_targets({"route": {"llm_on": {"accuracy": 0.95}}})}
    assert targets["router accuracy"]["met"] is True
    assert targets["translation chrF++"]["met"] is None


def respond(case_id: str, **changes):
    case = CASES[case_id]
    response = assist(AssistRequest(text=case.input, translate=True))
    return case, response.model_copy(update=changes)


def test_case_with_correct_native_passes_transliteration():
    case, response = respond("cm_001", native_script=CASES["cm_001"].expected_native)
    row = full.evaluate_case(case, response, None)
    assert row["native_ok"] is True


def test_wrong_route_is_f9_and_primary():
    case, response = respond(
        "cm_001",
        route=Route(stages=[Stage.ROUTE, Stage.TOKENIZE, Stage.REWRITE], reason="", source="llm"),
        rewritten=None,
        errors=[StageError(stage=Stage.REWRITE, message="rewrite rejected: length ratio 2.0")],
    )
    row = full.evaluate_case(case, response, None)
    assert row["failure"] == "F9"
    assert "F10" in row["failures"]


def test_unavailable_translator_is_not_a_failure():
    case, response = respond(
        "cm_001",
        native_script=CASES["cm_001"].expected_native,
        english=None,
        errors=[StageError(stage=Stage.TRANSLATE, message="model error: OSError")],
    )
    row = full.evaluate_case(case, response, None)
    assert row["translation_ok"] is None
    assert "F11" not in row["failures"] and "F12" not in row["failures"]


def test_crash_is_f12():
    row = full.evaluate_case(CASES["cm_001"], None, "RuntimeError")
    assert row["failure"] == "F12" and row["passed"] is False


def test_gold_intent():
    assert full.gold_intent(CASES["cm_001"]).value == "roman_hinglish"
    assert full.gold_intent(CASES["cm_034"]).value == "english"
    assert full.gold_intent(CASES["cm_050"]).value == "mixed_script"  # मेरा नाम Rohit है


def test_checkpoint_resume_gives_the_same_metrics(tmp_path):
    checkpoint = tmp_path / "unit_checkpoint.tmp"
    first, _ = run_all("unit", checkpoint)
    assert checkpoint.exists() and "resumed_phases" not in first["meta"]
    resumed, _ = run_all("unit", checkpoint)
    assert resumed["meta"]["resumed_phases"] == [
        "transliterate",
        "tokenize",
        "rerank_route",
        "rewrite",
    ]
    assert full.compare(resumed, json.loads(json.dumps(first))) == []
