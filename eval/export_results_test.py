import csv
import json

import pytest

from eval import full
from eval.export_results import export
from eval.run_eval import run_all, write_human_rubric


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    raw = tmp_path_factory.mktemp("raw")
    metrics, rows = run_all("unit")
    full.write_results("unit", metrics, rows, out_dir=raw)
    write_human_rubric("unit", metrics["rewrite"]["cases"], out_dir=raw)
    out = tmp_path_factory.mktemp("results") / "results"
    return export("unit", out=out, results=raw, human=raw), metrics, raw


def test_folder_has_everything_for_the_report(exported):
    out, _, _ = exported
    for name in [
        "README.md",
        "metrics.json",
        "cases.csv",
        "examples.md",
        "human_rubric.csv",
        "SHA256SUMS",
        "test_sets/codemixed_cases.yaml",
        "tables/prd_targets.csv",
        "tables/failure_buckets.csv",
        "tables/latency.csv",
    ]:
        assert (out / name).exists(), name
    readme = (out / "README.md").read_text(encoding="utf-8")
    assert "`unit`" in readme and "MOCK RUN" in readme
    assert "Not scored yet" in readme


def test_tables_match_the_metrics(exported):
    out, metrics, _ = exported
    with (out / "tables" / "failure_buckets.csv").open(encoding="utf-8") as f:
        buckets = {r["code"]: int(r["cases (primary)"]) for r in csv.DictReader(f)}
    assert buckets == {
        code: metrics["end_to_end"]["failures_primary"].get(code, 0) for code in buckets
    }
    assert json.loads((out / "metrics.json").read_text(encoding="utf-8")) == json.loads(
        json.dumps(metrics)
    )


def test_reexport_keeps_the_written_failure_analysis(exported):
    out, _, raw = exported
    (out / "FAILURE_ANALYSIS.md").write_text("hand written", encoding="utf-8")
    export("unit", out=out, results=raw, human=raw)
    assert (out / "FAILURE_ANALYSIS.md").read_text(encoding="utf-8") == "hand written"


def test_missing_run_is_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="no run"):
        export("nope", out=tmp_path / "r", results=tmp_path, human=tmp_path)


def test_run_without_torch_exports(exported, tmp_path):
    """A core-only install (MOCK_MODELS=1, no torch) records no torch version."""
    _, _, raw = exported
    for f in raw.iterdir():
        (tmp_path / f.name).write_bytes(f.read_bytes())
    metrics_path = tmp_path / "unit_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["meta"]["hardware"].pop("torch", None)
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    out = export("unit", out=tmp_path / "results", results=tmp_path, human=tmp_path)
    assert "torch" not in (out / "README.md").read_text(encoding="utf-8").split("| machine |")[1]
