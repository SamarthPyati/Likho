import csv

import pytest

from eval.human import load_sheet, score

AXES_COLS = ["meaning", "register", "fluency"]


def write_sheet(path, rows):
    cols = ["run", "case", "register", "original", "rewritten"] + [
        f"{r}_{a}" for r in ("rater_a", "rater_b") for a in AXES_COLS
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)


def row(case, a, b):
    out = {"run": "r", "case": case, "register": "formal", "original": "x", "rewritten": "y"}
    out.update({f"rater_a_{ax}": a for ax in AXES_COLS})
    out.update({f"rater_b_{ax}": b for ax in AXES_COLS})
    return out


def test_empty_cells_are_refused(tmp_path):
    path = tmp_path / "s.csv"
    write_sheet(path, [row("cm_001", 4, "")])
    with pytest.raises(ValueError, match="empty score cells"):
        load_sheet(path)


def test_scores_agreement_and_leniency(tmp_path):
    path = tmp_path / "s.csv"
    write_sheet(path, [row("cm_001", 3, 3), row("cm_002", 2, 3), row("cm_003", 5, 4)])
    judged = {
        (c, "formal"): {f"judge_{a}": j for a in AXES_COLS}
        for c, j in [("cm_001", 4), ("cm_002", 4), ("cm_003", 5)]
    }
    result = score(load_sheet(path), judged)["axes"]["meaning"]
    assert result["human_mean"] == pytest.approx(3.33, abs=0.01)
    assert result["judge_minus_human"] == pytest.approx(1.0, abs=0.01)
    assert result["raters_exact_agreement"] == pytest.approx(1 / 3)
    assert result["raters_within_one"] == 1.0
    assert result["judge_human_pearson"] is not None


def test_constant_judge_has_no_correlation(tmp_path):
    path = tmp_path / "s.csv"
    write_sheet(path, [row("cm_001", 3, 3), row("cm_002", 2, 3)])
    judged = {(c, "formal"): {f"judge_{a}": 4 for a in AXES_COLS} for c in ("cm_001", "cm_002")}
    assert score(load_sheet(path), judged)["axes"]["meaning"]["judge_human_pearson"] is None
