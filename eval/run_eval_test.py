from eval.run_eval import (
    TUNE_GRID,
    eval_rerank,
    eval_rewrite,
    eval_route,
    eval_transliterate,
    load_aksharantar_sample,
    tune_rerank,
)
from likho.rerank import DEFAULT_WEIGHTS
from likho.schemas import Register


def test_sample_is_50_pairs_across_four_categories():
    rows = load_aksharantar_sample()
    assert len(rows) == 50
    assert {r["category"] for r in rows} == {"ak_freq", "ak_nef", "ak_nei", "dakshina"}


def test_transliterate_eval_reports_overall_and_per_category():
    results = eval_transliterate(top_k=5, rescore=True)
    assert list(results)[0] == "overall"
    assert results["overall"]["n"] == 50
    assert sum(results[c]["n"] for c in results if c != "overall") == 50
    for row in results.values():
        assert 0.0 <= row["top1"] <= row["top5"] <= 1.0
        assert row["cer"] >= 0.0


def test_rerank_eval_reports_on_and_off_from_one_call():
    result = eval_rerank("dev", top_k=5, weights=DEFAULT_WEIGHTS)
    words = result["aksharantar_words"]
    assert {"top1_off", "top1_on"} <= set(words)
    s = result["dev_sentences"]
    assert s["n_hindi_tokens"] > 0
    assert 0.0 <= s["top1_off"] <= s["topk_ceiling"] <= 1.0
    assert s["f8_in_topk_not_chosen"] == len(s["f8_cases"])


def test_tuning_runs_on_dev_grid():
    results = tune_rerank(top_k=5)
    assert len(results) == len(TUNE_GRID) ** 2


def test_route_eval_reports_accuracy_and_fallback_rate():
    result = eval_route(use_llm=True)
    assert result["n"] == 30
    assert 0.0 <= result["accuracy"] <= 1.0
    assert 0.0 <= result["fallback_rate"] <= 1.0


def test_rule_router_on_committed_cases():
    # Guards the rule fallback itself; the LLM router is measured, not asserted.
    assert eval_route(use_llm=False)["accuracy"] >= 0.8


def test_rewrite_eval_counts_and_judges():
    result = eval_rewrite([Register.FORMAL], limit=5)
    formal = result["registers"]["formal"]
    assert formal["accepted"] + formal["rejected"] == 5
    assert formal["judge_mean"]["meaning"] is not None
