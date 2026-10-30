from eval.run_eval import eval_transliterate, load_aksharantar_sample


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
