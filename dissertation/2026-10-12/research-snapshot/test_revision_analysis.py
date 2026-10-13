"""Synthetic checks of prespecified follow-up contrasts and transformations."""

import pytest
import revision_analysis as analysis


def rows():
    return [
        {
            "record_id": str(index),
            "registrable_domain": f"domain-{index}.test",
            "is_phishing": label,
            "collected_at": "2025-01-01" if index < 2 else "2025-01-02",
            "stage1_probability": structural,
            "transformer_probability": transformer,
        }
        for index, (label, structural, transformer) in enumerate(
            [
                (1, 0.1, 0.9),
                (1, 0.9, 0.9),
                (1, 0.28, 0.01),
                (0, 0.3, 0.01),
                (0, 0.8, 0.9),
                (0, 0.01, 0.01),
            ]
        )
    ]


def test_forced_scheme_preserves_every_byte_after_colon():
    assert (
        analysis.force_scheme("HTTPS://MiXeD.test/a%2FB?q=Z#T", "http")
        == "http://MiXeD.test/a%2FB?q=Z#T"
    )
    assert analysis.force_scheme("http://same.test/", "http") == "http://same.test/"
    for raw in ("ftp://bad.test", "https:bad.test", "bad.test"):
        with pytest.raises(ValueError):
            analysis.force_scheme(raw, "https")


def test_routes_and_paired_counts_preserve_repairs_and_harms():
    result = analysis.analyze_routing(rows(), transformer_threshold=0.5)
    systems = result["systems"]
    assert systems["structural"]["true_positive"] == 2
    assert systems["symmetric"]["true_positive"] == 2
    assert systems["symmetric"]["false_positive"] == 1
    assert systems["symmetric"]["logical_calls"] == 3
    assert systems["symmetric"]["positive_repairs"] == 1
    assert systems["symmetric"]["positive_harms"] == 1
    contrast = result["contrasts"]["symmetric_minus_structural"]
    assert contrast["cells"] == {
        "both_positive": 1,
        "candidate_only": 1,
        "reference_only": 1,
        "both_negative": 0,
    }
    assert contrast["exact"]["lower"] < 0 < contrast["exact"]["upper"]
    assert (
        result["primary_statements"]["symmetric_benefit_and_budget"]["supported"]
        is False
    )
    assert result["original_hypotheses_reclassified"] is False


def test_identical_small_sample_is_not_automatically_noninferior():
    sample = rows()[:2] + rows()[-1:]
    for row in sample:
        row["stage1_probability"] = 0.9 if row["is_phishing"] else 0.01
        row["transformer_probability"] = row["stage1_probability"]
    report = analysis.analyze_routing(sample, transformer_threshold=0.5)
    assert report["primary_statements"]["original_noninferiority"]["supported"] is False


def test_missing_positive_dates_duplicate_domains_or_nan_hold_analysis():
    for modification in ("date", "domain", "nan"):
        sample = rows()
        if modification == "date":
            sample[0]["collected_at"] = ""
        elif modification == "domain":
            sample[1]["registrable_domain"] = sample[0]["registrable_domain"]
        else:
            sample[0]["stage1_probability"] = float("nan")
        with pytest.raises(ValueError):
            analysis.analyze_routing(sample, transformer_threshold=0.5)


def test_date_sensitivity_is_reproducible_and_keeps_whole_clusters():
    sample = rows()[:3]
    first = analysis.date_sensitivity(sample, [1, 1, 0], [0, 0, 1])
    second = analysis.date_sensitivity(sample, [1, 1, 0], [0, 0, 1])
    assert first == second
    assert first["groups"] == 2 and first["largest_group"] == 2
    assert first["lower"] == -1 and first["upper"] == 1
    assert first["replicates"] == 2000 and first["inferential_decision"] is False


def test_matched_control_has_equal_work_without_correctness_inputs():
    decisions, masks = analysis.routing_decisions(rows(), 0.5)
    assert sum(masks["symmetric_control"]) == sum(masks["symmetric"])
    assert sum(masks["directional_control"]) == sum(masks["directional"])
    relabelled = [{**row, "is_phishing": 1 - row["is_phishing"]} for row in rows()]
    assert analysis.routing_decisions(relabelled, 0.5) == (decisions, masks)
