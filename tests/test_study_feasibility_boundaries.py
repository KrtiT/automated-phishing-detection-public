"""Exact frozen capacity boundaries, not new power or outcome gates."""

import pytest
from study_feasibility_fixtures import assess, external_rows, internal_rows, shortages


@pytest.mark.parametrize("positive_rows,domains", [(0, 0), (1, 1), (20, 1), (2, 2)])
def test_clustered_intervals_need_two_positive_domains_not_two_rows(
    positive_rows, domains
):
    result = assess(internal_rows(1, positive_rows, max(domains, 1)))
    counts = result["counts"]["internal"]
    assert (
        counts["positive_rows"] == positive_rows
        and counts["positive_domains"] == domains
    )
    assert ("internal_positive_domains" in shortages(result)) == (domains < 2)


@pytest.mark.parametrize("domains", [1, 2])
def test_external_gold_cluster_capacity_is_independent_of_row_count(domains):
    result = assess(external=external_rows(("gold",) * 20, gold_domains=domains))
    assert result["counts"]["external"]["gold_positive_domains"] == domains
    assert ("gold_positive_domains" in shortages(result)) == (domains < 2)


@pytest.mark.parametrize(
    "rows,windows",
    [(0, 0), (255, 0), (256, 1), (319, 1), (320, 2), (999, 12), (1000, 12)],
)
def test_offline_windows_and_live_warmup_are_separate(rows, windows):
    result = assess(external=external_rows(("gold",) * rows))
    assert result["counts"]["external"]["complete_windows"] == windows
    found = shortages(result)
    assert ("external_complete_windows" in found) == (windows == 0)
    assert ("shift_warmup_rows" in found) == (rows < 1000)
    if rows < 1000:
        assert found["shift_warmup_rows"]["category"] == "descriptive"


@pytest.mark.parametrize(
    "prevalence,negatives,positives,category",
    [
        (100, 9900, 100, "primary"),
        (10, 9990, 10, "sensitivity"),
        (500, 9500, 500, "sensitivity"),
    ],
)
@pytest.mark.parametrize("label", [0, 1])
def test_each_replay_class_capacity_is_checked_without_sampling(
    prevalence, negatives, positives, category, label
):
    key = f"http_{prevalence}_{'negative' if label == 0 else 'positive'}_rows"
    below = internal_rows(negatives - (label == 0), positives - (label == 1), 2)
    found = shortages(assess(below))
    assert found[key] == {
        "requirement": key,
        "category": category,
        "required": negatives if label == 0 else positives,
        "available": (negatives if label == 0 else positives) - 1,
    }
    assert key not in shortages(assess(internal_rows(negatives, positives, 2)))


def test_sensitivity_shortage_does_not_reclassify_primary_capacity():
    found = shortages(assess(internal_rows(9900, 100, 2)))
    assert (
        "http_100_negative_rows" not in found and "http_100_positive_rows" not in found
    )
    assert found["http_10_negative_rows"]["category"] == "sensitivity"
    assert found["http_500_positive_rows"]["category"] == "sensitivity"
