"""Hypothetical power calculations must use the declared unit counts."""

from design_power import build_report


def test_available_domain_counts_do_not_become_row_counts():
    result = build_report(positive_sizes=(1392,), negative_sizes=(1262,))
    assert {row["independent_positive_pairs"] for row in result["paired_recall"]} == {
        1392
    }
    assert {
        row["independent_negative_units"]
        for row in result["negative_precision_sensitivity"]
    } == {1262}
    assert all(0 <= row["power"] <= 1 for row in result["paired_recall"])
