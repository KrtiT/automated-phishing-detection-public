"""Published examples and boundary checks for independent matched pairs."""

import math
from dataclasses import replace

import pytest
from scipy.stats import multinomial

from automated_phishing_detection import paired_inference as inference
from automated_phishing_detection.paired_evaluation import (
    BinaryPrediction,
    EvaluationRecord,
)


@pytest.mark.parametrize(
    "cells,expected",
    [
        ((36, 12, 2, 0), (0.0569, 0.3404)),
        ((20, 12, 2, 16), (0.0562, 0.3292)),
        ((18, 12, 2, 18), (0.0562, 0.3290)),
        ((36, 14, 0, 0), (0.1528, 0.4167)),
        ((35, 14, 0, 1), (0.1461, 0.4175)),
        ((18, 14, 0, 18), (0.1441, 0.3963)),
        ((2, 97, 1, 0), (0.8721, 0.9854)),
        ((0, 29, 1, 0), (0.6666, 0.9882)),
        ((2, 98, 0, 0), (0.9178, 0.9945)),
        ((1, 98, 0, 1), (0.9171, 0.9916)),
        ((0, 30, 0, 0), (0.8395, 1.0)),
        ((54, 0, 0, 0), (-0.0664, 0.0664)),
        ((53, 0, 0, 1), (-0.0729, 0.0729)),
        ((30, 0, 0, 24), (-0.0358, 0.0358)),
        ((29, 0, 0, 25), (-0.0354, 0.0354)),
        ((28, 0, 0, 26), (-0.0352, 0.0352)),
        ((27, 0, 0, 27), (-0.0351, 0.0351)),
    ],
)
def test_newcombe_table_three_method_ten(cells, expected):
    result = inference.paired_score_interval(inference.PairedTable(*cells))
    assert (result.lower, result.upper) == pytest.approx(expected, abs=0.000051)
    assert result.method == "newcombe_1998_method_10"


def test_published_negative_correlation_example_last_decimal_discrepancy():
    """Table III prints .8736 for method 10 and .8737 for identical method 8.

    The correlation is negative here, so method 10 leaves it unchanged. Equations
    give .8736721762; record the printed last-decimal discrepancy explicitly.
    """
    table = inference.PairedTable(1, 97, 1, 1)
    assert inference._corrected_correlation(table) == pytest.approx(-96 / 196)
    result = inference.paired_score_interval(table)
    assert result.lower == pytest.approx(0.8736721762007447, abs=1e-12)
    assert result.lower == pytest.approx(0.8736, abs=0.0001)
    assert result.upper == pytest.approx(0.9850, abs=0.000051)


@pytest.mark.parametrize("method", ["paired_score_interval", "paired_exact_envelope"])
def test_zero_discordances_do_not_establish_two_point_noninferiority(method):
    result = getattr(inference, method)(inference.PairedTable(69, 0, 0, 0))
    assert result.estimate == 0
    assert result.lower < -0.02 < 0.02 < result.upper
    assert result.confidence == 0.95


@pytest.mark.parametrize("method", ["paired_score_interval", "paired_exact_envelope"])
def test_sign_reversal_and_confidence_monotonicity(method):
    calculate = getattr(inference, method)
    table = inference.PairedTable(36, 12, 2, 0)
    result = calculate(table)
    reverse = calculate(inference.PairedTable(36, 2, 12, 0))
    wider = calculate(table, confidence=0.99)
    assert reverse.estimate == -result.estimate
    assert reverse.lower == pytest.approx(-result.upper)
    assert reverse.upper == pytest.approx(-result.lower)
    assert -1 <= wider.lower <= result.lower <= result.upper <= wider.upper <= 1


def test_exact_zero_event_envelope_matches_closed_form():
    result = inference.paired_exact_envelope(inference.PairedTable(69, 0, 0, 0))
    bound = 1 - (0.05 / 4) ** (1 / 69)
    assert (result.lower, result.upper) == pytest.approx((-bound, bound))
    assert result.method == "bonferroni_clopper_pearson_discordance"


@pytest.mark.parametrize("candidate_rate", [0, 0.01, 0.1, 0.4, 0.8, 1])
@pytest.mark.parametrize("reference_rate", [0, 0.01, 0.1, 0.4, 0.8, 1])
def test_exact_envelope_finite_multinomial_coverage(candidate_rate, reference_rate):
    if candidate_rate + reference_rate > 1:
        return
    probability = [candidate_rate, reference_rate, 1 - candidate_rate - reference_rate]
    target = candidate_rate - reference_rate
    coverage = 0.0
    for candidate_count in range(7):
        for reference_count in range(7 - candidate_count):
            concordant = 6 - candidate_count - reference_count
            table = inference.PairedTable(
                concordant, candidate_count, reference_count, 0
            )
            result = inference.paired_exact_envelope(table)
            if result.lower - 1e-12 <= target <= result.upper + 1e-12:
                coverage += multinomial.pmf(
                    [candidate_count, reference_count, concordant], 6, probability
                )
    assert coverage >= 0.95 - 1e-12


def test_record_pairing_validates_domains_ids_and_order():
    records = tuple(
        EvaluationRecord(f"row-{index}", f"d{index}.test", 1) for index in range(4)
    )
    candidate = tuple(
        BinaryPrediction(row.record_id, value)
        for row, value in zip(records, [1, 1, 0, 0])
    )
    reference = tuple(
        BinaryPrediction(row.record_id, value)
        for row, value in zip(records, [1, 0, 1, 0])
    )
    assert inference.count_independent_pairs(records, candidate, reference) == (
        inference.PairedTable(1, 1, 1, 1)
    )
    duplicate = (
        records[0],
        replace(records[1], registrable_domain="d0.test"),
        *records[2:],
    )
    with pytest.raises(ValueError, match="one positive record per domain"):
        inference.count_independent_pairs(duplicate, candidate, reference)
    with pytest.raises(ValueError, match="IDs or order"):
        inference.count_independent_pairs(records, candidate, reference[::-1])


@pytest.mark.parametrize(
    "cells",
    [(0, 0, 0, 0), (-1, 1, 1, 1), (True, 0, 0, 0), (1.0, 0, 0, 0), (math.nan, 0, 0, 0)],
)
def test_invalid_counts_rejected(cells):
    with pytest.raises(ValueError):
        inference.PairedTable(*cells)


@pytest.mark.parametrize("confidence", [0, 1, -0.5, True, "0.95", math.nan, math.inf])
@pytest.mark.parametrize("method", ["paired_score_interval", "paired_exact_envelope"])
def test_invalid_confidence_rejected(confidence, method):
    with pytest.raises(ValueError):
        getattr(inference, method)(
            inference.PairedTable(69, 0, 0, 0), confidence=confidence
        )
