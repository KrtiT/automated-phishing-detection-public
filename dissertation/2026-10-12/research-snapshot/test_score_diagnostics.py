"""Invented outcomes test the post-review, saved-score-only diagnosis."""

import importlib

import pytest


@pytest.fixture
def diagnostics():
    assert importlib.util.find_spec("score_diagnostics") is not None, (
        "saved-score diagnostic not implemented"
    )
    return importlib.import_module("score_diagnostics")


def test_empirical_curve_keeps_score_ties_together(diagnostics):
    result = diagnostics.empirical_low_fpr([0, 1, 1, 0], [0.9, 0.9, 0.8, 0.1], 0.25)
    assert result == {
        "false_positives": 0,
        "true_positives": 0,
        "recall": 0.0,
        "threshold": None,
        "negative_count": 2,
        "positive_count": 2,
    }


def test_empirical_curve_selects_recall_then_fewer_false_positives(diagnostics):
    result = diagnostics.empirical_low_fpr([0, 1, 1, 0], [0.9, 1.0, 0.8, 0.1], 0.5)
    assert result["false_positives"] == 1
    assert result["true_positives"] == 2
    assert result["threshold"] == 0.8


def test_observed_domain_weighting_does_not_replace_url_weighting(diagnostics):
    result = diagnostics.domain_rates(["one", "one", "one", "two"], [1, 1, 1, 0])
    assert result["url_weighted"] == 0.75
    assert result["equal_domain_weighted"] == 0.5
    assert result["domains"] == 2
    assert result["largest_domain_rows"] == 3


@pytest.mark.parametrize(
    "labels,scores",
    [
        ([0], [0.1]),
        ([1], [0.1]),
        ([True, 0], [0.1, 0.2]),
        ([1, 0], [2.0, 0.1]),
        ([1, 0], [0.5]),
    ],
)
def test_invalid_curve_inputs_fail(diagnostics, labels, scores):
    with pytest.raises(ValueError):
        diagnostics.empirical_low_fpr(labels, scores)
