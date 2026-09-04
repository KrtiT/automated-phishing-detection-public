import math

import numpy as np
import pytest

from automated_phishing_detection import baselines


def test_clopper_pearson_upper_matches_closed_form_boundaries():
    assert baselines.clopper_pearson_upper(0, 400) == pytest.approx(
        1.0 - 0.05 ** (1.0 / 400.0), rel=1e-13
    )
    assert baselines.clopper_pearson_upper(400, 400) == 1.0
    assert baselines.clopper_pearson_upper(1, 400) == pytest.approx(
        0.01180430445398703, rel=1e-12
    )


@pytest.mark.parametrize(
    ("false_positives", "negative_count"),
    ((-1, 400), (401, 400), (0, 0), (True, 400), (0, 400.0)),
)
def test_clopper_pearson_upper_rejects_invalid_exact_counts(
    false_positives, negative_count
):
    with pytest.raises(baselines.BaselineError):
        baselines.clopper_pearson_upper(false_positives, negative_count)


def test_threshold_selection_handles_score_ties_as_one_candidate():
    negative_scores = np.concatenate(
        (np.full(399, 0.1), np.array([0.8], dtype=np.float64))
    )
    positive_scores = np.array([0.2, 0.8, 0.8, 0.9], dtype=np.float64)
    scores = np.concatenate((negative_scores, positive_scores))
    labels = np.concatenate((np.zeros(400, dtype=np.int8), np.ones(4, dtype=np.int8)))

    selected = baselines.select_validation_threshold(scores, labels)

    assert selected == {
        "status": "selected",
        "threshold": 0.9,
        "candidate_count": 5,
        "counts": {
            "true_positive": 1,
            "false_positive": 0,
            "true_negative": 400,
            "false_negative": 3,
            "positive": 4,
            "negative": 400,
        },
        "recall": 0.25,
        "observed_fpr": 0.0,
        "fpr_upper_95": pytest.approx(1.0 - 0.05 ** (1.0 / 400.0)),
    }


def test_threshold_selection_uses_smaller_bound_then_higher_threshold_for_ties():
    scores = np.concatenate(
        (
            np.full(400, 0.1),
            np.array([0.3, 0.4, 0.4, 0.7], dtype=np.float64),
        )
    )
    labels = np.concatenate((np.zeros(400, dtype=np.int8), np.ones(4, dtype=np.int8)))

    selected = baselines.select_validation_threshold(scores, labels)

    assert selected["threshold"] == 0.3
    assert selected["recall"] == 1.0
    assert selected["counts"]["false_positive"] == 0


def test_threshold_selection_reports_small_negative_infeasibility():
    scores = np.array([0.1, 0.2, 0.8, 0.9], dtype=np.float64)
    labels = np.array([0, 0, 1, 1], dtype=np.int8)

    selected = baselines.select_validation_threshold(scores, labels)

    assert selected == {
        "status": "target_not_met",
        "threshold": None,
        "candidate_count": 5,
        "counts": None,
        "recall": None,
        "observed_fpr": None,
        "fpr_upper_95": None,
    }


@pytest.mark.parametrize(
    ("scores", "labels"),
    (
        ([0.1, math.nan], [0, 1]),
        ([0.1, math.inf], [0, 1]),
        ([-0.1, 0.8], [0, 1]),
        ([0.1, 1.1], [0, 1]),
        ([0.1], [0, 1]),
        ([[0.1], [0.2]], [0, 1]),
        ([0.1, 0.2], [0, 2]),
        ([0.1, 0.2], [0, 0]),
    ),
)
def test_threshold_selection_rejects_invalid_scores_and_labels(scores, labels):
    with pytest.raises(baselines.BaselineError):
        baselines.select_validation_threshold(np.asarray(scores), np.asarray(labels))
