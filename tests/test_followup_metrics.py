from importlib import import_module
from importlib.util import find_spec

import numpy as np
import pytest


def module():
    name = "automated_phishing_detection.followup_metrics"
    assert find_spec(name) is not None, "follow-up metrics are missing"
    return import_module(name)


def test_confusion_denominators_and_calibration_include_every_row():
    result = module().classification_metrics([0, 0, 1, 1], [0.1, 0.9, 0.2, 0.8], 0.5)
    assert result["counts"] == {
        "positive": 2,
        "negative": 2,
        "tp": 1,
        "fp": 1,
        "tn": 1,
        "fn": 1,
    }
    assert result["recall"] == result["fpr"] == result["precision"] == 0.5
    assert sum(row["count"] for row in result["calibration_bins"]) == 4
    assert result["brier"] == pytest.approx(0.375)


def test_identical_models_have_exact_zero_clustered_difference():
    result = module().paired_domain_intervals(
        [0, 1, 0, 1], [0, 1, 0, 1], [0, 1, 0, 1], ["a", "b", "c", "d"], replicates=100
    )
    assert result["recall_difference"]["interval_97_5"] == [0.0, 0.0]
    assert result["domain_count"] == 4
    assert result["replicates"] == 100
    assert result["undefined_recall_replicates"] > 0


def test_domain_resampling_keeps_paired_rows_together():
    result = module().paired_domain_intervals(
        [0, 1, 0, 1], [0, 0, 0, 0], [0, 1, 0, 1], ["a", "a", "b", "b"], replicates=100
    )
    assert result["recall_difference"]["point"] == 1.0
    assert result["recall_difference"]["interval_97_5"] == [1.0, 1.0]
    assert result["candidate_fpr"]["interval_97_5"] == [0.0, 0.0]
    assert result["undefined_recall_replicates"] == 0
    assert result["largest_domain_rows"] == 2


def test_no_alert_precision_is_undefined_not_zero():
    assert module().classification_metrics([0, 1], [0.0, 0.0], 1.0)["precision"] is None


@pytest.mark.parametrize("scores", [[0.1], [np.nan, 0.5], [-0.1, 0.5], [0.1, 1.1]])
def test_invalid_scores_cannot_be_summarized(scores):
    with pytest.raises(ValueError):
        module().classification_metrics([0, 1], scores, 0.5)
