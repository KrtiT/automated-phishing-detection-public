"""Prespecified paired follow-up summaries; no threshold fitting or selection."""

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def _binary(values, *, both=False):
    array = np.asarray(values)
    if (
        array.ndim != 1
        or not array.size
        or not np.issubdtype(array.dtype, np.integer)
        or not set(array.tolist()).issubset({0, 1})
        or (both and set(array.tolist()) != {0, 1})
    ):
        raise ValueError("expected exact integer binary observations")
    return array


def classification_metrics(labels, probabilities, threshold):
    labels = _binary(labels, both=True)
    scores = np.asarray(probabilities, dtype=np.float64)
    if (
        scores.shape != labels.shape
        or not np.all(np.isfinite(scores))
        or np.any((scores < 0) | (scores > 1))
        or not np.isfinite(threshold)
    ):
        raise ValueError("invalid probabilities, shape or threshold")
    decisions = scores >= threshold
    positive = labels == 1
    positives = int(positive.sum())
    negatives = len(labels) - positives
    true_positive = int(np.sum(decisions & positive))
    false_positive = int(np.sum(decisions & ~positive))
    bins = []
    allocation = np.minimum((scores * 10).astype(int), 9)
    for index in range(10):
        selected = allocation == index
        count = int(selected.sum())
        bins.append(
            {
                "bin": index,
                "count": count,
                "mean_probability": float(scores[selected].mean()) if count else None,
                "positive_fraction": float(labels[selected].mean()) if count else None,
            }
        )
    alerts = true_positive + false_positive
    return {
        "counts": {
            "positive": positives,
            "negative": negatives,
            "tp": true_positive,
            "fp": false_positive,
            "tn": negatives - false_positive,
            "fn": positives - true_positive,
        },
        "threshold": float(threshold),
        "recall": true_positive / positives,
        "fpr": false_positive / negatives,
        "precision": true_positive / alerts if alerts else None,
        "roc_auc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
        "brier": float(brier_score_loss(labels, scores)),
        "calibration_bins": bins,
    }


def _interval(values):
    return (
        np.quantile(values, [0.0125, 0.9875], method="linear").tolist()
        if values
        else None
    )


def paired_domain_intervals(labels, baseline, candidate, domains, *, replicates=10000):
    labels = _binary(labels, both=True)
    baseline, candidate = _binary(baseline), _binary(candidate)
    if (
        baseline.shape != labels.shape
        or candidate.shape != labels.shape
        or len(domains) != len(labels)
    ):
        raise ValueError("paired observations must have equal shapes")
    if any(type(domain) is not str or not domain for domain in domains):
        raise ValueError("domains must be nonempty strings")
    if type(replicates) is not int or replicates < 1:
        raise ValueError("replicate count must be positive")
    unique, allocation = np.unique(domains, return_inverse=True)
    grouped = np.zeros((len(unique), 5), dtype=np.int64)
    np.add.at(
        grouped,
        allocation,
        np.column_stack(
            [
                labels,
                1 - labels,
                labels * baseline,
                labels * candidate,
                (1 - labels) * candidate,
            ]
        ),
    )
    totals = grouped.sum(axis=0)
    point_difference = (totals[3] - totals[2]) / totals[0]
    point_fpr = totals[4] / totals[1]
    random = np.random.Generator(np.random.PCG64(20261001))
    differences, false_positive_rates = [], []
    for _ in range(replicates):
        sampled = grouped[random.integers(0, len(unique), size=len(unique))].sum(axis=0)
        if sampled[0]:
            differences.append(float((sampled[3] - sampled[2]) / sampled[0]))
        if sampled[1]:
            false_positive_rates.append(float(sampled[4] / sampled[1]))
    return {
        "domain_count": len(unique),
        "largest_domain_rows": int(np.bincount(allocation).max()),
        "replicates": replicates,
        "seed": 20261001,
        "method": "paired registrable-domain bootstrap; row-weighted percentile",
        "undefined_recall_replicates": replicates - len(differences),
        "undefined_fpr_replicates": replicates - len(false_positive_rates),
        "recall_difference": {
            "point": float(point_difference),
            "interval_97_5": _interval(differences),
        },
        "candidate_fpr": {
            "point": float(point_fpr),
            "interval_97_5": _interval(false_positive_rates),
        },
    }
