"""Descriptive diagnosis of hash-bound saved scores; no fitting or scoring."""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from audit_saved_predictions import SOURCE_PINS, decisions, verified_rows
from sklearn.metrics import average_precision_score, roc_auc_score

from automated_phishing_detection.url_features import (
    FEATURE_NAMES,
    extract_url_features,
)

SCORE_KEYS = {
    "logistic_l1": "stage1_probability",
    "transformer": "transformer_probability",
}
QUANTILES = [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1]


def empirical_low_fpr(labels, scores, maximum_fpr=0.01):
    if (
        len(labels) != len(scores)
        or any(type(label) is not int for label in labels)
        or set(labels) != {0, 1}
    ):
        raise ValueError("both binary classes with aligned scores are required")
    if not np.all(np.isfinite(scores)) or any(not 0 <= score <= 1 for score in scores):
        raise ValueError("scores must be finite probabilities")
    if not 0 <= maximum_fpr <= 1:
        raise ValueError("FPR must be a probability")
    positives, negatives = sum(labels), len(labels) - sum(labels)
    counts = defaultdict(lambda: [0, 0])
    for label, score in zip(labels, scores, strict=True):
        counts[score][label] += 1
    best, false_positives, true_positives = (0, 0, float("inf")), 0, 0
    for score in sorted(counts, reverse=True):
        false_positives += counts[score][0]
        true_positives += counts[score][1]
        if false_positives / negatives <= maximum_fpr:
            best = max(best, (true_positives, -false_positives, score))
    return {
        "false_positives": -best[1],
        "true_positives": best[0],
        "recall": best[0] / positives,
        "threshold": best[2] if np.isfinite(best[2]) else None,
        "positive_count": positives,
        "negative_count": negatives,
    }


def domain_rates(domains, alerts):
    counts = defaultdict(lambda: [0, 0])
    for domain, alert in zip(domains, alerts, strict=True):
        counts[domain][0] += alert
        counts[domain][1] += 1
    return {
        "url_weighted": sum(alerts) / len(alerts),
        "equal_domain_weighted": float(
            np.mean([hits / size for hits, size in counts.values()])
        ),
        "domains": len(counts),
        "largest_domain_rows": max(size for _, size in counts.values()),
    }


def _quantiles(values):
    return dict(
        zip(
            map(str, QUANTILES), map(float, np.quantile(values, QUANTILES)), strict=True
        )
    )


def _strata(rows):
    bins = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0]
    result = []
    for lower, upper in zip(bins[:-1], bins[1:], strict=True):
        selected = [
            row
            for row in rows
            if lower <= row.get("primary", row)["stage1_probability"]
            and (row.get("primary", row)["stage1_probability"] < upper or upper == 1.0)
        ]
        cells = Counter()
        for row in selected:
            label, prediction = row["record"]["is_phishing"], decisions(row)
            cells[
                (prediction["logistic_l1"] == label, prediction["transformer"] == label)
            ] += 1
        result.append(
            {
                "lower_inclusive": lower,
                "upper": upper,
                "upper_inclusive": upper == 1.0,
                "rows": len(selected),
                "transformer_repairs": cells[(False, True)],
                "transformer_introduced_errors": cells[(True, False)],
                "both_wrong": cells[(False, False)],
                "both_right": cells[(True, True)],
            }
        )
    return result


def _class_summary(rows):
    primary = [row.get("primary", row) for row in rows]
    features = np.asarray([row["features"] for row in primary], dtype=np.float64)
    domains = [row["record"]["registrable_domain"] for row in rows]
    return {
        "rows": len(rows),
        "domains": len(set(domains)),
        "scores": {
            name: _quantiles([row[key] for row in primary])
            for name, key in SCORE_KEYS.items()
        },
        "alert_rates": {
            name: domain_rates(domains, [decisions(row)[name] for row in rows])
            for name in SCORE_KEYS
        },
        "features": {
            name: _quantiles(features[:, index])
            for index, name in enumerate(FEATURE_NAMES)
        },
        "routing_strata": _strata(rows),
    }


def _population_summary(rows):
    labels = [row["record"]["is_phishing"] for row in rows]
    result = {
        "classes": {
            str(label): _class_summary(
                [row for row in rows if row["record"]["is_phishing"] == label]
            )
            for label in (0, 1)
        },
        "models": {},
    }
    for name, key in SCORE_KEYS.items():
        scores = [row.get("primary", row)[key] for row in rows]
        result["models"][name] = {
            "roc_auc": float(roc_auc_score(labels, scores)),
            "average_precision": float(average_precision_score(labels, scores)),
            "best_observed_recall_at_one_percent_fpr": empirical_low_fpr(
                labels, scores
            ),
        }
    return result


def _verify_published_metrics(report, aggregates):
    expected = list(csv.DictReader((aggregates / "secondary-metrics.csv").open()))
    for population, summary in report["populations"].items():
        for name, metrics in summary["models"].items():
            saved = next(
                row
                for row in expected
                if row["population"] == population and row["detector"] == name
            )
            for metric in ("roc_auc", "average_precision"):
                if abs(metrics[metric] - float(saved[metric + "_value"])) > 1e-12:
                    raise ValueError(
                        "saved score metrics differ from published aggregate"
                    )


def build_report(context, aggregates):
    retained = context / "gwu_working/study-urlnorm-v1-2026-09-30-attempt-4"
    sources = {
        name: verified_rows(retained / name / "evidence/predictions.jsonl", pin)
        for name, pin in SOURCE_PINS.items()
    }
    checked = 0
    for rows in sources.values():
        for row in rows:
            stored = row.get("primary", row)["features"]
            if list(extract_url_features(row["record"]["raw_url"])) != stored:
                raise ValueError(
                    "saved feature vector differs from pure extractor replay"
                )
            checked += 1
    populations = {
        "internal": sources["internal"],
        "gold_plus_certified": [
            row
            for row in sources["external"]
            if row["record"]["role"] in ("gold", "certified")
        ],
    }
    report = {
        "analysis_role": "retrospective_diagnostic_not_untouched_confirmation",
        "source_sha256": SOURCE_PINS,
        "feature_replay_exact_rows": checked,
        "uncertainty": "descriptive only; no IID confidence intervals on clustered URLs",
        "low_fpr_curve": "retrospective score frontier; not a deployment threshold or confirmation",
        "populations": {
            name: _population_summary(rows) for name, rows in populations.items()
        },
    }
    _verify_published_metrics(report, aggregates)
    report["published_ranking_metrics_verified"] = True
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", type=Path, required=True)
    parser.add_argument("--aggregates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(args.context, args.aggregates)
    with args.output.open("x") as stream:
        stream.write(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
    print(
        json.dumps(
            {
                "feature_replay_exact_rows": report["feature_replay_exact_rows"],
                "populations": {
                    name: result["models"]
                    for name, result in report["populations"].items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
