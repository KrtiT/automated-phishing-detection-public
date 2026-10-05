"""Recompute the saved follow-up predictions without fitting or new inference."""

import csv
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parent / "followup-20261001"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare(expected, actual, location="root"):
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or expected.keys() != actual.keys():
            raise ValueError(f"Key mismatch at {location}")
        for key in expected:
            compare(expected[key], actual[key], f"{location}.{key}")
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(expected) != len(actual):
            raise ValueError(f"Length mismatch at {location}")
        for index, (left, right) in enumerate(zip(expected, actual)):
            compare(left, right, f"{location}[{index}]")
    elif isinstance(expected, float):
        if not isinstance(actual, (int, float)) or not math.isclose(expected, actual, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f"Numeric mismatch at {location}: {expected} != {actual}")
    elif type(expected) is not type(actual) or expected != actual:
        raise ValueError(f"Value mismatch at {location}: {expected} != {actual}")


def metric_table(labels, scores, threshold):
    positive = labels == 1
    positives, negatives = int(positive.sum()), int((~positive).sum())
    predicted = scores >= threshold
    true_positive = int(np.count_nonzero(predicted & positive))
    false_positive = int(np.count_nonzero(predicted & ~positive))
    ranking = np.argsort(-scores, kind="stable")
    ranked = labels[ranking]
    endpoints = np.r_[np.flatnonzero(np.diff(scores[ranking]) != 0), len(labels) - 1]
    cumulative_positive = np.cumsum(ranked)[endpoints]
    recall_steps = np.diff(np.r_[0, cumulative_positive]) / positives
    precision_steps = cumulative_positive / (endpoints + 1)
    bins = []
    allocation = np.minimum((10 * scores).astype(int), 9)
    for index in range(10):
        selected = allocation == index
        count = int(selected.sum())
        bins.append({
            "bin": index, "count": count,
            "mean_probability": float(np.mean(scores[selected])) if count else None,
            "positive_fraction": float(np.mean(labels[selected])) if count else None,
        })
    return {
        "counts": {"positive": positives, "negative": negatives, "tp": true_positive,
                   "fp": false_positive, "tn": negatives - false_positive, "fn": positives - true_positive},
        "threshold": float(threshold), "recall": true_positive / positives,
        "fpr": false_positive / negatives,
        "precision": true_positive / (true_positive + false_positive) if true_positive + false_positive else None,
        "roc_auc": float((rankdata(scores)[positive].sum() - positives * (positives + 1) / 2) / (positives * negatives)),
        "average_precision": float(np.sum(recall_steps * precision_steps)),
        "brier": float(np.mean(np.square(scores - labels))), "calibration_bins": bins,
    }


def recompute(rows, thresholds, replicates=10000):
    identifiers = [row["record_id"] for row in rows]
    if not identifiers or len(set(identifiers)) != len(identifiers):
        raise ValueError("Duplicate or missing observations")
    if any(type(row["is_phishing"]) is not int or row["is_phishing"] not in (0, 1)
           or type(row["candidate_feature_equal"]) is not bool
           or not isinstance(row["registrable_domain"], str) or not row["registrable_domain"] for row in rows):
        raise ValueError("Invalid label, invariance or domain identity")
    labels = np.array([row["is_phishing"] for row in rows], dtype=np.int64)
    if set(labels.tolist()) != {0, 1}:
        raise ValueError("Both classes required")
    domains = [row["registrable_domain"] for row in rows]
    ordered = sorted(set(domains))
    domain_index = {domain: index for index, domain in enumerate(ordered)}
    allocation = np.array([domain_index[domain] for domain in domains])
    metrics, decisions, invariance = {}, {}, {}
    for model in ("baseline", "candidate"):
        original = np.array([row["original"][model] for row in rows])
        swapped = np.array([row["scheme_swap"][model] for row in rows])
        if any(not np.all(np.isfinite(scores)) or np.any((scores < 0) | (scores > 1)) for scores in (original, swapped)):
            raise ValueError("Invalid score")
        threshold = thresholds[model]
        if not math.isfinite(threshold):
            raise ValueError("Invalid threshold")
        decisions[model] = (original >= threshold).astype(np.int64)
        metrics[model] = metric_table(labels, original, threshold)
        invariance[model] = {
            "exact_score_matches": int(np.count_nonzero(original == swapped)),
            "decision_flips": int(np.count_nonzero((original >= threshold) != (swapped >= threshold))),
            "max_absolute_score_difference": float(np.max(np.abs(original - swapped))),
            "denominator": len(rows),
        }
    invariance["candidate"]["exact_feature_matches"] = sum(row["candidate_feature_equal"] for row in rows)
    random = np.random.Generator(np.random.PCG64(20261001))
    differences, false_positive_rates = [], []
    positive_change = labels * (decisions["candidate"] - decisions["baseline"])
    negative_alerts = (1 - labels) * decisions["candidate"]
    for unused in range(replicates):
        multiplicity = np.bincount(random.integers(len(ordered), size=len(ordered)), minlength=len(ordered))[allocation]
        positives = int(multiplicity @ labels)
        negatives = int(multiplicity @ (1 - labels))
        if positives:
            differences.append(float((multiplicity @ positive_change) / positives))
        if negatives:
            false_positive_rates.append(float((multiplicity @ negative_alerts) / negatives))

    def interval(values):
        return np.quantile(values, [0.0125, 0.9875], method="linear").tolist() if values else None

    paired = {
        "domain_count": len(ordered), "largest_domain_rows": max(Counter(domains).values()),
        "replicates": replicates, "seed": 20261001,
        "method": "paired registrable-domain bootstrap; row-weighted percentile",
        "undefined_recall_replicates": replicates - len(differences),
        "undefined_fpr_replicates": replicates - len(false_positive_rates),
        "recall_difference": {"point": float(positive_change.sum() / labels.sum()), "interval_97_5": interval(differences)},
        "candidate_fpr": {"point": metrics["candidate"]["fpr"], "interval_97_5": interval(false_positive_rates)},
    }
    recall_interval = paired["recall_difference"]["interval_97_5"]
    requirements = {
        "recall_gain_at_least_five_points": paired["recall_difference"]["point"] >= 0.05,
        "recall_interval_lower_above_zero": recall_interval is not None and recall_interval[0] > 0,
        "observed_fpr_at_most_one_percent": metrics["candidate"]["fpr"] <= 0.01,
        "candidate_exact_scheme_invariance": invariance["candidate"]["exact_feature_matches"] == len(rows) and invariance["candidate"]["exact_score_matches"] == len(rows),
    }
    return {"metrics": metrics, "paired_uncertainty": paired, "scheme_invariance": invariance,
            "requirements": requirements, "D_requirement_met": all(requirements.values())}


def save(path, value):
    with path.open("x") as output:
        json.dump(value, output, indent=2, sort_keys=True, allow_nan=False)
        output.write("\n")


def export_csv(path, rows):
    with path.open("x", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    manifest_path = ROOT / "execution-manifest-v1.json"
    manifest = json.loads(manifest_path.read_bytes())
    fitting = ROOT / "detection-fit-v1"
    evaluation = ROOT / "detection-evaluate-v1"
    population = ROOT / manifest["population_directory"]
    saved = json.loads((evaluation / "completion.json").read_bytes())
    frozen = json.loads((fitting / "frozen-models.json").read_bytes())
    candidate = json.loads((fitting / "candidate.json").read_bytes())
    admission = json.loads((population / "summary.json").read_bytes())
    for directory in (fitting, evaluation):
        if (directory / "failure.json").exists():
            raise ValueError("Failed scientific attempt cannot be accepted")
        intent = json.loads((directory / "intent.json").read_bytes())
        if intent["execution_manifest_sha256"] != digest(manifest_path):
            raise ValueError("Execution binding changed")
    if saved["status"] != "evaluated" or saved["original_hypothesis_decisions_changed"] is not False:
        raise ValueError("Wrong completion identity")
    if digest(population / "summary.json") != manifest["population_summary_sha256"] or not admission["admitted"]:
        raise ValueError("Admission identity changed")
    if digest(population / "retained.jsonl") != admission["output_hashes"]["retained.jsonl"]:
        raise ValueError("Retained population changed")
    if frozen != saved["models"] or digest(fitting / "candidate.json") != frozen["candidate_sha256"]:
        raise ValueError("Frozen model identity changed")
    baseline_path = ROOT.parents[1] / "gwu_working/study-only-v1-reviewed-inputs/primary/logistic-l1.json"
    if digest(baseline_path) != frozen["baseline_sha256"] or frozen["baseline_sha256"] != manifest["baseline_file_sha256"]:
        raise ValueError("Comparator identity changed")
    baseline = json.loads(baseline_path.read_bytes())
    if frozen["candidate_threshold"] != candidate["validation_threshold"]["threshold"]:
        raise ValueError("Candidate threshold changed")
    if frozen["baseline_threshold"] != baseline["validation_threshold"]["threshold"]:
        raise ValueError("Comparator threshold changed")
    intent = json.loads((evaluation / "intent.json").read_bytes())
    if datetime.fromisoformat(intent["started_at"]) <= datetime.fromisoformat(frozen["frozen_before_new_benchmark_scoring_at"]):
        raise ValueError("Evaluation preceded threshold freeze")
    predictions = evaluation / "predictions.jsonl"
    if digest(predictions) != saved["predictions_sha256"]:
        raise ValueError("Prediction identity changed")
    rows = [json.loads(line) for line in predictions.read_bytes().splitlines()]
    retained = [json.loads(line) for line in (population / "retained.jsonl").read_bytes().splitlines()]
    if len(rows) != len(retained) or len(rows) != admission["retained_rows"]:
        raise ValueError("Population coverage incomplete")
    for result, source in zip(rows, retained):
        if any(result[key] != source[key] for key in ("record_id", "registrable_domain", "is_phishing")):
            raise ValueError("Prediction order, label or identity mismatch")
    recomputed = recompute(rows, {name: frozen[f"{name}_threshold"] for name in ("baseline", "candidate")})
    for name, actual in recomputed.items():
        compare(saved[name], actual, name)
    destination = ROOT / "verified-detection-v1"
    destination.mkdir(mode=0o700)
    exports = []
    calibration = []
    for model, metrics in recomputed["metrics"].items():
        exports.append({"model": model, **metrics["counts"], **{name: value for name, value in metrics.items() if name not in ("counts", "calibration_bins")}})
        calibration.extend({"model": model, **entry} for entry in metrics["calibration_bins"])
    export_csv(destination / "detection-metrics.csv", exports)
    export_csv(destination / "calibration-bins.csv", calibration)
    export_csv(destination / "scheme-invariance.csv", [{"model": name, "exact_feature_matches": value.get("exact_feature_matches"), **value} for name, value in recomputed["scheme_invariance"].items()])
    sizes = Counter(row["registrable_domain"] for row in rows)
    export_csv(destination / "domain-size-distribution.csv", [{"rows_per_domain": size, "domain_count": count} for size, count in sorted(Counter(sizes.values()).items())])
    report = {
        "status": "verified", "completed_at": datetime.now(timezone.utc).isoformat(),
        "scope": "Separate arithmetic implementation on saved paired predictions; not external replication or fresh inference.",
        "records": len(rows), "source_order_and_labels_match": True, **recomputed,
        "source_hashes": {str(path.relative_to(ROOT)): digest(path) for path in [manifest_path, fitting / "candidate.json", fitting / "completion.json", fitting / "frozen-models.json", evaluation / "intent.json", evaluation / "completion.json", predictions, population / "summary.json", population / "retained.jsonl"]},
        "export_hashes": {path.name: digest(path) for path in destination.glob("*.csv")},
    }
    save(destination / "verification.json", report)
    print(json.dumps({key: report[key] for key in ("status", "records", "D_requirement_met", "paired_uncertainty")}, indent=2))


if __name__ == "__main__":
    main()
