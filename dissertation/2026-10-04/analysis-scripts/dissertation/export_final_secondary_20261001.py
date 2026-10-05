"""Check and export every saved secondary population without executing models."""

import json
import math

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from scipy.stats import beta

from verify_final_evidence_20261001 import (
    CONTEXT, ORIGIN, OUTPUT, close, digest, read, require, write_csv,
)


def load_outcomes(source):
    outcomes = []
    with (ORIGIN / source / "evidence/predictions.jsonl").open() as saved:
        for line in saved:
            row = json.loads(line)
            primary = row.get("primary", row)
            values = {}
            for detector, prefix in (("length_only", "length"), ("logistic_l1", "stage1"), ("transformer", "transformer"), ("cascade", "cascade")):
                values[detector] = (primary[prefix + "_decision"], primary[prefix + "_probability"])
            if source == "external":
                values["policy"] = (row["policy_decision"], row["policy_probability"])
            for tabular in row["secondary_tabular"]:
                values["tabular." + tabular["name"]] = (tabular["decision"], tabular["probability"])
            for seed in row["secondary_seeds"]:
                for detector in ("transformer", "cascade"):
                    values[f"seed_{seed['seed']}.{detector}"] = (seed[detector + "_decision"], seed[detector + "_probability"])
            outcomes.append((row["record"], values, row["secondary_seeds"]))
    return outcomes


def select_population(outcomes, population):
    if population == "internal":
        return outcomes
    if population == "gold_plus_certified":
        return [row for row in outcomes if row[0]["role"] in ("gold", "certified")]
    if population in ("gold", "certified", "tranco"):
        return [row for row in outcomes if row[0]["role"] == population]
    tier = "silver" if population == "ncsc_silver" else "bronze"
    return [row for row in outcomes if row[0]["confidence_tier"] == tier]


def nullable(value):
    if isinstance(value, dict):
        return value.get("value", value.get("estimate"))
    return value


def verify_metric(rows, name, metric):
    labels = np.array([row[0]["is_phishing"] for row in rows])
    decisions = np.array([row[1][name][0] for row in rows])
    probabilities = np.array([row[1][name][1] for row in rows])
    if "control_alert_rate" in metric:
        control = metric["control_alert_rate"]
        require(control["numerator"] == int(decisions.sum()) and control["denominator"] == len(rows), "Tranco alerts")
        require(set(metric) == {"control_alert_rate"}, "no labeled Tranco metrics")
        verify_rate(control, int(decisions.sum()), len(rows))
        return
    positive = labels == 1
    counts = {"true_positives": int(((decisions == 1) & positive).sum()),
              "false_positives": int(((decisions == 1) & ~positive).sum()),
              "false_negatives": int(((decisions == 0) & positive).sum()),
              "true_negatives": int(((decisions == 0) & ~positive).sum())}
    positives, negatives = int(positive.sum()), int((~positive).sum())
    if set(metric) == {"recall"}:
        require(positives == len(rows), "recall-only positive population")
        verify_rate(metric["recall"], counts["true_positives"], positives)
        return
    saved_counts = metric.get("counts", metric)
    for field, expected in counts.items():
        require(saved_counts[field] == expected, "secondary confusion counts")
    verify_rate(saved_counts["recall"], counts["true_positives"], positives)
    verify_rate(saved_counts["fpr"], counts["false_positives"], negatives)
    if "counts" not in metric:
        require(positives == 0 or negatives == 0, "single-class count-only population")
        return
    require(len(rows) == metric["row_count"], "secondary population size")
    require(len({row[0]["registrable_domain"] for row in rows}) == metric["domain_count"], "secondary domains")
    true_positive, false_positive = counts["true_positives"], counts["false_positives"]
    true_negative, false_negative = counts["true_negatives"], counts["false_negatives"]
    denominators = {
        "precision": (true_positive, true_positive + false_positive),
        "f2": (5 * true_positive, 5 * true_positive + 4 * false_negative + false_positive),
        "mcc": (true_positive * true_negative - false_positive * false_negative,
                math.sqrt((true_positive + false_positive) * positives * negatives * (true_negative + false_negative))),
    }
    for field, (numerator, denominator) in denominators.items():
        if denominator:
            close(numerator / denominator, nullable(metric[field]), field)
        else:
            require(nullable(metric[field]) is None, "undefined " + field)
    close((true_positive / positives + true_negative / negatives) / 2, nullable(metric["balanced_accuracy"]), "balanced accuracy")
    bins = metric["calibration_bins"]
    require(len(bins) == 10 and sum(item["count"] for item in bins) == len(rows), "calibration coverage")
    require(sum(item["positive_count"] for item in bins) == int(positive.sum()), "calibration positives")
    bin_ids = np.minimum((probabilities * 10).astype(int), 9)
    for item in bins:
        selected = bin_ids == item["index"]
        require(item["count"] == int(selected.sum()), "bin count")
        require(item["positive_count"] == int(positive[selected].sum()), "bin positive count")
        close(float(probabilities[selected].sum()), item["probability_sum"], "bin probability sum")
        close(float(((probabilities[selected] - labels[selected]) ** 2).sum()), item["squared_error_sum"], "bin squared errors")
    close(float(np.mean((probabilities - labels) ** 2)), nullable(metric["brier"]), "Brier")
    close(sum(item["squared_error_sum"] for item in bins) / len(rows), nullable(metric["brier"]), "bin Brier")
    calibration_error = sum(abs(item["probability_sum"] - item["positive_count"]) for item in bins) / len(rows)
    close(calibration_error, nullable(metric["calibration_error"]), "ECE")
    if positive.any() and (~positive).any():
        close(float(average_precision_score(labels, probabilities)), nullable(metric["average_precision"]), "stepwise AP")
    else:
        require(nullable(metric["average_precision"]) is None, "undefined AP")
    if positive.any() and (~positive).any():
        close(float(roc_auc_score(labels, probabilities)), nullable(metric["roc_auc"]), "ROC AUC")
        curve = metric["recall_at_fpr"]
        selected = probabilities >= curve["threshold"] if curve["threshold"] is not None else np.zeros(len(rows), dtype=bool)
        require(int((selected & positive).sum()) == curve["true_positives"], "score-curve TP")
        require(int((selected & ~positive).sum()) == curve["false_positives"], "score-curve FP")
        require(curve["false_positives"] / int((~positive).sum()) <= .01, "curve FPR")
        order = np.argsort(-probabilities, kind="stable")
        ordered_scores = probabilities[order]
        tie_ends = np.r_[np.flatnonzero(ordered_scores[:-1] != ordered_scores[1:]), len(rows) - 1]
        candidate_tp = np.cumsum(positive[order])[tie_ends]
        candidate_fp = np.cumsum(~positive[order])[tie_ends]
        eligible_tp = candidate_tp[candidate_fp / negatives <= .01]
        require(curve["true_positives"] == max([0, *eligible_tp.tolist()]), "saved score curve maximizes recall without splitting ties")
    else:
        require(nullable(metric["roc_auc"]) is None, "undefined AUC")
    for projection in metric["prevalence_projections"]:
        if positive.any() and (~positive).any():
            recall = counts["true_positives"] / int(positive.sum())
            fpr = counts["false_positives"] / int((~positive).sum())
            prevalence = projection["prevalence"]
            close(nullable(projection["alerts"]), 10000 * (prevalence * recall + (1 - prevalence) * fpr), "projected alerts")
            close(nullable(projection["misses"]), 10000 * prevalence * (1 - recall), "projected misses")
            close(nullable(projection["false_alerts"]), 10000 * (1 - prevalence) * fpr, "projected false alerts")


def verify_rate(rate, numerator, denominator):
    require(rate["numerator"] == numerator and rate["denominator"] == denominator, "rate counts")
    if not denominator:
        require(rate["estimate"] is None and rate["upper_95"] is None, "undefined rate")
        return
    close(rate["estimate"], numerator / denominator, "rate estimate")
    upper = 1.0 if numerator == denominator else float(beta.ppf(.95, numerator + 1, denominator - numerator))
    close(rate["upper_95"], upper, "CP upper bound")


def main():
    require(read(OUTPUT / "verification.json")["status"] == "verified", "primary prerequisite")
    internal = read(ORIGIN / "internal/evidence/secondary.json")
    external = read(ORIGIN / "external/evidence/secondary.json")
    sources = {name: load_outcomes(name) for name in ("internal", "external")}
    require(len(internal["metrics"]) == 21 and len(external["detector_columns"]) == 22, "all detector columns")
    populations = {"internal": internal["metrics"]} | {name: value["detectors"] for name, value in external["populations"].items()}
    require(len(populations) == 7, "all populations")
    metric_rows, calibration_rows, projection_rows, curve_rows, seed_rows = [], [], [], [], []
    for population, detectors in populations.items():
        rows = select_population(sources["internal" if population == "internal" else "external"], population)
        for name, metric in detectors.items():
            verify_metric(rows, name, metric)
            base = {"population": population, "detector": name, "rows": len(rows), "domains": len({row[0]["registrable_domain"] for row in rows})}
            flat = dict(base)
            for field, value in metric.items():
                if field in ("calibration_bins", "prevalence_projections", "recall_at_fpr", "counts"):
                    continue
                if isinstance(value, dict):
                    flat.update({field + "_" + key: item for key, item in value.items()})
                else:
                    flat[field] = value
            if "counts" in metric:
                for field, value in metric["counts"].items():
                    if isinstance(value, dict):
                        flat.update({field + "_" + key: item for key, item in value.items()})
                    else:
                        flat[field] = value
                calibration_rows.extend(base | item for item in metric["calibration_bins"])
                for projection in metric["prevalence_projections"]:
                    projection_rows.append(base | {key: nullable(value) for key, value in projection.items()} | {"source_recall": metric["counts"]["recall"]["estimate"], "source_fpr": metric["counts"]["fpr"]["estimate"]})
                curve = metric["recall_at_fpr"]
                curve_rows.append(base | {key: nullable(value) for key, value in curve.items()} | {"undefined_reason": curve["recall"].get("reason")})
            metric_rows.append(flat)
        for seed in range(42, 47):
            count = sum(next(value for value in row[2] if value["seed"] == seed)["band_selected"] for row in rows)
            seed_rows.append({"population": population, "seed": seed, "logical_band_count": count, "rows": len(rows), "fraction": count / len(rows)})
    require(len(metric_rows) == 153 and len(calibration_rows) == 430, "secondary coverage")
    require(len(curve_rows) == 43 and len(projection_rows) == 129, "mixed-class metric coverage")
    for name, rows in (("secondary-metrics.csv", metric_rows), ("calibration-bins.csv", calibration_rows),
                       ("prevalence-projections.csv", projection_rows), ("low-fpr-score-curves.csv", curve_rows),
                       ("seed-logical-invocations.csv", seed_rows), ("source-contingency.csv", external["source_contingency"])):
        write_csv(name, rows)
    monitors = read(ORIGIN / "external/evidence/monitors.json")
    monitor_rows, feature_rows = [], []
    for monitor in monitors:
        require(len(monitor["windows"]) == 132, "monitor windows")
        for window in monitor["windows"]:
            require(window["alert"] == (window["score"] > monitor["threshold"]), "strict alert boundary")
            base = {"monitor": monitor["name"], "threshold": monitor["threshold"]}
            monitor_rows.append(base | {key: value for key, value in window.items() if key != "feature_scores"})
            if monitor["name"] == "psi":
                require(len(window["feature_scores"]) == 26, "PSI feature coverage")
                feature_rows.extend(base | {"start_position": window["start_position"], "feature_index": index, "score": score} for index, score in enumerate(window["feature_scores"]))
    write_csv("external-monitor-windows.csv", monitor_rows)
    write_csv("external-psi-features.csv", feature_rows)
    historical = {}
    for name in ("secondary-development-correction-v2-summary.json", "secondary-seed-probe-correction-v1-summary.json", "rq2-gmm-development-v1-summary.json"):
        path = CONTEXT / "gwu_working/automated-phishing-detection-public/reports" / name
        historical[name] = {"sha256": digest(path), "data": read(path)}
    bundle = {"internal": internal, "external": external, "monitors": monitors, "historical": historical}
    (OUTPUT / "complete-secondary-results.json").write_text(json.dumps(bundle, indent=2) + "\n")
    report = {"status": "verified_and_exported", "scope": "Saved probabilities and decisions only; confusion counts, AP/AUC, Brier/ECE and bin totals, low-FPR cutoff counts, prevalence arithmetic, monitor strict boundaries and coverage. Historical accepted records retained with their original verification limitations.",
              "populations": list(populations), "population_detector_rows": len(metric_rows), "labeled_metric_rows": len(curve_rows),
              "label_free_control_rows": 22, "calibration_bins": len(calibration_rows), "projection_rows": len(projection_rows),
              "monitor_windows": len(monitor_rows), "psi_feature_scores": len(feature_rows), "seed_population_rows": len(seed_rows),
              "inputs_sha256": {source: digest(ORIGIN / source / "evidence/secondary.json") for source in ("internal", "external")}}
    (OUTPUT / "secondary-verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
