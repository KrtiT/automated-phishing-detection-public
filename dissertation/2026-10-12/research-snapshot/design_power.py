"""Power sensitivity for hypothetical paired outcomes, without reading research data."""

import argparse
import json
from pathlib import Path

from scipy.stats import beta, binom

from automated_phishing_detection.paired_power import noninferiority_power


def build_report(
    positive_sizes=(300, 500, 1000, 1500, 2000, 3000, 4000, 5000, 7500, 10000),
    negative_sizes=(300, 500, 1000, 1500, 2000, 3000, 5000),
):
    rows = []
    for discordance in (0.0, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20):
        for difference in (0.0, -0.005, -0.01):
            if abs(difference) > discordance:
                continue
            for total in positive_sizes:
                power = noninferiority_power(
                    total,
                    (discordance + difference) / 2,
                    (discordance - difference) / 2,
                )
                rows.append(
                    {
                        "independent_positive_pairs": total,
                        "total_discordance_probability": discordance,
                        "true_recall_difference": difference,
                        "power": power,
                        "meets_90_percent": power >= 0.9,
                    }
                )
    fpr_rows = []
    for total in negative_sizes:
        accepted = [
            count
            for count in range(total)
            if beta.isf(0.05, count + 1, total - count) <= 0.01
        ]
        maximum = max(accepted, default=-1)
        for truth in (0.001, 0.003, 0.005, 0.008, 0.01):
            fpr_rows.append(
                {
                    "independent_negative_units": total,
                    "true_fpr": truth,
                    "maximum_false_positives": maximum,
                    "probability_one_sided_upper_below_one_percent": float(
                        binom.cdf(maximum, total, truth)
                    ),
                }
            )
    return {
        "status": "hypothetical_design_sensitivity_not_a_frozen_sample_size",
        "paired_method": "bonferroni_clopper_pearson_discordance",
        "paired_confidence": 0.95,
        "paired_margin": 0.02,
        "power_target": 0.9,
        "assumptions": "IID accurately labeled pairs; no exclusions, clustering or sequential selection",
        "paired_recall": rows,
        "negative_precision_sensitivity": fpr_rows,
        "fpr_note": "Precision planning only; original FPR gates use observed rates, not confidence bounds.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--positive-sizes", nargs="+", type=int)
    parser.add_argument("--negative-sizes", nargs="+", type=int)
    args = parser.parse_args()
    options = {}
    if args.positive_sizes is not None:
        options["positive_sizes"] = tuple(args.positive_sizes)
    if args.negative_sizes is not None:
        options["negative_sizes"] = tuple(args.negative_sizes)
    report = build_report(**options)
    with args.output.open("x") as stream:
        stream.write(json.dumps(report, sort_keys=True, indent=2) + "\n")
    for discordance in (0.01, 0.02, 0.05, 0.10, 0.20):
        for difference in (0.0, -0.005, -0.01):
            candidates = [
                row["independent_positive_pairs"]
                for row in report["paired_recall"]
                if row["total_discordance_probability"] == discordance
                and row["true_recall_difference"] == difference
                and row["meets_90_percent"]
            ]
            print(
                discordance,
                difference,
                "first tested size meeting 90%:",
                min(candidates, default=None),
            )


if __name__ == "__main__":
    main()
