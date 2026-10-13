"""Reconcile original decisions with a separately identified corrected analysis."""

import argparse
import csv
import hashlib
import io
import json
import operator
from collections import Counter
from pathlib import Path

from automated_phishing_detection.paired_inference import (
    PairedTable,
    paired_exact_envelope,
)

PINS = {
    "primary-gates.csv": "35f2c142d3858ea5e9d791c7b513a9c2946fb858079943f092dd5a72603a07f6",
    "paired-and-dependence-audit.json": "86e8dc03c923e1a0810558e08292f3ea480ebf9c9b862c5583135b857d8928d6",
}
OPERATORS = {"<": operator.lt, "<=": operator.le, ">": operator.gt, ">=": operator.ge}
GOLD_FIXED = {
    "gold.logistic_l1_minus_length_only": (">", 0.0),
    "gold.cascade_minus_logistic_l1": (">", 0.0),
    "gold.cascade_minus_transformer": (">=", -0.02),
}
DEGENERATE = {
    "internal.cascade_minus_logistic_l1": "domain_clustered_degenerate_bootstrap",
    "gold.policy_minus_cascade": "stateful_policy_sequence_dependence",
}


def corrected_rows(gates, audit):
    identifiers = [(row["hypothesis"], row["name"]) for row in gates]
    if (
        len(set(identifiers)) != 22
        or len(gates) != 22
        or Counter(row["hypothesis"] for row in gates) != {"H1": 10, "H2": 4, "H3": 8}
        or not set(GOLD_FIXED).issubset(row["name"] for row in gates)
        or audit.get("published_contrast_margins_verified") is not True
    ):
        raise ValueError("original gate family or verified counts differ")
    result = []
    for original in gates:
        name = original["name"]
        row = {
            "hypothesis": original["hypothesis"],
            "name": name,
            "original_status": original["status"],
            "corrected_status": original["status"],
            "original_estimate": original["estimate"],
            "original_operator": original["operator"],
            "original_threshold": original["threshold"],
            "corrected_threshold": original["threshold"],
            "corrected_estimate": float(original["estimate"]),
            "corrected_lower": None,
            "corrected_upper": None,
            "method": "unchanged_original_rule",
            "interpretation": "Original metric and decision retained; not a new experiment.",
        }
        if name in GOLD_FIXED:
            comparison, threshold = GOLD_FIXED[name]
            if (original["operator"], float(original["threshold"])) != (
                comparison,
                threshold,
            ):
                raise ValueError("original recall margin or direction differs")
            entry = audit["paired_contrasts"][name]
            table = PairedTable(**entry["cells"])
            if entry["rows"] != table.total or entry["domains"] != table.total:
                raise ValueError("independent-pair counts or domain alignment differ")
            interval = paired_exact_envelope(table)
            row.update(
                corrected_estimate=interval.estimate,
                corrected_lower=interval.lower,
                corrected_upper=interval.upper,
                corrected_status="pass"
                if OPERATORS[comparison](interval.lower, threshold)
                else "fail",
                method=interval.method,
                interpretation="Conditional on independent labeled positive pairs; distinct domains do not prove campaign independence. Fail means criterion not established, not inferiority proved.",
            )
        elif name in DEGENERATE:
            entry = audit["paired_contrasts"][name]
            table = PairedTable(**entry["cells"])
            if (
                table.candidate_only
                or table.reference_only
                or original["status"] != "fail"
            ):
                raise ValueError("degenerate contrast outcomes differ")
            row.update(
                corrected_estimate=0.0,
                method=DEGENERATE[name],
                interpretation="Observed paired difference is zero; not population certainty. No valid positive lower bound established; original superiority criterion remains unmet.",
            )
        elif name.endswith((".fpr", ".alert_rate")):
            row.update(
                method="unchanged_observed_rate_gate",
                interpretation="Observed URL rate decides the original rule. Historical row-wise binomial bound is not certified for dependent URLs.",
            )
        elif name == "internal.logistic_l1_minus_length_only":
            entry = audit["paired_contrasts"][name]
            table = PairedTable(**entry["cells"])
            row.update(
                corrected_estimate=(table.candidate_only - table.reference_only)
                / table.total,
                corrected_lower=entry["historical_interval"][0],
                corrected_upper=entry["historical_interval"][1],
                method="original_domain_clustered_bootstrap",
                interpretation="Original nondegenerate whole-domain resampling retained, conditional on independently sampled domains; estimates URL-weighted recall difference.",
            )
        elif name.endswith("window_alerts"):
            row["interpretation"] = (
                "Observed fraction of overlapping windows; windows are not independent Bernoulli trials. Original descriptive rate rule retained."
            )
        result.append(row)
    return result


def read_verified(path):
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != PINS[path.name]:
        raise ValueError("analysis input identity differs")
    return content.decode("utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    gates = list(csv.DictReader(io.StringIO(read_verified(args.original))))
    rows = corrected_rows(gates, json.loads(read_verified(args.audit)))
    with args.output.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(
        json.dumps(
            {
                "checks": len(rows),
                "original_passes": sum(
                    row["original_status"] == "pass" for row in rows
                ),
                "corrected_passes_under_stated_assumptions": sum(
                    row["corrected_status"] == "pass" for row in rows
                ),
                "joint_supported": {
                    hypothesis: all(
                        row["corrected_status"] == "pass"
                        for row in rows
                        if row["hypothesis"] == hypothesis
                    )
                    for hypothesis in ("H1", "H2", "H3")
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
