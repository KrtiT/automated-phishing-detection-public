"""Read hash-bound saved outcomes only; never load models or make predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from automated_phishing_detection.paired_inference import (
    PairedTable,
    paired_exact_envelope,
    paired_score_interval,
)

SOURCE_PINS = {
    "external": "c819d7e713c6f27fbdc89ee0b6e3d523db23254c3a303027dd4a38f4cb5f758f",
    "internal": "827323a5525992332247011e1cb55a661e2b90d4ee350d65eaee0d136dc52c39",
}
MODELS = {
    "length_only": "length_decision",
    "logistic_l1": "stage1_decision",
    "cascade": "cascade_decision",
    "transformer": "transformer_decision",
}
CONTRASTS = (
    ("gold", "cascade", "logistic_l1"),
    ("gold", "cascade", "transformer"),
    ("gold", "logistic_l1", "length_only"),
    ("gold", "policy", "cascade"),
    ("internal", "cascade", "logistic_l1"),
    ("internal", "logistic_l1", "length_only"),
)


def verified_rows(path: Path, expected_sha256: str) -> list[dict]:
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != expected_sha256:
        raise ValueError("saved prediction content differs from published verification")
    rows = [json.loads(line) for line in content.splitlines()]
    identifiers = [row["record"]["record_id"] for row in rows]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("saved prediction record IDs are not unique")
    return rows


def decisions(row: dict) -> dict[str, int]:
    primary = row.get("primary", row)
    result = {name: primary[key] for name, key in MODELS.items()}
    if "policy_decision" in row:
        result["policy"] = row["policy_decision"]
    if any(type(value) is not int or value not in (0, 1) for value in result.values()):
        raise ValueError("saved outcome is not a binary integer")
    return result


def paired_summary(rows: list[dict], candidate: str, reference: str) -> dict:
    domain_counts = Counter(row["record"]["registrable_domain"] for row in rows)
    cells = Counter(
        (decisions(row)[candidate], decisions(row)[reference]) for row in rows
    )
    table = PairedTable(*(cells[key] for key in ((1, 1), (1, 0), (0, 1), (0, 0))))
    result = {
        "cells": asdict(table),
        "rows": len(rows),
        "domains": len(domain_counts),
        "maximum_rows_per_domain": max(domain_counts.values()),
        "intervals": [],
    }
    if len(domain_counts) == len(rows):
        result["inference_condition"] = (
            "independent sampled domains assumed; campaigns unverified"
        )
        result["intervals"] = [
            asdict(method(table))
            for method in (paired_score_interval, paired_exact_envelope)
        ]
    else:
        result["inference_condition"] = (
            "clustered rows; independent-pair methods not applied"
        )
    return result


def population_summary(rows: list[dict]) -> dict:
    domains = Counter(row["record"]["registrable_domain"] for row in rows)
    return {
        "rows": len(rows),
        "domains": len(domains),
        "maximum_rows_per_domain": max(domains.values()),
        "alert_counts": {
            model: sum(decisions(row)[model] for row in rows)
            for model in decisions(rows[0])
        },
        "transformer_band_selections": sum(
            row.get("primary", row)["band_selected"] for row in rows
        ),
    }


def verify_published_margins(report: dict, aggregates: Path) -> None:
    original = json.loads((aggregates / "original/primary-results.json").read_text())[
        "primary"
    ]
    for name, entry in report["paired_contrasts"].items():
        cells, saved = entry["cells"], original["contrasts"][name]
        checks = {
            "candidate_true_positives": cells["both_positive"]
            + cells["candidate_only"],
            "reference_true_positives": cells["both_positive"]
            + cells["reference_only"],
            "positive_count": entry["rows"],
            "domain_count": entry["domains"],
        }
        if any(saved[key] != value for key, value in checks.items()):
            raise ValueError(f"published contrast margin differs: {name}")
        entry["historical_interval"] = [saved["lower"], saved["upper"]]
    report["published_contrast_margins_verified"] = True


def build_report(context: Path, aggregates: Path) -> dict:
    retained = context / "gwu_working/study-urlnorm-v1-2026-09-30-attempt-4"
    rows_by_source = {
        role: verified_rows(retained / role / "evidence/predictions.jsonl", pin)
        for role, pin in SOURCE_PINS.items()
    }
    groups = {
        role: [
            row for row in rows_by_source["external"] if row["record"]["role"] == role
        ]
        for role in ("gold", "certified", "tranco", "secondary")
    }
    groups["internal"] = rows_by_source["internal"]
    for role, label in (("gold", 1), ("certified", 0)):
        if any(row["record"]["is_phishing"] != label for row in groups[role]):
            raise ValueError("outcome labels differ from required population")
    positives = {
        role: [row for row in rows if row["record"]["is_phishing"] == 1]
        for role, rows in groups.items()
    }
    contrasts = {
        f"{role}.{candidate}_minus_{reference}": paired_summary(
            positives[role], candidate, reference
        )
        for role, candidate, reference in CONTRASTS
    }
    negative_overlap = paired_summary(groups["certified"], "logistic_l1", "transformer")
    negative_overlap["observed_best_possible_routing_fpr"] = negative_overlap["cells"][
        "both_positive"
    ] / len(groups["certified"])
    report = {
        "stage": "retrospective_analysis_method_comparison_not_new_confirmation",
        "source_sha256": SOURCE_PINS,
        "paired_contrasts": contrasts,
        "certified_error_overlap": negative_overlap,
        "populations": {
            role: population_summary(rows) for role, rows in groups.items()
        },
    }
    verify_published_margins(report, aggregates)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", required=True, type=Path)
    parser.add_argument("--aggregates", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = build_report(args.context, args.aggregates)
    with args.output.open("x") as stream:
        stream.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
