"""Bounded development-only routing replay; never fit, predict or measure time."""

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from audit_saved_predictions import SOURCE_PINS, verified_rows

THRESHOLD = 0.2670846328466124
WIDTHS = (0.025, 0.05, 0.10, 0.20)
GATE_SHA256 = "7ac88c784dbc299d436a029904f0c6bde5a8cf741e62028a7635e8672e55119c"


def gate_mask(scores, family, width, *, threshold=THRESHOLD):
    if family not in {"symmetric", "below", "above"}:
        raise ValueError("unknown gate family")
    if not math.isfinite(width) or not 0 <= width <= 1 or not 0 <= threshold <= 1:
        raise ValueError("invalid gate boundary")
    if any(type(score) not in (int, float) or not 0 <= score <= 1 for score in scores):
        raise ValueError("invalid stage-one score")
    if family == "symmetric":
        return [abs(score - threshold) <= width for score in scores]
    if family == "below":
        return [threshold - width <= score < threshold for score in scores]
    return [threshold <= score <= threshold + width for score in scores]


def random_mask(identifiers, count):
    if (
        len(set(identifiers)) != len(identifiers)
        or type(count) is not int
        or not 0 <= count <= len(identifiers)
    ):
        raise ValueError("invalid matched-budget control")
    ranked = sorted(
        identifiers,
        key=lambda identifier: (
            hashlib.sha256(
                ("routing-control-20261012:" + identifier).encode()
            ).hexdigest(),
            identifier,
        ),
    )
    selected = set(ranked[:count])
    return [identifier in selected for identifier in identifiers]


def summarize(rows, mask):
    if (
        not rows
        or len(rows) != len(mask)
        or any(type(selected) is not bool for selected in mask)
    ):
        raise ValueError("nonempty aligned boolean routing required")
    counts = Counter()
    domains = set()
    for row, selected in zip(rows, mask, strict=True):
        primary = row.get("primary", row)
        record = row["record"]
        structural, expert = primary["stage1_decision"], primary["transformer_decision"]
        if any(
            type(value) is not int or value not in (0, 1)
            for value in (structural, expert)
        ):
            raise ValueError("expert decisions must be binary integers")
        decision = expert if selected else structural
        domains.add(record["registrable_domain"])
        counts["logical_calls"] += selected
        counts["alerts"] += decision
        label = None if record.get("role") == "tranco" else record["is_phishing"]
        if label is None:
            counts["unlabelled"] += 1
            continue
        if type(label) is not int or label not in (0, 1):
            raise ValueError("invalid reference label")
        group = "positive" if label else "negative"
        counts[group] += 1
        counts[
            ("true_" if decision == label else "false_")
            + ("positive" if decision else "negative")
        ] += 1
        counts[group + "_repairs"] += (
            selected and structural != label and expert == label
        )
        counts[group + "_harms"] += selected and structural == label and expert != label
    fields = (
        "logical_calls",
        "alerts",
        "unlabelled",
        "positive",
        "negative",
        "true_positive",
        "false_negative",
        "false_positive",
        "true_negative",
        "positive_repairs",
        "positive_harms",
        "negative_repairs",
        "negative_harms",
    )
    return {name: counts[name] for name in fields} | {
        "rows": len(rows),
        "domains": len(domains),
        "logical_call_fraction": counts["logical_calls"] / len(rows),
        "recall": counts["true_positive"] / counts["positive"]
        if counts["positive"]
        else None,
        "fpr": counts["false_positive"] / counts["negative"]
        if counts["negative"]
        else None,
        "tp_change_from_structural": counts["positive_repairs"]
        - counts["positive_harms"],
        "fp_change_from_structural": counts["negative_harms"]
        - counts["negative_repairs"],
    }


def verify_original(rows, *, threshold=THRESHOLD):
    primary = [row.get("primary", row) for row in rows]
    scores = [row["stage1_probability"] for row in primary]
    mask = gate_mask(scores, "symmetric", 0, threshold=threshold)
    for row, selected in zip(primary, mask, strict=True):
        expected = row["transformer_decision"] if selected else row["stage1_decision"]
        if (
            type(row["band_selected"]) is not bool
            or row["band_selected"] != selected
            or row["cascade_decision"] != expected
        ):
            raise ValueError("original cascade replay differs")
    return {
        "rows": len(rows),
        "exact_threshold_ties": sum(mask),
        "minimum_distance_to_threshold": min(
            abs(score - threshold) for score in scores
        ),
        "saved_mask_and_decisions_verified": True,
    }


def select(candidates, *, reference_recall):
    eligible = [
        entry
        for entry in candidates
        if entry["metrics"]["fpr"] <= 0.01
        and entry["metrics"]["recall"] >= reference_recall - 0.02
        and entry["metrics"]["logical_call_fraction"] <= 0.30
    ]
    if not eligible:
        return None
    return min(
        eligible,
        key=lambda entry: (
            -entry["metrics"]["true_positive"],
            entry["metrics"]["false_positive"],
            entry["metrics"]["logical_calls"],
            entry["width"],
            entry["name"],
        ),
    )["name"]


def _candidate_definitions():
    return [
        {
            "name": f"{direction}-{width:.3f}",
            "family": "symmetric" if direction == "symmetric" else "directional",
            "direction": direction,
            "width": width,
        }
        for direction in ("symmetric", "below", "above")
        for width in WIDTHS
    ]


def _by_population(rows, mask, source):
    populations = {"all": list(range(len(rows)))}
    if source == "external":
        for role in ("gold", "certified", "secondary", "tranco"):
            populations[role] = [
                index for index, row in enumerate(rows) if row["record"]["role"] == role
            ]
        populations["gold_plus_certified"] = (
            populations["gold"] + populations["certified"]
        )
    return {
        name: summarize(
            [rows[index] for index in indices], [mask[index] for index in indices]
        )
        for name, indices in populations.items()
        if indices
    }


def build_report(context):
    retained = context / "gwu_working/study-urlnorm-v1-2026-09-30-attempt-4"
    gate = (
        context
        / "gwu_working/study-only-v1-reviewed-inputs/primary/transformer-bundle/cascade.json"
    ).read_bytes()
    if hashlib.sha256(gate).hexdigest() != GATE_SHA256:
        raise ValueError("accepted gate metadata changed")
    calibration = json.loads(gate)["calibration"]
    if calibration["half_width"] != 0.0:
        raise ValueError("accepted band no longer zero")
    sources = {
        name: verified_rows(retained / name / "evidence/predictions.jsonl", pin)
        for name, pin in SOURCE_PINS.items()
    }
    original = {name: verify_original(rows) for name, rows in sources.items()}
    definitions = _candidate_definitions()
    results = {}
    masks = {}
    for source, rows in sources.items():
        scores = [row.get("primary", row)["stage1_probability"] for row in rows]
        source_masks = {
            "structural": [False] * len(rows),
            "transformer": [True] * len(rows),
            "original": gate_mask(scores, "symmetric", 0),
        }
        source_masks.update(
            {
                entry["name"]: gate_mask(scores, entry["direction"], entry["width"])
                for entry in definitions
            }
        )
        masks[source] = source_masks
        results[source] = {
            name: _by_population(rows, mask, source)
            for name, mask in source_masks.items()
        }
    reference = results["internal"]["transformer"]["all"]["recall"]
    selection = {
        family: select(
            [
                entry | {"metrics": results["internal"][entry["name"]]["all"]}
                for entry in definitions
                if entry["family"] == family
            ],
            reference_recall=reference,
        )
        for family in ("symmetric", "directional")
    }
    controls = {}
    for source, rows in sources.items():
        identifiers = [row["record"]["record_id"] for row in rows]
        controls[source] = {
            family: _by_population(
                rows, random_mask(identifiers, sum(masks[source][name])), source
            )
            for family, name in selection.items()
            if name is not None
        }
    certified = [
        row for row in sources["external"] if row["record"]["role"] == "certified"
    ]
    both_wrong = sum(
        row["primary"]["stage1_decision"] == row["primary"]["transformer_decision"] == 1
        for row in certified
    )
    specification = (
        Path(__file__).with_name("SAVED_SCORE_ROUTING_SPEC.txt").read_bytes()
    )
    return {
        "analysis_role": "post_review_development_replay_not_confirmation",
        "source_sha256": SOURCE_PINS,
        "gate_sha256": GATE_SHA256,
        "specification_sha256": hashlib.sha256(specification).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "original_validation_calibration": calibration,
        "original_replay": original,
        "candidates": definitions,
        "internal_development_selection": selection,
        "results": results,
        "matched_budget_controls": controls,
        "certified_fixed_decision_oracle": {
            "shared_errors": both_wrong,
            "negatives": len(certified),
            "fpr_floor": both_wrong / len(certified),
        },
        "actual_model_calls_measured": False,
        "service_cost_measured": False,
        "privacy": "Aggregate counts only; no URLs or individual identifiers.",
        "interpretation": "Exposed outcomes; no population confidence claims, timing inference or primary hypothesis replacement. External all-row mixture is descriptive; use gold recall and certified FPR separately.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(args.context)
    with args.output.open("x") as stream:
        stream.write(
            json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n"
        )
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "original_replay",
                    "internal_development_selection",
                    "certified_fixed_decision_oracle",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
