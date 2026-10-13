"""Independent count reconstruction from saved decisions, without replay imports."""

import hashlib
import json
from pathlib import Path


def independently_count(rows, selected):
    positives = [row for row in rows if row["label"] == 1]
    negatives = [row for row in rows if row["label"] == 0]
    decisions = {
        row["id"]: row["expert"] if row["id"] in selected else row["structural"]
        for row in rows
    }
    true_positive = sum(decisions[row["id"]] for row in positives)
    false_positive = sum(decisions[row["id"]] for row in negatives)
    repairs = {
        label: sum(
            row["id"] in selected
            and row["structural"] != label
            and row["expert"] == label
            for row in rows
            if row["label"] == label
        )
        for label in (0, 1)
    }
    harms = {
        label: sum(
            row["id"] in selected
            and row["structural"] == label
            and row["expert"] != label
            for row in rows
            if row["label"] == label
        )
        for label in (0, 1)
    }
    calls = sum(row["id"] in selected for row in rows)
    return {
        "rows": len(rows),
        "domains": len({row["domain"] for row in rows}),
        "logical_calls": calls,
        "logical_call_fraction": calls / len(rows),
        "alerts": sum(decisions.values()),
        "unlabelled": sum(row["label"] is None for row in rows),
        "positive": len(positives),
        "negative": len(negatives),
        "true_positive": true_positive,
        "false_negative": len(positives) - true_positive,
        "false_positive": false_positive,
        "true_negative": len(negatives) - false_positive,
        "positive_repairs": repairs[1],
        "positive_harms": harms[1],
        "negative_repairs": repairs[0],
        "negative_harms": harms[0],
        "tp_change_from_structural": repairs[1] - harms[1],
        "fp_change_from_structural": harms[0] - repairs[0],
        "recall": true_positive / len(positives) if positives else None,
        "fpr": false_positive / len(negatives) if negatives else None,
    }


def check(context, report_path):
    report_content = report_path.read_bytes()
    report = json.loads(report_content)
    expected_pins = {
        "internal": "827323a5525992332247011e1cb55a661e2b90d4ee350d65eaee0d136dc52c39",
        "external": "c819d7e713c6f27fbdc89ee0b6e3d523db23254c3a303027dd4a38f4cb5f758f",
    }
    assert report["source_sha256"] == expected_pins
    assert report["actual_model_calls_measured"] is False
    assert report["service_cost_measured"] is False
    for key, filename in (
        ("script_sha256", "routing_replay.py"),
        ("specification_sha256", "SAVED_SCORE_ROUTING_SPEC.txt"),
    ):
        assert (
            hashlib.sha256(report_path.with_name(filename).read_bytes()).hexdigest()
            == report[key]
        )
    comparisons = 0
    selections = {}
    for source, digest in expected_pins.items():
        path = (
            context
            / "gwu_working/study-urlnorm-v1-2026-09-30-attempt-4"
            / source
            / "evidence/predictions.jsonl"
        )
        content = path.read_bytes()
        assert hashlib.sha256(content).hexdigest() == digest
        rows = []
        for line in content.splitlines():
            saved = json.loads(line)
            primary = saved["primary"] if "primary" in saved else saved
            record = saved["record"]
            rows.append(
                {
                    "id": record["record_id"],
                    "domain": record["registrable_domain"],
                    "role": record.get("role", "internal"),
                    "label": None
                    if record.get("role") == "tranco"
                    else record["is_phishing"],
                    "score": primary["stage1_probability"],
                    "structural": primary["stage1_decision"],
                    "expert": primary["transformer_decision"],
                }
            )
        assert len({row["id"] for row in rows}) == len(rows)
        masks = {
            "structural": set(),
            "transformer": {row["id"] for row in rows},
            "original": {
                row["id"] for row in rows if row["score"] == 0.2670846328466124
            },
        }
        for direction in ("symmetric", "below", "above"):
            for width in (0.025, 0.05, 0.10, 0.20):
                selected = set()
                for row in rows:
                    score = row["score"]
                    threshold = 0.2670846328466124
                    eligible = (
                        abs(score - threshold) <= width
                        if direction == "symmetric"
                        else threshold - width <= score < threshold
                        if direction == "below"
                        else threshold <= score <= threshold + width
                    )
                    if eligible:
                        selected.add(row["id"])
                masks[f"{direction}-{width:.3f}"] = selected
        assert set(report["results"][source]) == set(masks)
        for name, selected in masks.items():
            for population, expected in report["results"][source][name].items():
                included = (
                    rows
                    if population == "all"
                    else [
                        row
                        for row in rows
                        if row["role"] == population
                        or (
                            population == "gold_plus_certified"
                            and row["role"] in {"gold", "certified"}
                        )
                    ]
                )
                assert independently_count(included, selected) == expected, (
                    source,
                    name,
                    population,
                )
                comparisons += 1
        if source == "internal":
            reference = independently_count(rows, masks["transformer"])["recall"]
            for family, directions in (
                ("symmetric", ("symmetric",)),
                ("directional", ("below", "above")),
            ):
                eligible_candidates = []
                for direction in directions:
                    for width in (0.025, 0.05, 0.10, 0.20):
                        name = f"{direction}-{width:.3f}"
                        metrics = independently_count(rows, masks[name])
                        if (
                            metrics["fpr"] <= 0.01
                            and metrics["recall"] >= reference - 0.02
                            and metrics["logical_call_fraction"] <= 0.30
                        ):
                            eligible_candidates.append(
                                (
                                    -metrics["true_positive"],
                                    metrics["false_positive"],
                                    metrics["logical_calls"],
                                    width,
                                    name,
                                )
                            )
                selections[family] = (
                    min(eligible_candidates)[-1] if eligible_candidates else None
                )
            assert selections == report["internal_development_selection"]
        ordering = sorted(
            (
                hashlib.sha256(
                    ("routing-control-20261012:" + row["id"]).encode()
                ).hexdigest(),
                row["id"],
            )
            for row in rows
        )
        for family, name in selections.items():
            if name is None:
                continue
            selected = {identifier for _, identifier in ordering[: len(masks[name])]}
            for population, expected in report["matched_budget_controls"][source][
                family
            ].items():
                included = (
                    rows
                    if population == "all"
                    else [
                        row
                        for row in rows
                        if row["role"] == population
                        or (
                            population == "gold_plus_certified"
                            and row["role"] in {"gold", "certified"}
                        )
                    ]
                )
                assert independently_count(included, selected) == expected, (
                    source,
                    family,
                    population,
                )
                comparisons += 1
    return {
        "status": "all_reconstructed_counts_and_selections_agree",
        "aggregate_comparisons": comparisons,
        "report_sha256": hashlib.sha256(report_content).hexdigest(),
        "verifier_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "independent_of_replay_implementation": True,
        "new_model_predictions_or_fits": False,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    result = check(arguments.context, arguments.report)
    with arguments.output.open("x") as output:
        output.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2))
