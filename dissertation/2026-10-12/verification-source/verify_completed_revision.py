"""Read-only verification of the sole completed P/F/R/C schedule."""

import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from statistics import median

from automated_phishing_detection import audited_permutation_evidence, execution_receipt
from revision_runner import PINS_DEVELOPMENT, inputs
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parent
RUN = ROOT / "execution-v1"
PROFILE_HASH = "84e4d47f2c7c6249766dc677b6b7c2cad5be8fdaefb65195d786be418463c4d6"
THRESHOLD = 0.2670846328466124
PATHS = (
    "structural",
    "transformer",
    "original",
    "symmetric",
    "directional",
    "symmetric_control",
)


def read(path):
    return json.loads(path.read_bytes())


def lines(path):
    return [json.loads(line) for line in path.read_bytes().splitlines()]


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def quantile(values, probability):
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (position - lower) * (ordered[upper] - ordered[lower])


def close(left, right):
    assert math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12), (left, right)


def masks_for(rows):
    scores = [row["stage1_probability"] for row in rows]
    masks = {
        "structural": [False] * len(rows),
        "transformer": [True] * len(rows),
        "original": [value == THRESHOLD for value in scores],
        "symmetric": [abs(value - THRESHOLD) <= 0.20 for value in scores],
        "directional": [THRESHOLD - 0.20 <= value < THRESHOLD for value in scores],
    }
    ordered = sorted(
        (row["record_id"] for row in rows),
        key=lambda value: (
            hashlib.sha256(("routing-control-20261012:" + value).encode()).hexdigest(),
            value,
        ),
    )
    for name in ("symmetric", "directional"):
        selected = set(ordered[: sum(masks[name])])
        masks[name + "_control"] = [row["record_id"] in selected for row in rows]
    return masks


def verify_permutations(entries):
    family = read(RUN / "P-summary.json")
    assert family["scheduled_seeds"] == list(range(42, 47))
    for name, expected in family["private_sha256"].items():
        assert digest(RUN / "P/evidence" / name) == expected
    diagnostics = []
    for seed in range(42, 47):
        directory = RUN / f"P/seed-{seed}"
        verification = audited_permutation_evidence.verify_completed_member(
            seed=seed,
            attempt=execution_receipt.Attempt(
                directory, digest(directory / "reservation.json")
            ),
            public_path=RUN / f"P/seed-{seed}-summary.json",
            pins=PINS_DEVELOPMENT,
            preparation_summary=entries["preparation"][0].read_bytes(),
            suffix_rules=entries["suffix"][0].read_bytes(),
            train_content=entries["train"][0].read_bytes(),
            validation_content=entries["validation"][0].read_bytes(),
        )
        saved = next(member for member in family["members"] if member["seed"] == seed)
        assert verification == saved["verification"]
        model = read(directory / "evidence/model.json")
        predictions = lines(directory / "evidence/validation-predictions.jsonl")
        scores = [row["probability"] for row in predictions]
        state = model["state"]
        diagnostics.append(
            {
                "seed": seed,
                "iterations": state["n_iter"],
                "iteration_limit": model["parameters"]["classifier"]["max_iter"],
                "coefficient_l2": math.sqrt(
                    sum(value * value for value in state["coefficients"][0])
                ),
                "score_quantiles": {
                    str(probability): quantile(scores, probability)
                    for probability in (0, 0.01, 0.25, 0.5, 0.75, 0.99, 1)
                },
                "permuted_label_changes": verification["permuted_label_changes"],
                "result": saved["result"],
            }
        )
    return diagnostics


def verify_formatting(entries):
    originals = lines(entries["saved"][0])
    rows = lines(RUN / "F/formatting-predictions.jsonl")
    events = lines(RUN / "F/formatting-events.jsonl")
    summary = read(RUN / "F/formatting-summary.json")
    assert len(originals) == len(rows) == summary["rows"] == 8701
    assert len(events) == 3 * len(rows)
    for index, (original, row) in enumerate(zip(originals, rows, strict=True)):
        assert row["record_id"] == original["record"]["record_id"]
        primary = original["primary"]
        assert row["variants"]["original"] == {
            "features": primary["features"],
            "probability": primary["stage1_probability"],
            "decision": primary["stage1_decision"],
        }
        for offset, variant in enumerate(("original", "http", "https")):
            result = row["variants"][variant]
            assert events[3 * index + offset] == {
                "record_id": row["record_id"],
                "variant": variant,
                **result,
            }
            assert result["decision"] == int(result["probability"] >= THRESHOLD)
    report = {}
    for role, saved in summary["groups"].items():
        selected = [row for row in rows if row["role"] == role]
        report[role] = {}
        for variant, expected in saved.items():
            values = [row["variants"][variant] for row in selected]
            cells = Counter(
                (row["is_phishing"], value["decision"])
                for row, value in zip(selected, values, strict=True)
            )
            for name, pair in (
                ("true_positive", (1, 1)),
                ("false_negative", (1, 0)),
                ("false_positive", (0, 1)),
                ("true_negative", (0, 0)),
            ):
                assert expected[name] == cells[pair]
            assert expected["alerts"] == sum(value["decision"] for value in values)
            assert expected["decision_flips"] == sum(
                value["decision"] != row["variants"]["original"]["decision"]
                for row, value in zip(selected, values, strict=True)
            )
            shifts = [
                value["probability"] - row["variants"]["original"]["probability"]
                for row, value in zip(selected, values, strict=True)
            ]
            for name, value in (
                ("min", min(shifts)),
                ("max", max(shifts)),
                ("median", median(shifts)),
                ("mean", sum(shifts) / len(shifts)),
            ):
                close(expected["probability_shift"][name], value)
            report[role][variant] = {
                key: expected[key]
                for key in (
                    "rows",
                    "true_positive",
                    "false_positive",
                    "recall",
                    "fpr",
                    "alerts",
                    "decision_flips",
                )
            }
    return report


def verify_routing(expert_threshold):
    rows = lines(RUN / "R/routing-predictions.jsonl")
    cohort = lines(ROOT / "population-v1/cohort.jsonl")
    assert len(rows) == len(cohort) == 2654
    assert len({row["registrable_domain"] for row in rows}) == len(rows)
    assert len({row["record_id"] for row in rows}) == len(rows)
    for row, original in zip(rows, cohort, strict=True):
        assert {key: row[key] for key in original} == original
    summary = read(RUN / "R/routing-summary.json")
    masks = masks_for(rows)
    structural = [int(row["stage1_probability"] >= THRESHOLD) for row in rows]
    expert = [int(row["transformer_probability"] >= expert_threshold) for row in rows]
    labels = [row["is_phishing"] for row in rows]
    assert sum(labels) == 1392
    for name, mask in masks.items():
        decisions = [
            expensive if use else cheap
            for expensive, cheap, use in zip(expert, structural, mask, strict=True)
        ]
        counts = Counter(zip(labels, decisions, strict=True))
        saved = summary["systems"][name]
        for field, pair in (
            ("true_positive", (1, 1)),
            ("false_negative", (1, 0)),
            ("false_positive", (0, 1)),
            ("true_negative", (0, 0)),
        ):
            assert saved[field] == counts[pair]
        assert saved["logical_calls"] == sum(mask)
        close(saved["fpr"], counts[(0, 1)] / 1262)
        close(saved["recall"], counts[(1, 1)] / 1392)
        for label, population in ((0, "negative"), (1, "positive")):
            assert saved[population + "_repairs"] == sum(
                truth == label and original != label and decision == label
                for truth, original, decision in zip(
                    labels, structural, decisions, strict=True
                )
            )
            assert saved[population + "_harms"] == sum(
                truth == label and original == label and decision != label
                for truth, original, decision in zip(
                    labels, structural, decisions, strict=True
                )
            )
    for saved in summary["contrasts"].values():
        assert saved["cells"] == {
            "both_positive": 1392,
            "candidate_only": 0,
            "reference_only": 0,
            "both_negative": 0,
        }
        radius = 1 - (0.05 / 4) ** (1 / 1392)
        close(saved["exact"]["lower"], -radius)
        close(saved["exact"]["upper"], radius)
    for name, column in (
        ("structural", "stage1_probability"),
        ("transformer", "transformer_probability"),
    ):
        scores = [row[column] for row in rows]
        close(summary["ranking"][name]["roc_auc"], roc_auc_score(labels, scores))
        close(
            summary["ranking"][name]["average_precision"],
            average_precision_score(labels, scores),
        )
    assert read(RUN / "R/actual-counts.json") == {
        "transformer_forward_attempts": 2654,
        "successful_transformer_scores": 2654,
        "completed_requests": 2654,
        "failed_requests": 0,
    }
    return rows, summary


def verify_cost(all_rows, expert_threshold):
    by_id = {row["record_id"]: row for row in all_rows}
    workload = lines(ROOT / "population-v1/workload.jsonl")
    rows = [by_id[row["record_id"]] for row in workload]
    assert len(rows) == 1000 and sum(row["is_phishing"] for row in rows) == 10
    membership = read(RUN / "C/cost-membership.json")
    assert membership["record_ids"] == [row["record_id"] for row in rows]
    masks = masks_for(rows)
    assert membership["masks"] == {name: masks[name] for name in PATHS}
    summary = read(RUN / "C/cost-summary.json")
    assert len(summary["arms"]) == 30
    schedule = read(RUN / "C/cost-started.json")["schedule"]
    assert schedule == [
        {"repetition": repeat, "path": path}
        for repeat in range(5)
        for path in PATHS[repeat:] + PATHS[:repeat]
    ]
    report = {}
    for arm in summary["arms"]:
        path, repeat = arm["path"], arm["repetition"]
        assert arm == read(RUN / f"C/arm-{repeat}-{path}-summary.json")
        records = lines(RUN / f"C/arm-{repeat}-{path}.jsonl")
        assert len(records) == 1000 and arm["warmup_requests_excluded"] == 100
        for index, (record, row) in enumerate(zip(records, rows, strict=True)):
            assert record["record_id"] == row["record_id"]
            assert record["selected"] == masks[path][index]
            selected = record["selected"]
            assert record["stage1_probability"] == (
                None if path == "transformer" else row["stage1_probability"]
            )
            assert record["transformer_probability"] == (
                row["transformer_probability"] if selected else None
            )
            assert (
                record["decision"]
                == int(row["transformer_probability"] >= expert_threshold)
                if selected
                else record["decision"] == int(row["stage1_probability"] >= THRESHOLD)
            )
        durations = [record["elapsed_ns"] for record in records]
        assert all(type(value) is int and value > 0 for value in durations)
        close(arm["summed_scoring_seconds"], sum(durations) / 1e9)
        assert arm["whole_arm_seconds"] >= arm["summed_scoring_seconds"]
        for name, probability in (("p50", 0.5), ("p95", 0.95), ("p99", 0.99)):
            close(arm["latency_ms"][name], quantile(durations, probability) / 1e6)
        close(arm["implied_serial_scoring_requests_per_second"], 1e12 / sum(durations))
        expected_stage1 = 0 if path == "transformer" else 1000
        calls = sum(masks[path])
        assert arm["counts"] == {
            "stage1_started": expected_stage1,
            "stage1_completed": expected_stage1,
            "stage2_started": calls,
            "stage2_completed": calls,
            "requests_completed": 1000,
            "requests_failed": 0,
        }
        assert arm["underlying_transformer_forwards"] == {
            "started": calls,
            "completed": calls,
        }
    for path in PATHS:
        selected = [
            next(
                arm
                for arm in summary["arms"]
                if arm["path"] == path and arm["repetition"] == repeat
            )
            for repeat in range(5)
        ]
        structural = [
            next(
                arm
                for arm in summary["arms"]
                if arm["path"] == "structural" and arm["repetition"] == repeat
            )
            for repeat in range(5)
        ]
        ratios = [
            left["summed_scoring_seconds"] / right["summed_scoring_seconds"]
            for left, right in zip(selected, structural, strict=True)
        ]
        assert summary["elapsed_ratios_to_structural"][path]["paired_ratios"] == ratios
        close(summary["elapsed_ratios_to_structural"][path]["median"], median(ratios))
        report[path] = {
            "calls_per_1000": sum(masks[path]),
            "median_elapsed_ratio": median(ratios),
            "ratio_range": [min(ratios), max(ratios)],
            "median_run_quantiles_ms": {
                key: median(arm["latency_ms"][key] for arm in selected)
                for key in ("p50", "p95", "p99")
            },
        }
    actual = read(RUN / "C/actual-counts-including-warmups.json")
    total = 5 * sum(sum(masks[path]) + sum(masks[path][:100]) for path in PATHS)
    assert actual["transformer"] == {"started": total, "completed": total}
    assert actual["engine"] == {
        "stage1_started": 27500,
        "stage1_completed": 27500,
        "stage2_started": total,
        "stage2_completed": total,
        "requests_completed": 33000,
        "requests_failed": 0,
    }
    return report


def main():
    profile = read(ROOT / "execution-profile-v1.json")
    assert digest(ROOT / "execution-profile-v1.json") == PROFILE_HASH
    assert read(RUN / "profile.json") == profile
    assert read(ROOT / "execution-v1-completed.json") == {
        "all_tasks": ["P", "F", "R", "C"],
        "profile_sha256": PROFILE_HASH,
        "session_closed_cleanly": True,
    }
    assert read(RUN / "parent-completed.json")["completed"] == ["P", "F", "R", "C"]
    assert (
        not list(RUN.rglob("*failure*"))
        and not (ROOT / "execution-v1-failure.json").exists()
    )
    for name, expected in profile["files"].items():
        assert digest(Path(name)) == expected, name
    entries = inputs()
    expert_threshold = read(entries["bundle_transformer.json"][0])[
        "validation_threshold"
    ]["threshold"]
    observations = lines(ROOT / "execution-v1-host.jsonl")
    assert all(
        not row["competing_known_workload_pids"] and "AC Power" in row["battery"]
        for row in observations
    )
    assert all(
        "No thermal warning level has been recorded" in row["thermal"]
        and "No performance warning level has been recorded" in row["thermal"]
        for row in observations
    )
    report = {
        "status": "verified_saved_evidence",
        "new_fits_or_predictions": False,
        "profile_sha256": PROFILE_HASH,
        "bound_files_checked": len(profile["files"]),
        "host_observations": len(observations),
        "host_guard_scope": "AC_thermal_known_command_patterns_not_all_OS_activity",
    }
    report["P"] = verify_permutations(entries)
    report["F"] = verify_formatting(entries)
    rows, report["R"] = verify_routing(expert_threshold)
    report["C"] = verify_cost(rows, expert_threshold)
    report["retained_output_hashes"] = {
        str(path.relative_to(RUN)): digest(path)
        for path in sorted(RUN.rglob("*"))
        if path.is_file()
    }
    destination = ROOT / "completed-evidence-verification.json"
    if destination.exists():
        assert read(destination) == report, (
            "Reconstructed evidence differs from saved verification"
        )
    else:
        with destination.open("x") as stream:
            json.dump(report, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
    print("P/F/R/C saved evidence verified; no model fitting or new predictions.")


if __name__ == "__main__":
    main()
