"""Fit once, then evaluate once under the distinct bounded-follow-up manifest."""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
import threadpoolctl

from automated_phishing_detection import (
    baselines,
    fixed_cascade,
    transport_neutral_model,
)
from automated_phishing_detection.followup_metrics import (
    classification_metrics,
    paired_domain_intervals,
)
from automated_phishing_detection.transport_neutral_features import (
    extract_transport_neutral_features,
)

PARTITION_HASHES = {
    "train": "575f2fb13a0766020e29d78bf8e633a185b381abde7060bdd1ed04cc4a5e38a0",
    "validation": "970c6568a6400a1fc265b7809ef7bd9d1c297632799cbb88801313bc34ac415a",
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def execution_versions():
    return {
        "python": platform.python_version(),
        "executable": str(Path(sys.executable).resolve()),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": {
            name: version(name)
            for name in (
                "numpy",
                "scipy",
                "scikit-learn",
                "threadpoolctl",
                "httpx",
                "httpcore",
                "h11",
                "fastapi",
                "pydantic",
                "uvicorn",
                "torch",
            )
        },
    }


def write_json(path, value):
    content = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with path.open("x") as output:
        output.write(content)
    path.chmod(0o600)


def checked_json(path, expected):
    if digest(path) != expected:
        raise ValueError(f"input hash mismatch: {path.name}")
    return json.loads(path.read_bytes())


def verify_execution(context, manifest_path):
    manifest = json.loads(manifest_path.read_bytes())
    repository = Path(__file__).resolve().parents[1]
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repository, text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=repository, text=True
    )
    if dirty or revision != manifest["code_revision"]:
        raise ValueError("follow-up source is not the exact clean frozen commit")
    if manifest["protocol"] != "bounded-followup-comparison-v1":
        raise ValueError("wrong extension identity")
    if manifest["software_versions"] != baselines._software_versions():
        raise ValueError("runtime differs from frozen execution manifest")
    if manifest["runtime_identity"] != execution_versions():
        raise ValueError("execution runtime differs from frozen manifest")
    followup = context / "dissertation/followup-20261001"
    if (
        digest(followup / "comparison-specification-v1.md")
        != manifest["specification_sha256"]
    ):
        raise ValueError("scientific specification changed")
    for name, expected in manifest["clarification_hashes"].items():
        if Path(name).name != name or digest(followup / name) != expected:
            raise ValueError("implementation clarification changed")
    population_path = followup / manifest["population_directory"]
    population = checked_json(
        population_path / "summary.json", manifest["population_summary_sha256"]
    )
    if population["admitted"] is not True:
        raise ValueError("whole-study hold: required population is absent")
    if (
        digest(population_path / "retained.jsonl")
        != population["output_hashes"]["retained.jsonl"]
    ):
        raise ValueError("admitted population changed")
    return manifest, followup, population_path


def partition(context, split):
    directory = (
        context
        / "gwu_working/automated-phishing-detection-public/data/processed/phiusiil-v1-source-schema-v2-label-map-v1-20260903"
    )
    path = directory / f"{split}.jsonl"
    if digest(path) != PARTITION_HASHES[split]:
        raise ValueError("original development partition hash mismatch")
    records = [json.loads(line) for line in path.read_bytes().splitlines()]
    if any(
        row["split"] != split or type(row["is_phishing"]) is not int for row in records
    ):
        raise ValueError("invalid development partition schema")
    return [row["raw_url"] for row in records], np.asarray(
        [row["is_phishing"] for row in records], dtype=np.int8
    )


def fit(context, attempt, baseline_path):
    baseline = fixed_cascade.load_logistic_l1_artifact(baseline_path)
    candidate = transport_neutral_model.fit_model(
        *partition(context, "train"), *partition(context, "validation")
    )
    write_json(attempt / "candidate.json", candidate)
    if candidate["validation_threshold"]["status"] != "selected":
        raise ValueError("candidate failed the frozen validation threshold requirement")
    models = {
        "baseline_sha256": baseline.artifact_sha256,
        "baseline_threshold": baseline.validation_threshold_record["threshold"],
        "candidate_sha256": digest(attempt / "candidate.json"),
        "candidate_threshold": candidate["validation_threshold"]["threshold"],
        "training_partition_sha256": PARTITION_HASHES["train"],
        "validation_partition_sha256": PARTITION_HASHES["validation"],
        "frozen_before_new_benchmark_scoring_at": datetime.now(
            timezone.utc
        ).isoformat(),
    }
    write_json(attempt / "frozen-models.json", models)
    return {
        "status": "fitted_and_calibrated",
        "models": models,
        "external_predictions": 0,
    }


def swapped_scheme(raw):
    scheme, remainder = raw.split(":", 1)
    if scheme.lower() not in ("http", "https"):
        raise ValueError("only admitted absolute HTTP URLs may be transformed")
    return ("https" if scheme.lower() == "http" else "http") + ":" + remainder


def verify_thresholds(models, baseline, candidate):
    if (
        models["baseline_threshold"]
        != baseline.validation_threshold_record["threshold"]
        or candidate["validation_threshold"]["status"] != "selected"
        or models["candidate_threshold"]
        != candidate["validation_threshold"]["threshold"]
    ):
        raise ValueError("frozen threshold binding differs from model artifacts")


def evaluate(attempt, followup, population_path, baseline_path, fit_manifest_hash):
    fitting = followup / "detection-fit-v1"
    fit_intent = json.loads((fitting / "intent.json").read_bytes())
    if fit_intent["execution_manifest_sha256"] != fit_manifest_hash:
        raise ValueError("fit did not use this frozen execution manifest")
    completion = json.loads((fitting / "completion.json").read_bytes())
    if completion["status"] != "fitted_and_calibrated":
        raise ValueError("candidate fitting did not complete")
    models = json.loads((fitting / "frozen-models.json").read_bytes())
    if completion["models"] != models:
        raise ValueError("frozen model binding differs from completed fit")
    candidate = checked_json(fitting / "candidate.json", models["candidate_sha256"])
    baseline = fixed_cascade.load_logistic_l1_artifact(baseline_path)
    if baseline.artifact_sha256 != models["baseline_sha256"]:
        raise ValueError("baseline changed")
    verify_thresholds(models, baseline, candidate)
    rows = [
        json.loads(line)
        for line in (population_path / "retained.jsonl").read_bytes().splitlines()
    ]
    outputs = []
    with (attempt / "predictions.jsonl").open("x") as output:
        os.chmod(output.name, 0o600)
        for row in rows:
            record = {
                key: row[key]
                for key in ("record_id", "registrable_domain", "is_phishing")
            }
            record["candidate_feature_equal"] = extract_transport_neutral_features(
                row["raw_url"]
            ) == extract_transport_neutral_features(swapped_scheme(row["raw_url"]))
            for variant, raw in (
                ("original", row["raw_url"]),
                ("scheme_swap", swapped_scheme(row["raw_url"])),
            ):
                baseline_scores, baseline_audit = (
                    fixed_cascade.score_logistic_l1_authoritative(baseline, [raw])
                )
                candidate_scores, candidate_audit = transport_neutral_model.score_urls(
                    candidate, [raw]
                )
                record[variant] = {
                    "baseline": baseline_scores[0],
                    "candidate": candidate_scores[0],
                    "baseline_audit": baseline_audit,
                    "candidate_audit": candidate_audit,
                }
            output.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
            output.flush()
            outputs.append(record)
            if len(outputs) % 1000 == 0:
                print(
                    f"retained {len(outputs)}/{len(rows)} paired observations",
                    flush=True,
                )
    labels = np.asarray([row["is_phishing"] for row in outputs])
    domains = [row["registrable_domain"] for row in outputs]
    decisions, metrics, invariance = {}, {}, {}
    for name in ("baseline", "candidate"):
        scores = np.asarray([row["original"][name] for row in outputs])
        swapped = np.asarray([row["scheme_swap"][name] for row in outputs])
        threshold = models[f"{name}_threshold"]
        decisions[name] = (scores >= threshold).astype(int)
        metrics[name] = classification_metrics(labels, scores, threshold)
        invariance[name] = {
            "exact_score_matches": int(np.sum(scores == swapped)),
            "decision_flips": int(
                np.sum((scores >= threshold) != (swapped >= threshold))
            ),
            "max_absolute_score_difference": float(np.max(np.abs(scores - swapped))),
            "denominator": len(outputs),
        }
    invariance["candidate"]["exact_feature_matches"] = sum(
        row["candidate_feature_equal"] for row in outputs
    )
    paired = paired_domain_intervals(
        labels, decisions["baseline"], decisions["candidate"], domains
    )
    interval = paired["recall_difference"]["interval_97_5"]
    requirements = {
        "recall_gain_at_least_five_points": paired["recall_difference"]["point"]
        >= 0.05,
        "recall_interval_lower_above_zero": interval is not None and interval[0] > 0,
        "observed_fpr_at_most_one_percent": metrics["candidate"]["fpr"] <= 0.01,
        "candidate_exact_scheme_invariance": (
            invariance["candidate"]["exact_score_matches"] == len(outputs)
            and invariance["candidate"]["exact_feature_matches"] == len(outputs)
        ),
    }
    return {
        "status": "evaluated",
        "protocol": "bounded-followup-detection-v1",
        "models": models,
        "metrics": metrics,
        "paired_uncertainty": paired,
        "scheme_invariance": invariance,
        "requirements": requirements,
        "D_requirement_met": all(requirements.values()),
        "predictions_sha256": digest(attempt / "predictions.jsonl"),
        "original_hypothesis_decisions_changed": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--phase", choices=("fit", "evaluate"), required=True)
    arguments = parser.parse_args()
    context = arguments.context_root.resolve(strict=True)
    manifest, followup, population = verify_execution(context, arguments.manifest)
    attempt = followup / f"detection-{arguments.phase}-v1"
    attempt.mkdir(mode=0o700)
    manifest_hash = digest(arguments.manifest)
    write_json(
        attempt / "intent.json",
        {
            "phase": arguments.phase,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "execution_manifest_sha256": manifest_hash,
            "manifest": manifest,
        },
    )
    baseline = (
        context / "gwu_working/study-only-v1-reviewed-inputs/primary/logistic-l1.json"
    )
    try:
        with threadpoolctl.threadpool_limits(limits=1):
            if any(
                pool["num_threads"] != 1 for pool in threadpoolctl.threadpool_info()
            ):
                raise ValueError("one-thread numerical runtime not established")
            if arguments.phase == "fit":
                result = fit(context, attempt, baseline)
            else:
                result = evaluate(
                    attempt, followup, population, baseline, manifest_hash
                )
        result["completed_at"] = datetime.now(timezone.utc).isoformat()
        write_json(attempt / "completion.json", result)
        print(json.dumps(result, indent=2, allow_nan=False))
    except BaseException as error:
        write_json(
            attempt / "failure.json",
            {
                "type": type(error).__name__,
                "message": str(error),
                "at": datetime.now(timezone.utc).isoformat(),
            },
        )
        raise


if __name__ == "__main__":
    main()
