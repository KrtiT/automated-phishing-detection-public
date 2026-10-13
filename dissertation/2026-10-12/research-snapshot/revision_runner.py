"""One bounded private post-review run; no retries or standalone task authority."""

import argparse
import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import torch
from audit_saved_predictions import SOURCE_PINS, verified_rows
from retained_domain_preflight import PINS
from revision_admission import (
    TASKS,
    LiveParent,
    file_hash,
    fingerprint,
    guarded_session,
)
from revision_analysis import force_scheme
from revision_measurements import RoutingEngine, run_cost
from revision_population import (
    EXPECTED_COUNTS,
    _bytes,
    _write,
    checked_history,
    select_workload,
)
from revision_preflight import verify_selection
from revision_tasks import run_cohort, run_formatting
from routing_replay import THRESHOLD

from automated_phishing_detection import (
    audited_permutation_family,
    audited_permutation_member,
    execution_receipt,
    fixed_cascade,
    gmm_monitor,
)
from automated_phishing_detection.protocol_preflight import parse_suffix_rules
from automated_phishing_detection.secondary_development import DevelopmentPins
from automated_phishing_detection.selective_inference import SelectiveCascade
from automated_phishing_detection.transformer_inference import (
    load_transformer_cascade_bundle,
)
from automated_phishing_detection.url_features import (
    FEATURE_NAMES,
    extract_url_features,
)

REVISION = Path(__file__).resolve().parent
CONTEXT = REVISION.parents[1]
RESEARCH = CONTEXT / "gwu_working/dissertation-publication-20261004"
PREPARATION = CONTEXT / "gwu_working/study-urlnorm-v1-2026-09-30-attempt-4/preparation"
PRIMARY = CONTEXT / "gwu_working/study-only-v1-reviewed-inputs/primary"
POPULATION = REVISION / "population-v1"
PINS_DEVELOPMENT = DevelopmentPins(
    train_sha256="575f2fb13a0766020e29d78bf8e633a185b381abde7060bdd1ed04cc4a5e38a0",
    validation_sha256="970c6568a6400a1fc265b7809ef7bd9d1c297632799cbb88801313bc34ac415a",
    source_csv_sha256="a236549cd369cd80bd478ff8e1779cbf44c58d5c3f79f7a51a1adbed7d06d1c6",
    preparation_summary_sha256="1a85a7eecc0f5baa7c59e03a0cbde63fd4595409feb918dc5ff916ead7cd5c9e",
    suffix_rules_sha256=PINS["suffix"],
    logistic_l1_artifact_sha256="71a3e24a0283a31ba188bc7dd60b18c1b708370b9ca275d5ab1a1004680c968a",
    baseline_contract_sha256="05d6d0831def7d26448c8dbdc8117800ea2448cdfc2aca2ad95489f22d2d11ba",
    gmm_artifact_sha256="a01a8b143c2423df57a153462cc47e79822ad1c6768213dc69e03683d4009745",
    gmm_contract_sha256="22d32088b05e74432704f9671ab76ba28b4f573ead418846b23bc366315cb393",
)
BUNDLE_HASHES = {
    "cascade.json": "7ac88c784dbc299d436a029904f0c6bde5a8cf741e62028a7635e8672e55119c",
    "transformer-weights.npz": "1d4cdef31cb23cb84f093ca61c0afe0318142fa45ae559acd78cf10b49ee5de7",
    "transformer.json": "a13b6b7d554db6a9ee5b1689ecef44e2ad8069fcd25967f931101bdd3b256727",
    "vocabulary.json": "68bda780006d07b3b849abc366fbe3ccffe8a29e8984b092396d04f4eef43579",
    "SHA256SUMS": "a09ae653b8d518fe686da0a6fcdfabc9552f677dccaf55a4ec62548aa02c0e88",
}
SUMMARY_HASH = "41499aa388babe60442de7231b4087f67a53f96f340568a7cc58a3268606a2fd"
REFERENCE_HASH = "bf5b3a6f0fc705d26852da4dd0053c6111ffc3e500d7a2e95dfba5ad859b279c"


def execute(parent, operations):
    if set(operations) != set(TASKS) or not all(
        callable(item) for item in operations.values()
    ):
        raise ValueError("all_task_implementations_required")
    parent.start()
    try:
        for name in TASKS:
            callback = parent.admit(name)
            print(f"{name}: started", flush=True)
            operations[name](callback)
            parent.complete(name)
            print(f"{name}: completed and retained", flush=True)
        parent.finish()
    except BaseException as error:
        parent.stop(error)
        raise


def git(*arguments):
    return subprocess.check_output(["git", *arguments], cwd=RESEARCH, text=True).strip()


def runtime():
    gmm_monitor._require_runtime()
    if platform.python_version() != "3.10.19" or not torch.backends.mps.is_available():
        raise ValueError("required_python_or_mps_unavailable")
    if torch.__version__.split("+")[0] != "2.7.1":
        raise ValueError("required_torch_unavailable")
    return {
        "python": platform.python_version(),
        "executable": str(Path(sys.executable).resolve()),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": {
            name: version(name)
            for name in ("numpy", "scipy", "scikit-learn", "threadpoolctl", "torch")
        },
        "blas": gmm_monitor._numpy_build_configuration(),
        "numerical_threads": 1,
        "device": "mps",
        "singleton": True,
    }


def inputs():
    partitions = (
        CONTEXT
        / "gwu_working/automated-phishing-detection-public/data/processed/phiusiil-v1"
    )
    entries = {
        "train": (partitions / "train.jsonl", PINS_DEVELOPMENT.train_sha256),
        "validation": (
            partitions / "validation.jsonl",
            PINS_DEVELOPMENT.validation_sha256,
        ),
        "preparation": (
            RESEARCH / "reports/phiusiil-preparation-summary.json",
            PINS_DEVELOPMENT.preparation_summary_sha256,
        ),
        "suffix": (PREPARATION / "suffix-rules.dat", PINS["suffix"]),
        "source": (PREPARATION / "publisher-source.json", PINS["source"]),
        "overlap": (PREPARATION / "source-overlap.json", PINS["overlap"]),
        "followup": (
            CONTEXT
            / "dissertation/followup-20261001/population-admission-v1-attempt-2/retained.jsonl",
            PINS["followup"],
        ),
        "stage1": (
            PRIMARY / "logistic-l1.json",
            PINS_DEVELOPMENT.logistic_l1_artifact_sha256,
        ),
        "summary": (
            RESEARCH / "reports/rq1-transformer-cascade-v2-summary.json",
            SUMMARY_HASH,
        ),
        "reference": (
            RESEARCH / "reports/rq1-baseline-v2-summary.json",
            REFERENCE_HASH,
        ),
        "saved": (
            PREPARATION.parent / "external/evidence/predictions.jsonl",
            SOURCE_PINS["external"],
        ),
        "cohort": (
            POPULATION / "cohort.jsonl",
            "7853a877fdfba4be4bcade69ddb425bcaa5b02cf725b24d0d41323b7fcf37c67",
        ),
        "workload": (
            POPULATION / "workload.jsonl",
            "03cbba8e6201a76af3c57cf4989080ef8ba2e60c25972ba66e00fb5356cf3dae",
        ),
        "roster": (
            REVISION / "retained-exposure-audit-v2.json",
            "e5d06e06c733b80cbc6ca1e49bd5faf6a0c79139e9e46feeb51a137b39951717",
        ),
    }
    for name, digest in BUNDLE_HASHES.items():
        entries["bundle_" + name] = (PRIMARY / "transformer-bundle" / name, digest)
    return entries


def rows(path):
    return [json.loads(line) for line in path.read_bytes().splitlines()]


def prepare(profile_path):
    if git("status", "--porcelain"):
        raise ValueError("academic_source_must_be_clean_before_profile")
    entries = inputs()
    files = {str(path): expected for path, expected in entries.values()}
    for path, expected in files.items():
        if file_hash(path) != expected:
            raise ValueError("pinned_input_changed:" + Path(path).name)
    print(
        "Known input hashes verified; reconstructing retained eligibility.", flush=True
    )
    roster = json.loads(entries["roster"][0].read_bytes())
    rules = parse_suffix_rules(entries["suffix"][0].read_text())
    excluded = checked_history(CONTEXT, roster, rules)
    excluded.update(json.loads(entries["overlap"][0].read_bytes())["domains"])
    excluded.update(row["registrable_domain"] for row in rows(entries["followup"][0]))
    cohort = rows(entries["cohort"][0])
    verify_selection(
        json.loads(entries["source"][0].read_bytes()), rules, excluded, cohort
    )
    observed_counts = {
        name: sum(row["population"] == name for row in cohort)
        for name in EXPECTED_COUNTS
    }
    if observed_counts != EXPECTED_COUNTS or select_workload(cohort) != rows(
        entries["workload"][0]
    ):
        raise ValueError("frozen_population_or_workload_differs")
    for entry in roster["inventory"]:
        files[str(CONTEXT / entry["path"])] = entry["sha256"]
    print("Independent selection agrees; verifying P and F prerequisites.", flush=True)
    audited_permutation_member._validated_rows(
        entries["train"][0].read_bytes(),
        entries["validation"][0].read_bytes(),
        entries["preparation"][0].read_bytes(),
        entries["suffix"][0].read_bytes(),
        PINS_DEVELOPMENT,
    )
    saved = verified_rows(entries["saved"][0], SOURCE_PINS["external"])
    if len(saved) != 8701:
        raise ValueError("formatting_population_changed")
    for row in saved:
        for scheme in ("http", "https"):
            force_scheme(row["record"]["raw_url"], scheme)
    specifications = [
        "EXPERIMENT_SPEC.txt",
        "PERMUTATION_DIAGNOSTIC_PROTOCOL.txt",
        "SAVED_SCORE_ROUTING_SPEC.txt",
        "SOURCE_PROVENANCE_REVIEW.txt",
        "EXECUTION_CLARIFICATION.txt",
    ]
    local = [
        *sorted(REVISION.glob("*.py")),
        *(REVISION / name for name in specifications),
        POPULATION / "summary.json",
    ]
    source = [
        RESEARCH / name
        for name in git(
            "ls-files", "src", "scripts", "pyproject.toml", "uv.lock"
        ).splitlines()
    ]
    for path in local + source:
        files[str(path)] = file_hash(path)
    profile = {
        "kind": "advisor_revision_20261012_PFRC_single_attempt",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "code_revision": git("rev-parse", "HEAD"),
        "runtime": runtime(),
        "files": files,
        "readiness": dict.fromkeys(TASKS, True),
        "task_order": TASKS,
        "expected_cohort": EXPECTED_COUNTS,
        "independent_selection_verified": True,
        "new_predictions_at_freeze": 0,
        "authorization": {
            "basis": "User instruction to proceed on the approved revision plan and delegated implementation decisions in this conversation on 2026-10-12.",
            "advisor_approval_claimed": False,
            "user_review_of_profile_digest_claimed": False,
            "scope": "Retained inputs only; one P/F/R/C schedule; preserve all outcomes; no destination visits, original archive reopening, automatic restart or publication of raw identities.",
        },
        "population_specification_sha256": json.loads(
            (POPULATION / "summary.json").read_bytes()
        )["specification_sha256"],
        "current_specification_sha256": file_hash(REVISION / "EXPERIMENT_SPEC.txt"),
        "timing_clarification_sha256": file_hash(
            REVISION / "EXECUTION_CLARIFICATION.txt"
        ),
        "retry_authorized": False,
    }
    for path, expected in files.items():
        if file_hash(path) != expected:
            raise ValueError("input_changed_during_preparation")
    _write(profile_path, _bytes(profile))
    print(
        "Complete P/F/R/C readiness profile frozen; no fits or predictions performed.",
        flush=True,
    )


def perform(profile_path, expected_hash):
    if file_hash(profile_path) != expected_hash:
        raise ValueError("profile_identity_mismatch")
    profile = json.loads(profile_path.read_bytes())
    if (
        git("status", "--porcelain")
        or git("rev-parse", "HEAD") != profile["code_revision"]
    ):
        raise ValueError("frozen_academic_revision_changed")
    if runtime() != profile["runtime"]:
        raise ValueError("frozen_runtime_changed")
    root = REVISION / "execution-v1"
    claim = REVISION / "execution-v1.claim.json"
    _write(claim, _bytes({"profile_sha256": expected_hash, "owner_pid": os.getpid()}))
    profile_identity = fingerprint(profile_path)
    entries = inputs()
    parent = None
    try:
        with guarded_session(REVISION / "execution-v1-host.jsonl") as session_check:

            def check_session():
                session_check()
                if fingerprint(profile_path) != profile_identity:
                    raise ValueError("live_profile_changed")

            def check_source():
                if (
                    git("status", "--porcelain")
                    or git("rev-parse", "HEAD") != profile["code_revision"]
                ):
                    raise ValueError("frozen_academic_revision_changed")

            parent = LiveParent(profile, root, check_session, check_source)
            parent.verify_files(full=True)
            loaded = load_transformer_cascade_bundle(
                PRIMARY / "transformer-bundle",
                entries["summary"][0],
                entries["stage1"][0],
                expected_public_summary_sha256=SUMMARY_HASH,
                device=torch.device("mps"),
            )
            if loaded.stage1_threshold != THRESHOLD or loaded.half_width != 0:
                raise ValueError("original_threshold_or_gate_changed")
            shared = {}

            def permutation(callback):
                attempt = execution_receipt.reserve_attempt(
                    root / "P",
                    identity={
                        "kind": "post_review_permutation_family",
                        "profile_sha256": expected_hash,
                    },
                )
                audited_permutation_family.run_reserved_family(
                    attempt=attempt,
                    public_path=root / "P-summary.json",
                    pins=PINS_DEVELOPMENT,
                    preparation_summary=entries["preparation"][0].read_bytes(),
                    suffix_rules=entries["suffix"][0].read_bytes(),
                    train_content=entries["train"][0].read_bytes(),
                    validation_content=entries["validation"][0].read_bytes(),
                    protocol_sha256=profile["files"][
                        str(REVISION / "PERMUTATION_DIAGNOSTIC_PROTOCOL.txt")
                    ],
                    accepted_comparison=entries["reference"][0].read_bytes(),
                    accepted_comparison_sha256=REFERENCE_HASH,
                    require_admission=callback,
                )

            def structural(raw):
                probabilities, unused = fixed_cascade.score_logistic_l1_authoritative(
                    loaded.stage1_model, (raw,)
                )
                return probabilities[0]

            def formatting(callback):
                destination = root / "F"
                destination.mkdir(mode=0o700)
                run_formatting(
                    verified_rows(entries["saved"][0], SOURCE_PINS["external"]),
                    structural,
                    extract_url_features,
                    FEATURE_NAMES,
                    destination,
                    callback,
                )

            with SelectiveCascade(loaded) as scorer:

                def forwards():
                    count = scorer.counts
                    return {
                        "started": count.transformer_forward_attempts,
                        "completed": count.successful_transformer_scores,
                    }

                def cohort(callback):
                    destination = root / "R"
                    destination.mkdir(mode=0o700)
                    before = asdict(scorer.counts)
                    predictions, unused = run_cohort(
                        rows(entries["cohort"][0]),
                        scorer,
                        loaded.transformer_threshold,
                        destination,
                        callback,
                    )
                    delta = {
                        name: value - before[name]
                        for name, value in asdict(scorer.counts).items()
                    }
                    if delta != {
                        "transformer_forward_attempts": 2654,
                        "successful_transformer_scores": 2654,
                        "completed_requests": 2654,
                        "failed_requests": 0,
                    }:
                        raise ValueError("cohort_actual_forward_counts_differ")
                    _write(destination / "actual-counts.json", _bytes(delta))
                    shared["predictions"] = predictions

                def cost(callback):
                    destination = root / "C"
                    destination.mkdir(mode=0o700)
                    predicted = {row["record_id"]: row for row in shared["predictions"]}
                    workload = [
                        predicted[row["record_id"]]
                        for row in rows(entries["workload"][0])
                    ]
                    engine = RoutingEngine(
                        structural,
                        lambda raw: scorer.scan_transformer(raw).probability,
                        transformer_threshold=loaded.transformer_threshold,
                        forward_counts=forwards,
                    )
                    before = forwards()
                    run_cost(workload, engine, destination, callback)
                    actual = {
                        name: value - before[name] for name, value in forwards().items()
                    }
                    if actual != {
                        "started": engine.counts["stage2_started"],
                        "completed": engine.counts["stage2_completed"],
                    }:
                        raise ValueError("whole_cost_actual_forward_counts_differ")
                    _write(
                        destination / "actual-counts-including-warmups.json",
                        _bytes({"engine": engine.counts, "transformer": actual}),
                    )

                execute(
                    parent, {"P": permutation, "F": formatting, "R": cohort, "C": cost}
                )
    except BaseException as error:
        _write(
            REVISION / "execution-v1-failure.json",
            _bytes(
                {
                    "error_type": type(error).__name__,
                    "reason": str(error),
                    "parent_started": parent is not None and parent.started,
                }
            ),
        )
        raise
    _write(
        REVISION / "execution-v1-completed.json",
        _bytes(
            {
                "profile_sha256": expected_hash,
                "all_tasks": list(TASKS),
                "session_closed_cleanly": True,
            }
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run"))
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--profile-sha256")
    arguments = parser.parse_args()
    if arguments.action == "prepare":
        prepare(arguments.profile.resolve())
    else:
        perform(arguments.profile.resolve(), arguments.profile_sha256)


if __name__ == "__main__":
    main()
