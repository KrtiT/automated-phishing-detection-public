"""One no-fit, development-only comparison of the two frozen inference paths."""

from __future__ import annotations

import argparse
import ast
import io
import json
import math
import os
import platform
import stat
import subprocess
import sys
import tempfile
import time
import warnings
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

MODELS = ("length-only", "Logistic-L1", "transformer", "cascade")
PROBABILITY_MODELS = MODELS[:3]
ROOT = Path(__file__).resolve().parents[1]
REPORT = "reports/inference-compatibility-v1.json"
CONTRACT = "data/evaluation-contract-v1.json"
TRANSFORMER_DIR = "data/processed/rq1-transformer-cascade-v2"
BASELINE_DIR = "data/processed/rq1-baselines-v2"
GMM_DIR = "data/processed/rq2-gmm-development-v1"
VALIDATION = "data/processed/phiusiil-v1/validation.jsonl"
PREPARATION = "reports/phiusiil-preparation-summary.json"
BASELINE_SUMMARY = "reports/rq1-baseline-v2-summary.json"
TRANSFORMER_SUMMARY = "reports/rq1-transformer-cascade-v2-summary.json"
GMM_SUMMARY = "reports/rq2-gmm-development-v1-summary.json"
INPUTS = {
    VALIDATION: "970c6568a6400a1fc265b7809ef7bd9d1c297632799cbb88801313bc34ac415a",
    PREPARATION: "1a85a7eecc0f5baa7c59e03a0cbde63fd4595409feb918dc5ff916ead7cd5c9e",
    BASELINE_SUMMARY: "bf5b3a6f0fc705d26852da4dd0053c6111ffc3e500d7a2e95dfba5ad859b279c",
    TRANSFORMER_SUMMARY: "41499aa388babe60442de7231b4087f67a53f96f340568a7cc58a3268606a2fd",
    GMM_SUMMARY: "6f695138a302e854e1e5af590152e289486affe8ccdf75510ca9a5dcaad3b523",
    f"{BASELINE_DIR}/length-only.json": "b8b92cfbe29160e769e5e7d80712becc8fc0680cdfd45a44b839ef9bada87799",
    f"{BASELINE_DIR}/logistic-l1.json": "71a3e24a0283a31ba188bc7dd60b18c1b708370b9ca275d5ab1a1004680c968a",
    f"{TRANSFORMER_DIR}/cascade.json": "7ac88c784dbc299d436a029904f0c6bde5a8cf741e62028a7635e8672e55119c",
    f"{TRANSFORMER_DIR}/transformer-weights.npz": "1d4cdef31cb23cb84f093ca61c0afe0318142fa45ae559acd78cf10b49ee5de7",
    f"{TRANSFORMER_DIR}/transformer.json": "a13b6b7d554db6a9ee5b1689ecef44e2ad8069fcd25967f931101bdd3b256727",
    f"{TRANSFORMER_DIR}/vocabulary.json": "68bda780006d07b3b849abc366fbe3ccffe8a29e8984b092396d04f4eef43579",
    f"{GMM_DIR}/gmm.json": "a01a8b143c2423df57a153462cc47e79822ad1c6768213dc69e03683d4009745",
    f"{GMM_DIR}/validation-audit.json": "fed5058ef72565629eed65c19f3f639634dd6734cba67e7fe22ac80190b19d6a",
    "data/rq1-baseline-contract-v2.json": "05d6d0831def7d26448c8dbdc8117800ea2448cdfc2aca2ad95489f22d2d11ba",
    "data/rq1-transformer-cascade-contract-v2.json": "686c0d86b33b8a6c2e09cd6e174003db0bd2f7c30b087faf5470e6a270524213",
    "data/rq2-gmm-development-contract-v1.json": "22d32088b05e74432704f9671ab76ba28b4f573ead418846b23bc366315cb393",
}
RUNTIME_REQUIREMENTS = {
    "device": "mps",
    "inference_batch_size": 1,
    "numpy_blas": {"name": "scipy-openblas", "version": "0.3.29"},
    "threadpoolctl_limits": 1,
}
BRIDGE_REQUIREMENTS = {
    "id": "development-inference-compatibility-v1",
    "reference": "accelerate_mps_original_validation_batch512",
    "candidate": "openblas_mps_singleton",
    "acceptance": "exact_decisions_band_masks_and_separate_stream_alerts",
    "reference_counts": "accepted_four_model_validation_counts",
    "gmm_reference": "exact_saved_calibration_and_audit_window_scores",
    "report": REPORT,
    "retry": False,
    "protected_evaluation_ready": False,
}
CORRECTION_REQUIREMENTS = {
    "id": "development-inference-compatibility-v1-preflight-correction",
    "report": "reports/inference-compatibility-v1-preflight-correction.json",
    "prior_receipt": REPORT,
    "prior_receipt_sha256": "6f27b23a88e40455a186ce21cddebec0f9ab827919c663b345a6a14ca14bb670",
    "prior_head": "6977cec7acaa5491caaad8b1bba140b4134a7fcf",
    "prior_contract_sha256": "0a42fc0f27abc611431cbbf1263bb9c2f02264942332ed8478b8a305e0dae9b9",
    "maximum_executions": 1,
    "automatic_retry": False,
}
REFERENCE_COMMIT = "e866441f2ff858472d031b8d358fd469897c6a65"
REFERENCE_FUNCTIONS = {
    "src/automated_phishing_detection/character_transformer.py": (
        "CharacterTransformer",
        "_evaluate_validation",
        "_device_flags_pass",
    ),
    "src/automated_phishing_detection/transformer_pipeline.py": ("_encode_partition",),
    "src/automated_phishing_detection/baselines.py": (
        "_audited_validation_scores",
        "_reference_score_differences",
        "_score_with_warning_policy",
    ),
    "src/automated_phishing_detection/fixed_cascade.py": (
        "score_logistic_l1_authoritative",
        "PortableLogisticL1",
    ),
}
_CHILD_STAGE = "arguments"
CHILD_STAGES = {
    "arguments",
    "environment_preflight",
    "contract_and_source_verification",
    "input_hash_verification",
    "validation_parsing",
    "artifact_loading",
    "gmm_reference_and_candidate",
    "reference_transformer_scoring",
    "candidate_transformer_scoring",
    "postread_verification",
    "private_publication",
}


def _read_verified(path, expected_sha256):
    if not stat.S_ISREG(path.lstat().st_mode):
        raise ValueError("input must be a regular file")
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), "rb") as stream:
        before = os.fstat(stream.fileno())
        content = stream.read()
        after = os.fstat(stream.fileno())
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise ValueError("input changed while being read")
    if sha256(content).hexdigest() != expected_sha256:
        raise ValueError("input SHA-256 mismatch")
    return content


def _json_bytes(value):
    return (
        json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()


def _write_private(path, value):
    content = _json_bytes(value)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _publish(path, value):
    content = _json_bytes(value)
    descriptor, temporary = tempfile.mkstemp(prefix=".compatibility-", dir=path.parent)
    temporary = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fchmod(stream.fileno(), 0o644)
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _verify_contract(path, digest):
    value = json.loads(_read_verified(path, digest))
    if value.get("protected_evaluation_ready") is not False:
        raise ValueError("bridge rules cannot declare protected evaluation ready")
    for section, expected in (
        ("runtime_candidate", RUNTIME_REQUIREMENTS),
        ("compatibility_bridge", BRIDGE_REQUIREMENTS),
    ):
        actual = value.get(section, {})
        for key, required in expected.items():
            if _json_bytes(actual.get(key)) != _json_bytes(required):
                raise ValueError(
                    "prospective bridge rules do not match this implementation"
                )
    return value


def _git(root, *arguments):
    return subprocess.run(
        ["git", "-C", str(root), *arguments], check=True, capture_output=True
    ).stdout


def _source_bindings(root):
    if _git(root, "status", "--porcelain=v1", "--untracked-files=all"):
        raise ValueError("bridge execution requires a clean committed checkout")
    head = _git(root, "rev-parse", "HEAD").decode().strip()
    paths = (
        _git(
            root,
            "ls-files",
            "src/*.py",
            "scripts/inference_compatibility.py",
            "uv.lock",
            "pyproject.toml",
            CONTRACT,
        )
        .decode()
        .splitlines()
    )
    bindings = {}
    for name in paths:
        committed = _git(root, "show", f"{head}:{name}")
        digest = sha256(committed).hexdigest()
        _read_verified(root / name, digest)
        bindings[name] = digest
    if "scripts/inference_compatibility.py" not in bindings:
        raise ValueError("bridge script must be committed")
    return {
        "head": head,
        "source_sha256": bindings,
        "original_reference_commit": REFERENCE_COMMIT,
        "original_function_sha256": _reference_fidelity(root),
    }


def _reference_fidelity(root):
    result = {}
    for path, names in REFERENCE_FUNCTIONS.items():
        original = ast.parse(_git(root, "show", f"{REFERENCE_COMMIT}:{path}"))
        current = ast.parse((root / path).read_bytes())
        for name in names:
            bodies = []
            for tree in (original, current):
                node = next(
                    (node for node in tree.body if getattr(node, "name", None) == name),
                    None,
                )
                if node is None:
                    raise ValueError("original reference function is absent")
                bodies.append(ast.dump(node, include_attributes=False).encode())
            if bodies[0] != bodies[1]:
                raise ValueError("original reference scoring function changed")
            result[f"{path}:{name}"] = sha256(bodies[0]).hexdigest()
    return result


def _accepted_counts(root):
    baseline = json.loads(
        _read_verified(root / BASELINE_SUMMARY, INPUTS[BASELINE_SUMMARY])
    )
    transformer = json.loads(
        _read_verified(root / TRANSFORMER_SUMMARY, INPUTS[TRANSFORMER_SUMMARY])
    )
    return {
        **{
            name: baseline["models"][name]["validation_threshold"]["counts"]
            for name in MODELS[:2]
        },
        "transformer": transformer["transformer"]["threshold"]["counts"],
        "cascade": transformer["cascade"]["counts"],
    }


def _accepted_band_count(root):
    summary = json.loads(
        _read_verified(root / TRANSFORMER_SUMMARY, INPUTS[TRANSFORMER_SUMMARY])
    )
    return summary["cascade"]["transformer_invocations"]


def _partition(content, preparation):
    from automated_phishing_detection import baselines, gmm_monitor

    features, labels, _, _, _ = baselines._load_partition(
        io.BytesIO(content),
        split="validation",
        declared=preparation["splits"]["validation"],
        source_csv_sha256=preparation["source_csv_sha256"],
    )
    records = [json.loads(line) for line in content.splitlines()]
    return {
        "features": features,
        "labels": labels,
        "raw_urls": tuple(row["raw_url"] for row in records),
        "domains": tuple(
            gmm_monitor._normalize_domain(row["registrable_domain"]) for row in records
        ),
        "record_ids": tuple(row["record_id"] for row in records),
    }


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _verify_preflight_correction(root, contract):
    policy = CORRECTION_REQUIREMENTS
    declared = contract.get("compatibility_bridge", {}).get("preflight_correction")
    if type(declared) is not dict or any(
        _json_bytes(declared.get(key)) != _json_bytes(value)
        for key, value in policy.items()
    ):
        raise ValueError(
            "the exact preflight correction must be prospectively declared"
        )
    prior = json.loads(
        _read_verified(root / policy["prior_receipt"], policy["prior_receipt_sha256"])
    )
    expected = {
        "schema_version": 1,
        "contract_id": BRIDGE_REQUIREMENTS["id"],
        "head": policy["prior_head"],
        "contract_sha256": policy["prior_contract_sha256"],
        "status": "failed",
        "failure_stage": "candidate_environment_preflight",
        "fit_performed": False,
        "protected_evaluation_ready": False,
    }
    if any(
        _json_bytes(prior.get(key)) != _json_bytes(value)
        for key, value in expected.items()
    ):
        raise ValueError("predecessor is not the authenticated preflight-only failure")
    processes = prior.get("processes")
    if type(processes) is not list or len(processes) != 2:
        raise ValueError("predecessor must contain exactly two preflight processes")
    for process, role, code in zip(
        processes, ("reference", "candidate"), (0, 2), strict=True
    ):
        expected_process = {"role": role, "phase": "preflight", "exit_code": code}
        if role == "candidate":
            expected_process["failure_stage"] = "environment_preflight"
        if any(
            _json_bytes(process.get(key)) != _json_bytes(value)
            for key, value in expected_process.items()
        ):
            raise ValueError("predecessor contains a different process outcome")
    scientific_fields = {
        "models",
        "gmm",
        "row_count",
        "comparison_execution",
        "band_mismatch_count",
        "reference_band_selected_count",
        "candidate_band_selected_count",
        "original_counts",
        "gmm_reference",
        "original_band_selected_count",
        "accepted_band_selected_count",
    }
    if scientific_fields & prior.keys():
        raise ValueError(
            "preflight correction cannot follow scientific scoring or comparison"
        )
    return {
        "prior_receipt": policy["prior_receipt"],
        "prior_receipt_sha256": policy["prior_receipt_sha256"],
        "prior_head": policy["prior_head"],
        "prior_contract_sha256": policy["prior_contract_sha256"],
        "change": "initialize_torch_thread_runtime_before_limiting",
        "maximum_executions": 1,
        "automatic_retry": False,
    }


def _run_bridge(
    reference_python,
    candidate_python,
    contract_hash,
    *,
    preflight_correction=False,
    _root=ROOT,
    _invoke=None,
):
    if type(preflight_correction) is not bool:
        raise ValueError("preflight correction flag must be boolean")
    destination = _root / (
        CORRECTION_REQUIREMENTS["report"] if preflight_correction else REPORT
    )
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("the compatibility receipt already exists")
    contract = _verify_contract(_root / CONTRACT, contract_hash)
    correction = (
        _verify_preflight_correction(_root, contract) if preflight_correction else None
    )
    bindings = _source_bindings(_root)
    invoke = _invoke or _invoke_child
    receipt = {
        "schema_version": 1,
        "contract_id": CORRECTION_REQUIREMENTS["id"]
        if preflight_correction
        else BRIDGE_REQUIREMENTS["id"],
        "analysis_stage": "development_validation_only",
        "protected_evaluation_ready": False,
        "fit_performed": False,
        "contract_sha256": contract_hash,
        **bindings,
        "input_sha256": INPUTS,
        "started_at": _utc_now(),
        "environments": {},
        "processes": [],
        "access": {
            "group_test_accessed": False,
            "phishvn_accessed": False,
            "scope": "this_process_and_its_children_only",
        },
    }
    if correction is not None:
        receipt["preflight_correction"] = correction
    start = time.monotonic()
    stage = "environment_preflight"
    try:
        for role, interpreter in (
            ("reference", reference_python),
            ("candidate", candidate_python),
        ):
            stage = f"{role}_environment_preflight"
            environment, process = invoke(role, interpreter, contract_hash)
            receipt["environments"][role] = environment
            receipt["processes"].append(process)
        stage = "accepted_counts_authentication"
        accepted = _accepted_counts(_root)
        accepted_band_count = _accepted_band_count(_root)
        with tempfile.TemporaryDirectory(prefix="phishing-compatibility-") as directory:
            os.chmod(directory, 0o700)
            lanes = {}
            for role, interpreter in (
                ("reference", reference_python),
                ("candidate", candidate_python),
            ):
                stage = f"{role}_scoring"
                output = Path(directory) / f"{role}.json"
                _, process = invoke(role, interpreter, contract_hash, output)
                receipt["processes"].append(process)
                if stat.S_IMODE(output.stat().st_mode) != 0o600:
                    raise ValueError("private evidence mode is invalid")
                lanes[role] = json.loads(output.read_bytes())
                if role == "reference":
                    original_counts = {
                        name: _counts(
                            lanes[role]["labels"], lanes[role]["decisions"][name]
                        )
                        for name in MODELS
                    }
                    if original_counts != accepted:
                        receipt.update(
                            status="not_equivalent",
                            reason="original_validation_counts_not_reproduced",
                            original_counts=original_counts,
                        )
                        break
                    if lanes[role]["band_selected"].count(True) != accepted_band_count:
                        receipt.update(
                            status="not_equivalent",
                            reason="original_band_selection_count_not_reproduced",
                            original_band_selected_count=lanes[role][
                                "band_selected"
                            ].count(True),
                            accepted_band_selected_count=accepted_band_count,
                        )
                        break
                elif lanes[role].get("reference_failure"):
                    receipt.update(
                        status="not_equivalent",
                        reason="original_gmm_traces_not_reproduced",
                        gmm_reference=lanes[role]["reference_failure"],
                    )
            if "status" not in receipt:
                stage = "paired_comparison"
                receipt.update(
                    _compare_lanes(lanes["reference"], lanes["candidate"], accepted)
                )
        stage = "final_source_verification"
        if _source_bindings(_root) != bindings:
            raise ValueError("source bindings changed during execution")
    except (Exception, KeyboardInterrupt) as error:
        receipt.update(
            status="failed", failure_type=type(error).__name__, failure_stage=stage
        )
        if isinstance(error, ChildProcessError):
            receipt["processes"].append(error.process_record)
    receipt.update(finished_at=_utc_now(), elapsed_seconds=time.monotonic() - start)
    _publish(destination, receipt)
    return receipt


def _vector(value, length, kind):
    if type(value) is not list or len(value) != length:
        raise ValueError("private lane vector length is invalid")
    if kind == "binary":
        valid = all(type(item) is int and item in (0, 1) for item in value)
    elif kind == "boolean":
        valid = all(type(item) is bool for item in value)
    else:
        valid = all(
            type(item) in (int, float) and math.isfinite(item) for item in value
        )
        if kind == "probability":
            valid = valid and all(0 <= item <= 1 for item in value)
    if not valid:
        raise ValueError("private lane vector values are invalid")
    return value


def _counts(labels, decisions):
    return {
        "true_positive": sum(
            label == 1 and decision == 1
            for label, decision in zip(labels, decisions, strict=True)
        ),
        "false_positive": sum(
            label == 0 and decision == 1
            for label, decision in zip(labels, decisions, strict=True)
        ),
        "true_negative": sum(
            label == 0 and decision == 0
            for label, decision in zip(labels, decisions, strict=True)
        ),
        "false_negative": sum(
            label == 1 and decision == 0
            for label, decision in zip(labels, decisions, strict=True)
        ),
        "positive": labels.count(1),
        "negative": labels.count(0),
    }


def _maximum_difference(left, right):
    return max(abs(a - b) for a, b in zip(left, right, strict=True))


def _execution_summary(reference, candidate, rows):
    counts = candidate["execution_counts"]
    expected = {
        "transformer_forward_attempts": rows,
        "successful_transformer_scores": rows,
        "completed_requests": rows,
        "failed_requests": 0,
    }
    if _json_bytes(counts) != _json_bytes(expected):
        raise ValueError("candidate physical execution counts are inconsistent")
    execution = reference["execution_counts"]
    if execution["reference_batch_size"] != 512:
        raise ValueError("reference batch size is invalid")
    audits = {}
    for name, field in (
        ("length-only", "length_scoring_audit"),
        ("Logistic-L1", "stage1_scoring_audit"),
    ):
        audit = execution[field]
        audits[name] = {"warning_count": len(audit["warning_records"])}
        for key in (
            "max_absolute_decision_difference",
            "max_absolute_probability_difference",
        ):
            _vector([audit[key]], 1, "finite")
            audits[name][key] = audit[key]
    return {
        "transformer_evaluated_for_every_validation_row": True,
        "reference_batch_size": 512,
        "candidate": expected,
        "reference_scoring_audits": audits,
    }


def _compare_lanes(reference, candidate, accepted_counts):
    ids = reference["record_ids"]
    if (
        type(ids) is not list
        or not ids
        or any(type(item) is not str or not item for item in ids)
        or len(set(ids)) != len(ids)
        or ids != candidate["record_ids"]
    ):
        raise ValueError("private lane record alignment is invalid")
    size = len(ids)
    for lane in (reference, candidate):
        _vector(lane["labels"], size, "binary")
        _vector(lane["band_selected"], size, "boolean")
        for name in MODELS:
            _vector(lane["decisions"][name], size, "binary")
        for name in PROBABILITY_MODELS:
            _vector(lane["probabilities"][name], size, "probability")
    if reference["labels"] != candidate["labels"]:
        raise ValueError("private lane label alignment is invalid")
    models = {}
    for name in MODELS:
        left, right = reference["decisions"][name], candidate["decisions"][name]
        models[name] = {
            "reference_counts": _counts(reference["labels"], left),
            "candidate_counts": _counts(reference["labels"], right),
            "reference_reproduces_accepted_counts": _counts(reference["labels"], left)
            == accepted_counts[name],
            "decision_mismatch_count": sum(
                a != b for a, b in zip(left, right, strict=True)
            ),
        }
        if name in PROBABILITY_MODELS:
            models[name]["max_absolute_probability_difference"] = _maximum_difference(
                reference["probabilities"][name], candidate["probabilities"][name]
            )
    band_mismatches = sum(
        a != b
        for a, b in zip(
            reference["band_selected"], candidate["band_selected"], strict=True
        )
    )
    gmm = candidate["gmm"]
    threshold = gmm["threshold"]
    _vector([threshold], 1, "finite")
    for name in ("reference_nll", "candidate_nll"):
        _vector(gmm[name], size, "finite")
    streams = {}
    for name in ("calibration", "audit"):
        left = gmm["streams"][name]["reference_scores"]
        right = gmm["streams"][name]["candidate_scores"]
        if not left:
            raise ValueError("GMM streams require complete windows")
        _vector(left, len(left), "finite")
        _vector(right, len(left), "finite")
        streams[name] = {
            "window_count": len(left),
            "reference_alert_count": sum(score > threshold for score in left),
            "candidate_alert_count": sum(score > threshold for score in right),
            "alert_mismatch_count": sum(
                (a > threshold) != (b > threshold)
                for a, b in zip(left, right, strict=True)
            ),
            "max_absolute_window_score_difference": _maximum_difference(left, right),
        }
    equivalent = (
        not band_mismatches
        and all(
            row["reference_reproduces_accepted_counts"]
            and not row["decision_mismatch_count"]
            for row in models.values()
        )
        and all(not row["alert_mismatch_count"] for row in streams.values())
        and gmm["reference_reproduces_saved_traces"] is True
    )
    return {
        "status": "equivalent_on_development_validation"
        if equivalent
        else "not_equivalent",
        "protected_evaluation_ready": False,
        "comparison_execution": _execution_summary(reference, candidate, size),
        "row_count": size,
        "models": models,
        "band_mismatch_count": band_mismatches,
        "reference_band_selected_count": reference["band_selected"].count(True),
        "candidate_band_selected_count": candidate["band_selected"].count(True),
        "gmm": {
            "reference_reproduces_saved_traces": gmm[
                "reference_reproduces_saved_traces"
            ],
            "max_absolute_nll_difference": _maximum_difference(
                gmm["reference_nll"], gmm["candidate_nll"]
            ),
            "streams": streams,
        },
    }


def _reference_transformer(model, tokens, padding, labels, device):
    from torch.utils.data import DataLoader, TensorDataset

    from automated_phishing_detection import character_transformer

    loader = DataLoader(
        TensorDataset(tokens, padding, labels),
        batch_size=character_transformer.VALIDATION_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )
    _, probabilities = character_transformer._evaluate_validation(model, loader, device)
    return probabilities


def _gmm_comparison(partition, stage1, artifact, traces, summary):
    import numpy as np

    from automated_phishing_detection import gmm_monitor as gm

    allocation = gm.allocate_validation_domains(partition["domains"])
    reference_nll = gm.score_feature_matrix(
        gm._build_features(partition, stage1), artifact
    )
    streams, reproduction, reproduced = {}, {}, True
    for name, indices in allocation.items():
        ends, scores = gm.window_scores(reference_nll[list(indices)])
        expected = {
            "domains": sorted({partition["domains"][i] for i in indices}),
            "record_ids": [partition["record_ids"][i] for i in indices],
            "input_row_positions": list(indices),
            "window_end_positions": list(ends),
        }
        if any(traces[name][key] != value for key, value in expected.items()):
            raise ValueError("saved GMM stream alignment is invalid")
        reproduced = reproduced and list(scores) == traces[name]["window_scores"]
        reproduced = reproduced and len(scores) == summary[f"{name}_window_count"]
        reproduction[name] = {
            "window_count": len(scores),
            "saved_window_count": len(traces[name]["window_scores"]),
            "exact_saved_window_scores_reproduced": list(scores)
            == traces[name]["window_scores"],
        }
        streams[name] = {"reference_scores": list(scores)}
    reproduced = (
        reproduced
        and sum(
            score > summary["threshold"]
            for score in streams["audit"]["reference_scores"]
        )
        == summary["audit_alert_count"]
    )
    if not reproduced:
        return {
            "reference_reproduces_saved_traces": False,
            "reference_failure": reproduction,
        }
    candidate_nll = []
    for index, raw_url in enumerate(partition["raw_urls"]):
        # Preserve feature 26's portable scorer; the classifier path is different.
        row = np.column_stack(
            (partition["features"][index : index + 1], stage1.score_urls((raw_url,)))
        )
        candidate_nll.append(float(gm.score_feature_matrix(row, artifact)[0]))
    for name, indices in allocation.items():
        _, scores = gm.window_scores(
            np.asarray(candidate_nll, dtype=np.float64)[list(indices)]
        )
        streams[name]["candidate_scores"] = list(scores)
    return {
        "reference_reproduces_saved_traces": bool(reproduced),
        "reference_nll": reference_nll.tolist(),
        "candidate_nll": candidate_nll,
        "threshold": summary["threshold"],
        "streams": streams,
    }


def _environment(role):
    if (
        os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") != "0"
        or os.environ.get("PYTORCH_MPS_FAST_MATH", "0") != "0"
    ):
        raise ValueError("MPS fallback and fast math must be disabled")
    import threadpoolctl
    import torch

    from automated_phishing_detection import gmm_monitor as gm

    versions = gm._software_versions()
    versions.update(
        python=platform.python_version(),
        pytorch=torch.__version__.split("+", maxsplit=1)[0],
    )
    required = {
        "python": "3.10.19",
        "pytorch": "2.7.1",
        "numpy": "2.2.6",
        "scipy": "1.15.3",
        "scikit-learn": "1.7.2",
        "threadpoolctl": "3.6.0",
    }
    if (
        versions != required
        or sys.platform != "darwin"
        or platform.machine() != "arm64"
        or not torch.backends.mps.is_available()
    ):
        raise ValueError("bridge requires the declared Apple silicon MPS versions")
    # PyTorch's first query can initialize OpenMP and overwrite an active limit.
    entry_torch_threads = torch.get_num_threads()
    blas = gm._numpy_build_configuration()
    if role == "reference":
        if blas["name"] != "accelerate":
            raise ValueError("reference requires original Accelerate runtime")
    elif role == "candidate":
        gm._require_runtime()
        with threadpoolctl.threadpool_limits(limits=1):
            pools = threadpoolctl.threadpool_info()
            if (
                not pools
                or any(pool.get("num_threads") != 1 for pool in pools)
                or torch.get_num_threads() != 1
            ):
                raise ValueError("candidate runtime must honor one numerical thread")
    else:
        raise ValueError("unknown runtime role")
    return {
        "role": role,
        "versions": versions,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy_blas": blas,
        "interpreter": sys.executable,
        "torch_intraop_threads_before_scoring": entry_torch_threads,
        "threadpools_before_scoring": [
            {
                key: pool.get(key)
                for key in ("internal_api", "prefix", "version", "num_threads")
            }
            for pool in threadpoolctl.threadpool_info()
        ],
    }


def _score_models(partition, loaded, length, role, *, _fixture_cpu=False):
    import torch

    from automated_phishing_detection import (
        character_transformer,
        fixed_cascade,
        length_inference,
        transformer_pipeline,
    )
    from automated_phishing_detection.selective_inference import SelectiveCascade

    probabilities = {name: [] for name in PROBABILITY_MODELS}
    candidate_band, candidate_fixed = [], []
    if role == "reference":
        with (
            warnings.catch_warnings(),
            torch.autocast(device_type=loaded.device.type, enabled=False),
        ):
            warnings.simplefilter("error")
            character_transformer.configure_deterministic_runtime()
            probabilities["length-only"], length_audit = (
                length_inference.score_length_only_authoritative(
                    length, partition["raw_urls"]
                )
            )
            probabilities["Logistic-L1"], stage1_audit = (
                fixed_cascade.score_logistic_l1_authoritative(
                    loaded.stage1_model, partition["raw_urls"]
                )
            )
            tokens, padding, labels = transformer_pipeline._encode_partition(
                SimpleNamespace(**partition), loaded.vocabulary
            )
            probabilities["transformer"] = _reference_transformer(
                loaded._model, tokens, padding, labels, loaded.device
            )
        execution = {
            "reference_batch_size": 512,
            "length_scoring_audit": length_audit,
            "stage1_scoring_audit": stage1_audit,
        }
    else:
        with SelectiveCascade(loaded, _fixture_cpu=_fixture_cpu) as scorer:
            for raw_url in partition["raw_urls"]:
                length_scores, _ = length_inference.score_length_only_authoritative(
                    length, (raw_url,)
                )
                result = scorer.score_all(raw_url)
                probabilities["length-only"].append(length_scores[0])
                probabilities["Logistic-L1"].append(result.stage1_probability)
                probabilities["transformer"].append(result.transformer_probability)
                candidate_band.append(result.band_selected)
                candidate_fixed.append(result.fixed_decision)
            from dataclasses import asdict

            execution = asdict(scorer.counts)
    band = [
        abs(score - loaded.stage1_threshold) <= loaded.half_width
        for score in probabilities["Logistic-L1"]
    ]
    thresholds = {
        "length-only": length.validation_threshold_record["threshold"],
        "Logistic-L1": loaded.stage1_threshold,
        "transformer": loaded.transformer_threshold,
    }
    decisions = {
        name: [int(score >= thresholds[name]) for score in probabilities[name]]
        for name in PROBABILITY_MODELS
    }
    decisions["cascade"] = [
        decisions["transformer"][index] if selected else decisions["Logistic-L1"][index]
        for index, selected in enumerate(band)
    ]
    if role == "candidate":
        band = candidate_band
        decisions["cascade"] = candidate_fixed
    return {
        "record_ids": list(partition["record_ids"]),
        "labels": partition["labels"].tolist(),
        "probabilities": {name: list(values) for name, values in probabilities.items()},
        "decisions": decisions,
        "band_selected": band,
        "execution_counts": execution,
    }


def _execute_lane(role, contract_hash, output):
    global _CHILD_STAGE
    import torch

    from automated_phishing_detection import baselines, gmm_monitor, length_inference
    from automated_phishing_detection.transformer_inference import (
        load_transformer_cascade_bundle,
    )

    _CHILD_STAGE = "contract_and_source_verification"
    _verify_contract(ROOT / CONTRACT, contract_hash)
    _source_bindings(ROOT)
    _CHILD_STAGE = "environment_preflight"
    _environment(role)
    # Authenticate every named input before decoding any partition or model.
    _CHILD_STAGE = "input_hash_verification"
    snapshots = {
        name: _read_verified(ROOT / name, digest) for name, digest in INPUTS.items()
    }
    _CHILD_STAGE = "validation_parsing"
    preparation = baselines._validate_preparation_summary(
        json.loads(snapshots[PREPARATION])
    )
    partition = _partition(snapshots[VALIDATION], preparation)
    _CHILD_STAGE = "artifact_loading"
    loaded = load_transformer_cascade_bundle(
        ROOT / TRANSFORMER_DIR,
        ROOT / TRANSFORMER_SUMMARY,
        ROOT / BASELINE_DIR / "logistic-l1.json",
        expected_public_summary_sha256=INPUTS[TRANSFORMER_SUMMARY],
        device=torch.device("mps"),
    )
    length = length_inference._load_length_only_artifact_bytes(
        snapshots[f"{BASELINE_DIR}/length-only.json"]
    )
    gmm = None
    if role == "candidate":
        _CHILD_STAGE = "gmm_reference_and_candidate"
        artifact = gmm_monitor.load_gmm_artifact_bytes(snapshots[f"{GMM_DIR}/gmm.json"])
        traces = json.loads(snapshots[f"{GMM_DIR}/validation-audit.json"])
        summary = json.loads(snapshots[GMM_SUMMARY])
        if (
            artifact["input_hashes"] != summary["input_hashes"]
            or traces["input_hashes"] != summary["input_hashes"]
        ):
            raise ValueError("GMM input provenance is inconsistent")
        gmm = _gmm_comparison(partition, loaded.stage1_model, artifact, traces, summary)
        if not gmm["reference_reproduces_saved_traces"]:
            _CHILD_STAGE = "private_publication"
            _write_private(output, {"reference_failure": gmm["reference_failure"]})
            return
    _CHILD_STAGE = f"{role}_transformer_scoring"
    result = _score_models(partition, loaded, length, role)
    if gmm is not None:
        result["gmm"] = gmm
    _CHILD_STAGE = "postread_verification"
    for name, digest in INPUTS.items():
        _read_verified(ROOT / name, digest)
    _CHILD_STAGE = "private_publication"
    _write_private(output, result)


class _ChildFailure(ChildProcessError):
    def __init__(self, record):
        super().__init__("bridge child stopped")
        self.process_record = record


def _invoke_child(role, interpreter, contract_hash, output=None):
    interpreter = os.path.abspath(interpreter)
    if not Path(interpreter).is_file() or not os.access(interpreter, os.X_OK):
        raise ValueError("explicit interpreter is not executable")
    command = [
        interpreter,
        "-s",
        str(Path(__file__).resolve()),
        "--_role",
        role,
        "--contract-sha256",
        contract_hash,
    ]
    command += ["--_output", str(output)] if output is not None else ["--_preflight"]
    environment = dict(os.environ, PYTHONPATH=str(ROOT / "src"), PYTHONNOUSERSITE="1")
    start, utc = time.monotonic(), _utc_now()
    child = subprocess.run(command, cwd=ROOT, env=environment, capture_output=True)
    record = {
        "role": role,
        "phase": "preflight" if output is None else "scoring",
        "interpreter": interpreter,
        "started_at": utc,
        "finished_at": _utc_now(),
        "elapsed_seconds": time.monotonic() - start,
        "exit_code": child.returncode,
        "stdout_sha256": sha256(child.stdout).hexdigest(),
        "stderr_sha256": sha256(child.stderr).hexdigest(),
    }
    if child.returncode or (output is not None and child.stdout):
        try:
            failure = json.loads(child.stdout)
            stage = failure.get("failure_stage")
        except (ValueError, AttributeError):
            stage = None
        record["failure_stage"] = (
            stage
            if type(stage) is str and stage in CHILD_STAGES
            else "unknown_child_stage"
        )
        raise _ChildFailure(record)
    return (json.loads(child.stdout) if output is None else None), record


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        raise ValueError("invalid bridge arguments")


def main(argv=None):
    global _CHILD_STAGE
    _CHILD_STAGE = "arguments"
    parser = _Parser(allow_abbrev=False)
    parser.add_argument("--reference-python")
    parser.add_argument("--candidate-python")
    parser.add_argument("--contract-sha256", required=True)
    parser.add_argument("--preflight-correction", action="store_true")
    parser.add_argument(
        "--_role", choices=("reference", "candidate"), help=argparse.SUPPRESS
    )
    parser.add_argument("--_preflight", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--_output", type=Path, help=argparse.SUPPRESS)
    try:
        arguments = parser.parse_args(argv)
        if arguments._role:
            if arguments.preflight_correction:
                raise ValueError("preflight correction is a parent execution option")
            if arguments._preflight:
                _CHILD_STAGE = "environment_preflight"
                print(_json_bytes(_environment(arguments._role)).decode(), end="")
            else:
                output = arguments._output
                if (
                    output is None
                    or output.name != f"{arguments._role}.json"
                    or not output.parent.name.startswith("phishing-compatibility-")
                    or stat.S_IMODE(output.parent.stat().st_mode) != 0o700
                ):
                    raise ValueError("private lane destination is invalid")
                _execute_lane(arguments._role, arguments.contract_sha256, output)
            return 0
        if (
            not arguments.reference_python
            or not arguments.candidate_python
            or arguments._preflight
            or arguments._output
        ):
            raise ValueError("both explicit interpreters are required")
        result = _run_bridge(
            arguments.reference_python,
            arguments.candidate_python,
            arguments.contract_sha256,
            preflight_correction=arguments.preflight_correction,
        )
    except (Exception, KeyboardInterrupt) as error:
        result = {
            "status": "failed",
            "failure_type": type(error).__name__,
            "failure_stage": _CHILD_STAGE,
            "protected_evaluation_ready": False,
        }
    print(_json_bytes(result).decode(), end="")
    return 0 if result["status"] == "equivalent_on_development_validation" else 2


if __name__ == "__main__":
    raise SystemExit(main())
