"""Bind frozen models to authenticated public summaries without fitting or scoring."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import torch

from . import (
    fixed_cascade,
    gmm_monitor,
    length_inference,
    transformer_inference,
    transformer_pipeline,
)
from .length_inference import LoadedLengthOnly
from .transformer_inference import LoadedTransformerCascade

_PUBLIC_SUMMARIES = {
    "baseline": (
        "reports/rq1-baseline-v2-summary.json",
        "bf5b3a6f0fc705d26852da4dd0053c6111ffc3e500d7a2e95dfba5ad859b279c",
    ),
    "transformer": (
        "reports/rq1-transformer-cascade-v2-summary.json",
        "41499aa388babe60442de7231b4087f67a53f96f340568a7cc58a3268606a2fd",
    ),
    "gmm": (
        "reports/rq2-gmm-development-v1-summary.json",
        "6f695138a302e854e1e5af590152e289486affe8ccdf75510ca9a5dcaad3b523",
    ),
}
_ARTIFACT_DIGESTS = {
    "length-only.json": length_inference.OFFICIAL_LENGTH_ONLY_SHA256,
    "logistic-l1.json": fixed_cascade.OFFICIAL_LOGISTIC_L1_SHA256,
    "gmm.json": "a01a8b143c2423df57a153462cc47e79822ad1c6768213dc69e03683d4009745",
}
_BASELINE_FIELDS = frozenset(
    {
        "access",
        "analysis_stage",
        "contract_id",
        "hypothesis_status",
        "input_counts",
        "input_hashes",
        "models",
        "pipeline",
        "schema_version",
        "software_versions",
    }
)
_MODEL_FIELDS = frozenset(
    {
        "artifact",
        "artifact_sha256",
        "feature_count",
        "n_iter",
        "validation_scoring_audit",
        "validation_threshold",
    }
)


class BoundModelsError(ValueError):
    """A fixed summary, artifact, or cross-model binding is invalid."""


@dataclass(frozen=True)
class ArtifactPaths:
    length_only: Path
    logistic_l1: Path
    transformer_bundle: Path
    gmm: Path


@dataclass(frozen=True)
class BoundModels:
    length_only: LoadedLengthOnly
    cascade: LoadedTransformerCascade
    gmm: dict
    monitor_boundary: float
    artifact_hashes: tuple[tuple[str, str], ...]


def _read_public(root: Path) -> dict[str, dict]:
    contents = {}
    # Authenticate every public input before parsing or touching private paths.
    for role, (relative, expected) in _PUBLIC_SUMMARIES.items():
        content = fixed_cascade._read_regular_file(root / relative)
        if sha256(content).hexdigest() != expected:
            raise BoundModelsError(f"{role} public summary SHA-256 mismatch")
        contents[role] = content
    summaries = {}
    for role, content in contents.items():
        if role == "transformer":
            summaries[role] = transformer_inference._canonical_json(content, role)
            continue
        value = json.loads(
            content.decode("utf-8"),
            object_pairs_hook=fixed_cascade._object_without_duplicate_keys,
            parse_constant=fixed_cascade._reject_json_constant,
        )
        if type(value) is not dict:
            raise BoundModelsError(f"{role} public summary must be an object")
        summaries[role] = value
    return summaries


def _hashes(value: object, expected_keys: set[str], field: str) -> dict:
    result = fixed_cascade._expect_fields(value, frozenset(expected_keys), field)
    for name, digest in result.items():
        fixed_cascade._lowercase_sha256(digest, f"{field}.{name}")
    return result


def _validate_baseline(summary: dict) -> None:
    fixed_cascade._expect_fields(summary, _BASELINE_FIELDS, "baseline summary")
    identity = {
        "schema_version": 2,
        "contract_id": "rq1-baselines-v2",
        "analysis_stage": "development_validation_only",
        "hypothesis_status": {"H1": "undecided", "H2": "undecided", "H3": "undecided"},
        "access": {"group_test_accessed": False, "phishvn_accessed": False},
    }
    if any(
        not fixed_cascade._matches_exactly(summary[key], value)
        for key, value in identity.items()
    ):
        raise BoundModelsError("baseline public summary identity is invalid")
    _hashes(
        summary["input_hashes"],
        {"train", "validation", "preparation_summary", "contract"},
        "baseline input hashes",
    )
    pipeline = fixed_cascade._expect_fields(
        summary["pipeline"],
        frozenset(
            {
                "scaler",
                "classifier",
                "convergence_warning_action",
                "scoring_integrity_policy_id",
                "score",
                "alert_rule",
                "threshold_constraint",
            }
        ),
        "baseline pipeline",
    )
    for field, expected in (
        ("scaler", fixed_cascade._SCALER_CONFIG),
        ("classifier", fixed_cascade._CLASSIFIER_CONFIG),
    ):
        if not fixed_cascade._matches_exactly(pipeline[field], expected):
            raise BoundModelsError(
                "baseline pipeline differs from frozen configuration"
            )
    models = fixed_cascade._expect_fields(
        summary["models"], frozenset({"length-only", "Logistic-L1"}), "baseline models"
    )
    for name, filename, width in (
        ("length-only", "length-only.json", 1),
        ("Logistic-L1", "logistic-l1.json", 25),
    ):
        model = fixed_cascade._expect_fields(models[name], _MODEL_FIELDS, name)
        if (
            model["artifact"] != filename
            or model["artifact_sha256"] != _ARTIFACT_DIGESTS[filename]
            or type(model["feature_count"]) is not int
            or model["feature_count"] != width
        ):
            raise BoundModelsError("baseline public artifact binding is invalid")
        fixed_cascade._validate_threshold_record(model["validation_threshold"], name)
        if (
            type(model["n_iter"]) is not list
            or len(model["n_iter"]) != 1
            or type(model["n_iter"][0]) is not int
            or not 1 <= model["n_iter"][0] < 5000
        ):
            raise BoundModelsError("baseline iteration count is invalid")


def _validate_transformer(summary: dict) -> dict:
    transformer_pipeline._validate_public_summary(summary)
    counts = transformer_inference._validate_public_summary_structure(summary)
    configuration = transformer_pipeline._canonical_json_bytes(summary["configuration"])
    if (
        sha256(configuration).hexdigest()
        != transformer_inference._EXPECTED_PUBLIC_CONFIGURATION_SHA256
    ):
        raise BoundModelsError("transformer public configuration differs")
    threshold = fixed_cascade._expect_fields(
        summary["transformer"]["threshold"],
        fixed_cascade._THRESHOLD_FIELDS - {"threshold"},
        "public transformer threshold",
    )
    # Validate published metrics without inventing or selecting the hidden threshold.
    fixed_cascade._validate_threshold_record(
        threshold | {"threshold": 0.0}, "public transformer metrics"
    )
    cascade = fixed_cascade._expect_fields(
        summary["cascade"],
        frozenset(
            {
                "schema_version",
                "status",
                "accepted_cascade",
                "reason",
                "threshold_statuses",
                "candidate_count",
                "transformer_invocations",
                "transformer_invocation_rate",
                "counts",
                "recall",
                "observed_fpr",
                "fpr_upper_95",
                "minimum_recall",
                "maximum_fpr_upper_95",
            }
        ),
        "public cascade",
    )
    if cascade["accepted_cascade"] is not True:
        raise BoundModelsError("transformer public cascade must be accepted")
    return counts


def _validate_public(summaries: dict[str, dict]) -> None:
    baseline, transformer, gmm = (
        summaries[name] for name in ("baseline", "transformer", "gmm")
    )
    _validate_baseline(baseline)
    counts = _validate_transformer(transformer)
    _hashes(
        transformer["artifact_hashes"],
        set(transformer_inference._HASHED_FILENAMES),
        "transformer artifact hashes",
    )
    gmm_monitor._validate_public_summary(gmm)
    for key in (
        "schema_version",
        "status",
        "analysis_stage",
        "hypothesis_status",
    ):
        if not fixed_cascade._matches_exactly(gmm[key], transformer[key]):
            raise BoundModelsError("GMM public summary identity is invalid")
    if not fixed_cascade._matches_exactly(
        gmm["access"],
        {
            "external_accessed": False,
            "group_test_accessed": False,
            "phishvn_accessed": False,
            "scope": "this_process_only",
        },
    ):
        raise BoundModelsError("GMM public access declaration is invalid")
    if (
        type(gmm["selected_component_count"]) is not int
        or not 1 <= gmm["selected_component_count"] <= 6
    ):
        raise BoundModelsError("GMM public component count is invalid")
    if type(gmm["threshold"]) not in (int, float) or not math.isfinite(
        gmm["threshold"]
    ):
        raise BoundModelsError("GMM public threshold must be finite")
    artifacts = _hashes(
        gmm["artifact_hashes"],
        {"gmm.json", "validation-audit.json"},
        "GMM artifact hashes",
    )
    if artifacts["gmm.json"] != _ARTIFACT_DIGESTS["gmm.json"]:
        raise BoundModelsError("GMM public artifact binding is invalid")
    common = {
        key: baseline["input_hashes"][key]
        for key in ("train", "validation", "preparation_summary")
    }
    common.update(
        baseline_contract=baseline["input_hashes"]["contract"],
        logistic_l1_artifact=_ARTIFACT_DIGESTS["logistic-l1.json"],
    )
    for summary, role in ((transformer, "transformer"), (gmm, "gmm")):
        contract = fixed_cascade._expect_fields(
            summary["contract"],
            frozenset(
                {
                    "id",
                    "protocol_version",
                    "sha256",
                }
            ),
            f"{role} public contract",
        )
        expected = common | {f"{role}_contract": contract["sha256"]}
        _hashes(summary["input_hashes"], set(expected), f"{role} input hashes")
        if summary["input_hashes"] != expected:
            raise BoundModelsError(f"{role} public input provenance differs")
    expected_counts = {
        partition: {
            name: value for name, value in record.items() if name != "domain_count"
        }
        for partition, record in counts.items()
    }
    if not fixed_cascade._matches_exactly(baseline["input_counts"], expected_counts):
        raise BoundModelsError("baseline and transformer public input counts differ")


def _validate_loaded(
    length: LoadedLengthOnly,
    cascade: LoadedTransformerCascade,
    summaries: dict[str, dict],
) -> None:
    baseline = summaries["baseline"]
    for model, role, filename in (
        (length, "length-only", "length-only.json"),
        (cascade.stage1_model, "Logistic-L1", "logistic-l1.json"),
    ):
        if (
            model.artifact_sha256 != _ARTIFACT_DIGESTS[filename]
            or model.contract_sha256 != baseline["input_hashes"]["contract"]
            or not fixed_cascade._matches_exactly(
                model.validation_threshold_record,
                baseline["models"][role]["validation_threshold"],
            )
        ):
            raise BoundModelsError("loaded baseline provenance or threshold differs")
    if (
        cascade.public_summary_sha256 != _PUBLIC_SUMMARIES["transformer"][1]
        or cascade.artifact_hashes
        != tuple(sorted(summaries["transformer"]["artifact_hashes"].items()))
        or cascade.stage1_threshold
        != baseline["models"]["Logistic-L1"]["validation_threshold"]["threshold"]
    ):
        raise BoundModelsError(
            "loaded transformer provenance or stage-one threshold differs"
        )


def load_bound_models(root: Path, paths: ArtifactPaths) -> BoundModels:
    """Load fixed authenticated artifacts on MPS, without entering a scorer session.

    The calling composition layer owns runtime preflight and invokes this function
    on its service owner thread. No alternate hashes, devices, or audit inputs are
    accepted by this public API.
    """
    try:
        if not isinstance(root, Path) or type(paths) is not ArtifactPaths:
            raise BoundModelsError("use a root Path and ArtifactPaths")
        if not all(
            isinstance(path, Path)
            for path in (
                paths.length_only,
                paths.logistic_l1,
                paths.transformer_bundle,
                paths.gmm,
            )
        ):
            raise BoundModelsError("all artifact paths must be Path values")
        summaries = _read_public(root)
        _validate_public(summaries)
        length = length_inference.load_length_only_artifact(paths.length_only)
        cascade = transformer_inference.load_transformer_cascade_bundle(
            paths.transformer_bundle,
            root / _PUBLIC_SUMMARIES["transformer"][0],
            paths.logistic_l1,
            expected_public_summary_sha256=_PUBLIC_SUMMARIES["transformer"][1],
            device=torch.device("mps"),
        )
        _validate_loaded(length, cascade, summaries)
        content = fixed_cascade._read_regular_file(paths.gmm)
        digest = sha256(content).hexdigest()
        if digest != summaries["gmm"]["artifact_hashes"]["gmm.json"]:
            raise BoundModelsError("GMM artifact SHA-256 mismatch")
        gmm = gmm_monitor.load_gmm_artifact_bytes(content)
        if gmm["input_hashes"] != summaries["gmm"]["input_hashes"]:
            raise BoundModelsError("GMM artifact input provenance differs")
        if gmm["mixture"]["components"] != summaries["gmm"]["selected_component_count"]:
            raise BoundModelsError("GMM artifact component count differs")
        artifacts = dict(cascade.artifact_hashes) | {
            "length-only.json": length.artifact_sha256,
            "logistic-l1.json": cascade.stage1_model.artifact_sha256,
            "gmm.json": digest,
        }
        return BoundModels(
            length,
            cascade,
            gmm,
            summaries["gmm"]["threshold"],
            tuple(sorted(artifacts.items())),
        )
    except BoundModelsError:
        raise
    except (ValueError, TypeError, KeyError, OverflowError, OSError) as exc:
        raise BoundModelsError(f"model binding failed: {exc}") from exc
