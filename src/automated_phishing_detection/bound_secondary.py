"""Bind accepted secondary artifacts for evaluation-only scoring.

This module neither authorizes protected-data access nor exposes fitting,
calibration, seed selection, or retry behavior. The source runner remains
responsible for the complete pre-access freeze and one-read experiment boundary.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import torch

from . import fixed_cascade, secondary_transformer
from .secondary_tabular import SecondaryModel, load_secondary_model_bytes
from .transformer_inference import LoadedTransformerCascade

PUBLIC_REPORTS = {
    "tabular": (
        "reports/secondary-development-correction-v2-summary.json",
        "663f1117cd33f949b70c35c56810764193f69cae3a37505004d0db27641d829d",
    ),
    "seeds": (
        "reports/secondary-seed-probe-correction-v1-summary.json",
        "d63a85792088871cfb667e7e5cbe86c6148dc3a6c7430ca2e4db29d788ab8e23",
    ),
}


class BoundSecondaryError(ValueError):
    """A report, artifact, operating point, or score binding is invalid."""


@dataclass(frozen=True)
class SecondaryArtifactPaths:
    formatting: Path
    permutation_42: Path
    permutation_43: Path
    permutation_44: Path
    permutation_45: Path
    permutation_46: Path
    random_forest: Path
    seed_43_weights: Path
    seed_44_weights: Path
    seed_45_weights: Path
    seed_46_weights: Path


@dataclass(frozen=True)
class BoundTabular:
    name: str
    model: SecondaryModel
    threshold: float
    artifact_sha256: str


@dataclass(frozen=True)
class BoundSeed:
    seed: int
    transformer_threshold: float
    half_width: float
    weights_sha256: str
    reuses_primary: bool
    _weights_bytes: bytes | None


@dataclass(frozen=True)
class BoundSecondary:
    tabular: tuple[BoundTabular, ...]
    seeds: tuple[BoundSeed, ...]
    stage1_threshold: float
    vocabulary_bytes: bytes
    device_type: str
    report_hashes: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class SecondaryTabularScore:
    name: str
    probability: float
    decision: int


@dataclass(frozen=True)
class SecondarySeedScore:
    seed: int
    transformer_probability: float
    transformer_decision: int
    cascade_probability: float
    cascade_decision: int
    band_selected: bool


@dataclass(frozen=True)
class SecondaryScoredRow:
    tabular: tuple[SecondaryTabularScore, ...]
    seeds: tuple[SecondarySeedScore, ...]


@dataclass(frozen=True)
class SecondaryInferenceCounts:
    tabular_singleton_calls: tuple[tuple[str, int], ...]
    transformer_singleton_calls: tuple[tuple[int, int], ...]
    reused_primary_transformer_scores: int


@dataclass(frozen=True)
class SecondaryScoring:
    rows: tuple[SecondaryScoredRow, ...]
    counts: SecondaryInferenceCounts


_TABULAR_NAMES = (
    "formatting",
    "permutation_42",
    "permutation_43",
    "permutation_44",
    "permutation_45",
    "permutation_46",
    "random_forest",
)
_SEEDS = (42, 43, 44, 45, 46)


def _require(condition, message):
    if not condition:
        raise BoundSecondaryError(message)


def _canonical_bytes(value):
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise BoundSecondaryError("secondary report contains invalid JSON") from exc


def _json(content, role):
    try:
        value = json.loads(
            content,
            object_pairs_hook=fixed_cascade._object_without_duplicate_keys,
            parse_constant=fixed_cascade._reject_json_constant,
        )
    except (ValueError, UnicodeError) as exc:
        raise BoundSecondaryError(f"{role} report is invalid") from exc
    _require(type(value) is dict, f"{role} report must be an object")
    return value


def _digest(value):
    return sha256(_canonical_bytes(value)).hexdigest()


def _read_reports(root):
    contents = {}
    for role in ("tabular", "seeds"):
        relative, expected = PUBLIC_REPORTS[role]
        content = fixed_cascade._read_regular_file(root / relative)
        _require(
            sha256(content).hexdigest() == expected,
            f"{role} accepted report SHA-256 mismatch",
        )
        contents[role] = content
    return {role: _json(content, role) for role, content in contents.items()}


def _selected_threshold(value, role):
    _require(type(value) is dict and value.get("status") == "selected", role)
    try:
        return fixed_cascade._threshold(value.get("threshold"), role)
    except fixed_cascade.FixedCascadeError as exc:
        raise BoundSecondaryError(role) from exc


def _tabular_points(report):
    try:
        completion = report["completion"]
        observation = report["execution_observation"]
        retained = completion["retained_audit"]
        audit = retained["result"]
        corrected = completion["corrected_random_forest"]
    except KeyError as exc:
        raise BoundSecondaryError("tabular accepted report is incomplete") from exc
    expected_exits = {"retained_audit": 0, "random_forest": 0}
    _require(
        report.get("status") == "accepted_development_evidence"
        and completion.get("status") == "completed_secondary_development_correction"
        and report.get("completion_summary_sha256") == _digest(completion)
        and type(observation.get("parent_exit_code")) is int
        and observation["parent_exit_code"] == 0
        and observation.get("worker_exit_codes") == expected_exits
        and completion.get("worker_exit_codes") == expected_exits
        and completion.get("analysis_stage") == "development_validation_only"
        and completion.get("protected_evaluation_authorized") is False
        and type(completion.get("new_fits")) is int
        and completion["new_fits"] == 1
        and type(completion.get("retries")) is int
        and completion["retries"] == 0,
        "tabular accepted status differs",
    )
    members = audit.get("members")
    expected_members = ("drift", *_TABULAR_NAMES[:-1])
    _require(
        retained.get("stage") == "retained_audit"
        and retained.get("status") == "development_correction_stage_completed"
        and audit.get("status") == "retained_development_members_audited"
        and audit.get("analysis_stage") == "development_validation_only"
        and audit.get("protected_evaluation_authorized") is False
        and type(audit.get("fits")) is int
        and audit["fits"] == 0
        and type(members) is list
        and tuple(member.get("member") for member in members) == expected_members,
        "retained tabular audit differs",
    )
    points = []
    for name, member in zip(_TABULAR_NAMES[:-1], members[1:], strict=True):
        summary = member.get("summary")
        result = summary.get("result") if type(summary) is dict else None
        private = summary.get("private_sha256") if type(summary) is dict else None
        kind = "permutation" if name.startswith("permutation_") else "formatting"
        seed = int(name.rsplit("_", 1)[1]) if kind == "permutation" else 42
        _require(
            type(summary) is dict
            and member.get("public_summary_sha256") == _digest(summary)
            and summary.get("status") == "development_member_completed"
            and summary.get("member") == name
            and type(result) is dict
            and result.get("analysis_role") == "descriptive_secondary_not_primary"
            and result.get("model_kind") == kind
            and type(result.get("seed")) is int
            and result["seed"] == seed
            and type(private) is dict
            and set(private) >= {"model.json"},
            "retained tabular member differs",
        )
        fixed_cascade._lowercase_sha256(
            private["model.json"], f"{name} artifact SHA-256"
        )
        points.append(
            (
                name,
                private["model.json"],
                _selected_threshold(
                    result.get("validation_threshold"), f"{name} threshold"
                ),
            )
        )
    result = corrected.get("result") if type(corrected) is dict else None
    private = corrected.get("private_sha256") if type(corrected) is dict else None
    receipt = report.get("receipt_sha256")
    _require(
        type(corrected) is dict
        and corrected.get("stage") == "random_forest"
        and corrected.get("status") == "development_correction_stage_completed"
        and type(receipt) is dict
        and receipt.get("random_forest.json") == _digest(corrected)
        and type(result) is dict
        and result.get("analysis_role") == "descriptive_secondary_not_primary"
        and result.get("model_kind") == "random_forest"
        and type(result.get("seed")) is int
        and result["seed"] == 42
        and type(private) is dict
        and set(private) >= {"model.json"},
        "corrected Random Forest member differs",
    )
    fixed_cascade._lowercase_sha256(
        private["model.json"], "random_forest artifact SHA-256"
    )
    points.append(
        (
            "random_forest",
            private["model.json"],
            _selected_threshold(
                result.get("validation_threshold"), "random_forest threshold"
            ),
        )
    )
    return tuple(points)


def _seed_points(report, primary):
    try:
        completion = report["completion"]
        observation = report["execution_observation"]
        retained = completion["retained_seed_audit"]
        audit = retained["result"]
    except KeyError as exc:
        raise BoundSecondaryError("seed accepted report is incomplete") from exc
    expected_exits = {"retained_seed_audit": 0, "probes": 0}
    _require(
        report.get("status") == "accepted_development_evidence"
        and completion.get("status") == "completed_secondary_seed_probe_correction"
        and report.get("completion_summary_sha256") == _digest(completion)
        and type(observation.get("parent_exit_code")) is int
        and observation["parent_exit_code"] == 0
        and observation.get("worker_exit_codes") == expected_exits
        and completion.get("worker_exit_codes") == expected_exits
        and completion.get("analysis_stage") == "development_validation_only"
        and completion.get("protected_evaluation_authorized") is False
        and type(completion.get("new_fits")) is int
        and completion["new_fits"] == 0
        and type(completion.get("seed_stage_executions")) is int
        and completion["seed_stage_executions"] == 0
        and type(completion.get("retries")) is int
        and completion["retries"] == 0,
        "seed accepted status differs",
    )
    members = audit.get("members")
    stages = ("seed_42_calibration", "seed_43", "seed_44", "seed_45", "seed_46")
    _require(
        retained.get("status") == "completed_secondary_seed_probe_correction_stage"
        and audit.get("status") == "retained_seed_stages_audited"
        and audit.get("analysis_stage") == "development_validation_only"
        and audit.get("protected_evaluation_authorized") is False
        and audit.get("primary_artifacts_changed") is False
        and type(audit.get("fits")) is int
        and audit["fits"] == 0
        and audit.get("seed_selection_performed") is False
        and type(members) is list
        and tuple(member.get("stage") for member in members) == stages,
        "retained seed audit differs",
    )
    points = []
    for seed, stage, member in zip(_SEEDS, stages, members, strict=True):
        summary = member.get("summary")
        result = summary.get("result") if type(summary) is dict else None
        calibration = result.get("calibration") if type(result) is dict else None
        stage1 = calibration.get("stage1") if type(calibration) is dict else None
        band = calibration.get("cascade_band") if type(calibration) is dict else None
        _require(
            type(summary) is dict
            and member.get("summary_sha256") == _digest(summary)
            and summary.get("stage") == stage
            and summary.get("status") == "completed_secondary_seed_probe_stage"
            and type(result) is dict
            and type(result.get("seed")) is int
            and result["seed"] == seed
            and result.get("new_fit") is (seed != 42)
            and result.get("primary_artifacts_changed") is False
            and type(stage1) is dict
            and stage1.get("threshold") == primary.stage1_threshold
            and type(band) is dict
            and band.get("status") == "selected"
            and band.get("accepted_cascade") is True,
            "retained seed member differs",
        )
        weights = result.get("weights_sha256")
        fixed_cascade._lowercase_sha256(weights, f"seed {seed} weights SHA-256")
        try:
            half_width = fixed_cascade._finite_number(
                band.get("half_width"), f"seed {seed} half_width"
            )
        except fixed_cascade.FixedCascadeError as exc:
            raise BoundSecondaryError("invalid seed half_width") from exc
        _require(half_width >= 0, "invalid seed half_width")
        points.append(
            (
                seed,
                weights,
                _selected_threshold(
                    calibration.get("transformer_threshold"),
                    f"seed {seed} transformer threshold",
                ),
                half_width,
            )
        )
    return tuple(points)


def _validate_primary(primary):
    _require(
        type(primary) is LoadedTransformerCascade,
        "primary must be a LoadedTransformerCascade",
    )
    _require(
        primary.device.type == "mps" and primary.device.index is None,
        "primary device differs from frozen runtime",
    )
    try:
        stage1 = fixed_cascade._threshold(
            primary.stage1_threshold, "primary stage1 threshold"
        )
        artifacts = dict(primary.artifact_hashes)
        weights = fixed_cascade._lowercase_sha256(
            artifacts["transformer-weights.npz"], "primary transformer weights"
        )
        vocabulary_digest = fixed_cascade._lowercase_sha256(
            artifacts["vocabulary.json"], "primary vocabulary"
        )
        vocabulary_bytes = primary.vocabulary.to_json().encode("utf-8")
    except (
        KeyError,
        AttributeError,
        UnicodeError,
        fixed_cascade.FixedCascadeError,
    ) as exc:
        raise BoundSecondaryError("primary transformer binding differs") from exc
    _require(
        sha256(vocabulary_bytes).hexdigest() == vocabulary_digest,
        "primary vocabulary bytes differ from artifact binding",
    )
    return stage1, weights, vocabulary_bytes


def load_bound_secondary(root, paths, primary):
    """Authenticate accepted reports, then snapshot exact secondary artifacts."""
    try:
        _require(isinstance(root, Path), "root must be a Path")
        _require(
            type(paths) is SecondaryArtifactPaths
            and all(
                isinstance(getattr(paths, field), Path)
                for field in paths.__dataclass_fields__
            ),
            "secondary paths must be explicit Path values",
        )
        stage1, primary_weights, vocabulary_bytes = _validate_primary(primary)
        reports = _read_reports(root)
        tabular_points = _tabular_points(reports["tabular"])
        seed_points = _seed_points(reports["seeds"], primary)
        _require(
            seed_points[0][1] == primary_weights,
            "seed 42 weights differ from the accepted primary",
        )
        tabular = []
        for name, expected, threshold in tabular_points:
            content = fixed_cascade._read_regular_file(getattr(paths, name))
            _require(
                sha256(content).hexdigest() == expected,
                f"{name} artifact SHA-256 mismatch",
            )
            tabular.append(
                BoundTabular(
                    name,
                    load_secondary_model_bytes(content),
                    threshold,
                    expected,
                )
            )
        seeds = [
            BoundSeed(
                seed_points[0][0],
                seed_points[0][2],
                seed_points[0][3],
                seed_points[0][1],
                True,
                None,
            )
        ]
        for seed, expected, threshold, half_width in seed_points[1:]:
            content = fixed_cascade._read_regular_file(
                getattr(paths, f"seed_{seed}_weights")
            )
            _require(
                sha256(content).hexdigest() == expected,
                f"seed {seed} weights SHA-256 mismatch",
            )
            seeds.append(
                BoundSeed(seed, threshold, half_width, expected, False, content)
            )
        return BoundSecondary(
            tuple(tabular),
            tuple(seeds),
            stage1,
            vocabulary_bytes,
            primary.device.type,
            tuple((role, PUBLIC_REPORTS[role][1]) for role in ("tabular", "seeds")),
        )
    except BoundSecondaryError:
        raise
    except (OSError, TypeError, ValueError, KeyError, OverflowError) as exc:
        raise BoundSecondaryError("secondary artifact binding failed") from exc


def score_bound_secondary(bound, raw_urls, stage1_probabilities, seed_42_probabilities):
    """Score the accepted secondary family over one retained ordered URL tuple."""
    try:
        _validate_bound(bound)
        _require(
            type(raw_urls) is tuple
            and bool(raw_urls)
            and all(type(value) is str for value in raw_urls),
            "raw_urls must be a nonempty tuple of exact strings",
        )
        count = len(raw_urls)
        stage1 = _probabilities(stage1_probabilities, count, "stage1 probabilities")
        seed_42 = _probabilities(seed_42_probabilities, count, "seed 42 probabilities")
        tabular_columns = []
        for member in bound.tabular:
            values = _probabilities(
                member.model.score_urls_singleton_ordered(raw_urls),
                count,
                f"{member.name} probabilities",
            )
            tabular_columns.append(
                tuple(
                    SecondaryTabularScore(
                        member.name, value, int(value >= member.threshold)
                    )
                    for value in values
                )
            )
        seed_columns = []
        vocabulary_sha256 = sha256(bound.vocabulary_bytes).hexdigest()
        for member in bound.seeds:
            if member.reuses_primary:
                transformer = seed_42
            else:
                loaded = secondary_transformer.load_secondary_transformer_bytes(
                    member._weights_bytes,
                    bound.vocabulary_bytes,
                    seed=member.seed,
                    device=torch.device(bound.device_type),
                )
                _require(
                    loaded.seed == member.seed
                    and loaded.weights_sha256 == member.weights_sha256
                    and loaded.vocabulary_sha256 == vocabulary_sha256
                    and loaded.device == torch.device(bound.device_type),
                    "loaded secondary transformer differs from binding",
                )
                transformer = _probabilities(
                    secondary_transformer.score_secondary_transformer_urls(
                        loaded, raw_urls
                    ),
                    count,
                    f"seed {member.seed} probabilities",
                )
                del loaded
            cascade = fixed_cascade.score_fixed_cascade(
                stage1,
                transformer,
                stage1_threshold=bound.stage1_threshold,
                transformer_threshold=member.transformer_threshold,
                half_width=member.half_width,
            )
            seed_columns.append(
                tuple(
                    SecondarySeedScore(
                        member.seed,
                        transformer[index],
                        int(transformer[index] >= member.transformer_threshold),
                        transformer[index]
                        if cascade.transformer_invoked[index]
                        else stage1[index],
                        cascade.decisions[index],
                        cascade.transformer_invoked[index],
                    )
                    for index in range(count)
                )
            )
        rows = tuple(
            SecondaryScoredRow(
                tuple(column[index] for column in tabular_columns),
                tuple(column[index] for column in seed_columns),
            )
            for index in range(count)
        )
        counts = SecondaryInferenceCounts(
            tuple((member.name, count) for member in bound.tabular),
            tuple(
                (member.seed, 0 if member.reuses_primary else count)
                for member in bound.seeds
            ),
            count,
        )
        return SecondaryScoring(rows, counts)
    except BoundSecondaryError:
        raise
    except (
        fixed_cascade.FixedCascadeError,
        secondary_transformer.SecondaryTransformerError,
        TypeError,
        ValueError,
        OverflowError,
    ) as exc:
        raise BoundSecondaryError("secondary scoring failed") from exc


def _probabilities(values, count, field):
    _require(
        type(values) is tuple and len(values) == count,
        f"{field} must align with raw_urls",
    )
    result = []
    for value in values:
        _require(
            type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1,
            f"{field} must contain finite probabilities",
        )
        result.append(float(value))
    return tuple(result)


def _validate_bound(bound):
    _require(type(bound) is BoundSecondary, "use a BoundSecondary")
    _require(
        type(bound.tabular) is tuple
        and tuple(member.name for member in bound.tabular) == _TABULAR_NAMES
        and all(type(member) is BoundTabular for member in bound.tabular),
        "bound tabular inventory differs",
    )
    for member in bound.tabular:
        _require(
            type(member.model) is SecondaryModel,
            "bound tabular model type differs",
        )
        fixed_cascade._threshold(member.threshold, f"{member.name} threshold")
        fixed_cascade._lowercase_sha256(
            member.artifact_sha256, f"{member.name} artifact SHA-256"
        )
    _require(
        type(bound.seeds) is tuple
        and tuple(member.seed for member in bound.seeds) == _SEEDS
        and all(type(member) is BoundSeed for member in bound.seeds),
        "bound seed inventory differs",
    )
    for member in bound.seeds:
        fixed_cascade._threshold(
            member.transformer_threshold,
            f"seed {member.seed} transformer threshold",
        )
        half_width = fixed_cascade._finite_number(
            member.half_width, f"seed {member.seed} half_width"
        )
        fixed_cascade._lowercase_sha256(
            member.weights_sha256, f"seed {member.seed} weights SHA-256"
        )
        _require(half_width >= 0, "seed half_width must be nonnegative")
        if member.seed == 42:
            _require(
                member.reuses_primary is True and member._weights_bytes is None,
                "seed 42 must reuse primary scores",
            )
        else:
            _require(
                member.reuses_primary is False
                and type(member._weights_bytes) is bytes
                and sha256(member._weights_bytes).hexdigest() == member.weights_sha256,
                "retained seed weights differ from binding",
            )
    fixed_cascade._threshold(bound.stage1_threshold, "secondary stage1 threshold")
    _require(
        type(bound.vocabulary_bytes) is bytes
        and bool(bound.vocabulary_bytes)
        and bound.device_type == "mps"
        and type(bound.report_hashes) is tuple
        and tuple(role for role, _ in bound.report_hashes) == ("tabular", "seeds"),
        "secondary shared binding differs",
    )
    for _, digest in bound.report_hashes:
        fixed_cascade._lowercase_sha256(digest, "accepted report SHA-256")
