"""No-fit validation and loading for the frozen transformer cascade."""

from __future__ import annotations

import io
import os
import stat
import zipfile
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path

import numpy as np
import torch

from . import (
    character_sequence,
    character_transformer,
    fixed_cascade,
    transformer_pipeline,
)

_BUNDLE_FILENAMES = frozenset(
    {
        "SHA256SUMS",
        "cascade.json",
        "transformer-weights.npz",
        "transformer.json",
        "vocabulary.json",
    }
)
_HASHED_FILENAMES = tuple(sorted(_BUNDLE_FILENAMES - {"SHA256SUMS"}))
_TRANSFORMER_FIELDS = frozenset(
    {
        "access",
        "analysis_stage",
        "artifact_type",
        "configuration",
        "contract_id",
        "contract_sha256",
        "fit",
        "input_hashes",
        "runtime_observed",
        "schema_version",
        "software_versions",
        "validation_threshold",
        "vocabulary_sha256",
        "warnings",
        "weights_sha256",
    }
)
_CASCADE_FIELDS = frozenset(
    {
        "access",
        "analysis_stage",
        "artifact_type",
        "calibration",
        "contract_id",
        "contract_sha256",
        "input_hashes",
        "schema_version",
        "stage1",
        "transformer_metadata_sha256",
        "warnings",
    }
)
_INPUT_HASH_FIELDS = frozenset(
    {
        "train",
        "validation",
        "preparation_summary",
        "baseline_contract",
        "logistic_l1_artifact",
        "transformer_contract",
    }
)
_ACCESS = {"group_test_accessed": False, "phishvn_accessed": False}
_FIT_FIELDS = frozenset(
    {
        "best_epoch",
        "best_validation_average_precision",
        "epochs_completed",
        "history",
        "positive_class_weight",
        "stopped_early",
    }
)
_EPOCH_FIELDS = frozenset({"epoch", "training_loss", "validation_average_precision"})
_PUBLIC_TRANSFORMER_FIELDS = frozenset({"fit", "threshold"})
_PUBLIC_FIT_FIELDS = _FIT_FIELDS - {"history"}
_PUBLIC_VOCABULARY_FIELDS = frozenset({"observed_character_count", "sha256", "size"})
_PUBLIC_INPUT_COUNT_FIELDS = frozenset({"train", "validation"})
_PUBLIC_PARTITION_COUNT_FIELDS = frozenset({"0", "1", "domain_count", "rows"})
_EXPECTED_PRIVATE_CONFIGURATION_SHA256 = (
    "71713de6bd0bb8a5d86ca524107b2d121f870bb3a428b0bfb98af4b325d22b02"
)
_EXPECTED_PUBLIC_CONFIGURATION_SHA256 = (
    "32f548d91ecd11ffc1fb408570ce626de8ad4b14115cb22b2d62dc482bce20d5"
)


class TransformerInferenceError(ValueError):
    """A frozen inference artifact or input violates its contract."""


@dataclass(frozen=True)
class _BundleHashPolicy:
    public_summary_sha256: str
    train_sha256: str
    validation_sha256: str
    preparation_summary_sha256: str
    transformer_contract_sha256: str
    baseline_contract_sha256: str
    logistic_l1_artifact_sha256: str


@dataclass(frozen=True)
class LoadedTransformerCascade:
    """Hash-bound artifact references; scorer execution controls are separate."""

    stage1_model: fixed_cascade.PortableLogisticL1
    vocabulary: character_sequence.CharacterVocabulary
    stage1_threshold: float
    transformer_threshold: float
    half_width: float
    device: torch.device
    public_summary_sha256: str
    artifact_hashes: tuple[tuple[str, str], ...]
    _model: character_transformer.CharacterTransformer = field(
        repr=False, compare=False
    )


def _lowercase_sha256(value: object, field_name: str) -> str:
    try:
        return fixed_cascade._lowercase_sha256(value, field_name)
    except fixed_cascade.FixedCascadeError as exc:
        raise TransformerInferenceError(str(exc)) from exc


def _expect_fields(value: object, expected: frozenset[str], field_name: str) -> dict:
    if type(value) is not dict or frozenset(value) != expected:
        raise TransformerInferenceError(
            f"{field_name} fields do not match the frozen schema"
        )
    return value


def _canonical_json(content: bytes, field_name: str) -> dict:
    try:
        value = transformer_pipeline._load_json(content, field_name)
        if type(value) is not dict:
            raise TransformerInferenceError(f"{field_name} must be an object")
        if transformer_pipeline._canonical_json_bytes(value) != content:
            raise TransformerInferenceError(f"{field_name} is not canonical JSON")
        return value
    except transformer_pipeline.TransformerPipelineError as exc:
        raise TransformerInferenceError(str(exc)) from exc


def _path_identity(path: Path) -> tuple[int, int]:
    metadata = path.lstat()
    return metadata.st_dev, metadata.st_ino


def _read_regular_file(path: Path, field_name: str) -> bytes:
    try:
        return fixed_cascade._read_regular_file(path)
    except (OSError, fixed_cascade.FixedCascadeError) as exc:
        raise TransformerInferenceError(f"{field_name}: {exc}") from exc


def _snapshot_files(
    bundle_dir: Path, public_summary_path: Path, logistic_l1_artifact_path: Path
) -> tuple[dict[str, bytes], bytes, bytes]:
    try:
        directory_metadata = bundle_dir.lstat()
    except OSError as exc:
        raise TransformerInferenceError("bundle directory does not exist") from exc
    if not stat.S_ISDIR(directory_metadata.st_mode):
        raise TransformerInferenceError("bundle must be a real directory, not an alias")
    try:
        entries = {entry.name: Path(entry.path) for entry in os.scandir(bundle_dir)}
    except OSError as exc:
        raise TransformerInferenceError("bundle directory cannot be inspected") from exc
    if frozenset(entries) != _BUNDLE_FILENAMES:
        raise TransformerInferenceError("bundle files do not match the frozen schema")

    all_paths = [entries[name] for name in sorted(entries)] + [
        public_summary_path,
        logistic_l1_artifact_path,
    ]
    try:
        identities = [_path_identity(path) for path in all_paths]
    except OSError as exc:
        raise TransformerInferenceError(
            "an inference input cannot be inspected"
        ) from exc
    if len(set(identities)) != len(identities):
        raise TransformerInferenceError("inference inputs must not alias one another")

    bundle = {
        name: _read_regular_file(entries[name], f"bundle {name}")
        for name in sorted(entries)
    }
    summary = _read_regular_file(public_summary_path, "public summary")
    stage1 = _read_regular_file(logistic_l1_artifact_path, "stage-one artifact")
    try:
        if [_path_identity(path) for path in all_paths] != identities:
            raise TransformerInferenceError("an inference input changed while loading")
    except OSError as exc:
        raise TransformerInferenceError(
            "an inference input changed while loading"
        ) from exc
    return bundle, summary, stage1


def _validate_hash_policy(policy: _BundleHashPolicy) -> None:
    if type(policy) is not _BundleHashPolicy:
        raise TransformerInferenceError("bundle hash policy is invalid")
    for field_name, value in vars(policy).items():
        _lowercase_sha256(value, field_name)


def _artifact_hashes(bundle: dict[str, bytes]) -> dict[str, str]:
    hashes = {name: sha256(bundle[name]).hexdigest() for name in _HASHED_FILENAMES}
    expected_manifest = "".join(
        f"{hashes[name]}  {name}\n" for name in _HASHED_FILENAMES
    ).encode("ascii")
    if bundle["SHA256SUMS"] != expected_manifest:
        raise TransformerInferenceError("SHA256SUMS does not match the bundle")
    return hashes


def _validate_input_hashes(value: object, policy: _BundleHashPolicy) -> dict:
    hashes = _expect_fields(value, _INPUT_HASH_FIELDS, "input_hashes")
    for name, digest in hashes.items():
        _lowercase_sha256(digest, f"input_hashes.{name}")
    expected = {
        "train": policy.train_sha256,
        "validation": policy.validation_sha256,
        "preparation_summary": policy.preparation_summary_sha256,
        "baseline_contract": policy.baseline_contract_sha256,
        "logistic_l1_artifact": policy.logistic_l1_artifact_sha256,
        "transformer_contract": policy.transformer_contract_sha256,
    }
    if any(hashes[name] != digest for name, digest in expected.items()):
        raise TransformerInferenceError(
            "artifact input hashes violate the active policy"
        )
    return hashes


def _validate_threshold(value: object, field_name: str) -> dict:
    try:
        record = fixed_cascade._validate_threshold_record(value, field_name)
    except fixed_cascade.FixedCascadeError as exc:
        raise TransformerInferenceError(str(exc)) from exc
    if record["status"] != "selected":
        raise TransformerInferenceError(f"{field_name} was not selected")
    return record


def _validate_fit(value: object) -> dict:
    fit = _expect_fields(value, _FIT_FIELDS, "transformer fit")
    try:
        best_epoch = fixed_cascade._exact_integer(
            fit["best_epoch"], "transformer fit best_epoch", minimum=1
        )
        epochs_completed = fixed_cascade._exact_integer(
            fit["epochs_completed"],
            "transformer fit epochs_completed",
            minimum=1,
        )
        best_average_precision = fixed_cascade._finite_number(
            fit["best_validation_average_precision"],
            "transformer fit best_validation_average_precision",
        )
        positive_class_weight = fixed_cascade._finite_number(
            fit["positive_class_weight"],
            "transformer fit positive_class_weight",
        )
    except fixed_cascade.FixedCascadeError as exc:
        raise TransformerInferenceError(str(exc)) from exc
    history = fit["history"]
    if (
        type(history) is not list
        or len(history) != epochs_completed
        or epochs_completed > character_transformer.MAX_EPOCHS
        or type(fit["stopped_early"]) is not bool
        or positive_class_weight <= 0.0
        or not 0.0 <= best_average_precision <= 1.0
        or best_epoch > epochs_completed
    ):
        raise TransformerInferenceError("transformer fit history is invalid")
    for expected_epoch, value in enumerate(history, start=1):
        record = _expect_fields(value, _EPOCH_FIELDS, "transformer epoch record")
        try:
            epoch = fixed_cascade._exact_integer(
                record["epoch"], "transformer epoch", minimum=1
            )
            training_loss = fixed_cascade._finite_number(
                record["training_loss"], "transformer training loss"
            )
            validation_average_precision = fixed_cascade._finite_number(
                record["validation_average_precision"],
                "transformer validation average precision",
            )
        except fixed_cascade.FixedCascadeError as exc:
            raise TransformerInferenceError(str(exc)) from exc
        if (
            epoch != expected_epoch
            or training_loss < 0.0
            or not 0.0 <= validation_average_precision <= 1.0
        ):
            raise TransformerInferenceError("transformer fit history is invalid")
    if (
        history[best_epoch - 1]["validation_average_precision"]
        != fit["best_validation_average_precision"]
    ):
        raise TransformerInferenceError(
            "transformer fit summary does not match its history"
        )
    return fit


def _validate_transformer_metadata(
    value: object,
    *,
    policy: _BundleHashPolicy,
    artifact_hashes: dict[str, str],
    device: torch.device,
) -> tuple[dict, dict]:
    metadata = _expect_fields(value, _TRANSFORMER_FIELDS, "transformer metadata")
    expected_identity = {
        "access": _ACCESS,
        "analysis_stage": transformer_pipeline.ANALYSIS_STAGE,
        "artifact_type": "rq1-character-transformer",
        "contract_id": transformer_pipeline.CONTRACT_ID,
        "contract_sha256": policy.transformer_contract_sha256,
        "schema_version": 1,
        "warnings": [],
    }
    if any(metadata[name] != expected for name, expected in expected_identity.items()):
        raise TransformerInferenceError("transformer metadata identity is invalid")
    configuration_sha256 = sha256(
        transformer_pipeline._canonical_json_bytes(metadata["configuration"])
    ).hexdigest()
    if configuration_sha256 != _EXPECTED_PRIVATE_CONFIGURATION_SHA256:
        raise TransformerInferenceError(
            "transformer configuration does not match the frozen contract"
        )
    _validate_input_hashes(metadata["input_hashes"], policy)
    if metadata["vocabulary_sha256"] != artifact_hashes["vocabulary.json"]:
        raise TransformerInferenceError("vocabulary hash binding is invalid")
    if metadata["weights_sha256"] != artifact_hashes["transformer-weights.npz"]:
        raise TransformerInferenceError("weights hash binding is invalid")
    if metadata["software_versions"] != transformer_pipeline._software_versions():
        raise TransformerInferenceError("transformer software versions have changed")
    if metadata["runtime_observed"] != {
        "device": device.type,
        "dtype": "float32",
    }:
        raise TransformerInferenceError("transformer runtime evidence does not match")
    threshold = _validate_threshold(
        metadata["validation_threshold"], "transformer validation threshold"
    )
    _validate_fit(metadata["fit"])
    return metadata, threshold


def _validate_cascade_metadata(
    value: object,
    *,
    policy: _BundleHashPolicy,
    artifact_hashes: dict[str, str],
    transformer_metadata: dict,
    transformer_threshold: dict,
) -> tuple[dict, dict]:
    metadata = _expect_fields(value, _CASCADE_FIELDS, "cascade metadata")
    expected_identity = {
        "access": _ACCESS,
        "analysis_stage": transformer_pipeline.ANALYSIS_STAGE,
        "artifact_type": "rq1-fixed-cascade",
        "contract_id": transformer_pipeline.CONTRACT_ID,
        "contract_sha256": policy.transformer_contract_sha256,
        "schema_version": 1,
    }
    if any(metadata[name] != expected for name, expected in expected_identity.items()):
        raise TransformerInferenceError("cascade metadata identity is invalid")
    if metadata["input_hashes"] != transformer_metadata["input_hashes"]:
        raise TransformerInferenceError("cascade and transformer inputs differ")
    _validate_input_hashes(metadata["input_hashes"], policy)
    if metadata["transformer_metadata_sha256"] != artifact_hashes["transformer.json"]:
        raise TransformerInferenceError("cascade transformer binding is invalid")
    if metadata["stage1"] != {
        "artifact_sha256": policy.logistic_l1_artifact_sha256,
        "baseline_contract_sha256": policy.baseline_contract_sha256,
        "model": "Logistic-L1",
        "refit": False,
    }:
        raise TransformerInferenceError("cascade stage-one binding is invalid")
    calibration = metadata["calibration"]
    required = {
        "schema_version",
        "status",
        "accepted_cascade",
        "reason",
        "threshold_statuses",
        "candidate_count",
        "half_width",
        "transformer_invocations",
        "transformer_invocation_rate",
        "counts",
        "recall",
        "observed_fpr",
        "fpr_upper_95",
        "minimum_recall",
        "maximum_fpr_upper_95",
    }
    if type(calibration) is not dict or set(calibration) != required:
        raise TransformerInferenceError("cascade calibration schema is invalid")
    if (
        calibration["schema_version"] != 1
        or calibration["status"] != "selected"
        or calibration["accepted_cascade"] is not True
        or calibration["reason"] != "constraints_met"
        or calibration["threshold_statuses"]
        != {"stage1": "selected", "transformer": "selected"}
        or type(calibration["half_width"]) not in {int, float}
        or not np.isfinite(calibration["half_width"])
        or calibration["half_width"] < 0
    ):
        raise TransformerInferenceError(
            "cascade calibration is not an accepted selection"
        )
    metric_record = {
        "status": "selected",
        "threshold": transformer_threshold["threshold"],
        "candidate_count": calibration["candidate_count"],
        "counts": calibration["counts"],
        "recall": calibration["recall"],
        "observed_fpr": calibration["observed_fpr"],
        "fpr_upper_95": calibration["fpr_upper_95"],
    }
    _validate_threshold(metric_record, "cascade calibration")
    counts = metric_record["counts"]
    validation_rows = counts["positive"] + counts["negative"]
    try:
        candidate_count = fixed_cascade._exact_integer(
            calibration["candidate_count"], "cascade candidate_count", minimum=1
        )
        transformer_invocations = fixed_cascade._exact_integer(
            calibration["transformer_invocations"],
            "cascade transformer_invocations",
            minimum=1,
        )
        invocation_rate = fixed_cascade._finite_number(
            calibration["transformer_invocation_rate"],
            "cascade transformer_invocation_rate",
        )
        minimum_recall = fixed_cascade._finite_number(
            calibration["minimum_recall"], "cascade minimum_recall"
        )
    except fixed_cascade.FixedCascadeError as exc:
        raise TransformerInferenceError(str(exc)) from exc
    if candidate_count > validation_rows:
        raise TransformerInferenceError("cascade candidate_count is invalid")
    if transformer_invocations > validation_rows:
        raise TransformerInferenceError("cascade transformer_invocations is invalid")
    expected_rate = transformer_invocations / validation_rows
    if not np.isclose(invocation_rate, expected_rate, rtol=1e-15, atol=1e-15):
        raise TransformerInferenceError("cascade invocation rate is inconsistent")
    if not fixed_cascade._matches_exactly(
        calibration["maximum_fpr_upper_95"], fixed_cascade.MAXIMUM_FPR_UPPER_95
    ):
        raise TransformerInferenceError("cascade maximum FPR gate is invalid")
    expected_minimum_recall = (
        transformer_threshold["recall"] - fixed_cascade.RECALL_TOLERANCE
    )
    if not np.isclose(minimum_recall, expected_minimum_recall, rtol=1e-15, atol=1e-15):
        raise TransformerInferenceError("cascade minimum recall is inconsistent")
    if calibration["recall"] < minimum_recall:
        raise TransformerInferenceError("cascade does not satisfy its recall gate")
    return metadata, calibration


def _validate_public_summary_structure(summary: dict) -> dict[str, dict]:
    expected_identity = {
        "access": _ACCESS,
        "analysis_stage": transformer_pipeline.ANALYSIS_STAGE,
        "hypothesis_status": {
            "H1": "undecided",
            "H2": "undecided",
            "H3": "undecided",
        },
        "schema_version": 1,
        "status": "completed_development_validation",
    }
    if any(
        not fixed_cascade._matches_exactly(summary[name], expected)
        for name, expected in expected_identity.items()
    ):
        raise TransformerInferenceError("public summary identity is invalid")
    transformer = _expect_fields(
        summary["transformer"], _PUBLIC_TRANSFORMER_FIELDS, "public transformer"
    )
    _expect_fields(transformer["fit"], _PUBLIC_FIT_FIELDS, "public transformer fit")
    _expect_fields(
        summary["vocabulary"], _PUBLIC_VOCABULARY_FIELDS, "public vocabulary"
    )
    input_counts = _expect_fields(
        summary["input_counts"], _PUBLIC_INPUT_COUNT_FIELDS, "public input_counts"
    )
    validated = {}
    for partition_name in ("train", "validation"):
        counts = _expect_fields(
            input_counts[partition_name],
            _PUBLIC_PARTITION_COUNT_FIELDS,
            f"public input_counts.{partition_name}",
        )
        try:
            negative = fixed_cascade._exact_integer(
                counts["0"], f"public input_counts.{partition_name}.0"
            )
            positive = fixed_cascade._exact_integer(
                counts["1"], f"public input_counts.{partition_name}.1"
            )
            domains = fixed_cascade._exact_integer(
                counts["domain_count"],
                f"public input_counts.{partition_name}.domain_count",
                minimum=1,
            )
            rows = fixed_cascade._exact_integer(
                counts["rows"],
                f"public input_counts.{partition_name}.rows",
                minimum=1,
            )
        except fixed_cascade.FixedCascadeError as exc:
            raise TransformerInferenceError(str(exc)) from exc
        if rows != negative + positive or domains > rows:
            raise TransformerInferenceError(
                f"public input_counts.{partition_name} is inconsistent"
            )
        if partition_name == "validation" and (negative < 1 or positive < 1):
            raise TransformerInferenceError(
                "public validation counts must contain both classes"
            )
        validated[partition_name] = counts
    return validated


def _validate_public_projection(
    summary: dict,
    *,
    artifact_hashes: dict[str, str],
    transformer: dict,
    transformer_threshold: dict,
    cascade: dict,
    calibration: dict,
    vocabulary: character_sequence.CharacterVocabulary,
) -> dict:
    try:
        transformer_pipeline._validate_public_summary(summary)
    except transformer_pipeline.TransformerPipelineError as exc:
        raise TransformerInferenceError(str(exc)) from exc
    input_counts = _validate_public_summary_structure(summary)
    configuration_sha256 = sha256(
        transformer_pipeline._canonical_json_bytes(summary["configuration"])
    ).hexdigest()
    if configuration_sha256 != _EXPECTED_PUBLIC_CONFIGURATION_SHA256:
        raise TransformerInferenceError(
            "public configuration does not match the frozen contract"
        )
    exact_bindings = (
        summary["access"] == _ACCESS,
        summary["analysis_stage"] == transformer_pipeline.ANALYSIS_STAGE,
        summary["artifact_hashes"] == artifact_hashes,
        summary["input_hashes"]
        == transformer["input_hashes"]
        == cascade["input_hashes"],
        summary["software_versions"] == transformer["software_versions"],
        summary["warnings"] == cascade["warnings"],
        summary["contract"]
        == {
            "id": transformer_pipeline.CONTRACT_ID,
            "protocol_version": CONTRACT_PROTOCOL_VERSION,
            "sha256": transformer["contract_sha256"],
        },
        summary["cascade"] == transformer_pipeline._public_cascade_record(calibration),
        summary["transformer"]["threshold"]
        == transformer_pipeline._public_threshold_record(transformer_threshold),
        summary["vocabulary"]
        == {
            "observed_character_count": len(vocabulary.characters),
            "sha256": artifact_hashes["vocabulary.json"],
            "size": vocabulary.size,
        },
    )
    if not all(exact_bindings):
        raise TransformerInferenceError(
            "public and private artifact projections differ"
        )
    private_fit = transformer["fit"]
    expected_fit = {
        "best_epoch": private_fit["best_epoch"],
        "best_validation_average_precision": private_fit[
            "best_validation_average_precision"
        ],
        "epochs_completed": private_fit["epochs_completed"],
        "positive_class_weight": private_fit["positive_class_weight"],
        "stopped_early": private_fit["stopped_early"],
    }
    if summary["transformer"]["fit"] != expected_fit:
        raise TransformerInferenceError("transformer fit projection differs")
    for name in (
        "architecture",
        "normalization",
        "runtime",
        "training",
        "validation",
        "vocabulary",
    ):
        if summary["configuration"][name] != transformer["configuration"][name]:
            raise TransformerInferenceError(
                "transformer configuration projection differs"
            )
    validation_counts = input_counts["validation"]
    transformer_counts = transformer_threshold["counts"]
    if (
        transformer_counts["negative"] != validation_counts["0"]
        or transformer_counts["positive"] != validation_counts["1"]
    ):
        raise TransformerInferenceError(
            "transformer threshold counts differ from public validation counts"
        )
    cascade_counts = calibration["counts"]
    if (
        cascade_counts["negative"] != validation_counts["0"]
        or cascade_counts["positive"] != validation_counts["1"]
    ):
        raise TransformerInferenceError(
            "cascade counts differ from public validation counts"
        )
    return validation_counts


CONTRACT_PROTOCOL_VERSION = "1.9"


def _load_vocabulary(content: bytes) -> character_sequence.CharacterVocabulary:
    try:
        serialized = content.decode("utf-8")
        vocabulary = character_sequence.CharacterVocabulary.from_json(serialized)
    except (UnicodeDecodeError, character_sequence.CharacterSequenceError) as exc:
        raise TransformerInferenceError(f"vocabulary is invalid: {exc}") from exc
    if vocabulary.to_json().encode("utf-8") != content:
        raise TransformerInferenceError("vocabulary is not canonical JSON")
    return vocabulary


def _load_model(
    content: bytes,
    vocabulary_size: int,
    device: torch.device,
) -> character_transformer.CharacterTransformer:
    try:
        with torch.random.fork_rng(devices=[]):
            model = character_transformer.CharacterTransformer(vocabulary_size)
        expected_state = model.state_dict()
        expected_names = [f"{name}.npy" for name in sorted(expected_state)]
        arrays: dict[str, np.ndarray] = {}
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            if archive.namelist() != expected_names:
                raise TransformerInferenceError("weights keys do not match the model")
            for info, (name, expected) in zip(
                archive.infolist(), sorted(expected_state.items())
            ):
                member = archive.read(info)
                array = np.load(io.BytesIO(member), allow_pickle=False)
                if (
                    array.dtype != np.dtype("<f4")
                    or not array.flags.c_contiguous
                    or array.shape != tuple(expected.shape)
                    or not np.all(np.isfinite(array))
                ):
                    raise TransformerInferenceError(
                        f"weight {name!r} has invalid shape, dtype, or values"
                    )
                arrays[name] = np.asarray(array, dtype="<f4").copy(order="C")
        transformer_pipeline._validate_state_dict_npz(content, arrays)
        padding_row = arrays["token_embedding.weight"][0]
        if not np.array_equal(padding_row, np.zeros_like(padding_row)):
            raise TransformerInferenceError(
                "token embedding padding row must remain exactly zero"
            )
        model.load_state_dict(
            {name: torch.from_numpy(array) for name, array in arrays.items()},
            strict=True,
        )
        model.to(device=device, dtype=torch.float32)
        model.eval()
        model.requires_grad_(False)
        return model
    except TransformerInferenceError:
        raise
    except (
        EOFError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        zipfile.BadZipFile,
        transformer_pipeline.TransformerPipelineError,
    ) as exc:
        raise TransformerInferenceError(f"weights archive is invalid: {exc}") from exc


def _validate_device(device: torch.device, fixture_cpu: bool) -> torch.device:
    if type(device) is not torch.device or device.index is not None:
        raise TransformerInferenceError(
            "inference device must be an exact unindexed device"
        )
    if device.type == "cpu" and fixture_cpu is True:
        return device
    if fixture_cpu is not False:
        raise TransformerInferenceError(
            "fixture CPU permission must be an exact boolean"
        )
    if (
        device.type != "mps"
        or torch.__version__.split("+", maxsplit=1)[0] != "2.7.1"
        or not hasattr(torch.backends, "mps")
        or not torch.backends.mps.is_available()
    ):
        raise TransformerInferenceError(
            "official inference requires available MPS with PyTorch 2.7.1"
        )
    return device


def _load_transformer_cascade_bundle(
    bundle_dir: str | os.PathLike[str],
    public_summary_path: str | os.PathLike[str],
    logistic_l1_artifact_path: str | os.PathLike[str],
    *,
    _hash_policy: _BundleHashPolicy,
    _device: torch.device,
    _fixture_cpu: bool = False,
) -> LoadedTransformerCascade:
    """Private fixture seam; the public loader fixes all scientific hashes."""
    _validate_hash_policy(_hash_policy)
    device = _validate_device(_device, _fixture_cpu)
    try:
        bundle_path = Path(bundle_dir)
        summary_path = Path(public_summary_path)
        stage1_path = Path(logistic_l1_artifact_path)
    except TypeError as exc:
        raise TransformerInferenceError(
            "all inference inputs must be path-like"
        ) from exc
    bundle, summary_bytes, stage1_bytes = _snapshot_files(
        bundle_path, summary_path, stage1_path
    )
    if sha256(summary_bytes).hexdigest() != _hash_policy.public_summary_sha256:
        raise TransformerInferenceError("public summary SHA-256 mismatch")
    if sha256(stage1_bytes).hexdigest() != _hash_policy.logistic_l1_artifact_sha256:
        raise TransformerInferenceError("stage-one artifact SHA-256 mismatch")
    artifact_hashes = _artifact_hashes(bundle)
    summary = _canonical_json(summary_bytes, "public summary")
    transformer_value = _canonical_json(bundle["transformer.json"], "transformer.json")
    cascade_value = _canonical_json(bundle["cascade.json"], "cascade.json")
    vocabulary = _load_vocabulary(bundle["vocabulary.json"])
    transformer, transformer_threshold = _validate_transformer_metadata(
        transformer_value,
        policy=_hash_policy,
        artifact_hashes=artifact_hashes,
        device=device,
    )
    cascade, calibration = _validate_cascade_metadata(
        cascade_value,
        policy=_hash_policy,
        artifact_hashes=artifact_hashes,
        transformer_metadata=transformer,
        transformer_threshold=transformer_threshold,
    )
    validation_counts = _validate_public_projection(
        summary,
        artifact_hashes=artifact_hashes,
        transformer=transformer,
        transformer_threshold=transformer_threshold,
        cascade=cascade,
        calibration=calibration,
        vocabulary=vocabulary,
    )
    try:
        stage1_model = fixed_cascade._load_logistic_l1_artifact_bytes(
            stage1_bytes,
            expected_sha256=_hash_policy.logistic_l1_artifact_sha256,
            expected_contract_sha256=_hash_policy.baseline_contract_sha256,
        )
    except fixed_cascade.FixedCascadeError as exc:
        raise TransformerInferenceError(str(exc)) from exc
    stage1_threshold = _validate_threshold(
        stage1_model.validation_threshold_record, "stage-one validation threshold"
    )
    stage1_counts = stage1_threshold["counts"]
    if (
        stage1_counts["negative"] != validation_counts["0"]
        or stage1_counts["positive"] != validation_counts["1"]
    ):
        raise TransformerInferenceError(
            "stage-one threshold counts differ from public validation counts"
        )
    model = _load_model(bundle["transformer-weights.npz"], vocabulary.size, device)
    return LoadedTransformerCascade(
        stage1_model=stage1_model,
        vocabulary=vocabulary,
        stage1_threshold=float(stage1_threshold["threshold"]),
        transformer_threshold=float(transformer_threshold["threshold"]),
        half_width=float(calibration["half_width"]),
        device=device,
        public_summary_sha256=_hash_policy.public_summary_sha256,
        artifact_hashes=tuple(sorted(artifact_hashes.items())),
        _model=model,
    )


def load_transformer_cascade_bundle(
    bundle_dir: str | os.PathLike[str],
    public_summary_path: str | os.PathLike[str],
    logistic_l1_artifact_path: str | os.PathLike[str],
    *,
    expected_public_summary_sha256: str,
    device: torch.device,
) -> LoadedTransformerCascade:
    """Validate and load the active artifacts on MPS without fitting.

    Deterministic execution and request routing remain scorer responsibilities.
    """
    active_inputs = transformer_pipeline._OFFICIAL_INPUT_HASH_POLICY
    policy = _BundleHashPolicy(
        public_summary_sha256=expected_public_summary_sha256,
        train_sha256=active_inputs.train_sha256,
        validation_sha256=active_inputs.validation_sha256,
        preparation_summary_sha256=active_inputs.preparation_summary_sha256,
        transformer_contract_sha256=(
            transformer_pipeline.OFFICIAL_TRANSFORMER_CONTRACT_SHA256
        ),
        baseline_contract_sha256=fixed_cascade.OFFICIAL_BASELINE_CONTRACT_SHA256,
        logistic_l1_artifact_sha256=fixed_cascade.OFFICIAL_LOGISTIC_L1_SHA256,
    )
    return _load_transformer_cascade_bundle(
        bundle_dir,
        public_summary_path,
        logistic_l1_artifact_path,
        _hash_policy=policy,
        _device=device,
        _fixture_cpu=False,
    )
