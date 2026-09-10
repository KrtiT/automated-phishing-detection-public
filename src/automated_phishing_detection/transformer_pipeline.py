"""Frozen train/validation pipeline for the RQ1 transformer and cascade."""

from __future__ import annotations

import io
import json
import os
import platform
import shutil
import stat
import tempfile
import warnings
import zipfile
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import BinaryIO

import numpy as np
import sklearn
import torch
from torch import Tensor, nn

from . import baselines, character_sequence, character_transformer, fixed_cascade

ANALYSIS_STAGE = "development_validation_only"
CONTRACT_ID = "rq1-transformer-cascade-v1"
OFFICIAL_TRANSFORMER_CONTRACT_SHA256 = (
    "aeaa84534c4cadf0459cf6d2f010dc802684d4801cce563ce18242f36359fb54"
)
_EXPECTED_TRANSFORMER_CONTRACT_CANONICAL_SHA256 = (
    "70f528ac9e9a4838c0ed8304c3c8a10a1e1ec472cbef70e375046f35f49b3b74"
)
_PRIVATE_FILENAMES = (
    "cascade.json",
    "transformer-weights.npz",
    "transformer.json",
    "vocabulary.json",
)
_LOWERCASE_HEX = frozenset("0123456789abcdef")

_write_file = baselines._write_file
_write_summary_temp = baselines._write_summary_temp
_publish_path_without_replace = baselines._publish_path_without_replace
_path_identity = baselines._path_identity
_remove_if_identity = baselines._remove_if_identity


class TransformerPipelineError(ValueError):
    """Raised when the frozen pipeline cannot complete without protocol drift."""


@dataclass(frozen=True)
class _InputHashPolicy:
    train_sha256: str
    validation_sha256: str
    preparation_summary_sha256: str
    baseline_contract_sha256: str
    logistic_l1_artifact_sha256: str
    transformer_contract_sha256: str


_OFFICIAL_INPUT_HASH_POLICY = _InputHashPolicy(
    train_sha256=("575f2fb13a0766020e29d78bf8e633a185b381abde7060bdd1ed04cc4a5e38a0"),
    validation_sha256=(
        "970c6568a6400a1fc265b7809ef7bd9d1c297632799cbb88801313bc34ac415a"
    ),
    preparation_summary_sha256=(
        "1a85a7eecc0f5baa7c59e03a0cbde63fd4595409feb918dc5ff916ead7cd5c9e"
    ),
    baseline_contract_sha256=(
        "05d6d0831def7d26448c8dbdc8117800ea2448cdfc2aca2ad95489f22d2d11ba"
    ),
    logistic_l1_artifact_sha256=(
        "71a3e24a0283a31ba188bc7dd60b18c1b708370b9ca275d5ab1a1004680c968a"
    ),
    transformer_contract_sha256=OFFICIAL_TRANSFORMER_CONTRACT_SHA256,
)


@dataclass(frozen=True)
class _Partition:
    raw_urls: tuple[str, ...]
    labels: np.ndarray
    domains: frozenset[str]
    ordinals: frozenset[int]
    class_counts: dict[str, int]


def _lowercase_sha256(value: object, field: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _LOWERCASE_HEX for character in value)
    ):
        raise TransformerPipelineError(f"{field} must be a lowercase SHA-256")
    return value


def _object_without_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise TransformerPipelineError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str):
    raise TransformerPipelineError(f"JSON contains a nonfinite number: {value}")


def _load_json(content: bytes, field: str) -> object:
    try:
        return json.loads(
            content.decode("utf-8"),
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except UnicodeDecodeError as exc:
        raise TransformerPipelineError(f"{field} is not valid UTF-8") from exc
    except json.JSONDecodeError as exc:
        raise TransformerPipelineError(f"{field} is not valid JSON") from exc


def _canonical_json_bytes(value: object) -> bytes:
    try:
        serialized = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise TransformerPipelineError("artifact is not finite canonical JSON") from exc
    return f"{serialized}\n".encode()


def _validate_transformer_contract(value: object) -> dict:
    try:
        canonical = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise TransformerPipelineError(
            "transformer contract is not finite canonical JSON"
        ) from exc
    if sha256(canonical).hexdigest() != _EXPECTED_TRANSFORMER_CONTRACT_CANONICAL_SHA256:
        raise TransformerPipelineError(
            "transformer contract content does not match the frozen method"
        )
    if type(value) is not dict:
        raise TransformerPipelineError("transformer contract must be an object")
    return value


def _expected_input_hashes(policy: _InputHashPolicy) -> dict[str, str]:
    if type(policy) is not _InputHashPolicy:
        raise TransformerPipelineError("input hash policy is invalid")
    expected = {
        "train": policy.train_sha256,
        "validation": policy.validation_sha256,
        "preparation_summary": policy.preparation_summary_sha256,
        "baseline_contract": policy.baseline_contract_sha256,
        "logistic_l1_artifact": policy.logistic_l1_artifact_sha256,
        "transformer_contract": policy.transformer_contract_sha256,
    }
    for name, digest in expected.items():
        _lowercase_sha256(digest, f"{name} expected hash")
    return expected


def _verify_hashes(
    streams: dict[str, BinaryIO], policy: _InputHashPolicy
) -> dict[str, str]:
    expected = _expected_input_hashes(policy)
    observed = {
        name: baselines._hash_stream(stream) for name, stream in streams.items()
    }
    for name, expected_digest in expected.items():
        if observed[name] != expected_digest:
            raise TransformerPipelineError(
                f"{name} SHA-256 mismatch: expected {expected_digest}, "
                f"observed {observed[name]}"
            )
    return observed


def _require_real_directory(path: Path, field: str) -> None:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise TransformerPipelineError(f"{field} must already exist") from exc
    if not stat.S_ISDIR(metadata.st_mode):
        raise TransformerPipelineError(f"{field} must be a real directory")


def _validate_paths(
    *, input_paths: dict[str, Path], output_dir: Path, summary_path: Path
) -> None:
    for label, input_path in input_paths.items():
        try:
            metadata = input_path.lstat()
        except OSError as exc:
            raise TransformerPipelineError(f"{label} input does not exist") from exc
        if not stat.S_ISREG(metadata.st_mode):
            raise TransformerPipelineError(
                f"{label} input must be a regular file, not an alias"
            )
    try:
        baselines._validate_paths(
            input_paths=input_paths,
            output_dir=output_dir,
            summary_path=summary_path,
        )
    except (OSError, baselines.BaselineError) as exc:
        raise TransformerPipelineError(str(exc)) from exc
    _require_real_directory(output_dir.parent, "output directory parent")
    _require_real_directory(summary_path.parent, "summary parent")


def _read_raw_urls(stream: BinaryIO) -> tuple[str, ...]:
    stream.seek(0)
    rows = []
    for raw_line in stream:
        record = _load_json(raw_line, "partition record")
        rows.append(record["raw_url"])
    stream.seek(0)
    return tuple(rows)


def _load_partition(
    stream: BinaryIO,
    *,
    split: str,
    declared: dict,
    source_csv_sha256: str,
) -> _Partition:
    try:
        _, labels, domains, ordinals, class_counts = baselines._load_partition(
            stream,
            split=split,
            declared=declared,
            source_csv_sha256=source_csv_sha256,
        )
    except baselines.BaselineError as exc:
        raise TransformerPipelineError(str(exc)) from exc
    raw_urls = _read_raw_urls(stream)
    if len(raw_urls) != int(labels.shape[0]):
        raise TransformerPipelineError(f"{split} URL and label rows differ")
    return _Partition(
        raw_urls=raw_urls,
        labels=labels,
        domains=frozenset(domains),
        ordinals=frozenset(ordinals),
        class_counts=class_counts,
    )


def _encode_partition(
    partition: _Partition, vocabulary: character_sequence.CharacterVocabulary
) -> tuple[Tensor, Tensor, Tensor]:
    shape = (len(partition.raw_urls), character_sequence.MAX_SEQUENCE_LENGTH)
    token_array = np.empty(shape, dtype=np.int64)
    mask_array = np.empty(shape, dtype=np.bool_)
    for index, raw_url in enumerate(partition.raw_urls):
        encoded = character_sequence.encode_character_url(raw_url, vocabulary)
        token_array[index] = encoded.token_ids
        mask_array[index] = encoded.padding_mask
    token_ids = torch.from_numpy(token_array)
    padding_mask = torch.from_numpy(mask_array)
    labels = torch.from_numpy(partition.labels.copy()).to(dtype=torch.float32)
    return token_ids, padding_mask, labels


def _train_transformer(
    train_token_ids: Tensor,
    train_padding_mask: Tensor,
    train_labels: Tensor,
    validation_token_ids: Tensor,
    validation_padding_mask: Tensor,
    validation_labels: Tensor,
    *,
    vocabulary_size: int,
    device: torch.device | None,
) -> character_transformer.TransformerFit:
    if device is None:
        return character_transformer.fit_character_transformer(
            train_token_ids,
            train_padding_mask,
            train_labels,
            validation_token_ids,
            validation_padding_mask,
            validation_labels,
            vocabulary_size=vocabulary_size,
        )
    return character_transformer._fit_character_transformer_on_device(
        train_token_ids,
        train_padding_mask,
        train_labels,
        validation_token_ids,
        validation_padding_mask,
        validation_labels,
        vocabulary_size=vocabulary_size,
        device=device,
    )


def _state_dict_arrays(model: nn.Module) -> dict[str, np.ndarray]:
    if not isinstance(model, nn.Module):
        raise TransformerPipelineError("trained model is not a PyTorch module")
    state = model.state_dict()
    if not state:
        raise TransformerPipelineError("trained model state is empty")
    arrays = {}
    for name in sorted(state):
        if type(name) is not str or not name or "/" in name or "\\" in name:
            raise TransformerPipelineError("model state contains an unsafe key")
        value = state[name]
        if not isinstance(value, Tensor):
            raise TransformerPipelineError("model state contains a non-tensor value")
        try:
            array = value.detach().cpu().numpy().astype("<f4", copy=True)
        except (TypeError, RuntimeError, ValueError) as exc:
            raise TransformerPipelineError(
                f"model state tensor {name!r} cannot be serialized"
            ) from exc
        array = np.ascontiguousarray(array, dtype="<f4")
        if not np.all(np.isfinite(array)):
            raise TransformerPipelineError("model state contains a nonfinite value")
        arrays[name] = array
    return arrays


def _serialize_state_dict_npz(model: nn.Module) -> tuple[bytes, dict[str, np.ndarray]]:
    arrays = _state_dict_arrays(model)
    destination = io.BytesIO()
    with zipfile.ZipFile(
        destination, mode="w", compression=zipfile.ZIP_STORED
    ) as archive:
        archive.comment = b""
        for name, array in arrays.items():
            member = io.BytesIO()
            np.lib.format.write_array(member, array, version=(1, 0), allow_pickle=False)
            info = zipfile.ZipInfo(
                filename=f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0)
            )
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            info.external_attr = 0o600 << 16
            info.extra = b""
            info.comment = b""
            archive.writestr(info, member.getvalue())
    return destination.getvalue(), arrays


def _validate_state_dict_npz(
    content: bytes, expected_arrays: dict[str, np.ndarray]
) -> None:
    expected_names = [f"{name}.npy" for name in sorted(expected_arrays)]
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            if archive.comment != b"" or archive.namelist() != expected_names:
                raise TransformerPipelineError(
                    "weights archive member order is invalid"
                )
            if archive.testzip() is not None:
                raise TransformerPipelineError("weights archive checksum is invalid")
            for info, (name, expected) in zip(
                archive.infolist(), sorted(expected_arrays.items())
            ):
                if (
                    info.filename != f"{name}.npy"
                    or info.compress_type != zipfile.ZIP_STORED
                    or info.date_time != (1980, 1, 1, 0, 0, 0)
                    or info.create_system != 3
                    or info.external_attr != 0o600 << 16
                    or info.extra != b""
                    or info.comment != b""
                ):
                    raise TransformerPipelineError(
                        "weights archive metadata violates the frozen format"
                    )
                member = archive.read(info)
                if not member.startswith(b"\x93NUMPY\x01\x00"):
                    raise TransformerPipelineError("weight member is not NPY 1.0")
                loaded = np.load(io.BytesIO(member), allow_pickle=False)
                if (
                    loaded.dtype != np.dtype("<f4")
                    or not loaded.flags.c_contiguous
                    or loaded.shape != expected.shape
                    or not np.array_equal(loaded, expected)
                ):
                    raise TransformerPipelineError(
                        "weight member does not match the frozen tensor encoding"
                    )
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        if isinstance(exc, TransformerPipelineError):
            raise
        raise TransformerPipelineError("weights archive validation failed") from exc


def _software_versions() -> dict[str, str]:
    return {
        "numpy": np.__version__,
        "python": platform.python_version(),
        "pytorch": torch.__version__.split("+", maxsplit=1)[0],
        "scikit-learn": sklearn.__version__,
    }


def _partition_counts(partition: _Partition) -> dict[str, int]:
    return {
        "0": partition.class_counts["0"],
        "1": partition.class_counts["1"],
        "domain_count": len(partition.domains),
        "rows": len(partition.raw_urls),
    }


def _public_threshold_record(record: dict[str, object]) -> dict[str, object]:
    return {key: value for key, value in record.items() if key != "threshold"}


def _public_cascade_record(record: dict[str, object]) -> dict[str, object]:
    return {key: value for key, value in record.items() if key != "half_width"}


def _transformer_metadata(
    *,
    fit: character_transformer.TransformerFit,
    threshold: dict[str, object],
    input_hashes: dict[str, str],
    contract: dict,
    vocabulary_sha256: str,
    weights_sha256: str,
    training_device: torch.device | None,
) -> dict[str, object]:
    return {
        "access": {"group_test_accessed": False, "phishvn_accessed": False},
        "analysis_stage": ANALYSIS_STAGE,
        "artifact_type": "rq1-character-transformer",
        "configuration": {
            "architecture": contract["architecture"],
            "normalization": contract["normalization"],
            "runtime": contract["runtime"],
            "training": contract["training"],
            "validation": contract["validation"],
            "vocabulary": contract["vocabulary"],
        },
        "contract_id": CONTRACT_ID,
        "contract_sha256": input_hashes["transformer_contract"],
        "fit": {
            "best_epoch": fit.best_epoch,
            "best_validation_average_precision": (
                fit.best_validation_average_precision
            ),
            "epochs_completed": fit.epochs_completed,
            "history": [asdict(record) for record in fit.history],
            "positive_class_weight": fit.positive_class_weight,
            "stopped_early": fit.stopped_early,
        },
        "input_hashes": input_hashes,
        "runtime_observed": {
            "device": "mps" if training_device is None else training_device.type,
            "dtype": "float32",
        },
        "schema_version": 1,
        "software_versions": _software_versions(),
        "validation_threshold": threshold,
        "vocabulary_sha256": vocabulary_sha256,
        "warnings": [],
        "weights_sha256": weights_sha256,
    }


def _cascade_metadata(
    *,
    calibration: dict[str, object],
    input_hashes: dict[str, str],
    transformer_sha256: str,
) -> dict[str, object]:
    return {
        "access": {"group_test_accessed": False, "phishvn_accessed": False},
        "analysis_stage": ANALYSIS_STAGE,
        "artifact_type": "rq1-fixed-cascade",
        "calibration": calibration,
        "contract_id": CONTRACT_ID,
        "contract_sha256": input_hashes["transformer_contract"],
        "input_hashes": input_hashes,
        "schema_version": 1,
        "stage1": {
            "artifact_sha256": input_hashes["logistic_l1_artifact"],
            "baseline_contract_sha256": input_hashes["baseline_contract"],
            "model": "Logistic-L1",
            "refit": False,
        },
        "transformer_metadata_sha256": transformer_sha256,
        "warnings": [],
    }


def _public_summary(
    *,
    train: _Partition,
    validation: _Partition,
    vocabulary: character_sequence.CharacterVocabulary,
    transformer_fit: character_transformer.TransformerFit,
    transformer_threshold: dict[str, object],
    cascade: dict[str, object],
    contract: dict,
    input_hashes: dict[str, str],
    artifact_hashes: dict[str, str],
) -> dict[str, object]:
    status = (
        "completed_development_validation"
        if transformer_threshold["status"] == "selected"
        and cascade["status"] == "selected"
        else "target_not_met"
    )
    return {
        "access": {"group_test_accessed": False, "phishvn_accessed": False},
        "analysis_stage": ANALYSIS_STAGE,
        "artifact_hashes": artifact_hashes,
        "cascade": _public_cascade_record(cascade),
        "configuration": {
            "architecture": contract["architecture"],
            "cascade": contract["cascade"],
            "normalization": contract["normalization"],
            "runtime": contract["runtime"],
            "threshold_selection": contract["threshold_selection"],
            "training": contract["training"],
            "validation": contract["validation"],
            "vocabulary": contract["vocabulary"],
        },
        "contract": {
            "id": CONTRACT_ID,
            "protocol_version": contract["protocol_version"],
            "sha256": input_hashes["transformer_contract"],
        },
        "hypothesis_status": {"H1": "undecided", "H2": "undecided", "H3": "undecided"},
        "input_counts": {
            "train": _partition_counts(train),
            "validation": _partition_counts(validation),
        },
        "input_hashes": input_hashes,
        "schema_version": 1,
        "software_versions": _software_versions(),
        "status": status,
        "transformer": {
            "fit": {
                "best_epoch": transformer_fit.best_epoch,
                "best_validation_average_precision": (
                    transformer_fit.best_validation_average_precision
                ),
                "epochs_completed": transformer_fit.epochs_completed,
                "positive_class_weight": transformer_fit.positive_class_weight,
                "stopped_early": transformer_fit.stopped_early,
            },
            "threshold": _public_threshold_record(transformer_threshold),
        },
        "vocabulary": {
            "observed_character_count": len(vocabulary.characters),
            "sha256": artifact_hashes["vocabulary.json"],
            "size": vocabulary.size,
        },
        "warnings": [],
    }


def _validate_public_summary(summary: dict[str, object]) -> None:
    expected = {
        "access",
        "analysis_stage",
        "artifact_hashes",
        "cascade",
        "configuration",
        "contract",
        "hypothesis_status",
        "input_counts",
        "input_hashes",
        "schema_version",
        "software_versions",
        "status",
        "transformer",
        "vocabulary",
        "warnings",
    }
    if set(summary) != expected:
        raise TransformerPipelineError("public summary fields are not allowlisted")
    forbidden_keys = {
        "coefficient",
        "coefficients",
        "prediction",
        "predictions",
        "raw_url",
        "record_id",
        "registrable_domain",
        "sequence",
        "sequences",
        "weights",
    }

    def scan(value: object) -> None:
        if isinstance(value, dict):
            if set(value) & forbidden_keys:
                raise TransformerPipelineError("public summary contains row-level data")
            for child in value.values():
                scan(child)
        elif isinstance(value, list):
            for child in value:
                scan(child)

    scan(summary)
    _canonical_json_bytes(summary)


def _publish_artifacts(
    *,
    output_dir: Path,
    summary_path: Path,
    private_contents: dict[str, bytes],
    summary: dict[str, object],
    expected_arrays: dict[str, np.ndarray],
) -> dict[str, object]:
    temporary_output: Path | None = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent)
    )
    os.chmod(temporary_output, 0o700)
    temporary_summary: Path | None = None
    try:
        for filename in (*_PRIVATE_FILENAMES, "SHA256SUMS"):
            _write_file(temporary_output / filename, private_contents[filename], 0o600)
        if stat.S_IMODE(temporary_output.stat().st_mode) != 0o700 or any(
            stat.S_IMODE((temporary_output / name).stat().st_mode) != 0o600
            for name in (*_PRIVATE_FILENAMES, "SHA256SUMS")
        ):
            raise TransformerPipelineError("private artifact permissions are invalid")
        _validate_state_dict_npz(
            (temporary_output / "transformer-weights.npz").read_bytes(),
            expected_arrays,
        )
        for filename in ("vocabulary.json", "transformer.json", "cascade.json"):
            content = (temporary_output / filename).read_bytes()
            parsed = _load_json(content, filename)
            if _canonical_json_bytes(parsed) != content:
                raise TransformerPipelineError(f"{filename} is not canonical JSON")

        summary_bytes = _canonical_json_bytes(summary)
        temporary_summary = _write_summary_temp(summary_path, summary_bytes)
        output_identity = _path_identity(temporary_output)
        summary_identity = _path_identity(temporary_summary)
        try:
            _publish_path_without_replace(temporary_output, output_dir)
            temporary_output = None
            _publish_path_without_replace(temporary_summary, summary_path)
            temporary_summary = None
        except BaseException:
            _remove_if_identity(summary_path, summary_identity)
            _remove_if_identity(output_dir, output_identity)
            raise
        return summary
    finally:
        if temporary_output is not None:
            shutil.rmtree(temporary_output, ignore_errors=True)
        if temporary_summary is not None:
            temporary_summary.unlink(missing_ok=True)


def _fit_transformer_cascade(
    *,
    train_path: Path,
    validation_path: Path,
    preparation_summary_path: Path,
    baseline_contract_path: Path,
    logistic_l1_artifact_path: Path,
    transformer_contract_path: Path,
    output_dir: Path,
    summary_path: Path,
    _input_hash_policy: _InputHashPolicy,
    _training_device: torch.device | None,
) -> dict[str, object]:
    """Private execution seam used by deterministic, train/validation-only tests."""
    if (
        _input_hash_policy == _OFFICIAL_INPUT_HASH_POLICY
        and _training_device is not None
    ):
        raise TransformerPipelineError(
            "the official fit cannot override the MPS device"
        )
    if _training_device is not None and (
        not isinstance(_training_device, torch.device)
        or _training_device.type not in {"cpu", "mps"}
    ):
        raise TransformerPipelineError("fixture device must be CPU or MPS")

    try:
        input_paths = {
            "train": Path(train_path),
            "validation": Path(validation_path),
            "preparation_summary": Path(preparation_summary_path),
            "baseline_contract": Path(baseline_contract_path),
            "logistic_l1_artifact": Path(logistic_l1_artifact_path),
            "transformer_contract": Path(transformer_contract_path),
        }
        output_dir = Path(output_dir)
        summary_path = Path(summary_path)
    except TypeError as exc:
        raise TransformerPipelineError(
            "all inputs and outputs must be path-like"
        ) from exc
    _validate_paths(
        input_paths=input_paths, output_dir=output_dir, summary_path=summary_path
    )

    with ExitStack() as stack:
        try:
            streams, stability = baselines._open_input_streams(input_paths, stack)
        except baselines.BaselineError as exc:
            raise TransformerPipelineError(str(exc)) from exc
        input_hashes = _verify_hashes(streams, _input_hash_policy)
        try:
            baselines._require_stable_streams(streams, stability)
            preparation_summary = baselines._validate_preparation_summary(
                _load_json(streams["preparation_summary"].read(), "preparation summary")
            )
            baseline_contract = baselines._validate_contract(
                _load_json(streams["baseline_contract"].read(), "baseline contract")
            )
        except baselines.BaselineError as exc:
            raise TransformerPipelineError(str(exc)) from exc
        transformer_contract = _validate_transformer_contract(
            _load_json(streams["transformer_contract"].read(), "transformer contract")
        )
        if _input_hash_policy == _OFFICIAL_INPUT_HASH_POLICY:
            accepted = transformer_contract["inputs"]["accepted_roles"]
            expected_roles = {
                "train": input_hashes["train"],
                "validation": input_hashes["validation"],
                "preparation_summary": input_hashes["preparation_summary"],
                "logistic_l1_artifact": input_hashes["logistic_l1_artifact"],
                "contract": input_hashes["baseline_contract"],
            }
            if accepted != expected_roles:
                raise TransformerPipelineError(
                    "official input hashes do not match the transformer contract"
                )
        if baseline_contract["contract_id"] != "rq1-baselines-v2":
            raise TransformerPipelineError("baseline contract identity is invalid")
        if preparation_summary["output_hashes"]["train.jsonl"] != input_hashes["train"]:
            raise TransformerPipelineError(
                "train hash does not match the preparation summary"
            )
        if (
            preparation_summary["output_hashes"]["validation.jsonl"]
            != input_hashes["validation"]
        ):
            raise TransformerPipelineError(
                "validation hash does not match the preparation summary"
            )

        train = _load_partition(
            streams["train"],
            split="train",
            declared=preparation_summary["splits"]["train"],
            source_csv_sha256=preparation_summary["source_csv_sha256"],
        )
        validation = _load_partition(
            streams["validation"],
            split="validation",
            declared=preparation_summary["splits"]["validation"],
            source_csv_sha256=preparation_summary["source_csv_sha256"],
        )
        if train.domains & validation.domains:
            raise TransformerPipelineError(
                "registrable domain crosses train and validation"
            )
        if train.ordinals & validation.ordinals:
            raise TransformerPipelineError(
                "record identifier crosses train and validation"
            )
        try:
            stage1_model = fixed_cascade.load_logistic_l1_artifact(
                input_paths["logistic_l1_artifact"],
                expected_sha256=input_hashes["logistic_l1_artifact"],
                expected_contract_sha256=input_hashes["baseline_contract"],
            )
        except fixed_cascade.FixedCascadeError as exc:
            raise TransformerPipelineError(str(exc)) from exc
        try:
            baselines._require_stable_streams(streams, stability)
        except baselines.BaselineError as exc:
            raise TransformerPipelineError(str(exc)) from exc

    try:
        vocabulary = character_sequence.build_character_vocabulary(train.raw_urls)
        train_tensors = _encode_partition(train, vocabulary)
        validation_tensors = _encode_partition(validation, vocabulary)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            fit = _train_transformer(
                *train_tensors,
                *validation_tensors,
                vocabulary_size=vocabulary.size,
                device=_training_device,
            )
            validation_probabilities = np.asarray(
                fit.validation_probabilities, dtype=np.float64
            )
            transformer_threshold = baselines.select_validation_threshold(
                validation_probabilities, validation.labels
            )
            stage1_probabilities = stage1_model.score_urls(validation.raw_urls)
            cascade = fixed_cascade.calibrate_fixed_cascade(
                stage1_model,
                stage1_probabilities,
                validation_probabilities,
                validation.labels,
                transformer_threshold_record=transformer_threshold,
            )
    except Warning as exc:
        raise TransformerPipelineError(f"pipeline stopped on warning: {exc}") from exc
    except (
        baselines.BaselineError,
        character_sequence.CharacterSequenceError,
        character_transformer.TransformerTrainingError,
        fixed_cascade.FixedCascadeError,
        FloatingPointError,
        TypeError,
        ValueError,
    ) as exc:
        if isinstance(exc, TransformerPipelineError):
            raise
        raise TransformerPipelineError(str(exc)) from exc

    weights_bytes, expected_arrays = _serialize_state_dict_npz(fit.model)
    _validate_state_dict_npz(weights_bytes, expected_arrays)
    vocabulary_bytes = vocabulary.to_json().encode("utf-8")
    vocabulary_sha256 = sha256(vocabulary_bytes).hexdigest()
    weights_sha256 = sha256(weights_bytes).hexdigest()
    transformer = _transformer_metadata(
        fit=fit,
        threshold=transformer_threshold,
        input_hashes=input_hashes,
        contract=transformer_contract,
        vocabulary_sha256=vocabulary_sha256,
        weights_sha256=weights_sha256,
        training_device=_training_device,
    )
    transformer_bytes = _canonical_json_bytes(transformer)
    cascade_metadata = _cascade_metadata(
        calibration=cascade,
        input_hashes=input_hashes,
        transformer_sha256=sha256(transformer_bytes).hexdigest(),
    )
    private_contents = {
        "cascade.json": _canonical_json_bytes(cascade_metadata),
        "transformer-weights.npz": weights_bytes,
        "transformer.json": transformer_bytes,
        "vocabulary.json": vocabulary_bytes,
    }
    artifact_hashes = {
        name: sha256(private_contents[name]).hexdigest() for name in _PRIVATE_FILENAMES
    }
    checksum_bytes = "".join(
        f"{artifact_hashes[name]}  {name}\n" for name in sorted(artifact_hashes)
    ).encode("ascii")
    private_contents["SHA256SUMS"] = checksum_bytes
    summary = _public_summary(
        train=train,
        validation=validation,
        vocabulary=vocabulary,
        transformer_fit=fit,
        transformer_threshold=transformer_threshold,
        cascade=cascade,
        contract=transformer_contract,
        input_hashes=input_hashes,
        artifact_hashes=artifact_hashes,
    )
    _validate_public_summary(summary)
    try:
        return _publish_artifacts(
            output_dir=output_dir,
            summary_path=summary_path,
            private_contents=private_contents,
            summary=summary,
            expected_arrays=expected_arrays,
        )
    except baselines.BaselineError as exc:
        raise TransformerPipelineError(str(exc)) from exc


def fit_transformer_cascade(
    *,
    train_path: Path,
    validation_path: Path,
    preparation_summary_path: Path,
    baseline_contract_path: Path,
    logistic_l1_artifact_path: Path,
    transformer_contract_path: Path,
    output_dir: Path,
    summary_path: Path,
) -> dict[str, object]:
    """Run the exact official MPS fit with no held-out-data input surface."""
    return _fit_transformer_cascade(
        train_path=train_path,
        validation_path=validation_path,
        preparation_summary_path=preparation_summary_path,
        baseline_contract_path=baseline_contract_path,
        logistic_l1_artifact_path=logistic_l1_artifact_path,
        transformer_contract_path=transformer_contract_path,
        output_dir=output_dir,
        summary_path=summary_path,
        _input_hash_policy=_OFFICIAL_INPUT_HASH_POLICY,
        _training_device=None,
    )
