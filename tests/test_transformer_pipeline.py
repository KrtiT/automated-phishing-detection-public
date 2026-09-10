import inspect
import json
import os
import stat
import warnings
import zipfile
from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from automated_phishing_detection import (
    baselines,
    character_sequence,
    character_transformer,
    transformer_pipeline,
)
from automated_phishing_detection.url_features import FEATURE_NAMES

ROOT = Path(__file__).resolve().parents[1]
BASELINE_CONTRACT = ROOT / "data" / "rq1-baseline-contract-v2.json"
TRANSFORMER_CONTRACT = ROOT / "data" / "rq1-transformer-cascade-contract-v1.json"
SOURCE_SHA256 = "0" * 64


def _canonical_json_bytes(value):
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")


def _records(split, *, negatives, positives, ordinal_start):
    records = []
    marker = {"train": "tr", "validation": "va"}[split]
    for offset in range(negatives + positives):
        ordinal = ordinal_start + offset
        label = int(offset >= negatives)
        label_word = "phish" if label else "safe"
        validation_only_character = "~" if split == "validation" else ""
        raw_url = (
            f"https://{label_word}-{ordinal:04d}.{marker}-{ordinal:04d}.example/"
            f"{label_word}{validation_only_character}"
        )
        records.append(
            {
                "canonical_url_sha256": sha256(raw_url.encode("ascii")).hexdigest(),
                "is_phishing": label,
                "raw_url": raw_url,
                "record_id": f"phiusiil-row-v1:{SOURCE_SHA256}:{ordinal:016x}",
                "registrable_domain": f"{marker}-{ordinal:04d}.example",
                "split": split,
            }
        )
    return records


def _preparation_summary(train_rows, validation_rows, train_bytes, validation_bytes):
    def split_record(rows):
        return {
            "class_counts": {
                "0": sum(row["is_phishing"] == 0 for row in rows),
                "1": sum(row["is_phishing"] == 1 for row in rows),
            },
            "domain_count": len(rows),
            "row_count": len(rows),
        }

    train_split = split_record(train_rows)
    validation_split = split_record(validation_rows)
    return {
        "algorithms": {
            "allocation_basis": "unique_ascii_domain_groups",
            "allocation_version": "hamilton-largest-remainder-v1",
            "canonicalization_version": "canonical-url-v1",
            "domain_split_version": "phiusiil-domain-split-v1",
            "record_identifier_version": "phiusiil-row-v1",
            "seed": "20260816",
            "split_percentages": {"group_test": 15, "train": 70, "validation": 15},
        },
        "declared_sources": {
            "contract_id": "phiusiil-development-v1",
            "phiusiil": {
                "archive_sha256": "1" * 64,
                "archive_url": "https://example.invalid/archive.zip",
                "csv_filename": "fixture.csv",
                "csv_sha256": SOURCE_SHA256,
                "license": "CC BY 4.0",
                "native_label_semantics": {"0": "phishing", "1": "legitimate"},
                "page_url": "https://example.invalid/dataset",
                "paper_doi": "10.0000/fixture",
                "public_per_row_provenance_available": {
                    "independent_adjudication": False,
                    "snapshot": False,
                    "source": False,
                    "timestamp": False,
                },
                "publisher_reported_class_sources": {
                    "legitimate": ["fixture legitimate"],
                    "phishing": ["fixture phishing"],
                },
                "publisher_reported_legitimate_collection_window": None,
                "publisher_reported_phishing_retrieval_window": {
                    "end": "2023-05-21",
                    "start": "2022-10-01",
                },
                "reference_classification_basis": "publisher-provided/source-derived",
                "uci_dataset_id": 967,
            },
            "public_suffix_list": {
                "commit": "2" * 40,
                "license": "MPL-2.0",
                "sha256": "3" * 64,
                "upstream_url": "https://example.invalid/psl",
                "url": "https://example.invalid/psl-at-commit",
                "version": "commit-pinned snapshot",
            },
            "schema_version": 2,
        },
        "label_mapping": {
            "is_phishing_meanings": {"0": "legitimate", "1": "phishing"},
            "native_label_meanings": {"0": "phishing", "1": "legitimate"},
            "native_to_is_phishing": {"0": 1, "1": 0},
            "version": "phiusiil-native-label-map-v1",
        },
        "local_label_counts": {
            "0": train_split["class_counts"]["0"]
            + validation_split["class_counts"]["0"]
            + 1,
            "1": train_split["class_counts"]["1"]
            + validation_split["class_counts"]["1"]
            + 1,
        },
        "native_label_counts": {"0": 1, "1": 1, "invalid": 0},
        "output_hashes": {
            "SHA256SUMS": "4" * 64,
            "group_test.jsonl": "5" * 64,
            "quarantine.jsonl": "6" * 64,
            "train.jsonl": sha256(train_bytes).hexdigest(),
            "validation.jsonl": sha256(validation_bytes).hexdigest(),
        },
        "overall_counts": {
            "canonicalized_url_groups": len(train_rows) + len(validation_rows) + 2,
            "input_rows": len(train_rows) + len(validation_rows) + 2,
            "quarantined_rows": 0,
            "retained_domains": len(train_rows) + len(validation_rows) + 2,
            "retained_rows": len(train_rows) + len(validation_rows) + 2,
        },
        "quarantine_reason_counts": {
            "canonical_url_conflicting_mapping": 0,
            "canonical_url_duplicate_same_mapping": 0,
            "invalid_or_missing_url": 0,
            "invalid_phiusiil_native_label": 0,
        },
        "schema_version": 1,
        "source_spec_sha256": "7" * 64,
        "splits": {
            "group_test": {
                "class_counts": {"0": 1, "1": 1},
                "domain_count": 2,
                "row_count": 2,
            },
            "train": train_split,
            "validation": validation_split,
        },
    }


def _logistic_artifact(labels, *, baseline_contract_sha256):
    scores = np.full(labels.shape, 0.5, dtype=np.float64)
    threshold = baselines.select_validation_threshold(scores, labels)
    width = len(FEATURE_NAMES)
    return {
        "access": {"group_test_accessed": False, "phishvn_accessed": False},
        "analysis_stage": "development_validation_only",
        "artifact_type": "rq1-baseline-model",
        "classes": [0, 1],
        "classifier": {
            "coefficients": [[0.0] * width],
            "config": {
                "C": 1.0,
                "class": "LogisticRegression",
                "class_weight": "balanced",
                "fit_intercept": True,
                "max_iter": 5000,
                "penalty": "l1",
                "random_state": 42,
                "solver": "saga",
                "tol": 0.0001,
            },
            "intercept": [0.0],
            "n_iter": [12],
        },
        "contract_id": "rq1-baselines-v2",
        "contract_sha256": baseline_contract_sha256,
        "features": list(FEATURE_NAMES),
        "input_hashes": {
            "contract": baseline_contract_sha256,
            "preparation_summary": "8" * 64,
            "train": "9" * 64,
            "validation": "a" * 64,
        },
        "model_name": "Logistic-L1",
        "scaler": {
            "config": {
                "class": "StandardScaler",
                "fit_partition": "train",
                "with_mean": True,
                "with_std": True,
            },
            "mean": [0.0] * width,
            "n_samples_seen": 20,
            "scale": [1.0] * width,
            "variance": [1.0] * width,
        },
        "schema_version": 2,
        "software_versions": {
            "numpy": "2.2.6",
            "scikit-learn": "1.7.2",
            "scipy": "1.15.3",
        },
        "validation_scoring_audit": {
            "max_absolute_decision_difference": 0.0,
            "max_absolute_probability_difference": 0.0,
            "platform_identity": {
                "numpy_blas_name": "accelerate",
                "platform_machine": "arm64",
                "sys_platform": "darwin",
            },
            "warning_records": [],
        },
        "validation_threshold": threshold,
    }


def _write_fixture(directory, *, validation_negatives=400):
    directory.mkdir()
    train_rows = _records("train", negatives=12, positives=12, ordinal_start=1)
    validation_rows = _records(
        "validation", negatives=validation_negatives, positives=20, ordinal_start=1000
    )
    train_bytes = b"".join(_canonical_json_bytes(row) for row in train_rows)
    validation_bytes = b"".join(_canonical_json_bytes(row) for row in validation_rows)
    summary_bytes = (
        json.dumps(
            _preparation_summary(
                train_rows, validation_rows, train_bytes, validation_bytes
            ),
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")

    paths = {
        "train": directory / "train.jsonl",
        "validation": directory / "validation.jsonl",
        "preparation_summary": directory / "preparation-summary.json",
        "baseline_contract": directory / "rq1-baseline-contract-v2.json",
        "logistic_l1_artifact": directory / "logistic-l1.json",
        "transformer_contract": directory / "rq1-transformer-cascade-contract-v1.json",
        "output_dir": directory / "private-transformer",
        "summary": directory / "transformer-summary.json",
    }
    paths["train"].write_bytes(train_bytes)
    paths["validation"].write_bytes(validation_bytes)
    paths["preparation_summary"].write_bytes(summary_bytes)
    paths["baseline_contract"].write_bytes(BASELINE_CONTRACT.read_bytes())
    paths["transformer_contract"].write_bytes(TRANSFORMER_CONTRACT.read_bytes())
    baseline_contract_sha256 = sha256(
        paths["baseline_contract"].read_bytes()
    ).hexdigest()
    labels = np.asarray([row["is_phishing"] for row in validation_rows], dtype=np.int8)
    artifact_bytes = (
        json.dumps(
            _logistic_artifact(
                labels, baseline_contract_sha256=baseline_contract_sha256
            ),
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")
    paths["logistic_l1_artifact"].write_bytes(artifact_bytes)
    paths["policy"] = transformer_pipeline._InputHashPolicy(
        train_sha256=sha256(train_bytes).hexdigest(),
        validation_sha256=sha256(validation_bytes).hexdigest(),
        preparation_summary_sha256=sha256(summary_bytes).hexdigest(),
        baseline_contract_sha256=baseline_contract_sha256,
        logistic_l1_artifact_sha256=sha256(artifact_bytes).hexdigest(),
        transformer_contract_sha256=sha256(
            paths["transformer_contract"].read_bytes()
        ).hexdigest(),
    )
    return paths


class _FixtureModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1, dtype=torch.float32)
        with torch.no_grad():
            self.linear.weight.copy_(torch.tensor([[0.25, -0.5]]))
            self.linear.bias.copy_(torch.tensor([0.125]))


def _fixture_fit(validation_labels):
    probabilities = tuple(0.9 if int(label) else 0.1 for label in validation_labels)
    return character_transformer.TransformerFit(
        model=_FixtureModel(),
        history=(character_transformer.EpochRecord(1, 0.5, 1.0),),
        best_epoch=1,
        best_validation_average_precision=1.0,
        epochs_completed=1,
        stopped_early=False,
        positive_class_weight=1.0,
        validation_probabilities=probabilities,
    )


def _install_fixture_trainer(monkeypatch, observations=None, *, warning=False):
    def fake_train(
        train_token_ids,
        train_padding_mask,
        train_labels,
        validation_token_ids,
        validation_padding_mask,
        validation_labels,
        *,
        vocabulary_size,
        device,
    ):
        if warning:
            warnings.warn("fixture warning", RuntimeWarning)
        if observations is not None:
            observations.update(
                {
                    "device": device,
                    "train_rows": int(train_token_ids.shape[0]),
                    "validation_rows": int(validation_token_ids.shape[0]),
                    "validation_has_unk": bool(torch.any(validation_token_ids == 1)),
                    "vocabulary_size": vocabulary_size,
                }
            )
        return _fixture_fit(validation_labels)

    monkeypatch.setattr(transformer_pipeline, "_train_transformer", fake_train)


def _run_fixture(paths):
    return transformer_pipeline._fit_transformer_cascade(
        train_path=paths["train"],
        validation_path=paths["validation"],
        preparation_summary_path=paths["preparation_summary"],
        baseline_contract_path=paths["baseline_contract"],
        logistic_l1_artifact_path=paths["logistic_l1_artifact"],
        transformer_contract_path=paths["transformer_contract"],
        output_dir=paths["output_dir"],
        summary_path=paths["summary"],
        _input_hash_policy=paths["policy"],
        _training_device=torch.device("cpu"),
    )


def _rewrite_preparation_summary(paths, mutation):
    summary = json.loads(paths["preparation_summary"].read_text())
    mutation(summary)
    content = (json.dumps(summary, indent=2, sort_keys=True) + "\n").encode("ascii")
    paths["preparation_summary"].write_bytes(content)
    paths["policy"] = transformer_pipeline._InputHashPolicy(
        **{
            **vars(paths["policy"]),
            "preparation_summary_sha256": sha256(content).hexdigest(),
        }
    )


def _rewrite_partition(paths, partition, mutation):
    records = [json.loads(line) for line in paths[partition].read_text().splitlines()]
    mutation(records)
    content = b"".join(_canonical_json_bytes(record) for record in records)
    paths[partition].write_bytes(content)

    def update_summary(summary):
        summary["output_hashes"][f"{partition}.jsonl"] = sha256(content).hexdigest()

    _rewrite_preparation_summary(paths, update_summary)
    paths["policy"] = transformer_pipeline._InputHashPolicy(
        **{
            **vars(paths["policy"]),
            f"{partition}_sha256": sha256(content).hexdigest(),
        }
    )


def test_fixture_pipeline_is_train_only_validation_only_and_separates_outputs(
    tmp_path, monkeypatch
):
    paths = _write_fixture(tmp_path / "run")
    observed = {}
    _install_fixture_trainer(monkeypatch, observed)
    original_threshold = transformer_pipeline.baselines.select_validation_threshold
    original_cascade = transformer_pipeline.fixed_cascade.calibrate_fixed_cascade

    def threshold_spy(scores, labels):
        observed["threshold_rows"] = len(scores)
        observed["threshold_labels"] = tuple(int(value) for value in labels)
        return original_threshold(scores, labels)

    def cascade_spy(model, stage1, transformer, labels, **kwargs):
        observed["cascade_rows"] = len(transformer)
        observed["cascade_labels"] = tuple(int(value) for value in labels)
        observed["cascade_transformer"] = tuple(transformer)
        return original_cascade(model, stage1, transformer, labels, **kwargs)

    monkeypatch.setattr(
        transformer_pipeline.baselines, "select_validation_threshold", threshold_spy
    )
    monkeypatch.setattr(
        transformer_pipeline.fixed_cascade,
        "calibrate_fixed_cascade",
        cascade_spy,
    )

    summary = _run_fixture(paths)

    assert observed["device"] == torch.device("cpu")
    assert observed["train_rows"] == 24
    assert observed["validation_rows"] == 420
    assert observed["validation_has_unk"] is True
    assert observed["vocabulary_size"] == summary["vocabulary"]["size"]
    assert observed["threshold_rows"] == 420
    assert observed["cascade_rows"] == 420
    expected_labels = (0,) * 400 + (1,) * 20
    assert observed["threshold_labels"] == expected_labels
    assert observed["cascade_labels"] == expected_labels
    assert observed["cascade_transformer"] == (0.1,) * 400 + (0.9,) * 20
    assert summary["status"] == "completed_development_validation"
    assert summary["transformer"]["threshold"]["status"] == "selected"
    assert summary["cascade"]["status"] == "selected"
    assert summary["cascade"]["accepted_cascade"] is True
    assert summary["access"] == {
        "group_test_accessed": False,
        "phishvn_accessed": False,
    }
    assert json.loads(paths["summary"].read_text()) == summary

    private_names = {path.name for path in paths["output_dir"].iterdir()}
    assert private_names == {
        "SHA256SUMS",
        "cascade.json",
        "transformer-weights.npz",
        "transformer.json",
        "vocabulary.json",
    }
    assert stat.S_IMODE(paths["output_dir"].stat().st_mode) == 0o700
    assert all(
        stat.S_IMODE(path.stat().st_mode) == 0o600
        for path in paths["output_dir"].iterdir()
    )
    vocabulary_record = json.loads(
        (paths["output_dir"] / "vocabulary.json").read_text()
    )
    train_urls = [
        json.loads(line)["raw_url"] for line in paths["train"].read_text().splitlines()
    ]
    validation_urls = [
        json.loads(line)["raw_url"]
        for line in paths["validation"].read_text().splitlines()
    ]
    train_characters = set().union(
        *map(set, map(character_sequence.normalize_character_url, train_urls))
    )
    validation_characters = set().union(
        *map(set, map(character_sequence.normalize_character_url, validation_urls))
    )
    validation_only = validation_characters - train_characters
    assert validation_only
    assert set(vocabulary_record["characters"]) == train_characters
    assert set(vocabulary_record["characters"]).isdisjoint(validation_only)

    manifest_lines = (paths["output_dir"] / "SHA256SUMS").read_text().splitlines()
    expected_manifest_names = [
        "cascade.json",
        "transformer-weights.npz",
        "transformer.json",
        "vocabulary.json",
    ]
    assert [line[66:] for line in manifest_lines] == expected_manifest_names
    assert all(line[64:66] == "  " for line in manifest_lines)
    assert all(
        line[:64] == sha256((paths["output_dir"] / filename).read_bytes()).hexdigest()
        for line, filename in zip(manifest_lines, expected_manifest_names)
    )
    assert "SHA256SUMS" not in "\n".join(manifest_lines)

    transformer_record = json.loads(
        (paths["output_dir"] / "transformer.json").read_text()
    )
    cascade_record = json.loads((paths["output_dir"] / "cascade.json").read_text())
    assert set(vocabulary_record) == {
        "character_ids_start",
        "characters",
        "max_sequence_length",
        "normalization",
        "pad_id",
        "schema_version",
        "unk_id",
    }
    assert set(transformer_record) == {
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
    assert set(cascade_record) == {
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
    assert set(summary) == {
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

    serialized_public = paths["summary"].read_text()
    serialized_private_metadata = (
        paths["output_dir"] / "transformer.json"
    ).read_text() + (paths["output_dir"] / "cascade.json").read_text()
    for sensitive in (
        "https://safe-1000",
        f"phiusiil-row-v1:{SOURCE_SHA256}",
        "va-1000.example",
        "coefficients",
    ):
        assert sensitive not in serialized_public
        assert sensitive not in serialized_private_metadata
    assert "validation_probabilities" not in json.loads(serialized_public)
    assert "validation_probabilities" not in json.loads(
        (paths["output_dir"] / "transformer.json").read_text()
    )


def test_private_weights_are_deterministic_safe_npy_1_0_members(tmp_path, monkeypatch):
    paths = _write_fixture(tmp_path / "run")
    _install_fixture_trainer(monkeypatch)
    _run_fixture(paths)

    weights_path = paths["output_dir"] / "transformer-weights.npz"
    with zipfile.ZipFile(weights_path) as archive:
        assert archive.comment == b""
        assert archive.namelist() == ["linear.bias.npy", "linear.weight.npy"]
        for info in archive.infolist():
            assert info.compress_type == zipfile.ZIP_STORED
            assert info.date_time == (1980, 1, 1, 0, 0, 0)
            assert info.create_system == 3
            assert info.external_attr == 0o600 << 16
            assert info.extra == b""
            assert info.comment == b""
            member = archive.read(info)
            assert member.startswith(b"\x93NUMPY\x01\x00")
            array = np.load(__import__("io").BytesIO(member), allow_pickle=False)
            assert array.dtype == np.dtype("<f4")
            assert array.flags.c_contiguous

    first_bytes = {
        path.name: path.read_bytes() for path in paths["output_dir"].iterdir()
    }
    second = _write_fixture(tmp_path / "repeat")
    _run_fixture(second)
    assert first_bytes == {
        path.name: path.read_bytes() for path in second["output_dir"].iterdir()
    }
    assert paths["summary"].read_bytes() == second["summary"].read_bytes()


def test_transformer_target_not_met_is_published_without_a_cascade(
    tmp_path, monkeypatch
):
    paths = _write_fixture(tmp_path / "run", validation_negatives=100)
    _install_fixture_trainer(monkeypatch)

    summary = _run_fixture(paths)

    assert summary["status"] == "target_not_met"
    assert summary["transformer"]["threshold"]["status"] == "target_not_met"
    assert summary["cascade"]["status"] == "target_not_met"
    assert summary["cascade"]["accepted_cascade"] is False
    cascade = json.loads((paths["output_dir"] / "cascade.json").read_text())
    assert cascade["calibration"]["status"] == "target_not_met"
    assert cascade["calibration"]["accepted_cascade"] is False


@pytest.mark.parametrize(
    "mutation",
    (
        lambda contract: contract.update(protocol_version="1.9"),
        lambda contract: contract["architecture"]["encoder"].update(layers=5),
        lambda contract: contract["cascade"]["escalation_rule"].update(inclusive=False),
        lambda contract: contract["inputs"]["forbidden_roles"].pop(),
    ),
)
def test_contract_is_validated_semantically_even_under_fixture_hash_policy(
    tmp_path, monkeypatch, mutation
):
    paths = _write_fixture(tmp_path / "run")
    _install_fixture_trainer(monkeypatch)
    contract = json.loads(paths["transformer_contract"].read_text())
    mutation(contract)
    paths["transformer_contract"].write_bytes(_canonical_json_bytes(contract))
    paths["policy"] = transformer_pipeline._InputHashPolicy(
        **{
            **vars(paths["policy"]),
            "transformer_contract_sha256": sha256(
                paths["transformer_contract"].read_bytes()
            ).hexdigest(),
        }
    )

    with pytest.raises(transformer_pipeline.TransformerPipelineError, match="frozen"):
        _run_fixture(paths)
    assert not paths["output_dir"].exists()
    assert not paths["summary"].exists()


def test_reordered_whitespace_contract_is_semantically_equivalent_in_fixture(
    tmp_path, monkeypatch
):
    paths = _write_fixture(tmp_path / "run")
    _install_fixture_trainer(monkeypatch)
    contract = json.loads(paths["transformer_contract"].read_text())
    reordered = {key: contract[key] for key in reversed(contract)}
    content = (json.dumps(reordered, indent=4, ensure_ascii=False) + "\n").encode()
    paths["transformer_contract"].write_bytes(content)
    paths["policy"] = transformer_pipeline._InputHashPolicy(
        **{
            **vars(paths["policy"]),
            "transformer_contract_sha256": sha256(content).hexdigest(),
        }
    )

    assert _run_fixture(paths)["status"] == "completed_development_validation"


def test_official_hash_policy_matches_the_frozen_contract_and_file():
    contract_bytes = TRANSFORMER_CONTRACT.read_bytes()
    contract = json.loads(contract_bytes)
    policy = transformer_pipeline._OFFICIAL_INPUT_HASH_POLICY

    assert sha256(contract_bytes).hexdigest() == (
        transformer_pipeline.OFFICIAL_TRANSFORMER_CONTRACT_SHA256
    )
    assert policy.transformer_contract_sha256 == sha256(contract_bytes).hexdigest()
    assert contract["inputs"]["accepted_roles"] == {
        "train": policy.train_sha256,
        "validation": policy.validation_sha256,
        "preparation_summary": policy.preparation_summary_sha256,
        "logistic_l1_artifact": policy.logistic_l1_artifact_sha256,
        "contract": policy.baseline_contract_sha256,
    }


def test_hashes_are_checked_before_invalid_content_is_parsed(tmp_path, monkeypatch):
    paths = _write_fixture(tmp_path / "run")
    _install_fixture_trainer(monkeypatch)
    paths["train"].write_text("not JSON\n", encoding="ascii")

    with pytest.raises(transformer_pipeline.TransformerPipelineError, match="SHA-256"):
        _run_fixture(paths)
    assert not paths["output_dir"].exists()
    assert not paths["summary"].exists()


def test_last_input_hash_is_checked_before_its_invalid_content_is_parsed(
    tmp_path, monkeypatch
):
    paths = _write_fixture(tmp_path / "run")
    _install_fixture_trainer(monkeypatch)
    paths["transformer_contract"].write_text("not JSON\n", encoding="ascii")

    with pytest.raises(
        transformer_pipeline.TransformerPipelineError,
        match="transformer_contract SHA-256 mismatch",
    ):
        _run_fixture(paths)
    assert not paths["output_dir"].exists()
    assert not paths["summary"].exists()


def test_verified_input_snapshot_is_used_after_same_size_path_mutation(
    tmp_path, monkeypatch
):
    paths = _write_fixture(tmp_path / "run")
    _install_fixture_trainer(monkeypatch)
    artifact_path = paths["logistic_l1_artifact"]
    original_content = artifact_path.read_bytes()
    changed_content = original_content.replace(
        b'"n_iter": [\n      12', b'"n_iter": [\n      13'
    )
    assert changed_content != original_content
    assert len(changed_content) == len(original_content)
    original_metadata = artifact_path.stat()
    original_stability_check = baselines._require_stable_streams
    mutated = False

    def mutate_after_verification(streams, expected):
        nonlocal mutated
        original_stability_check(streams, expected)
        if not mutated:
            artifact_path.write_bytes(changed_content)
            os.utime(
                artifact_path,
                ns=(original_metadata.st_atime_ns, original_metadata.st_mtime_ns),
            )
            mutated = True

    monkeypatch.setattr(baselines, "_require_stable_streams", mutate_after_verification)

    summary = _run_fixture(paths)

    assert mutated is True
    assert artifact_path.read_bytes() == changed_content
    assert (
        summary["input_hashes"]["logistic_l1_artifact"]
        == sha256(original_content).hexdigest()
    )
    assert summary["status"] == "completed_development_validation"


def test_partition_validation_error_is_translated_and_leaves_no_outputs(
    tmp_path, monkeypatch
):
    paths = _write_fixture(tmp_path / "run")
    _install_fixture_trainer(monkeypatch)
    records = [json.loads(line) for line in paths["train"].read_text().splitlines()]
    records[0]["split"] = "validation"
    train_bytes = b"".join(_canonical_json_bytes(record) for record in records)
    paths["train"].write_bytes(train_bytes)
    summary = json.loads(paths["preparation_summary"].read_text())
    summary["output_hashes"]["train.jsonl"] = sha256(train_bytes).hexdigest()
    summary_bytes = (json.dumps(summary, indent=2, sort_keys=True) + "\n").encode(
        "ascii"
    )
    paths["preparation_summary"].write_bytes(summary_bytes)
    paths["policy"] = transformer_pipeline._InputHashPolicy(
        **{
            **vars(paths["policy"]),
            "train_sha256": sha256(train_bytes).hexdigest(),
            "preparation_summary_sha256": sha256(summary_bytes).hexdigest(),
        }
    )

    with pytest.raises(
        transformer_pipeline.TransformerPipelineError, match="incorrect split tag"
    ):
        _run_fixture(paths)
    assert not paths["output_dir"].exists()
    assert not paths["summary"].exists()


@pytest.mark.parametrize("mismatch", ("declared_rows", "domain_count"))
def test_declared_partition_counts_are_enforced(tmp_path, monkeypatch, mismatch):
    paths = _write_fixture(tmp_path / "run")
    _install_fixture_trainer(monkeypatch)

    def mutate(summary):
        train = summary["splits"]["train"]
        if mismatch == "declared_rows":
            train["row_count"] += 1
            train["class_counts"]["0"] += 1
        else:
            train["domain_count"] -= 1

    _rewrite_preparation_summary(paths, mutate)
    with pytest.raises(transformer_pipeline.TransformerPipelineError, match="declared"):
        _run_fixture(paths)
    assert not paths["output_dir"].exists()
    assert not paths["summary"].exists()


def test_cross_partition_domain_reuse_is_rejected(tmp_path, monkeypatch):
    paths = _write_fixture(tmp_path / "run")
    _install_fixture_trainer(monkeypatch)
    train_domain = json.loads(paths["train"].read_text().splitlines()[0])[
        "registrable_domain"
    ]
    _rewrite_partition(
        paths,
        "validation",
        lambda records: records[0].update(registrable_domain=train_domain),
    )

    with pytest.raises(
        transformer_pipeline.TransformerPipelineError,
        match="registrable domain crosses",
    ):
        _run_fixture(paths)


def test_adjacent_group_test_file_is_never_opened(tmp_path, monkeypatch):
    paths = _write_fixture(tmp_path / "run")
    _install_fixture_trainer(monkeypatch)
    sentinel = paths["train"].parent / "group_test.jsonl"
    sentinel.write_text("must remain unopened\n", encoding="ascii")
    original_open = os.open
    opened = []

    def guarded_open(path, *args, **kwargs):
        candidate = Path(path)
        if candidate == sentinel:
            raise AssertionError("group-test sentinel was opened")
        opened.append(candidate)
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(os, "open", guarded_open)

    assert _run_fixture(paths)["status"] == "completed_development_validation"
    assert sentinel not in opened


def test_inputs_and_outputs_cannot_alias_or_overwrite(tmp_path, monkeypatch):
    paths = _write_fixture(tmp_path / "hard-link")
    _install_fixture_trainer(monkeypatch)
    paths["validation"].unlink()
    os.link(paths["train"], paths["validation"])
    paths["policy"] = transformer_pipeline._InputHashPolicy(
        **{
            **vars(paths["policy"]),
            "validation_sha256": paths["policy"].train_sha256,
        }
    )
    with pytest.raises(transformer_pipeline.TransformerPipelineError, match="alias"):
        _run_fixture(paths)

    paths = _write_fixture(tmp_path / "existing-output")
    paths["summary"].write_text("competitor", encoding="ascii")
    with pytest.raises(transformer_pipeline.TransformerPipelineError, match="exists"):
        _run_fixture(paths)
    assert paths["summary"].read_text() == "competitor"
    assert not paths["output_dir"].exists()

    paths = _write_fixture(tmp_path / "same-output")
    paths["summary"] = paths["output_dir"]
    with pytest.raises(transformer_pipeline.TransformerPipelineError, match="alias"):
        _run_fixture(paths)
    assert not paths["output_dir"].exists()

    paths = _write_fixture(tmp_path / "existing-private")
    paths["output_dir"].mkdir()
    sentinel = paths["output_dir"] / "sentinel"
    sentinel.write_text("competitor", encoding="ascii")
    with pytest.raises(transformer_pipeline.TransformerPipelineError, match="exists"):
        _run_fixture(paths)
    assert sentinel.read_text() == "competitor"
    assert not paths["summary"].exists()


def test_symlink_input_is_rejected(tmp_path, monkeypatch):
    paths = _write_fixture(tmp_path / "run")
    _install_fixture_trainer(monkeypatch)
    real_contract = paths["transformer_contract"].with_suffix(".real.json")
    paths["transformer_contract"].rename(real_contract)
    paths["transformer_contract"].symlink_to(real_contract)

    with pytest.raises(
        transformer_pipeline.TransformerPipelineError, match="regular file|alias"
    ):
        _run_fixture(paths)


def test_missing_input_is_reported_as_a_pipeline_error(tmp_path, monkeypatch):
    paths = _write_fixture(tmp_path / "run")
    _install_fixture_trainer(monkeypatch)
    paths["train"].unlink()

    with pytest.raises(
        transformer_pipeline.TransformerPipelineError, match="does not exist"
    ):
        _run_fixture(paths)


def test_warning_or_publication_failure_leaves_no_outputs(tmp_path, monkeypatch):
    warning_paths = _write_fixture(tmp_path / "warning")
    _install_fixture_trainer(monkeypatch, warning=True)
    with pytest.raises(transformer_pipeline.TransformerPipelineError, match="warning"):
        _run_fixture(warning_paths)
    assert not warning_paths["output_dir"].exists()
    assert not warning_paths["summary"].exists()

    collision_paths = _write_fixture(tmp_path / "collision")
    _install_fixture_trainer(monkeypatch)
    original_publish = transformer_pipeline._publish_path_without_replace
    calls = 0

    def fail_second_publish(source, destination):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected summary publication failure")
        return original_publish(source, destination)

    monkeypatch.setattr(
        transformer_pipeline, "_publish_path_without_replace", fail_second_publish
    )
    with pytest.raises(OSError, match="injected"):
        _run_fixture(collision_paths)
    assert not collision_paths["output_dir"].exists()
    assert not collision_paths["summary"].exists()
    assert not list(collision_paths["summary"].parent.glob(".*.tmp-*"))


def test_training_runtime_error_is_translated_without_publication(
    tmp_path, monkeypatch
):
    paths = _write_fixture(tmp_path / "run")

    def deterministic_kernel_failure(*_args, **_kwargs):
        raise RuntimeError("deterministic MPS kernel unavailable")

    monkeypatch.setattr(
        transformer_pipeline, "_train_transformer", deterministic_kernel_failure
    )

    with pytest.raises(
        transformer_pipeline.TransformerPipelineError,
        match="deterministic MPS kernel unavailable",
    ):
        _run_fixture(paths)
    assert not paths["output_dir"].exists()
    assert not paths["summary"].exists()


def test_publish_time_competitor_survives_rollback(tmp_path, monkeypatch):
    paths = _write_fixture(tmp_path / "run")
    _install_fixture_trainer(monkeypatch)
    original_publish = transformer_pipeline._publish_path_without_replace
    calls = 0

    def install_competitor_before_summary(source, destination):
        nonlocal calls
        calls += 1
        if calls == 2:
            destination.write_text("competitor", encoding="ascii")
        return original_publish(source, destination)

    monkeypatch.setattr(
        transformer_pipeline,
        "_publish_path_without_replace",
        install_competitor_before_summary,
    )

    with pytest.raises(
        transformer_pipeline.TransformerPipelineError, match="already exists"
    ):
        _run_fixture(paths)
    assert paths["summary"].read_text() == "competitor"
    assert not paths["output_dir"].exists()


def test_public_entry_point_has_only_paths_and_forces_official_policy(monkeypatch):
    expected_parameters = {
        "train_path",
        "validation_path",
        "preparation_summary_path",
        "baseline_contract_path",
        "logistic_l1_artifact_path",
        "transformer_contract_path",
        "output_dir",
        "summary_path",
    }
    assert set(
        inspect.signature(transformer_pipeline.fit_transformer_cascade).parameters
    ) == (expected_parameters)
    observed = {}

    def fake_private(**kwargs):
        observed.update(kwargs)
        return {"status": "fixture"}

    monkeypatch.setattr(transformer_pipeline, "_fit_transformer_cascade", fake_private)
    paths = {name: Path(name) for name in expected_parameters}

    result = transformer_pipeline.fit_transformer_cascade(**paths)

    assert result == {"status": "fixture"}
    assert (
        observed["_input_hash_policy"]
        == transformer_pipeline._OFFICIAL_INPUT_HASH_POLICY
    )
    assert observed["_training_device"] is None
    assert all(observed[name] == paths[name] for name in expected_parameters)


def test_none_device_routes_to_public_trainer_and_official_cpu_override_is_rejected(
    monkeypatch,
):
    observed = {}

    def fake_public(*args, **kwargs):
        observed["args"] = args
        observed["kwargs"] = kwargs
        return "public-fit"

    monkeypatch.setattr(character_transformer, "fit_character_transformer", fake_public)
    tensor = torch.zeros((1, 256), dtype=torch.int64)
    mask = torch.zeros((1, 256), dtype=torch.bool)
    labels = torch.tensor([0.0])
    result = transformer_pipeline._train_transformer(
        tensor,
        mask,
        labels,
        tensor,
        mask,
        labels,
        vocabulary_size=3,
        device=None,
    )

    assert result == "public-fit"
    assert observed["args"] == (tensor, mask, labels, tensor, mask, labels)
    assert observed["kwargs"] == {"vocabulary_size": 3}
    with pytest.raises(
        transformer_pipeline.TransformerPipelineError, match="cannot override"
    ):
        transformer_pipeline._fit_transformer_cascade(
            train_path=Path("train"),
            validation_path=Path("validation"),
            preparation_summary_path=Path("summary"),
            baseline_contract_path=Path("baseline"),
            logistic_l1_artifact_path=Path("artifact"),
            transformer_contract_path=Path("transformer"),
            output_dir=Path("output"),
            summary_path=Path("public"),
            _input_hash_policy=transformer_pipeline._OFFICIAL_INPUT_HASH_POLICY,
            _training_device=torch.device("cpu"),
        )
