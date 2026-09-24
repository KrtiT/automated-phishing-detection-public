"""Invented partitions and numerical fixtures; no research inputs or fits."""

import importlib
from contextlib import nullcontext
from dataclasses import replace
from hashlib import sha256
from types import SimpleNamespace

import pytest
import torch
from test_fixed_cascade import _threshold_record
from test_secondary_development import _fixture

from automated_phishing_detection import (
    character_sequence,
    phiusiil,
)
from automated_phishing_detection import (
    character_transformer as trainer,
)
from automated_phishing_detection import (
    secondary_development as development,
)
from automated_phishing_detection import (
    secondary_transformer as secondary,
)


def _module():
    return importlib.import_module("automated_phishing_detection.seed_probe_seeds")


def test_safe_failure_identifiers_never_copy_unrecognized_details():
    module = _module()
    error = module.SeedStageError("invalid_seed_stage")
    assert error.check_id == "invalid_seed_stage"
    private = module.SeedStageError("secret.example/private-url")
    assert private.check_id == str(private) == "unclassified_check"
    assert "secret" not in str(private)


@pytest.fixture
def sample(monkeypatch, request):
    module = _module()
    validation_count = getattr(request, "param", 600)
    fixture = _fixture(development, train_count=8, validation_count=validation_count)
    arguments = fixture["arguments"]
    # Build accepted fixture bytes, not a production model file.
    from test_fixed_cascade import _artifact

    state = _artifact(arguments["pins"].baseline_contract_sha256)
    state["scaler"]["n_samples_seen"] = 8
    state["validation_threshold"] = (
        _threshold_record(
            positive=validation_count // 2, negative=validation_count // 2
        )
        if validation_count >= 600
        else _threshold_record("target_not_met")
    )
    state["input_hashes"].update(
        train=arguments["pins"].train_sha256,
        validation=arguments["pins"].validation_sha256,
        preparation_summary=arguments["pins"].preparation_summary_sha256,
    )
    artifact = development._json_bytes(state)
    pins = replace(
        arguments["pins"], logistic_l1_artifact_sha256=sha256(artifact).hexdigest()
    )
    vocabulary = character_sequence.build_character_vocabulary(
        [row["raw_url"] for row in fixture["train"]]
    )
    vocabulary_bytes = vocabulary.to_json().encode()
    weights = secondary.serialize_secondary_weights(
        trainer.CharacterTransformer(vocabulary.size)
    )
    artifacts = {
        "logistic-l1.json": artifact,
        "vocabulary.json": vocabulary_bytes,
        "transformer-weights.npz": weights,
    }
    binding = SimpleNamespace(
        pins=pins,
        preparation_bytes=arguments["preparation_summary"],
        primary_artifact_hashes=tuple(
            (key, sha256(value).hexdigest()) for key, value in artifacts.items()
        ),
        methods_sha256="a" * 64,
        profile_sha256="b" * 64,
    )
    monkeypatch.setattr(module, "_numerical_context", nullcontext)
    monkeypatch.setattr(
        trainer, "_resolve_training_device", lambda: torch.device("cpu")
    )
    labels = tuple(row["is_phishing"] for row in fixture["validation"])
    probabilities = tuple(0.9 if label else 0.1 for label in labels)
    monkeypatch.setattr(module, "_score_weights", lambda *args: probabilities)
    monkeypatch.setattr(
        secondary,
        "score_secondary_stage1_urls",
        lambda *args: secondary.SecondaryStage1Scores(
            probabilities, tuple({"singleton_fixture": True} for _ in labels)
        ),
    )
    return SimpleNamespace(
        module=module,
        binding=binding,
        artifacts=artifacts,
        fixture=fixture,
        probabilities=probabilities,
        labels=labels,
        arguments=dict(
            train_bytes=None,
            validation_bytes=fixture["validation_content"],
            suffix_rules_bytes=arguments["suffix_rules"],
            artifacts=artifacts,
            common_stage1=None,
        ),
    )


def _run(sample, stage="seed_42_calibration", **changes):
    retained = {}

    def retain(name, content):
        assert name not in retained
        retained[name] = content

    outputs, summary = sample.module.run_seed_stage(
        sample.binding, stage, retain=retain, **(sample.arguments | changes)
    )
    return outputs, summary, retained


def test_primary_recalibration_does_not_fit_or_read_training(sample, monkeypatch):
    monkeypatch.setattr(
        secondary,
        "fit_secondary_transformer",
        lambda *a, **k: pytest.fail("fit reached"),
    )
    outputs, summary, retained = _run(sample)
    assert summary["seed"] == 42 and summary["new_fit"] is False
    assert development._json(outputs["fit-input.json"])["train"] is None
    assert "restored-best.npz" not in retained
    assert "common-stage1.jsonl" in retained and "common-stage1.jsonl" not in outputs
    assert (
        sample.module.verify_seed_stage(
            sample.binding, "seed_42_calibration", outputs=outputs, auxiliary=retained
        )
        == summary
    )


@pytest.mark.parametrize("stage", ("seed_42", "seed_47", "train", 43, True))
def test_invalid_stage_rejected_before_inputs(sample, stage):
    with pytest.raises(sample.module.SeedStageError, match="invalid_seed_stage"):
        _run(sample, stage)


def test_stage42_rejects_training_input(sample):
    with pytest.raises(sample.module.SeedStageError, match="unexpected_training_input"):
        _run(sample, train_bytes=b"must not be used")


def test_changed_artifact_rejected_before_scoring(sample, monkeypatch):
    monkeypatch.setattr(
        sample.module, "_score_weights", lambda *a: pytest.fail("scoring reached")
    )
    artifacts = sample.artifacts | {"vocabulary.json": b"changed"}
    with pytest.raises(sample.module.SeedStageError, match="artifact_hash_mismatch"):
        _run(sample, artifacts=artifacts)


def test_prepare_accepts_phiusiil_ascii_escaped_unicode_source(sample):
    rows = [dict(row) for row in sample.fixture["validation"]]
    raw_url = f"{rows[0]['raw_url']}&place=Honolulu-例"
    rows[0].update(
        raw_url=raw_url,
        canonical_url_sha256=sha256(
            phiusiil.canonicalize_url(raw_url).encode("utf-8")
        ).hexdigest(),
    )
    validation_bytes = phiusiil._jsonl_bytes(rows)
    preparation = development._json(sample.binding.preparation_bytes)
    validation_sha256 = sha256(validation_bytes).hexdigest()
    preparation["output_hashes"]["validation.jsonl"] = validation_sha256
    preparation_bytes = development._json_bytes(preparation)
    binding = SimpleNamespace(
        **(
            vars(sample.binding)
            | {
                "pins": replace(
                    sample.binding.pins,
                    validation_sha256=validation_sha256,
                    preparation_summary_sha256=sha256(preparation_bytes).hexdigest(),
                ),
                "preparation_bytes": preparation_bytes,
            }
        )
    )
    vocabulary = character_sequence.build_character_vocabulary(
        [row["raw_url"] for row in rows]
    )

    _, parsed_rows, _, _, _ = sample.module._prepare(
        binding,
        42,
        None,
        validation_bytes,
        sample.arguments["suffix_rules_bytes"],
        vocabulary,
    )

    assert parsed_rows[0]["raw_url"] == raw_url


def test_saved_evidence_rejects_phiusiil_ascii_escaped_unicode():
    module = _module()
    content = phiusiil._jsonl_bytes([{"raw_url": "https://example.test/例"}])

    with pytest.raises(module.SeedStageError, match="invalid_evidence_json"):
        module._lines(content)


def test_saved_predictions_are_checked_without_source_scoring(sample, monkeypatch):
    outputs, summary, retained = _run(sample)
    monkeypatch.setattr(
        sample.module, "_score_weights", lambda *a: pytest.fail("scoring reached")
    )
    rows = [
        development._json(line)
        for line in outputs["validation-predictions.jsonl"].splitlines()
    ]
    rows[0]["cascade_decision"] = 1
    outputs["validation-predictions.jsonl"] = b"".join(
        development._json_bytes(row) for row in rows
    )
    with pytest.raises(sample.module.SeedStageError, match="prediction"):
        sample.module.verify_seed_stage(
            sample.binding, "seed_42_calibration", outputs=outputs, auxiliary=retained
        )


def test_raw_scores_survive_calibration_failure(sample, monkeypatch):
    retained = {}

    def fail(*args):
        raise RuntimeError("fixture calibration failed")

    monkeypatch.setattr(secondary, "calibrate_secondary_seed", fail)
    with pytest.raises(RuntimeError, match="fixture calibration"):
        sample.module.run_seed_stage(
            sample.binding,
            "seed_42_calibration",
            retain=retained.__setitem__,
            **sample.arguments,
        )
    assert "raw-scores.jsonl" in retained
    assert "fit-input.json" in retained


def _fit_fixture(sample, monkeypatch):
    monkeypatch.setattr(trainer, "_run_training_epoch", lambda *args: 0.25)
    monkeypatch.setattr(
        trainer, "_evaluate_validation", lambda *args: (1.0, sample.probabilities)
    )
    _, _, retained = _run(sample)
    common = retained["common-stage1.jsonl"]
    monkeypatch.setattr(
        secondary,
        "score_secondary_stage1_urls",
        lambda *args: pytest.fail("stage1 rescored for new fit"),
    )
    return {
        "train_bytes": sample.fixture["arguments"]["train_content"],
        "common_stage1": common,
    }


@pytest.mark.parametrize("seed", (43, 44, 45, 46))
def test_new_fit_preserves_epochs_checkpoints_and_common_stage1(
    sample, monkeypatch, seed
):
    arguments = _fit_fixture(sample, monkeypatch)
    outputs, summary, retained = _run(sample, f"seed_{seed}", **arguments)
    assert summary["new_fit"] is True and summary["seed"] == seed
    assert summary["best_epoch"] == 1 and summary["epochs_completed"] == 6
    assert retained["restored-best.npz"] == retained["best-001.npz"]
    assert [key for key in retained if key.startswith("epoch-")] == [
        f"epoch-{i:03d}.json" for i in range(1, 7)
    ]
    assert "restored-best.npz" not in outputs
    assert (
        sample.module.verify_seed_stage(
            sample.binding,
            f"seed_{seed}",
            outputs=outputs,
            auxiliary=retained,
            common_stage1=arguments["common_stage1"],
        )
        == summary
    )


@pytest.mark.parametrize("field", ("label", "record_id", "probability"))
def test_new_seed_requires_exact_common_identity_and_digest(sample, monkeypatch, field):
    arguments = _fit_fixture(sample, monkeypatch)
    outputs, _, retained = _run(sample, "seed_43", **arguments)
    rows = [development._json(line) for line in arguments["common_stage1"].splitlines()]
    rows[0][field] = {"label": 1, "record_id": "changed", "probability": 0.8}[field]
    altered = b"".join(development._json_bytes(row) for row in rows)
    with pytest.raises(sample.module.SeedStageError, match="fit_input_binding"):
        sample.module.verify_seed_stage(
            sample.binding,
            "seed_43",
            outputs=outputs,
            auxiliary=retained,
            common_stage1=altered,
        )


@pytest.mark.parametrize(
    "mutation",
    (
        "epoch_ap",
        "epoch_order",
        "missing_best",
        "extra_best",
        "restored_bytes",
        "restored_ap",
        "loss",
        "result_weight",
        "labels_digest",
        "tensor_labels",
    ),
)
def test_saved_fit_rejects_inconsistent_evidence(sample, monkeypatch, mutation):
    arguments = _fit_fixture(sample, monkeypatch)
    outputs, _, retained = _run(sample, "seed_43", **arguments)
    if mutation in {"epoch_ap", "epoch_order", "loss"}:
        value = development._json(retained["epoch-001.json"])
        if mutation == "epoch_ap":
            value["validation_average_precision"] = 0.5
        elif mutation == "epoch_order":
            value["validation_probabilities"].reverse()
        else:
            value["training_loss"] = -0.25
        retained["epoch-001.json"] = development._json_bytes(value)
    elif mutation == "missing_best":
        del retained["best-001.npz"]
    elif mutation == "extra_best":
        retained["best-002.npz"] = retained["best-001.npz"]
    elif mutation == "restored_bytes":
        retained["restored-best.npz"] += b"changed"
    elif mutation in {"restored_ap", "result_weight"}:
        value = development._json(outputs["training-result.json"])
        if mutation == "restored_ap":
            value["restored_validation_probabilities"].reverse()
        else:
            value["weights_sha256"] = "0" * 64
        outputs["training-result.json"] = development._json_bytes(value)
    else:
        value = development._json(outputs["fit-input.json"])
        if mutation == "labels_digest":
            value["validation"]["labels_sha256"] = "0" * 64
        else:
            value["validation"]["tensors"]["labels"]["sha256"] = "0" * 64
        outputs["fit-input.json"] = retained["fit-input.json"] = (
            development._json_bytes(value)
        )
    with pytest.raises(sample.module.SeedStageError):
        sample.module.verify_seed_stage(
            sample.binding,
            "seed_43",
            outputs=outputs,
            auxiliary=retained,
            common_stage1=arguments["common_stage1"],
        )


def test_later_training_failure_keeps_completed_checkpoint_and_epoch(
    sample, monkeypatch
):
    arguments = _fit_fixture(sample, monkeypatch)
    calls = 0

    def train(*args):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("fixture later epoch stopped")
        return 0.25

    monkeypatch.setattr(trainer, "_run_training_epoch", train)
    retained = {}
    with pytest.raises(RuntimeError, match="later epoch"):
        sample.module.run_seed_stage(
            sample.binding,
            "seed_43",
            retain=retained.__setitem__,
            **(sample.arguments | arguments),
        )
    assert {
        "fit-input.json",
        "best-001.npz",
        "best-001.json",
        "epoch-001.json",
    } <= retained.keys()
    assert "restored-best.npz" not in retained


def test_callback_failure_stops_before_next_epoch(sample, monkeypatch):
    arguments = _fit_fixture(sample, monkeypatch)
    completed = []

    def retain(name, content):
        completed.append(name)
        if name == "epoch-001.json":
            raise OSError("fixture durable sink failed")

    with pytest.raises(OSError, match="durable sink"):
        sample.module.run_seed_stage(
            sample.binding, "seed_43", retain=retain, **(sample.arguments | arguments)
        )
    assert "best-001.npz" in completed
    assert "epoch-002.json" not in completed


@pytest.mark.parametrize("sample", (8,), indirect=True)
def test_unavailable_thresholds_leave_decisions_and_masks_null(sample):
    outputs, summary, retained = _run(sample)
    assert summary["calibration"]["transformer_threshold"]["status"] == "target_not_met"
    assert summary["calibration"]["cascade_band"]["status"] == "target_not_met"
    rows = [
        development._json(line)
        for line in outputs["validation-predictions.jsonl"].splitlines()
    ]
    for row in rows:
        for field in (
            "stage1_decision",
            "transformer_decision",
            "cascade_decision",
            "cascade_probability",
            "logical_stage2_mask",
        ):
            assert row[field] is None
    assert (
        sample.module.verify_seed_stage(
            sample.binding, "seed_42_calibration", outputs=outputs, auxiliary=retained
        )
        == summary
    )


def test_verifier_neither_loads_source_partitions_nor_calls_inference(
    sample, monkeypatch
):
    outputs, summary, retained = _run(sample)

    def forbidden(*args, **kwargs):
        pytest.fail("verification reached source processing, training or inference")

    monkeypatch.setattr(sample.module, "_prepare", forbidden)
    monkeypatch.setattr(sample.module, "_score_weights", forbidden)
    monkeypatch.setattr(secondary, "score_secondary_stage1_urls", forbidden)
    monkeypatch.setattr(secondary, "fit_secondary_transformer", forbidden)
    assert (
        sample.module.verify_seed_stage(
            sample.binding, "seed_42_calibration", outputs=outputs, auxiliary=retained
        )
        == summary
    )


def test_training_domain_overlap_stops_before_fit(sample, monkeypatch):
    arguments = _fit_fixture(sample, monkeypatch)
    partition = development._partition
    seen = {}

    def overlapping(content, split, *args):
        result = partition(content, split, *args)
        if split == "validation":
            seen["domain"] = result.domains[0]
        else:
            result = replace(result, domains=(seen["domain"],) + result.domains[1:])
        return result

    monkeypatch.setattr(development, "_partition", overlapping)
    monkeypatch.setattr(
        secondary,
        "fit_secondary_transformer",
        lambda *a, **k: pytest.fail("fit reached"),
    )
    with pytest.raises(
        sample.module.SeedStageError, match="training_validation_overlap"
    ):
        _run(sample, "seed_43", **arguments)


@pytest.mark.parametrize(
    "mutation",
    (
        "training_identity",
        "training_labels",
        "training_tensor_labels",
        "training_counts",
        "token_shape",
        "mask_dtype",
        "mask_hash",
        "training_identity_order",
        "training_label_order",
        "extra_field",
        "bool_schema",
        "epoch_extra",
        "epoch_bool",
        "epoch_ap_bool",
        "epoch_score_bool",
    ),
)
def test_consumed_training_and_tensor_metadata_are_verified(
    sample, monkeypatch, mutation
):
    arguments = _fit_fixture(sample, monkeypatch)
    outputs, _, retained = _run(sample, "seed_43", **arguments)
    value = development._json(outputs["fit-input.json"])
    if mutation == "training_identity":
        value["train"]["identity_sha256"] = "0" * 64
    elif mutation == "training_labels":
        value["train"]["labels_sha256"] = "0" * 64
    elif mutation == "training_tensor_labels":
        value["train"]["tensors"]["labels"]["sha256"] = "0" * 64
    elif mutation == "training_counts":
        value["train"]["positive"] *= 2
        value["train"]["negative"] *= 2
        value["train"]["row_count"] *= 2
    elif mutation == "token_shape":
        value["validation"]["tensors"]["token_ids"]["shape"] = [600, 255]
    elif mutation == "mask_dtype":
        value["validation"]["tensors"]["padding_mask"]["dtype"] = "<f4"
    elif mutation == "mask_hash":
        value["validation"]["tensors"]["padding_mask"]["sha256"] = "private value"
    elif mutation == "extra_field":
        value["undeclared"] = True
    elif mutation == "bool_schema":
        value["schema_version"] = True
    elif mutation.startswith("epoch_"):
        epoch = development._json(retained["epoch-001.json"])
        if mutation == "epoch_extra":
            epoch["undeclared"] = True
        elif mutation == "epoch_bool":
            epoch["epoch"] = True
        elif mutation == "epoch_ap_bool":
            epoch["validation_average_precision"] = True
        else:
            epoch["validation_probabilities"][0] = False
        retained["epoch-001.json"] = development._json_bytes(epoch)
    else:
        rows = [
            development._json(line)
            for line in retained["training-identities.jsonl"].splitlines()
        ]
        if mutation == "training_identity_order":
            rows.reverse()
        else:
            rows[0]["label"], rows[-1]["label"] = rows[-1]["label"], rows[0]["label"]
        retained["training-identities.jsonl"] = b"".join(
            development._json_bytes(row) for row in rows
        )
    outputs["fit-input.json"] = retained["fit-input.json"] = development._json_bytes(
        value
    )
    with pytest.raises(sample.module.SeedStageError):
        sample.module.verify_seed_stage(
            sample.binding,
            "seed_43",
            outputs=outputs,
            auxiliary=retained,
            common_stage1=arguments["common_stage1"],
        )


def test_every_qualifying_checkpoint_is_a_valid_numeric_archive(sample, monkeypatch):
    arguments = _fit_fixture(sample, monkeypatch)
    calls = 0

    def evaluate(*args):
        nonlocal calls
        calls += 1
        return (0.5, (0.5,) * 600) if calls == 1 else (1.0, sample.probabilities)

    monkeypatch.setattr(trainer, "_evaluate_validation", evaluate)
    outputs, _, retained = _run(sample, "seed_43", **arguments)
    assert "best-002.npz" in retained
    retained["best-001.npz"] = b"not a numeric archive"
    metadata = development._json(retained["best-001.json"])
    metadata["weights_sha256"] = sha256(retained["best-001.npz"]).hexdigest()
    retained["best-001.json"] = development._json_bytes(metadata)
    with pytest.raises(sample.module.SeedStageError):
        sample.module.verify_seed_stage(
            sample.binding,
            "seed_43",
            outputs=outputs,
            auxiliary=retained,
            common_stage1=arguments["common_stage1"],
        )
