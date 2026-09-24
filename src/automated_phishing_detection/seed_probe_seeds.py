"""Byte-only seed execution and checks of retained development evidence.

The process runner authenticates inputs and supplies a create-only durable sink.
This module never opens a file, retries a fit, or changes a primary artifact.
"""

from __future__ import annotations

import math
import re
from contextlib import contextmanager
from dataclasses import asdict
from hashlib import sha256

import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from . import character_transformer as trainer
from . import (
    fixed_cascade,
    phiusiil,
    secondary_transformer,
    transformer_inference,
    transformer_scoring,
)
from . import secondary_development as development

SAFE_CHECKS = frozenset(
    {
        "artifact_hash_mismatch",
        "best_checkpoint_mismatch",
        "calibration_mismatch",
        "checkpoint_coverage_mismatch",
        "checkpoint_archive_invalid",
        "common_stage1_audit",
        "common_stage1_identity_mismatch",
        "common_stage1_requirement",
        "common_stage1_schema",
        "empty_prediction_evidence",
        "epoch_ap_mismatch",
        "epoch_count",
        "epoch_identity",
        "epoch_loss",
        "epoch_score_count",
        "fit_artifact_binding_mismatch",
        "fit_input_binding_mismatch",
        "fit_input_retention_mismatch",
        "fitted_checkpoint_metadata_mismatch",
        "fitted_restored_weights_mismatch",
        "invalid_evidence_bytes",
        "invalid_evidence_json",
        "invalid_retention_callback",
        "invalid_seed_stage",
        "noncanonical_evidence",
        "partition_binding_mismatch",
        "prediction_binding_mismatch",
        "prediction_decision_mismatch",
        "prediction_identity_invalid",
        "prediction_identity_mismatch",
        "population_binding_mismatch",
        "population_schema",
        "probability_schema",
        "primary_weight_binding_mismatch",
        "restored_ap_mismatch",
        "restored_checkpoint_mismatch",
        "restored_checkpoint_missing",
        "saved_seed_evidence_invalid",
        "seed_output_inventory",
        "training_after_patience",
        "training_result_mismatch",
        "training_stopped_too_early",
        "training_validation_overlap",
        "training_identity_schema",
        "tensor_descriptor_schema",
        "transformer_score_count",
        "unclassified_check",
        "unexpected_common_stage1",
        "unexpected_primary_fit_evidence",
        "unexpected_training_input",
    }
)


class SeedStageError(ValueError):
    """A safe identifier for a failed seed input or saved-evidence check."""

    def __init__(self, check_id):
        self.check_id = (
            check_id
            if type(check_id) is str and check_id in SAFE_CHECKS
            else "unclassified_check"
        )
        super().__init__(self.check_id)


def _require(condition, reason):
    if not condition:
        raise SeedStageError(reason)


def _seed(stage):
    stages = {"seed_42_calibration": 42, **{f"seed_{i}": i for i in range(43, 47)}}
    _require(type(stage) is str and stage in stages, "invalid_seed_stage")
    return stages[stage]


def _hash(content):
    return sha256(content).hexdigest()


def _same(actual, expected):
    return development._json_bytes(actual) == development._json_bytes(expected)


def _fields(value, expected, check):
    _require(type(value) is dict and set(value) == set(expected), check)


def _probabilities(values):
    _require(
        isinstance(values, (list, tuple))
        and bool(values)
        and all(
            type(value) is float and math.isfinite(value) and 0 <= value <= 1
            for value in values
        ),
        "probability_schema",
    )
    return values


def _json(content):
    _require(type(content) is bytes, "invalid_evidence_bytes")
    try:
        value = development._json(content)
        _require(development._json_bytes(value) == content, "noncanonical_evidence")
        return value
    except (ValueError, TypeError, UnicodeError) as error:
        raise SeedStageError("invalid_evidence_json") from error


def _lines(content):
    _require(type(content) is bytes and bool(content), "empty_prediction_evidence")
    return [_json(line) for line in content.splitlines(keepends=True)]


def _source_lines(content):
    _require(type(content) is bytes and bool(content), "empty_prediction_evidence")
    try:
        rows = [development._json(line) for line in content.splitlines(keepends=True)]
        _require(phiusiil._jsonl_bytes(rows) == content, "noncanonical_evidence")
        return rows
    except (ValueError, TypeError, UnicodeError) as error:
        raise SeedStageError("invalid_evidence_json") from error


def _line_bytes(rows):
    return b"".join(development._json_bytes(row) for row in rows)


@contextmanager
def _numerical_context():
    previous = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        with development._numerical_context():
            yield
    finally:
        torch.set_num_threads(previous)


def _artifact(binding, artifacts, name):
    value = artifacts.get(name)
    _require(
        type(value) is bytes
        and _hash(value) == dict(binding.primary_artifact_hashes).get(name),
        "artifact_hash_mismatch",
    )
    return value


def _stage1(binding, content):
    return fixed_cascade._load_logistic_l1_artifact_bytes(
        content,
        expected_sha256=binding.pins.logistic_l1_artifact_sha256,
        expected_contract_sha256=binding.pins.baseline_contract_sha256,
    )


def _identity(rows):
    return [{"record_id": row["record_id"], "label": row["label"]} for row in rows]


def _tensor_digest(value):
    array = value.detach().cpu().contiguous().numpy()
    array = array.astype(array.dtype.newbyteorder("<"), copy=False)
    return {
        "shape": list(array.shape),
        "dtype": array.dtype.str,
        "sha256": _hash(array.tobytes(order="C")),
    }


def _population(rows, tensors):
    labels = [row["label"] for row in rows]
    return {
        "row_count": len(rows),
        "positive": sum(labels),
        "negative": len(labels) - sum(labels),
        "identity_sha256": _hash(development._json_bytes(_identity(rows))),
        "labels_sha256": _hash(development._json_bytes(labels)),
        "tensors": {
            name: _tensor_digest(tensor)
            for name, tensor in zip(("token_ids", "padding_mask", "labels"), tensors)
        },
    }


def _encode(vocabulary, partition, rows):
    _, tokens, mask = transformer_scoring._prepare_character_inputs(
        vocabulary, partition.raw_urls
    )
    labels = torch.tensor([row["label"] for row in rows], dtype=torch.float32)
    return tokens, mask, labels


def _prepare(
    binding, seed, train_bytes, validation_bytes, suffix_rules_bytes, vocabulary
):
    try:
        declaration, rules = development._preparation(
            binding.preparation_bytes, suffix_rules_bytes, binding.pins
        )
        validation = development._partition(
            validation_bytes,
            "validation",
            declaration["splits"]["validation"],
            binding.pins,
            rules,
        )
        validation_rows = [
            {**row, "label": row["is_phishing"]}
            for row in _source_lines(validation_bytes)
        ]
        validation_tensors = _encode(vocabulary, validation, validation_rows)
        train, train_rows, train_tensors = None, None, None
        if seed != 42:
            train = development._partition(
                train_bytes,
                "train",
                declaration["splits"]["train"],
                binding.pins,
                rules,
            )
            train_rows = [
                {**row, "label": row["is_phishing"]}
                for row in _source_lines(train_bytes)
            ]
            _require(
                not set(train.domains).intersection(validation.domains)
                and not set(train.record_ids).intersection(validation.record_ids)
                and not {
                    row["canonical_url_sha256"] for row in train_rows
                }.intersection(row["canonical_url_sha256"] for row in validation_rows),
                "training_validation_overlap",
            )
            train_tensors = _encode(vocabulary, train, train_rows)
        return (
            validation,
            validation_rows,
            validation_tensors,
            train_rows,
            train_tensors,
        )
    except (development.SecondaryDevelopmentError, KeyError, TypeError) as error:
        raise SeedStageError("partition_binding_mismatch") from error


def _common_scores(content, rows):
    common = _lines(content)
    _require(
        _same(_identity(common), _identity(rows)), "common_stage1_identity_mismatch"
    )
    _require(
        all(
            set(row) == {"record_id", "label", "probability", "audit"} for row in common
        ),
        "common_stage1_schema",
    )
    _require(all(type(row["audit"]) is dict for row in common), "common_stage1_audit")
    scores = fixed_cascade._probability_vector(
        _probabilities([row["probability"] for row in common]), "stage1"
    )
    return tuple(float(value) for value in scores)


def _score_weights(weights, vocabulary, tensors, device):
    model = transformer_inference._load_model(weights, vocabulary.size, device)
    transformer_scoring._validate_transformer_state(model, vocabulary, device)
    return transformer_scoring._score_model_prepared(model, device, *tensors[:2], 1)


def _checkpoint_record(event, weights_hash):
    return {
        "seed": event.seed,
        "phase": event.phase,
        "weights_sha256": weights_hash,
        "history": [asdict(record) for record in event.history],
        "best_epoch": event.best_epoch,
        "best_validation_average_precision": event.best_validation_average_precision,
        "epochs_completed": event.epochs_completed,
        "stopped_early": event.stopped_early,
        "positive_class_weight": event.positive_class_weight,
    }


def _fit(seed, train, validation, vocabulary, retain):
    restored = None

    def epoch(event):
        retain(
            f"epoch-{event.record.epoch:03d}.json",
            development._json_bytes(
                {
                    "seed": seed,
                    **asdict(event.record),
                    "checkpoint_batch_size": event.checkpoint_batch_size,
                    "validation_probabilities": event.validation_probabilities,
                }
            ),
        )

    def checkpoint(event):
        nonlocal restored
        name = (
            f"best-{event.best_epoch:03d}"
            if event.phase == "best_update"
            else "restored-best"
        )
        retain(f"{name}.npz", event.weights_bytes)
        record = _checkpoint_record(event, _hash(event.weights_bytes))
        retain(f"{name}.json", development._json_bytes(record))
        if event.phase == "restored_best":
            restored = (event.weights_bytes, record)

    fitted = secondary_transformer.fit_secondary_transformer(
        *train,
        *validation,
        vocabulary_size=vocabulary.size,
        seed=seed,
        epoch_callback=epoch,
        checkpoint_callback=checkpoint,
    ).fit
    _require(restored is not None, "restored_checkpoint_missing")
    weights, record = restored
    _require(
        secondary_transformer.serialize_secondary_weights(fitted.model) == weights,
        "fitted_restored_weights_mismatch",
    )
    result = {
        **record,
        "new_fit": True,
        "weights_file": "restored-best.npz",
        "restored_validation_probabilities": fitted.validation_probabilities,
    }
    _require(
        result["history"] == [asdict(item) for item in fitted.history]
        and result["best_epoch"] == fitted.best_epoch
        and result["best_validation_average_precision"]
        == fitted.best_validation_average_precision
        and result["epochs_completed"] == fitted.epochs_completed
        and result["stopped_early"] == fitted.stopped_early
        and result["positive_class_weight"] == fitted.positive_class_weight,
        "fitted_checkpoint_metadata_mismatch",
    )
    retain("completed-fit.json", development._json_bytes(result))
    return weights, result


def _prediction_rows(raw, calibration):
    stage1 = [row["stage1_probability"] for row in raw]
    transformer = [row["transformer_probability"] for row in raw]
    threshold = calibration["transformer_threshold"]["threshold"]
    stage1_threshold = calibration["stage1"]["threshold"]
    band = calibration["cascade_band"]
    cascade = None
    if band["status"] == "selected":
        cascade = fixed_cascade.score_fixed_cascade(
            stage1,
            transformer,
            stage1_threshold=stage1_threshold,
            transformer_threshold=threshold,
            half_width=band["half_width"],
        )
    return [
        {
            **row,
            "stage1_decision": None
            if stage1_threshold is None
            else int(stage1[index] >= stage1_threshold),
            "transformer_decision": None
            if threshold is None
            else int(transformer[index] >= threshold),
            "cascade_probability": None
            if cascade is None
            else cascade.probabilities[index],
            "cascade_decision": None if cascade is None else cascade.decisions[index],
            "logical_stage2_mask": None
            if cascade is None
            else cascade.transformer_invoked[index],
        }
        for index, row in enumerate(raw)
    ]


def _summary(seed, outputs, calibration, result, raw):
    labels = [row["label"] for row in raw]
    scores = [row["transformer_probability"] for row in raw]
    return {
        "seed": seed,
        "new_fit": seed != 42,
        "result_role": "secondary_seed_runtime_sensitivity",
        "validation_rows": len(raw),
        "validation_positive": sum(labels),
        "validation_negative": len(labels) - sum(labels),
        "weights_sha256": result["weights_sha256"],
        "best_epoch": result.get("best_epoch"),
        "epochs_completed": result.get("epochs_completed", 0),
        "validation_average_precision": float(average_precision_score(labels, scores)),
        "validation_roc_auc": float(roc_auc_score(labels, scores)),
        "calibration": calibration,
        "output_sha256": {
            name: _hash(value) for name, value in sorted(outputs.items())
        },
        "primary_artifacts_changed": False,
        "pure_seed_effect_claim": False,
    }


def run_seed_stage(
    binding,
    stage,
    *,
    train_bytes,
    validation_bytes,
    suffix_rules_bytes,
    artifacts,
    common_stage1,
    retain,
):
    """Run one reserved stage; the caller durably saves every callback before return."""
    seed = _seed(stage)
    _require(callable(retain), "invalid_retention_callback")
    _require(seed != 42 or train_bytes is None, "unexpected_training_input")
    _require((seed == 42) == (common_stage1 is None), "common_stage1_requirement")
    logistic_bytes = _artifact(binding, artifacts, "logistic-l1.json")
    vocabulary_bytes = _artifact(binding, artifacts, "vocabulary.json")
    primary_weights = (
        _artifact(binding, artifacts, "transformer-weights.npz") if seed == 42 else None
    )
    with _numerical_context():
        vocabulary = transformer_inference._load_vocabulary(vocabulary_bytes)
        model = _stage1(binding, logistic_bytes)
        validation, rows, tensors, train_rows, train = _prepare(
            binding, seed, train_bytes, validation_bytes, suffix_rules_bytes, vocabulary
        )
        population = _population(rows, tensors)
        fit_input = {
            "schema_version": 1,
            "stage": stage,
            "seed": seed,
            "profile_sha256": binding.profile_sha256,
            "methods_sha256": binding.methods_sha256,
            "input_hashes": asdict(binding.pins),
            "artifact_hashes": {
                "logistic-l1.json": _hash(logistic_bytes),
                "vocabulary.json": _hash(vocabulary_bytes),
            },
            "validation": population,
            "train": None if train_rows is None else _population(train_rows, train),
            "common_stage1_sha256": None
            if common_stage1 is None
            else _hash(common_stage1),
        }
        fit_input_bytes = development._json_bytes(fit_input)
        retain("fit-input.json", fit_input_bytes)
        retain("input-logistic-l1.json", logistic_bytes)
        retain("input-vocabulary.json", vocabulary_bytes)
        if train_rows is not None:
            retain("training-identities.jsonl", _line_bytes(_identity(train_rows)))
        if seed == 42:
            scored = secondary_transformer.score_secondary_stage1_urls(
                model, validation.raw_urls
            )
            common_stage1 = _line_bytes(
                [
                    {**identity, "probability": probability, "audit": audit}
                    for identity, probability, audit in zip(
                        _identity(rows), scored.probabilities, scored.scoring_audits
                    )
                ]
            )
            retain("common-stage1.jsonl", common_stage1)
        stage1 = _common_scores(common_stage1, rows)
        if seed == 42:
            weights = primary_weights
            result = {
                "seed": 42,
                "new_fit": False,
                "weights_sha256": _hash(weights),
                "weights_file": "accepted_primary_transformer",
            }
        else:
            weights, result = _fit(seed, train, tensors, vocabulary, retain)
        transformer = _score_weights(
            weights, vocabulary, tensors, trainer._resolve_training_device()
        )
        _require(len(transformer) == len(rows), "transformer_score_count")
        fixed_cascade._probability_vector(transformer, "transformer")
        weights_hash = _hash(weights)
        raw = [
            {
                **identity,
                "seed": seed,
                "weights_sha256": weights_hash,
                "stage1_probability": first,
                "transformer_probability": second,
            }
            for identity, first, second in zip(_identity(rows), stage1, transformer)
        ]
        retain("raw-scores.jsonl", _line_bytes(raw))
        calibration = secondary_transformer.calibrate_secondary_seed(
            model, stage1, transformer, [row["label"] for row in rows]
        )
        outputs = {
            "fit-input.json": fit_input_bytes,
            "validation-predictions.jsonl": _line_bytes(
                _prediction_rows(raw, calibration)
            ),
            "calibration.json": development._json_bytes(calibration),
            "training-result.json": development._json_bytes(result),
        }
        return outputs, _summary(seed, outputs, calibration, result, raw)


def _verify_population(population, rows, binding, split):
    _fields(
        population,
        (
            "row_count",
            "positive",
            "negative",
            "identity_sha256",
            "labels_sha256",
            "tensors",
        ),
        "population_schema",
    )
    _require(
        all(
            type(population[key]) is int and population[key] > 0
            for key in ("row_count", "positive", "negative")
        ),
        "population_schema",
    )
    declaration = development._json(binding.preparation_bytes)["splits"][split]
    _require(
        population["row_count"] == declaration["row_count"]
        and population["positive"] == declaration["class_counts"]["1"]
        and population["negative"] == declaration["class_counts"]["0"],
        "population_binding_mismatch",
    )
    _fields(
        population["tensors"],
        ("token_ids", "padding_mask", "labels"),
        "tensor_descriptor_schema",
    )
    for key, dtype, shape in (
        ("token_ids", "<i8", [len(rows), 256]),
        ("padding_mask", "|b1", [len(rows), 256]),
        ("labels", "<f4", [len(rows)]),
    ):
        descriptor = population["tensors"][key]
        _fields(descriptor, ("shape", "dtype", "sha256"), "tensor_descriptor_schema")
        _require(
            _same(descriptor["shape"], shape)
            and descriptor["dtype"] == dtype
            and type(descriptor["sha256"]) is str
            and re.fullmatch(r"[0-9a-f]{64}", descriptor["sha256"]) is not None,
            "tensor_descriptor_schema",
        )
    labels = [row["label"] for row in rows]
    _require(
        all(type(label) is int and label in (0, 1) for label in labels)
        and set(labels) == {0, 1}
        and all(
            type(row["record_id"]) is str
            and re.fullmatch(
                r"phiusiil-row-v1:" + binding.pins.source_csv_sha256 + r":[0-9a-f]{16}",
                row["record_id"],
            )
            is not None
            for row in rows
        )
        and len({row["record_id"] for row in rows}) == len(rows),
        "prediction_identity_invalid",
    )
    _require(
        population["row_count"] == len(rows)
        and population["positive"] == sum(labels)
        and population["negative"] == len(labels) - sum(labels)
        and population["identity_sha256"]
        == _hash(development._json_bytes(_identity(rows)))
        and population["labels_sha256"] == _hash(development._json_bytes(labels))
        and population["tensors"]["labels"]
        == _tensor_digest(torch.tensor(labels, dtype=torch.float32)),
        "prediction_identity_mismatch",
    )


def _ap(probabilities, labels):
    scores = fixed_cascade._probability_vector(
        _probabilities(probabilities), "retained_epoch"
    )
    _require(len(scores) == len(labels), "epoch_score_count")
    return float(average_precision_score(labels, scores))


def _verify_training(seed, fit_input, result, auxiliary, labels, vocabulary):
    history, best_epoch, best_score, stale = [], None, None, 0
    count = result["epochs_completed"]
    _require(type(count) is int and 1 <= count <= trainer.MAX_EPOCHS, "epoch_count")
    expected_names = {"restored-best.json", "restored-best.npz", "completed-fit.json"}
    weight = fit_input["train"]["negative"] / fit_input["train"]["positive"]
    for number in range(1, count + 1):
        name = f"epoch-{number:03d}.json"
        expected_names.add(name)
        epoch = _json(auxiliary[name])
        _fields(
            epoch,
            (
                "seed",
                "epoch",
                "training_loss",
                "validation_average_precision",
                "checkpoint_batch_size",
                "validation_probabilities",
            ),
            "epoch_identity",
        )
        _require(
            _same(epoch["seed"], seed)
            and _same(epoch["epoch"], number)
            and _same(epoch["checkpoint_batch_size"], 512),
            "epoch_identity",
        )
        score = _ap(epoch["validation_probabilities"], labels)
        _require(
            _same(score, epoch["validation_average_precision"]), "epoch_ap_mismatch"
        )
        _require(
            type(epoch["training_loss"]) is float
            and math.isfinite(epoch["training_loss"])
            and epoch["training_loss"] >= 0,
            "epoch_loss",
        )
        history.append(
            {
                key: epoch[key]
                for key in ("epoch", "training_loss", "validation_average_precision")
            }
        )
        qualifies = best_score is None or score > best_score + trainer.MIN_DELTA
        if qualifies:
            best_epoch, best_score, stale = number, score, 0
            base = f"best-{number:03d}"
            expected_names.update((f"{base}.json", f"{base}.npz"))
            checkpoint = _json(auxiliary[f"{base}.json"])
            try:
                transformer_inference._load_model(
                    auxiliary[f"{base}.npz"], vocabulary.size, torch.device("cpu")
                )
            except transformer_inference.TransformerInferenceError as error:
                raise SeedStageError("checkpoint_archive_invalid") from error
            _require(
                _same(
                    checkpoint,
                    {
                        "seed": seed,
                        "phase": "best_update",
                        "weights_sha256": _hash(auxiliary[f"{base}.npz"]),
                        "history": history,
                        "best_epoch": number,
                        "best_validation_average_precision": score,
                        "epochs_completed": number,
                        "stopped_early": False,
                        "positive_class_weight": weight,
                    },
                ),
                "best_checkpoint_mismatch",
            )
        else:
            stale += 1
        _require(stale < trainer.PATIENCE or number == count, "training_after_patience")
    _require(
        stale == trainer.PATIENCE or count == trainer.MAX_EPOCHS,
        "training_stopped_too_early",
    )
    observed_names = {
        name
        for name in auxiliary
        if name.startswith(("epoch-", "best-", "restored-best", "completed-fit"))
    }
    _require(observed_names == expected_names, "checkpoint_coverage_mismatch")
    restored = _json(auxiliary["restored-best.json"])
    expected = {
        "seed": seed,
        "phase": "restored_best",
        "weights_sha256": _hash(auxiliary["restored-best.npz"]),
        "history": history,
        "best_epoch": best_epoch,
        "best_validation_average_precision": best_score,
        "epochs_completed": count,
        "stopped_early": stale == trainer.PATIENCE,
        "positive_class_weight": weight,
    }
    _require(
        _same(restored, expected)
        and auxiliary["restored-best.npz"] == auxiliary[f"best-{best_epoch:03d}.npz"],
        "restored_checkpoint_mismatch",
    )
    _require(
        _same(
            result,
            {
                **expected,
                "new_fit": True,
                "weights_file": "restored-best.npz",
                "restored_validation_probabilities": result[
                    "restored_validation_probabilities"
                ],
            },
        )
        and _same(_json(auxiliary["completed-fit.json"]), result),
        "training_result_mismatch",
    )
    _require(
        math.isclose(
            _ap(result["restored_validation_probabilities"], labels),
            best_score,
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "restored_ap_mismatch",
    )


def verify_seed_stage(binding, stage, *, outputs, auxiliary, common_stage1=None):
    """Recompute saved metrics and consumed labels, without source inference.

    Token/mask hashes identify producer-consumed arrays. Their descriptor schema
    is checked here; reconstructing those arrays would require source URLs again.
    """
    seed = _seed(stage)
    try:
        with _numerical_context():
            expected_outputs = {
                "fit-input.json",
                "validation-predictions.jsonl",
                "calibration.json",
                "training-result.json",
            }
            if seed == 42:
                _require(common_stage1 is None, "unexpected_common_stage1")
                common_stage1 = auxiliary["common-stage1.jsonl"]
            _require(set(outputs) == expected_outputs, "seed_output_inventory")
            fit_input = _json(outputs["fit-input.json"])
            _fields(
                fit_input,
                (
                    "schema_version",
                    "stage",
                    "seed",
                    "profile_sha256",
                    "methods_sha256",
                    "input_hashes",
                    "artifact_hashes",
                    "validation",
                    "train",
                    "common_stage1_sha256",
                ),
                "fit_input_binding_mismatch",
            )
            _require(
                outputs["fit-input.json"] == auxiliary["fit-input.json"],
                "fit_input_retention_mismatch",
            )
            _require(
                fit_input["stage"] == stage
                and _same(fit_input["seed"], seed)
                and _same(fit_input["schema_version"], 1)
                and _same(fit_input["input_hashes"], asdict(binding.pins))
                and _hash(binding.preparation_bytes)
                == binding.pins.preparation_summary_sha256
                and fit_input["profile_sha256"] == binding.profile_sha256
                and fit_input["methods_sha256"] == binding.methods_sha256
                and fit_input["common_stage1_sha256"]
                == (None if seed == 42 else _hash(common_stage1))
                and (fit_input["train"] is None) == (seed == 42),
                "fit_input_binding_mismatch",
            )
            logistic_bytes = _artifact(
                binding,
                {"logistic-l1.json": auxiliary["input-logistic-l1.json"]},
                "logistic-l1.json",
            )
            vocabulary_bytes = _artifact(
                binding,
                {"vocabulary.json": auxiliary["input-vocabulary.json"]},
                "vocabulary.json",
            )
            _require(
                fit_input["artifact_hashes"]
                == {
                    "logistic-l1.json": _hash(logistic_bytes),
                    "vocabulary.json": _hash(vocabulary_bytes),
                },
                "fit_artifact_binding_mismatch",
            )
            model = _stage1(binding, logistic_bytes)
            raw = _lines(auxiliary["raw-scores.jsonl"])
            _verify_population(fit_input["validation"], raw, binding, "validation")
            stage1 = _common_scores(common_stage1, raw)
            result = _json(outputs["training-result.json"])
            weight_hash = result["weights_sha256"]
            _require(
                all(
                    set(row)
                    == {
                        "record_id",
                        "label",
                        "seed",
                        "weights_sha256",
                        "stage1_probability",
                        "transformer_probability",
                    }
                    and _same(row["seed"], seed)
                    and row["weights_sha256"] == weight_hash
                    and row["stage1_probability"] == score
                    for row, score in zip(raw, stage1)
                ),
                "prediction_binding_mismatch",
            )
            labels = [row["label"] for row in raw]
            _probabilities([row["transformer_probability"] for row in raw])
            _probabilities([row["stage1_probability"] for row in raw])
            if seed == 42:
                _require(
                    _same(
                        result,
                        {
                            "seed": 42,
                            "new_fit": False,
                            "weights_sha256": dict(binding.primary_artifact_hashes)[
                                "transformer-weights.npz"
                            ],
                            "weights_file": "accepted_primary_transformer",
                        },
                    ),
                    "primary_weight_binding_mismatch",
                )
                _require(
                    not any(
                        name.startswith(
                            ("epoch-", "best-", "restored-best", "completed-fit")
                        )
                        for name in auxiliary
                    ),
                    "unexpected_primary_fit_evidence",
                )
            else:
                train_rows = _lines(auxiliary["training-identities.jsonl"])
                _require(
                    all(set(row) == {"record_id", "label"} for row in train_rows),
                    "training_identity_schema",
                )
                _verify_population(fit_input["train"], train_rows, binding, "train")
                _require(
                    not {row["record_id"] for row in train_rows}.intersection(
                        row["record_id"] for row in raw
                    ),
                    "training_validation_overlap",
                )
                vocabulary = transformer_inference._load_vocabulary(vocabulary_bytes)
                _verify_training(seed, fit_input, result, auxiliary, labels, vocabulary)
            calibration = secondary_transformer.calibrate_secondary_seed(
                model, stage1, [row["transformer_probability"] for row in raw], labels
            )
            _require(
                outputs["calibration.json"] == development._json_bytes(calibration),
                "calibration_mismatch",
            )
            _require(
                outputs["validation-predictions.jsonl"]
                == _line_bytes(_prediction_rows(raw, calibration)),
                "prediction_decision_mismatch",
            )
            return _summary(seed, outputs, calibration, result, raw)
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as error:
        if isinstance(error, SeedStageError):
            raise
        raise SeedStageError("saved_seed_evidence_invalid") from error
