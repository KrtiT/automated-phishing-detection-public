"""Synthetic-only tests for secondary seed methods, never research fits."""

import importlib
import random
from inspect import signature

import numpy as np
import pytest
import torch
from test_character_transformer import _encoded_rows

from automated_phishing_detection import character_transformer as trainer


def _secondary():
    return importlib.import_module("automated_phishing_detection.secondary_transformer")


@pytest.mark.parametrize("seed", (43, 44, 45, 46))
def test_secondary_entry_routes_exact_seed_device_and_mandatory_callbacks(
    monkeypatch, seed
):
    secondary = _secondary()
    seen = {}
    result = object()

    def core(*args, **kwargs):
        seen.update(kwargs)
        return result

    monkeypatch.setattr(
        trainer, "_resolve_training_device", lambda: torch.device("cpu")
    )
    monkeypatch.setattr(trainer, "_fit_character_transformer_on_device", core)

    def epoch(record):
        pass

    def checkpoint(record):
        pass

    fitted = secondary.fit_secondary_transformer(
        *_encoded_rows(),
        *_encoded_rows(),
        vocabulary_size=24,
        seed=seed,
        epoch_callback=epoch,
        checkpoint_callback=checkpoint,
    )
    assert fitted.seed == seed
    assert fitted.fit is result
    assert seen["seed"] == seed
    assert seen["device"] == torch.device("cpu")
    assert callable(seen["epoch_callback"])
    assert callable(seen["checkpoint_callback"])
    assert "seed" not in signature(trainer.fit_character_transformer).parameters


@pytest.mark.parametrize("seed", (True, False, 42, 47, 43.0, "43", np.int64(43)))
def test_secondary_rejects_nonsecondary_or_inexact_seeds_before_runtime(
    monkeypatch, seed
):
    secondary = _secondary()
    monkeypatch.setattr(
        trainer, "_resolve_training_device", lambda: pytest.fail("runtime reached")
    )
    with pytest.raises(secondary.SecondaryTransformerError, match="seed"):
        secondary.fit_secondary_transformer(
            *_encoded_rows(),
            *_encoded_rows(),
            vocabulary_size=24,
            seed=seed,
            epoch_callback=lambda record: None,
            checkpoint_callback=lambda record: None,
        )


@pytest.mark.parametrize("callback", ("epoch_callback", "checkpoint_callback"))
def test_secondary_requires_callable_evidence_sinks(monkeypatch, callback):
    secondary = _secondary()
    monkeypatch.setattr(
        trainer, "_resolve_training_device", lambda: pytest.fail("runtime reached")
    )
    callbacks = {
        "epoch_callback": lambda record: None,
        "checkpoint_callback": lambda record: None,
    }
    callbacks[callback] = None
    with pytest.raises(secondary.SecondaryTransformerError, match="callback"):
        secondary.fit_secondary_transformer(
            *_encoded_rows(),
            *_encoded_rows(),
            vocabulary_size=24,
            seed=43,
            **callbacks,
        )


def test_private_seed_seam_routes_all_rngs_and_loader_without_changing_default(
    monkeypatch,
):
    mps_seeds = []
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(torch.mps, "manual_seed", mps_seeds.append)
    values = []
    for seed in (42, 43, 43):
        trainer._configure_deterministic_runtime(seed=seed)
        values.append((random.random(), np.random.random(), torch.rand(1).item()))
    assert values[0] != values[1] == values[2]
    # torch.manual_seed also reaches MPS in the pinned runtime.
    assert mps_seeds == [42, 42, 43, 43, 43, 43]
    batch = trainer._validated_tensor_batch(
        *_encoded_rows(tuple(index % 2 for index in range(300))),
        vocabulary_size=24,
        partition="train",
    )
    default, validation = trainer._build_data_loaders(batch, batch)
    seed42, _ = trainer._build_data_loaders(batch, batch, seed=42)
    seed43, _ = trainer._build_data_loaders(batch, batch, seed=43)
    assert list(default.sampler) == list(seed42.sampler)
    assert list(seed43.sampler) != list(
        trainer._build_data_loaders(batch, batch)[0].sampler
    )
    assert default.batch_size == 256 and validation.batch_size == 512


def _synthetic_core(monkeypatch, *, final_failure=False):
    monkeypatch.setattr(
        trainer, "_resolve_training_device", lambda: torch.device("cpu")
    )
    epoch = 0
    validations = 0

    def train(model, *args):
        nonlocal epoch
        epoch += 1
        with torch.no_grad():
            model.output.bias.fill_(epoch)
        return 0.25

    def validation(model, *args):
        nonlocal validations
        validations += 1
        if validations == 7 and final_failure:
            raise RuntimeError("final restored score failed")
        return (0.75 if model.output.bias.item() == 1.0 else 0.5), (0.1, 0.9)

    monkeypatch.setattr(trainer, "_run_training_epoch", train)
    monkeypatch.setattr(trainer, "_evaluate_validation", validation)


def test_restored_checkpoint_is_retained_before_final_score_failure(monkeypatch):
    secondary = _secondary()
    _synthetic_core(monkeypatch, final_failure=True)
    epochs, checkpoints = [], []
    with pytest.raises(RuntimeError, match="final restored score"):
        secondary.fit_secondary_transformer(
            *_encoded_rows(),
            *_encoded_rows((0, 1)),
            vocabulary_size=24,
            seed=43,
            epoch_callback=epochs.append,
            checkpoint_callback=checkpoints.append,
        )
    assert [event.record.epoch for event in epochs] == list(range(1, 7))
    assert all(event.validation_probabilities == (0.1, 0.9) for event in epochs)
    assert [checkpoint.phase for checkpoint in checkpoints] == [
        "best_update",
        "restored_best",
    ]
    checkpoint = checkpoints[-1]
    assert checkpoint.seed == 43 and checkpoint.best_epoch == 1
    assert checkpoint.stopped_early and checkpoint.epochs_completed == 6
    assert checkpoint.best_validation_average_precision == 0.75
    assert checkpoint.history == tuple(event.record for event in epochs)
    from automated_phishing_detection.transformer_inference import _load_model

    restored = _load_model(checkpoint.weights_bytes, 24, torch.device("cpu"))
    assert restored.output.bias.item() == 1.0


def test_later_training_failure_preserves_completed_best_checkpoint(monkeypatch):
    secondary = _secondary()
    _synthetic_core(monkeypatch)
    prior_train = trainer._run_training_epoch
    epochs, checkpoints = [], []

    def failing_train(model, *args):
        if epochs:
            raise RuntimeError("later epoch failed")
        return prior_train(model, *args)

    monkeypatch.setattr(trainer, "_run_training_epoch", failing_train)
    with pytest.raises(RuntimeError, match="later epoch failed"):
        secondary.fit_secondary_transformer(
            *_encoded_rows(),
            *_encoded_rows((0, 1)),
            vocabulary_size=24,
            seed=43,
            epoch_callback=epochs.append,
            checkpoint_callback=checkpoints.append,
        )
    assert len(epochs) == len(checkpoints) == 1
    assert checkpoints[0].phase == "best_update"
    assert checkpoints[0].best_epoch == 1
    assert checkpoints[0].weights_bytes
    assert epochs[0].validation_probabilities == (0.1, 0.9)


def test_secondary_epoch_preserves_checkpoint_scores_order_without_extra_scoring(
    monkeypatch,
):
    from dataclasses import FrozenInstanceError

    secondary = _secondary()
    evaluate = trainer._evaluate_validation
    _synthetic_core(monkeypatch)
    epochs, checkpoints, probabilities = [], [], []

    def synthetic_forward(model, token_ids, padding_mask):
        return token_ids[:, 1].float() * -0.25 + model.output.bias[0] * 0.1

    def recording_validation(model, loader, device):
        assert loader.batch_size == 512
        score, vector = evaluate(model, loader, device)
        probabilities.append(vector)
        return score, vector

    def retain_epoch(event):
        assert type(event) is secondary.SecondaryEpoch
        assert checkpoints[0].phase == "best_update"
        assert event.record.epoch == len(epochs) + 1
        assert event.validation_probabilities is probabilities[-1]
        epochs.append(event)

    monkeypatch.setattr(trainer, "_evaluate_validation", recording_validation)
    monkeypatch.setattr(trainer.CharacterTransformer, "forward", synthetic_forward)
    result = secondary.fit_secondary_transformer(
        *_encoded_rows(),
        *_encoded_rows((0, 1)),
        vocabulary_size=24,
        seed=43,
        epoch_callback=retain_epoch,
        checkpoint_callback=checkpoints.append,
    )
    assert len(probabilities) == 7
    assert [event.validation_probabilities for event in epochs] == probabilities[:6]
    assert all(event.checkpoint_batch_size == 512 for event in epochs)
    assert probabilities[0][0] > probabilities[0][1]
    assert probabilities[1] != probabilities[0]
    assert all(
        event.record.validation_average_precision
        == trainer.average_precision_score(
            (0, 1),
            event.validation_probabilities,
        )
        for event in epochs
    )
    assert tuple(event.record for event in epochs) == result.fit.history
    assert result.fit.validation_probabilities == probabilities[-1]
    with pytest.raises(FrozenInstanceError):
        epochs[0].checkpoint_batch_size = 1


def test_primary_fit_defaults_route_seed42_and_keep_original_result(monkeypatch):
    _synthetic_core(monkeypatch)
    seeds, loader_seeds = [], []
    configure = trainer._configure_deterministic_runtime
    loaders = trainer._build_data_loaders

    def seed_runtime(*, seed=42):
        seeds.append(seed)
        return configure(seed=seed)

    def seeded_loaders(train, validation, *, seed=42):
        loader_seeds.append(seed)
        return loaders(train, validation, seed=seed)

    monkeypatch.setattr(trainer, "_configure_deterministic_runtime", seed_runtime)
    monkeypatch.setattr(trainer, "_build_data_loaders", seeded_loaders)
    result = trainer.fit_character_transformer(
        *_encoded_rows(),
        *_encoded_rows((0, 1)),
        vocabulary_size=24,
    )
    assert type(result) is trainer.TransformerFit
    assert seeds == loader_seeds == [42]
    assert result.validation_probabilities == (0.1, 0.9)
    assert result.best_epoch == 1


def test_secondary_checkpoint_survives_restored_ap_mismatch(monkeypatch):
    secondary = _secondary()
    _synthetic_core(monkeypatch)
    evaluate = trainer._evaluate_validation
    checkpoints = []

    def wrong_ap(*args):
        if len(checkpoints) == 2:
            return 0.25, (0.1, 0.9)
        return evaluate(*args)

    monkeypatch.setattr(trainer, "_evaluate_validation", wrong_ap)
    with pytest.raises(trainer.TransformerTrainingError, match="restored checkpoint"):
        secondary.fit_secondary_transformer(
            *_encoded_rows(),
            *_encoded_rows((0, 1)),
            vocabulary_size=24,
            seed=43,
            epoch_callback=lambda record: None,
            checkpoint_callback=checkpoints.append,
        )
    assert [item.phase for item in checkpoints] == ["best_update", "restored_best"]
    assert checkpoints[-1].weights_bytes == checkpoints[0].weights_bytes


@pytest.mark.parametrize("sink", ("epoch", "checkpoint"))
def test_evidence_callback_failure_propagates_without_retry(monkeypatch, sink):
    secondary = _secondary()
    _synthetic_core(monkeypatch)
    calls = []

    def fail(record):
        calls.append(record)
        raise RuntimeError("evidence sink failed")

    with pytest.raises(RuntimeError, match="evidence sink failed"):
        secondary.fit_secondary_transformer(
            *_encoded_rows(),
            *_encoded_rows((0, 1)),
            vocabulary_size=24,
            seed=43,
            epoch_callback=fail if sink == "epoch" else lambda record: None,
            checkpoint_callback=fail if sink == "checkpoint" else lambda record: None,
        )
    assert len(calls) == 1


def _stage1(positive=50, negative=400):
    import json
    from hashlib import sha256

    from test_fixed_cascade import _artifact, _threshold_record

    from automated_phishing_detection import fixed_cascade

    artifact = _artifact()
    artifact["validation_threshold"] = _threshold_record(
        positive=positive, negative=negative
    )
    content = json.dumps(artifact).encode()
    return fixed_cascade._load_logistic_l1_artifact_bytes(
        content,
        expected_sha256=sha256(content).hexdigest(),
        expected_contract_sha256="0" * 64,
    )


def test_secondary_calibration_keeps_historical_cutoff_with_current_descriptive_counts():
    secondary = _secondary()
    from automated_phishing_detection import baselines, fixed_cascade

    model = _stage1()
    labels = [0] * 400 + [1] * 50
    stage1 = [0.2] * 400 + [0.55] * 49 + [0.3]
    transformer = [0.1] * 400 + [0.9] * 50
    selected = secondary.calibrate_secondary_seed(model, stage1, transformer, labels)
    assert (
        selected["result_role"]
        == "secondary_seed_sensitivity_historical_42_runtime_confounded"
    )
    assert selected["stage1"]["threshold"] == 0.5
    assert selected["stage1"]["threshold_source"] == "fixed_historical_artifact"
    assert selected["stage1"]["counts"]["true_positive"] == 49
    assert (
        selected["stage1"]["historical_threshold_record"]
        == model.validation_threshold_record
    )
    assert selected["transformer_threshold"] == baselines.select_validation_threshold(
        transformer, labels
    )
    band = selected["cascade_band"]
    assert band["status"] == "selected"
    assert band["transformer_invocations"] == 49
    assert band["half_width"] == abs(0.55 - 0.5)
    assert band["recall"] == 0.98
    assert band["minimum_recall"] == 0.98
    assert band["fpr_upper_95"] <= 0.01
    # The primary historical-score binding is still strict on the same inputs.
    with pytest.raises(
        fixed_cascade.FixedCascadeError, match="stage-one artifact threshold"
    ):
        fixed_cascade.calibrate_fixed_cascade(
            model,
            stage1,
            transformer,
            labels,
            transformer_threshold_record=selected["transformer_threshold"],
        )


def test_secondary_transformer_cp_tie_and_band_distance_ties():
    secondary = _secondary()
    labels = [0] * 400 + [1, 1]
    selected = secondary.calibrate_secondary_seed(
        _stage1(positive=2),
        [0.25] * 400 + [0.75, 0.75],
        [0.1] * 400 + [0.9, 0.9],
        labels,
    )
    assert selected["transformer_threshold"]["threshold"] == 0.9
    # Equal distances route as a group, including both sides of the cutoff.
    assert selected["cascade_band"]["transformer_invocations"] == 402
    assert selected["cascade_band"]["half_width"] == 0.25


def test_secondary_returns_explicit_target_not_met_without_reselecting_stage1():
    secondary = _secondary()
    model = _stage1(positive=1)
    import json
    from dataclasses import replace

    from test_fixed_cascade import _threshold_record

    # With too few negatives, neither threshold may be manufactured as usable.
    model = replace(
        model,
        _threshold_record_json=json.dumps(_threshold_record(status="target_not_met")),
    )
    selected = secondary.calibrate_secondary_seed(model, [0.2, 0.8], [0.1, 0.9], [0, 1])
    assert selected["transformer_threshold"]["status"] == "target_not_met"
    assert selected["cascade_band"]["status"] == "target_not_met"
    assert selected["cascade_band"]["half_width"] is None


def test_secondary_rejects_changed_validation_class_totals():
    secondary = _secondary()
    with pytest.raises(secondary.SecondaryTransformerError, match="class totals"):
        secondary.calibrate_secondary_seed(
            _stage1(),
            [0.2] * 401 + [0.8] * 49,
            [0.1] * 401 + [0.9] * 49,
            [0] * 401 + [1] * 49,
        )


def test_secondary_records_current_stage1_failure_without_claiming_historical_gate():
    secondary = _secondary()
    selected = secondary.calibrate_secondary_seed(
        _stage1(positive=1),
        [0.6] * 400 + [0.8],
        [0.1] * 400 + [0.9],
        [0] * 400 + [1],
    )
    assert selected["stage1"]["threshold"] == 0.5
    assert selected["stage1"]["counts"]["false_positive"] == 400
    assert selected["stage1"]["fpr_upper_95"] > 0.01
    assert selected["cascade_band"]["status"] == "selected"
    assert selected["cascade_band"]["transformer_invocations"] == 400


@pytest.mark.parametrize(
    "stage1,transformer,labels",
    (
        ([0.2, float("nan")], [0.1, 0.9], [0, 1]),
        ([0.2, 0.8], [0.1, float("inf")], [0, 1]),
        ([0.2, 0.8], [0.1], [0, 1]),
        ([0.2, 0.8], [0.1, 0.9], [1, 1]),
        ([0.2, 0.8], [0.1, 0.9], [0.0, 1.0]),
    ),
)
def test_secondary_calibration_rejects_invalid_inputs(stage1, transformer, labels):
    secondary = _secondary()
    with pytest.raises(secondary.SecondaryTransformerError):
        secondary.calibrate_secondary_seed(_stage1(), stage1, transformer, labels)


@pytest.fixture
def secondary_model():
    secondary = _secondary()
    from automated_phishing_detection import character_sequence

    vocabulary = character_sequence.build_character_vocabulary(
        ("https://safe.example/path",)
    )
    trainer.configure_deterministic_runtime()
    model = trainer.CharacterTransformer(vocabulary.size)
    content = secondary.serialize_secondary_weights(model)
    loaded = secondary.load_secondary_transformer_bytes(
        content,
        vocabulary.to_json().encode(),
        seed=43,
        device=torch.device("cpu"),
    )
    return model, loaded, content, vocabulary


def test_secondary_weights_round_trip_and_remain_distinct_from_primary(secondary_model):
    secondary = _secondary()
    from automated_phishing_detection.transformer_inference import (
        LoadedTransformerCascade,
    )

    model, loaded, content, vocabulary = secondary_model
    assert type(loaded) is secondary.SecondaryTransformerModel
    assert not isinstance(loaded, LoadedTransformerCascade)
    assert loaded.model_identity == "secondary-transformer-v1" and loaded.seed == 43
    assert secondary.serialize_secondary_weights(loaded._model) == content
    assert loaded.vocabulary == vocabulary
    assert all(not parameter.requires_grad for parameter in loaded._model.parameters())
    for name, value in model.state_dict().items():
        torch.testing.assert_close(
            value, loaded._model.state_dict()[name], rtol=0, atol=0
        )


def test_primary_and_secondary_share_identical_singleton_numerics_on_invented_weights(
    tmp_path,
    monkeypatch,
):
    from test_transformer_inference import _build_fixture, _load_fixture

    from automated_phishing_detection import transformer_scoring

    secondary = _secondary()
    primary = _load_fixture(_build_fixture(tmp_path))
    loaded = secondary.load_secondary_transformer_bytes(
        secondary.serialize_secondary_weights(primary._model),
        primary.vocabulary.to_json().encode(),
        seed=43,
        device=torch.device("cpu"),
    )
    score = transformer_scoring._score_model_prepared
    calls = []

    def traced_score(model, device, tokens, mask, batch_size):
        calls.append((model, batch_size))
        return score(model, device, tokens, mask, batch_size)

    monkeypatch.setattr(transformer_scoring, "_score_model_prepared", traced_score)
    urls = ("https://safe.example/path", "https://phish.example/account")
    primary_scores = transformer_scoring.score_transformer_urls(
        primary, urls, batch_size=1
    )
    secondary_scores = secondary.score_secondary_transformer_urls(loaded, urls)
    assert primary_scores == secondary_scores
    assert calls == [(primary._model, 1), (loaded._model, 1)]


def test_secondary_scoring_is_singleton_ordered_and_no_fit(
    secondary_model, monkeypatch
):
    secondary = _secondary()
    from automated_phishing_detection import character_sequence

    _, loaded, _, vocabulary = secondary_model
    urls = (
        "HTTPS://SAFE.example:443",
        "https://safe.example/b",
        "https://safe.example/a",
    )
    expected = [
        character_sequence.encode_character_url(url, vocabulary).token_ids
        for url in urls
    ]
    calls = []

    def forward(tokens, mask):
        assert len(tokens) == len(mask) == 1 and torch.is_inference_mode_enabled()
        calls.append(tuple(tokens[0].tolist()))
        return torch.tensor([float(len(calls))], dtype=torch.float32)

    monkeypatch.setattr(loaded._model, "forward", forward)
    monkeypatch.setattr(
        trainer, "fit_character_transformer", lambda *a, **k: pytest.fail("fit")
    )
    result = secondary.score_secondary_transformer_urls(loaded, urls)
    assert calls == expected
    assert result == tuple(torch.sigmoid(torch.tensor([1.0, 2.0, 3.0])).tolist())


def test_secondary_stage1_scoring_is_authoritative_singleton_with_per_row_audits(
    monkeypatch,
):
    secondary = _secondary()
    from automated_phishing_detection import fixed_cascade

    urls = (
        "HTTPS://SAFE.example:443",
        "https://safe.example/b",
        "https://safe.example/a",
    )
    calls = []

    def score(model, supplied):
        assert type(supplied) is tuple and len(supplied) == 1
        calls.append(supplied[0])
        return (len(calls) / 10,), {"row": len(calls)}

    monkeypatch.setattr(fixed_cascade, "score_logistic_l1_authoritative", score)
    result = secondary.score_secondary_stage1_urls(_stage1(), urls)
    assert calls == list(urls)
    assert result.probabilities == (0.1, 0.2, 0.3)
    assert result.scoring_audits == ({"row": 1}, {"row": 2}, {"row": 3})


@pytest.mark.parametrize(
    "urls",
    (
        [],
        "https://safe.example",
        {"https://safe.example"},
        ("https://safe.example", None),
    ),
)
def test_secondary_validates_all_urls_before_forward(
    secondary_model, monkeypatch, urls
):
    secondary = _secondary()
    _, loaded, _, _ = secondary_model
    monkeypatch.setattr(loaded._model, "forward", lambda *a: pytest.fail("forward"))
    with pytest.raises(secondary.SecondaryTransformerError):
        secondary.score_secondary_transformer_urls(loaded, urls)


def test_secondary_rejects_nonfinite_logits_and_malformed_archive(
    secondary_model, monkeypatch
):
    secondary = _secondary()
    _, loaded, _, vocabulary = secondary_model
    monkeypatch.setattr(
        loaded._model, "forward", lambda *a: torch.tensor([float("nan")])
    )
    with pytest.raises(secondary.SecondaryTransformerError, match="nonfinite"):
        secondary.score_secondary_transformer_urls(loaded, ("https://safe.example",))
    with pytest.raises(secondary.SecondaryTransformerError, match="weights"):
        secondary.load_secondary_transformer_bytes(
            b"not npz",
            vocabulary.to_json().encode(),
            seed=43,
            device=torch.device("cpu"),
        )


@pytest.mark.parametrize("scores", ((0.1, 0.2), (), (float("nan"),)))
def test_stage1_singleton_rejects_malformed_score_outputs(monkeypatch, scores):
    secondary = _secondary()
    from automated_phishing_detection import fixed_cascade

    monkeypatch.setattr(
        fixed_cascade, "score_logistic_l1_authoritative", lambda *a: (scores, {})
    )
    with pytest.raises(secondary.SecondaryTransformerError):
        secondary.score_secondary_stage1_urls(_stage1(), ("https://safe.example",))


@pytest.mark.parametrize("seed", (True, 42, 47, 43.0))
def test_secondary_loader_rejects_invalid_seed_without_reading_bytes(seed):
    secondary = _secondary()
    with pytest.raises(secondary.SecondaryTransformerError, match="seed"):
        secondary.load_secondary_transformer_bytes(
            b"", b"", seed=seed, device=torch.device("cpu")
        )


def test_secondary_weight_serializer_rejects_lossy_dtype_and_nonfinite_values(
    secondary_model,
):
    secondary = _secondary()
    model, _, _, _ = secondary_model
    model.to(dtype=torch.float64)
    with pytest.raises(secondary.SecondaryTransformerError, match="float32"):
        secondary.serialize_secondary_weights(model)
    model.to(dtype=torch.float32)
    with torch.no_grad():
        model.output.bias.fill_(float("nan"))
    with pytest.raises(secondary.SecondaryTransformerError, match="nonfinite"):
        secondary.serialize_secondary_weights(model)
