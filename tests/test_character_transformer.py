import math
import random
import warnings
from inspect import signature

import numpy as np
import pytest
import torch

from automated_phishing_detection import character_transformer


def _encoded_rows(labels=(0, 1, 0, 1), *, vocabulary_size=24):
    row_count = len(labels)
    token_ids = torch.zeros((row_count, 256), dtype=torch.int64)
    padding_mask = torch.ones((row_count, 256), dtype=torch.bool)
    for row in range(row_count):
        length = 3 + row % 8
        token_ids[row, :length] = (
            torch.arange(2, 2 + length) % (vocabulary_size - 2) + 2
        )
        token_ids[row, 1] = 2 + row % (vocabulary_size - 2)
        padding_mask[row, :length] = False
    return token_ids, padding_mask, torch.tensor(labels, dtype=torch.float32)


def test_model_matches_the_frozen_architecture_and_initialization():
    character_transformer.configure_deterministic_runtime()
    original_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        model = character_transformer.CharacterTransformer(vocabulary_size=32)
    finally:
        torch.set_default_dtype(original_dtype)

    assert model.token_embedding.num_embeddings == 32
    assert model.token_embedding.embedding_dim == 192
    assert model.token_embedding.padding_idx == 0
    assert model.position_embedding.num_embeddings == 256
    assert model.position_embedding.embedding_dim == 192
    assert len(model.encoder.layers) == 4
    assert isinstance(model.final_norm, torch.nn.LayerNorm)
    assert model.output.in_features == 192
    assert model.output.out_features == 1

    first, second = model.encoder.layers[:2]
    assert first.norm_first is True
    assert first.self_attn.num_heads == 6
    assert first.linear1.in_features == 192
    assert first.linear1.out_features == 768
    assert first.activation.__name__ == "gelu"
    assert first.self_attn.dropout == pytest.approx(0.1)
    assert first.dropout.p == pytest.approx(0.1)
    assert first.dropout1.p == pytest.approx(0.1)
    assert first.dropout2.p == pytest.approx(0.1)
    assert first.linear1.weight.data_ptr() != second.linear1.weight.data_ptr()
    assert not torch.equal(first.linear1.weight, second.linear1.weight)
    assert {parameter.dtype for parameter in model.parameters()} == {torch.float32}

    assert torch.count_nonzero(model.token_embedding.weight[0]) == 0
    for module in model.modules():
        if isinstance(module, torch.nn.Linear) and module.bias is not None:
            assert torch.count_nonzero(module.bias) == 0
        if isinstance(module, torch.nn.LayerNorm):
            torch.testing.assert_close(module.weight, torch.ones_like(module.weight))
            torch.testing.assert_close(module.bias, torch.zeros_like(module.bias))


def test_forward_excludes_padded_positions_and_returns_one_logit_per_row():
    character_transformer.configure_deterministic_runtime()
    model = character_transformer.CharacterTransformer(vocabulary_size=32).eval()
    token_ids, padding_mask, _ = _encoded_rows(labels=(0, 1), vocabulary_size=32)
    changed_padding = token_ids.clone()
    changed_padding[padding_mask] = 31

    with torch.no_grad():
        original = model(token_ids, padding_mask)
        changed = model(changed_padding, padding_mask)

    assert original.shape == (2,)
    assert original.dtype == torch.float32
    torch.testing.assert_close(original, changed)


def test_forward_rejects_malformed_tensor_shapes_with_a_contract_error():
    model = character_transformer.CharacterTransformer(vocabulary_size=8)

    with pytest.raises(character_transformer.TransformerTrainingError):
        model(torch.tensor(1, dtype=torch.int64), torch.tensor(False))


def test_seed_control_repeats_python_numpy_torch_and_model_initialization():
    character_transformer.configure_deterministic_runtime()
    first_draws = (random.random(), np.random.random(), torch.rand(2))
    first_model = character_transformer.CharacterTransformer(vocabulary_size=16)

    character_transformer.configure_deterministic_runtime()
    second_draws = (random.random(), np.random.random(), torch.rand(2))
    second_model = character_transformer.CharacterTransformer(vocabulary_size=16)

    assert first_draws[0] == second_draws[0]
    assert first_draws[1] == second_draws[1]
    torch.testing.assert_close(first_draws[2], second_draws[2])
    for first, second in zip(first_model.parameters(), second_model.parameters()):
        torch.testing.assert_close(first, second)
    assert torch.are_deterministic_algorithms_enabled()
    assert not torch.is_deterministic_algorithms_warn_only_enabled()


def test_positive_class_weight_is_the_exact_training_class_ratio():
    labels = torch.tensor([0, 0, 0, 1], dtype=torch.float32)

    assert character_transformer.positive_class_weight(labels) == 3.0


@pytest.mark.parametrize(
    "labels",
    (
        torch.tensor([0, 0], dtype=torch.float32),
        torch.tensor([1, 1], dtype=torch.float32),
        torch.tensor([0, 2, 1], dtype=torch.float32),
        torch.tensor([0, torch.nan, 1], dtype=torch.float32),
    ),
)
def test_positive_class_weight_rejects_missing_or_invalid_classes(labels):
    with pytest.raises(character_transformer.TransformerTrainingError):
        character_transformer.positive_class_weight(labels)


def test_tensor_batch_validates_labels_before_float32_conversion():
    token_ids, padding_mask, _ = _encoded_rows(labels=(0, 1))
    labels = torch.tensor([0.0, 1.000000000001], dtype=torch.float64)

    with pytest.raises(
        character_transformer.TransformerTrainingError,
        match="binary",
    ):
        character_transformer._validated_tensor_batch(
            token_ids,
            padding_mask,
            labels,
            vocabulary_size=24,
            partition="train",
        )


def test_early_stopping_uses_strict_minimum_delta_and_restores_earliest_best():
    model = torch.nn.Linear(1, 1, bias=False)
    tracker = character_transformer._BestCheckpoint()

    with torch.no_grad():
        model.weight.fill_(1.0)
    assert tracker.observe(epoch=1, score=0.5, model=model) is False

    with torch.no_grad():
        model.weight.fill_(2.0)
    assert tracker.observe(epoch=2, score=0.5001, model=model) is False

    with torch.no_grad():
        model.weight.fill_(3.0)
    assert tracker.observe(epoch=3, score=0.50010001, model=model) is False
    assert tracker.best_epoch == 3

    for epoch in range(4, 9):
        with torch.no_grad():
            model.weight.fill_(float(epoch))
        should_stop = tracker.observe(epoch=epoch, score=0.50010001, model=model)
        assert should_stop is (epoch == 8)

    tracker.restore(model)
    torch.testing.assert_close(model.weight, torch.tensor([[3.0]]))
    assert tracker.best_epoch == 3
    assert tracker.best_score == pytest.approx(0.50010001)


def test_data_loader_training_order_is_seeded_once_and_advances_each_epoch():
    labels = tuple(index % 2 for index in range(300))
    train = character_transformer._validated_tensor_batch(
        *_encoded_rows(labels, vocabulary_size=24),
        vocabulary_size=24,
        partition="train",
    )
    validation = character_transformer._validated_tensor_batch(
        *_encoded_rows((0, 1), vocabulary_size=24),
        vocabulary_size=24,
        partition="validation",
    )

    first_train, _ = character_transformer._build_data_loaders(train, validation)
    second_train, _ = character_transformer._build_data_loaders(train, validation)
    first_epoch = torch.cat([batch[0][:, 1] for batch in first_train])
    second_epoch = torch.cat([batch[0][:, 1] for batch in first_train])
    repeated_first = torch.cat([batch[0][:, 1] for batch in second_train])
    repeated_second = torch.cat([batch[0][:, 1] for batch in second_train])

    torch.testing.assert_close(first_epoch, repeated_first)
    torch.testing.assert_close(second_epoch, repeated_second)
    assert not torch.equal(first_epoch, second_epoch)
    assert first_train.batch_size == 256
    assert first_train.num_workers == 0


def test_tensor_batch_rejects_interleaved_padding():
    token_ids, padding_mask, labels = _encoded_rows(labels=(0, 1))
    token_ids[0, 1] = 0
    padding_mask[0, 1] = True

    with pytest.raises(
        character_transformer.TransformerTrainingError,
        match="right padding",
    ):
        character_transformer._validated_tensor_batch(
            token_ids,
            padding_mask,
            labels,
            vocabulary_size=24,
            partition="train",
        )


def test_official_device_requires_mps_and_cpu_is_fixture_only(monkeypatch):
    public_parameters = signature(
        character_transformer.fit_character_transformer
    ).parameters
    assert "device" not in public_parameters
    assert "_fixture_cpu" not in public_parameters

    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    with pytest.raises(
        character_transformer.TransformerTrainingError,
        match="MPS",
    ):
        character_transformer._resolve_training_device()


def test_fit_uses_frozen_optimizer_weight_and_restores_best_checkpoint(monkeypatch):
    train = _encoded_rows(labels=(0, 0, 0, 1))
    validation = _encoded_rows(labels=(0, 1))
    optimizer_arguments = {}
    real_adamw = torch.optim.AdamW

    def recording_adamw(parameters, **kwargs):
        optimizer_arguments.update(kwargs)
        return real_adamw(parameters, **kwargs)

    epoch = 0

    def synthetic_epoch(model, loader, optimizer, loss_function, device):
        nonlocal epoch
        epoch += 1
        assert loader.batch_size == 256
        assert loss_function.pos_weight.item() == pytest.approx(3.0)
        with torch.no_grad():
            model.output.bias.fill_(float(epoch))
        return 0.25

    def synthetic_validation(model, loader, device):
        assert loader.batch_size == 512
        if epoch > 0 and model.output.bias.item() == 1.0:
            return 0.75, (0.1, 0.9)
        return 0.75, (0.2, 0.8)

    monkeypatch.setattr(torch.optim, "AdamW", recording_adamw)
    monkeypatch.setattr(character_transformer, "_run_training_epoch", synthetic_epoch)
    monkeypatch.setattr(
        character_transformer, "_evaluate_validation", synthetic_validation
    )

    fitted = character_transformer._fit_character_transformer_on_device(
        *train,
        *validation,
        vocabulary_size=24,
        device=torch.device("cpu"),
    )

    assert optimizer_arguments == {
        "lr": 1e-4,
        "betas": (0.9, 0.999),
        "eps": 1e-8,
        "weight_decay": 0.01,
    }
    assert fitted.best_epoch == 1
    assert fitted.epochs_completed == 6
    assert fitted.stopped_early is True
    assert fitted.positive_class_weight == 3.0
    assert fitted.validation_probabilities == (0.1, 0.9)
    assert fitted.model.output.bias.item() == 1.0


def test_training_epoch_clips_the_global_gradient_norm_at_one(monkeypatch):
    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.logit = torch.nn.Parameter(torch.tensor(0.0))

        def forward(self, token_ids, padding_mask):
            return self.logit.expand(token_ids.shape[0])

    batch = character_transformer._validated_tensor_batch(
        *_encoded_rows(labels=(0, 1)),
        vocabulary_size=24,
        partition="train",
    )
    loader, _ = character_transformer._build_data_loaders(batch, batch)
    model = TinyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss_function = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(1.0))
    calls = []
    real_clip = torch.nn.utils.clip_grad_norm_

    def recording_clip(parameters, max_norm, *args, **kwargs):
        calls.append(max_norm)
        return real_clip(parameters, max_norm, *args, **kwargs)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", recording_clip)

    loss = character_transformer._run_training_epoch(
        model, loader, optimizer, loss_function, torch.device("cpu")
    )

    assert math.isfinite(loss)
    assert calls == [1.0]


def test_outer_autocast_cannot_change_repeatable_fixture_training(monkeypatch):
    train = _encoded_rows(labels=(0, 1))
    validation = _encoded_rows(labels=(0, 1))
    monkeypatch.setattr(character_transformer, "MAX_EPOCHS", 1)
    observed_autocast = []
    real_training_epoch = character_transformer._run_training_epoch
    real_validation = character_transformer._evaluate_validation

    def observed_training_epoch(*args, **kwargs):
        observed_autocast.append(torch.is_autocast_enabled("cpu"))
        return real_training_epoch(*args, **kwargs)

    def observed_validation(*args, **kwargs):
        observed_autocast.append(torch.is_autocast_enabled("cpu"))
        return real_validation(*args, **kwargs)

    monkeypatch.setattr(
        character_transformer, "_run_training_epoch", observed_training_epoch
    )
    monkeypatch.setattr(
        character_transformer, "_evaluate_validation", observed_validation
    )

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        assert torch.is_autocast_enabled("cpu")
        first = character_transformer._fit_character_transformer_on_device(
            *train,
            *validation,
            vocabulary_size=24,
            device=torch.device("cpu"),
        )
    second = character_transformer._fit_character_transformer_on_device(
        *train,
        *validation,
        vocabulary_size=24,
        device=torch.device("cpu"),
    )

    assert observed_autocast and not any(observed_autocast)
    assert first.history == second.history
    assert first.validation_probabilities == second.validation_probabilities
    assert first.best_epoch == second.best_epoch == 1
    for name, first_value in first.model.state_dict().items():
        torch.testing.assert_close(
            first_value,
            second.model.state_dict()[name],
            rtol=0,
            atol=0,
        )


def test_fit_treats_every_warning_as_a_failure(monkeypatch):
    train = _encoded_rows(labels=(0, 1))
    validation = _encoded_rows(labels=(0, 1))

    def warning_epoch(*args, **kwargs):
        warnings.warn("synthetic training warning", RuntimeWarning)

    monkeypatch.setattr(character_transformer, "_run_training_epoch", warning_epoch)

    with pytest.raises(
        character_transformer.TransformerTrainingError,
        match="warning",
    ):
        character_transformer._fit_character_transformer_on_device(
            *train,
            *validation,
            vocabulary_size=24,
            device=torch.device("cpu"),
        )


def test_fit_rejects_nonfinite_training_and_validation_values(monkeypatch):
    train = _encoded_rows(labels=(0, 1))
    validation = _encoded_rows(labels=(0, 1))
    monkeypatch.setattr(
        character_transformer,
        "_run_training_epoch",
        lambda *args, **kwargs: float("nan"),
    )

    with pytest.raises(
        character_transformer.TransformerTrainingError,
        match="nonfinite",
    ):
        character_transformer._fit_character_transformer_on_device(
            *train,
            *validation,
            vocabulary_size=24,
            device=torch.device("cpu"),
        )

    monkeypatch.setattr(
        character_transformer,
        "_run_training_epoch",
        lambda *args, **kwargs: 0.25,
    )
    monkeypatch.setattr(
        character_transformer,
        "_evaluate_validation",
        lambda *args, **kwargs: (float("nan"), (0.1, 0.9)),
    )
    with pytest.raises(
        character_transformer.TransformerTrainingError,
        match="nonfinite",
    ):
        character_transformer._fit_character_transformer_on_device(
            *train,
            *validation,
            vocabulary_size=24,
            device=torch.device("cpu"),
        )
