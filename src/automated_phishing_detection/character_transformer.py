"""Frozen character-transformer model and development-only trainer."""

from __future__ import annotations

import math
import random
import warnings
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

SEED = 42
SEQUENCE_LENGTH = 256
EMBEDDING_WIDTH = 192
ENCODER_LAYERS = 4
ATTENTION_HEADS = 6
FEED_FORWARD_WIDTH = 768
DROPOUT = 0.1
LEARNING_RATE = 1e-4
ADAM_BETAS = (0.9, 0.999)
ADAM_EPSILON = 1e-8
WEIGHT_DECAY = 0.01
TRAIN_BATCH_SIZE = 256
VALIDATION_BATCH_SIZE = 512
MAX_EPOCHS = 40
GRADIENT_CLIP_MAX_NORM = 1.0
MIN_DELTA = 1e-4
PATIENCE = 5


class TransformerTrainingError(ValueError):
    """Raised when the frozen model or training procedure cannot be honored."""


@dataclass(frozen=True)
class _TensorBatch:
    token_ids: Tensor
    padding_mask: Tensor
    labels: Tensor


@dataclass(frozen=True)
class EpochRecord:
    """Aggregate development metrics for one completed training epoch."""

    epoch: int
    training_loss: float
    validation_average_precision: float


@dataclass(frozen=True)
class _EpochEvidence:
    """Completed-epoch metrics and ordered probabilities used to calculate AP."""

    record: EpochRecord
    validation_probabilities: tuple[float, ...]


@dataclass(frozen=True)
class TransformerFit:
    """Restored best model and the validation outputs needed downstream."""

    model: CharacterTransformer
    history: tuple[EpochRecord, ...]
    best_epoch: int
    best_validation_average_precision: float
    epochs_completed: int
    stopped_early: bool
    positive_class_weight: float
    validation_probabilities: tuple[float, ...]


@dataclass(frozen=True)
class _CheckpointEvidence:
    """Synchronous snapshot seam, before another epoch or restored-score checks."""

    phase: str
    model: CharacterTransformer
    history: tuple[EpochRecord, ...]
    best_epoch: int
    best_validation_average_precision: float
    epochs_completed: int
    stopped_early: bool
    positive_class_weight: float


class CharacterTransformer(nn.Module):
    """The exact character encoder frozen for the RQ1 development fit."""

    def __init__(self, vocabulary_size: int) -> None:
        super().__init__()
        if type(vocabulary_size) is not int or vocabulary_size < 3:
            raise TransformerTrainingError("vocabulary_size must be an integer >= 3")

        self.vocabulary_size = vocabulary_size
        self.token_embedding = nn.Embedding(
            vocabulary_size,
            EMBEDDING_WIDTH,
            padding_idx=0,
            dtype=torch.float32,
        )
        self.position_embedding = nn.Embedding(
            SEQUENCE_LENGTH,
            EMBEDDING_WIDTH,
            dtype=torch.float32,
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=EMBEDDING_WIDTH,
            nhead=ATTENTION_HEADS,
            dim_feedforward=FEED_FORWARD_WIDTH,
            dropout=DROPOUT,
            activation="gelu",
            batch_first=True,
            norm_first=True,
            dtype=torch.float32,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=ENCODER_LAYERS,
            enable_nested_tensor=False,
        )
        self.final_norm = nn.LayerNorm(EMBEDDING_WIDTH, dtype=torch.float32)
        self.output = nn.Linear(EMBEDDING_WIDTH, 1, dtype=torch.float32)
        self._initialize_parameters()

    def _initialize_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.MultiheadAttention):
                if module.in_proj_weight is not None:
                    nn.init.xavier_uniform_(module.in_proj_weight)
                if module.q_proj_weight is not None:
                    nn.init.xavier_uniform_(module.q_proj_weight)
                if module.k_proj_weight is not None:
                    nn.init.xavier_uniform_(module.k_proj_weight)
                if module.v_proj_weight is not None:
                    nn.init.xavier_uniform_(module.v_proj_weight)
                if module.in_proj_bias is not None:
                    nn.init.zeros_(module.in_proj_bias)
                if module.bias_k is not None:
                    nn.init.zeros_(module.bias_k)
                if module.bias_v is not None:
                    nn.init.zeros_(module.bias_v)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

        with torch.no_grad():
            self.token_embedding.weight[0].zero_()

    def forward(self, token_ids: Tensor, padding_mask: Tensor) -> Tensor:
        if not isinstance(token_ids, Tensor) or token_ids.dtype != torch.int64:
            raise TransformerTrainingError("token_ids must be an int64 tensor")
        if not isinstance(padding_mask, Tensor) or padding_mask.dtype != torch.bool:
            raise TransformerTrainingError("padding_mask must be a boolean tensor")
        if token_ids.ndim != 2:
            raise TransformerTrainingError("token_ids must have shape (rows, 256)")
        expected_shape = (token_ids.shape[0], SEQUENCE_LENGTH)
        if tuple(token_ids.shape) != expected_shape:
            raise TransformerTrainingError("token_ids must have shape (rows, 256)")
        if tuple(padding_mask.shape) != expected_shape:
            raise TransformerTrainingError("padding_mask must match token_ids")
        if token_ids.numel() == 0 or token_ids.shape[0] == 0:
            raise TransformerTrainingError("a model batch must not be empty")
        if torch.any(token_ids < 0) or torch.any(token_ids >= self.vocabulary_size):
            raise TransformerTrainingError("token_ids contain an out-of-range ID")
        if torch.any(padding_mask.all(dim=1)):
            raise TransformerTrainingError("each sequence must contain a token")

        positions = torch.arange(SEQUENCE_LENGTH, device=token_ids.device)
        encoded = (
            self.token_embedding(token_ids)
            + self.position_embedding(positions)[None, :, :]
        )
        encoded = self.encoder(encoded, src_key_padding_mask=padding_mask)
        encoded = self.final_norm(encoded)
        retained = (~padding_mask).unsqueeze(-1).to(dtype=encoded.dtype)
        pooled = (encoded * retained).sum(dim=1) / retained.sum(dim=1)
        return self.output(pooled).squeeze(-1)


class _BestCheckpoint:
    """Track the earliest checkpoint satisfying the frozen improvement rule."""

    def __init__(self) -> None:
        self.best_epoch: int | None = None
        self.best_score: float | None = None
        self.nonqualifying_epochs = 0
        self._state: dict[str, Tensor] | None = None

    def observe(self, *, epoch: int, score: float, model: nn.Module) -> bool:
        if type(epoch) is not int or epoch < 1:
            raise TransformerTrainingError("epoch must be a positive integer")
        if not math.isfinite(score):
            raise TransformerTrainingError("validation metric is nonfinite")

        qualifies = self.best_score is None or score > self.best_score + MIN_DELTA
        if qualifies:
            self.best_epoch = epoch
            self.best_score = score
            self.nonqualifying_epochs = 0
            self._state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
        else:
            self.nonqualifying_epochs += 1
        return self.nonqualifying_epochs >= PATIENCE

    def restore(self, model: nn.Module) -> None:
        if self._state is None:
            raise TransformerTrainingError("no validation checkpoint was recorded")
        model.load_state_dict(self._state, strict=True)


def configure_deterministic_runtime() -> None:
    """Seed every used generator and require deterministic kernels in error mode."""
    _configure_deterministic_runtime()


def _configure_deterministic_runtime(*, seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=False)


def _resolve_training_device() -> torch.device:
    if torch.__version__.split("+", maxsplit=1)[0] != "2.7.1":
        raise TransformerTrainingError("the official fit requires PyTorch 2.7.1")
    if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
        raise TransformerTrainingError(
            "the official fit requires an available MPS device"
        )
    return torch.device("mps")


def positive_class_weight(labels: Tensor) -> float:
    """Return training negatives divided by training positives."""
    if not isinstance(labels, Tensor) or labels.ndim != 1 or labels.numel() == 0:
        raise TransformerTrainingError(
            "labels must be a nonempty one-dimensional tensor"
        )
    if labels.is_complex():
        raise TransformerTrainingError("labels must contain only finite binary values")
    original = labels.detach().to(device="cpu")
    if not torch.isfinite(original).all() or not torch.all(
        (original == 0) | (original == 1)
    ):
        raise TransformerTrainingError("labels must contain only finite binary values")
    negative_count = int(torch.count_nonzero(original == 0).item())
    positive_count = int(torch.count_nonzero(original == 1).item())
    if negative_count == 0 or positive_count == 0:
        raise TransformerTrainingError("labels must contain both binary classes")
    return negative_count / positive_count


def _validated_tensor_batch(
    token_ids: Tensor,
    padding_mask: Tensor,
    labels: Tensor,
    *,
    vocabulary_size: int,
    partition: str,
) -> _TensorBatch:
    if type(partition) is not str or not partition:
        raise TransformerTrainingError("partition must be a nonempty string")
    if type(vocabulary_size) is not int or vocabulary_size < 3:
        raise TransformerTrainingError("vocabulary_size must be an integer >= 3")
    if not isinstance(token_ids, Tensor) or token_ids.dtype != torch.int64:
        raise TransformerTrainingError(f"{partition} token_ids must be int64")
    if not isinstance(padding_mask, Tensor) or padding_mask.dtype != torch.bool:
        raise TransformerTrainingError(f"{partition} padding_mask must be boolean")
    if not isinstance(labels, Tensor) or labels.ndim != 1:
        raise TransformerTrainingError(f"{partition} labels must be one-dimensional")
    if token_ids.ndim != 2 or token_ids.shape[1] != SEQUENCE_LENGTH:
        raise TransformerTrainingError(
            f"{partition} token_ids must have shape (rows, 256)"
        )
    if padding_mask.shape != token_ids.shape:
        raise TransformerTrainingError(f"{partition} padding_mask must match token_ids")
    if token_ids.shape[0] == 0 or labels.shape[0] != token_ids.shape[0]:
        raise TransformerTrainingError(f"{partition} tensors have inconsistent rows")

    positive_class_weight(labels)
    token_ids = token_ids.detach().to(device="cpu").contiguous()
    padding_mask = padding_mask.detach().to(device="cpu").contiguous()
    labels = labels.detach().to(device="cpu", dtype=torch.float32).contiguous()
    if torch.any(token_ids < 0) or torch.any(token_ids >= vocabulary_size):
        raise TransformerTrainingError(f"{partition} token_ids are out of range")
    if torch.any(padding_mask.all(dim=1)):
        raise TransformerTrainingError(f"{partition} contains an empty sequence")
    if torch.any(token_ids[padding_mask] != 0) or torch.any(
        token_ids[~padding_mask] == 0
    ):
        raise TransformerTrainingError(
            f"{partition} token_ids and padding_mask violate right padding"
        )
    padding_started = padding_mask.to(dtype=torch.int8).cumsum(dim=1) > 0
    if torch.any((~padding_mask) & padding_started):
        raise TransformerTrainingError(
            f"{partition} token_ids and padding_mask violate right padding"
        )
    return _TensorBatch(token_ids, padding_mask, labels)


def _build_data_loaders(
    train: _TensorBatch,
    validation: _TensorBatch,
    *,
    seed: int = SEED,
) -> tuple[DataLoader, DataLoader]:
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(
        TensorDataset(train.token_ids, train.padding_mask, train.labels),
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        generator=generator,
        num_workers=0,
        drop_last=False,
    )
    validation_loader = DataLoader(
        TensorDataset(
            validation.token_ids,
            validation.padding_mask,
            validation.labels,
        ),
        batch_size=VALIDATION_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )
    return train_loader, validation_loader


def _device_flags_pass(flags: list[Tensor], device: torch.device) -> bool:
    """Combine device-side checks before making one host-side decision."""
    if not flags:
        raise TransformerTrainingError("at least one finite-value check is required")
    combined = torch.stack(
        [flag.to(device=device, dtype=torch.bool) for flag in flags]
    ).all()
    return bool(combined.detach().cpu().item())


def _run_training_epoch(
    model: CharacterTransformer,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_function: nn.BCEWithLogitsLoss,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    row_count = 0
    parameters = tuple(model.parameters())
    for token_ids, padding_mask, labels in loader:
        token_ids = token_ids.to(device=device, dtype=torch.int64)
        padding_mask = padding_mask.to(device=device, dtype=torch.bool)
        labels = labels.to(device=device, dtype=torch.float32)
        optimizer.zero_grad(set_to_none=True)
        logits = model(token_ids, padding_mask)
        loss = loss_function(logits, labels)
        loss.backward()
        gradient_norm = nn.utils.clip_grad_norm_(parameters, GRADIENT_CLIP_MAX_NORM)
        optimizer.step()
        optimizer_tensors = [
            value
            for state in optimizer.state.values()
            for value in state.values()
            if isinstance(value, Tensor)
        ]
        finite_flags = [
            torch.isfinite(value).all()
            for value in (
                logits,
                loss,
                gradient_norm,
                *(
                    parameter.grad
                    for parameter in parameters
                    if parameter.grad is not None
                ),
                *parameters,
                *optimizer_tensors,
            )
        ]
        if not _device_flags_pass(finite_flags, device):
            raise TransformerTrainingError("training state contains a nonfinite value")
        batch_rows = int(labels.shape[0])
        total_loss += float(loss.detach().cpu().item()) * batch_rows
        row_count += batch_rows

    average_loss = total_loss / row_count
    if not math.isfinite(average_loss):
        raise TransformerTrainingError("training loss is nonfinite")
    return average_loss


def _evaluate_validation(
    model: CharacterTransformer,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, tuple[float, ...]]:
    model.eval()
    probabilities: list[Tensor] = []
    labels: list[Tensor] = []
    finite_flags: list[Tensor] = []
    with torch.no_grad():
        for token_ids, padding_mask, batch_labels in loader:
            logits = model(
                token_ids.to(device=device, dtype=torch.int64),
                padding_mask.to(device=device, dtype=torch.bool),
            )
            batch_probabilities = torch.sigmoid(logits)
            finite_flags.extend(
                (
                    torch.isfinite(logits).all(),
                    torch.isfinite(batch_probabilities).all(),
                )
            )
            probabilities.append(batch_probabilities.detach().cpu())
            labels.append(batch_labels.detach().cpu())

    if not _device_flags_pass(finite_flags, device):
        raise TransformerTrainingError("validation output contains a nonfinite value")
    probability_array = torch.cat(probabilities).numpy().astype(np.float64, copy=False)
    label_array = torch.cat(labels).numpy().astype(np.int8, copy=False)
    score = float(average_precision_score(label_array, probability_array))
    if not math.isfinite(score):
        raise TransformerTrainingError("validation metric is nonfinite")
    return score, tuple(float(value) for value in probability_array)


def _fit_character_transformer_on_device(
    train_token_ids: Tensor,
    train_padding_mask: Tensor,
    train_labels: Tensor,
    validation_token_ids: Tensor,
    validation_padding_mask: Tensor,
    validation_labels: Tensor,
    *,
    vocabulary_size: int,
    device: torch.device,
    seed: int = SEED,
    epoch_callback: Callable[[_EpochEvidence], None] | None = None,
    checkpoint_callback: Callable[[_CheckpointEvidence], None] | None = None,
) -> TransformerFit:
    if not isinstance(device, torch.device) or device.type not in {"cpu", "mps"}:
        raise TransformerTrainingError("fixture device must be CPU or MPS")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with torch.autocast(device_type=device.type, enabled=False):
                _configure_deterministic_runtime(seed=seed)
                train = _validated_tensor_batch(
                    train_token_ids,
                    train_padding_mask,
                    train_labels,
                    vocabulary_size=vocabulary_size,
                    partition="train",
                )
                validation = _validated_tensor_batch(
                    validation_token_ids,
                    validation_padding_mask,
                    validation_labels,
                    vocabulary_size=vocabulary_size,
                    partition="validation",
                )
                class_weight = positive_class_weight(train.labels)
                train_loader, validation_loader = _build_data_loaders(
                    train, validation, seed=seed
                )

                model = CharacterTransformer(vocabulary_size).to(
                    device=device, dtype=torch.float32
                )
                optimizer = torch.optim.AdamW(
                    model.parameters(),
                    lr=LEARNING_RATE,
                    betas=ADAM_BETAS,
                    eps=ADAM_EPSILON,
                    weight_decay=WEIGHT_DECAY,
                )
                loss_function = nn.BCEWithLogitsLoss(
                    pos_weight=torch.tensor(
                        class_weight, dtype=torch.float32, device=device
                    )
                )
                checkpoint = _BestCheckpoint()
                history: list[EpochRecord] = []
                stopped_early = False

                def retain_checkpoint(phase: str) -> None:
                    if checkpoint_callback is not None:
                        checkpoint_callback(
                            _CheckpointEvidence(
                                phase=phase,
                                model=model,
                                history=tuple(history),
                                best_epoch=checkpoint.best_epoch,
                                best_validation_average_precision=checkpoint.best_score,
                                epochs_completed=len(history),
                                stopped_early=stopped_early,
                                positive_class_weight=class_weight,
                            )
                        )

                for epoch in range(1, MAX_EPOCHS + 1):
                    training_loss = _run_training_epoch(
                        model, train_loader, optimizer, loss_function, device
                    )
                    if not math.isfinite(training_loss):
                        raise TransformerTrainingError("training loss is nonfinite")
                    validation_ap, epoch_probabilities = _evaluate_validation(
                        model, validation_loader, device
                    )
                    if not math.isfinite(validation_ap):
                        raise TransformerTrainingError("validation metric is nonfinite")
                    history.append(EpochRecord(epoch, training_loss, validation_ap))
                    stopped_early = checkpoint.observe(
                        epoch=epoch, score=validation_ap, model=model
                    )
                    if checkpoint.best_epoch == epoch:
                        retain_checkpoint("best_update")
                    if epoch_callback is not None:
                        epoch_callback(_EpochEvidence(history[-1], epoch_probabilities))
                    if stopped_early:
                        break

                checkpoint.restore(model)
                retain_checkpoint("restored_best")
                best_ap, validation_probabilities = _evaluate_validation(
                    model, validation_loader, device
                )
                if checkpoint.best_epoch is None or checkpoint.best_score is None:
                    raise TransformerTrainingError(
                        "no best validation checkpoint exists"
                    )
                if not math.isclose(
                    best_ap, checkpoint.best_score, rel_tol=0.0, abs_tol=1e-12
                ):
                    raise TransformerTrainingError(
                        "restored checkpoint validation metric changed"
                    )
                model = model.to(device="cpu", dtype=torch.float32)
                return TransformerFit(
                    model=model,
                    history=tuple(history),
                    best_epoch=checkpoint.best_epoch,
                    best_validation_average_precision=checkpoint.best_score,
                    epochs_completed=len(history),
                    stopped_early=stopped_early,
                    positive_class_weight=class_weight,
                    validation_probabilities=validation_probabilities,
                )
    except Warning as error:
        raise TransformerTrainingError(
            f"training stopped on warning: {error}"
        ) from error


def fit_character_transformer(
    train_token_ids: Tensor,
    train_padding_mask: Tensor,
    train_labels: Tensor,
    validation_token_ids: Tensor,
    validation_padding_mask: Tensor,
    validation_labels: Tensor,
    *,
    vocabulary_size: int,
) -> TransformerFit:
    """Fit once using the verified PyTorch 2.7.1 float32 MPS runtime."""
    return _fit_character_transformer_on_device(
        train_token_ids,
        train_padding_mask,
        train_labels,
        validation_token_ids,
        validation_padding_mask,
        validation_labels,
        vocabulary_size=vocabulary_size,
        device=_resolve_training_device(),
    )
