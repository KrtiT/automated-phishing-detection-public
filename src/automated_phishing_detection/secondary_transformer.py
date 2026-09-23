"""Bounded secondary seed methods, not an authenticated research runner.

Future fits require the caller's execution-binding-v2 one-thread runtime receipt.
Historical seed 42 is a runtime-confounded comparator, never a new primary fit or
a candidate for best-seed selection. No function here accesses research files.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from hashlib import sha256
from typing import ClassVar

import torch
from torch import Tensor

from . import (
    baselines,
    character_sequence,
    fixed_cascade,
    transformer_inference,
    transformer_pipeline,
    transformer_scoring,
)
from . import character_transformer as trainer
from .url_features import FeatureExtractionError, extract_url_features

SECONDARY_SEEDS = (43, 44, 45, 46)
MODEL_IDENTITY = "secondary-transformer-v1"


class SecondaryTransformerError(ValueError):
    """A supplied secondary-method input violates its bounded contract."""


@dataclass(frozen=True)
class SecondaryTransformerFit:
    seed: int
    fit: trainer.TransformerFit


@dataclass(frozen=True)
class SecondaryEpoch:
    """Ordered checkpoint-selection scores, not later singleton calibration scores."""

    record: trainer.EpochRecord
    validation_probabilities: tuple[float, ...]
    checkpoint_batch_size: int = trainer.VALIDATION_BATCH_SIZE


@dataclass(frozen=True)
class SecondaryCheckpoint:
    """Numeric retained weights; the caller must durably record each event."""

    seed: int
    phase: str
    weights_bytes: bytes
    history: tuple[trainer.EpochRecord, ...]
    best_epoch: int
    best_validation_average_precision: float
    epochs_completed: int
    stopped_early: bool
    positive_class_weight: float


def _secondary_seed(seed: object) -> int:
    if type(seed) is not int or seed not in SECONDARY_SEEDS:
        raise SecondaryTransformerError("seed must be an exact integer in 43..46")
    return seed


def fit_secondary_transformer(
    train_token_ids: Tensor,
    train_padding_mask: Tensor,
    train_labels: Tensor,
    validation_token_ids: Tensor,
    validation_padding_mask: Tensor,
    validation_labels: Tensor,
    *,
    vocabulary_size: int,
    seed: int,
    epoch_callback: Callable[[SecondaryEpoch], None],
    checkpoint_callback: Callable[[SecondaryCheckpoint], None],
) -> SecondaryTransformerFit:
    """Fit a secondary seed on the frozen 2.7.1 MPS procedure, with no retries.

    Each qualifying checkpoint is retained before the epoch callback and next
    epoch. Restored-best retention precedes final validation scoring/AP checks.
    Epoch evidence retains the same ordered probabilities used to calculate AP
    with checkpoint batch size 512; it never adds a scoring pass.
    Callback errors abort immediately. Callbacks provide evidence seams, not
    durable-write or runtime-authentication guarantees; those belong to a runner.
    """
    seed = _secondary_seed(seed)
    if not callable(epoch_callback) or not callable(checkpoint_callback):
        raise SecondaryTransformerError("both evidence callbacks must be callable")

    def retain_epoch(evidence: trainer._EpochEvidence) -> None:
        epoch_callback(
            SecondaryEpoch(evidence.record, evidence.validation_probabilities)
        )

    def retain(evidence: trainer._CheckpointEvidence) -> None:
        checkpoint_callback(
            SecondaryCheckpoint(
                seed=seed,
                phase=evidence.phase,
                weights_bytes=serialize_secondary_weights(evidence.model),
                history=evidence.history,
                best_epoch=evidence.best_epoch,
                best_validation_average_precision=evidence.best_validation_average_precision,
                epochs_completed=evidence.epochs_completed,
                stopped_early=evidence.stopped_early,
                positive_class_weight=evidence.positive_class_weight,
            )
        )

    fitted = trainer._fit_character_transformer_on_device(
        train_token_ids,
        train_padding_mask,
        train_labels,
        validation_token_ids,
        validation_padding_mask,
        validation_labels,
        vocabulary_size=vocabulary_size,
        device=trainer._resolve_training_device(),
        seed=seed,
        epoch_callback=retain_epoch,
        checkpoint_callback=retain,
    )
    return SecondaryTransformerFit(seed=seed, fit=fitted)


@dataclass(frozen=True)
class SecondaryTransformerModel:
    """Numeric secondary state, not an authenticated primary artifact bundle."""

    model_identity: ClassVar[str] = MODEL_IDENTITY
    seed: int
    vocabulary: character_sequence.CharacterVocabulary
    device: torch.device
    weights_sha256: str
    vocabulary_sha256: str
    _model: trainer.CharacterTransformer


@dataclass(frozen=True)
class SecondaryStage1Scores:
    probabilities: tuple[float, ...]
    scoring_audits: tuple[dict[str, object], ...]


def serialize_secondary_weights(model: trainer.CharacterTransformer) -> bytes:
    """Return canonical float32 numeric NPZ bytes, never a pickle artifact."""
    if type(model) is not trainer.CharacterTransformer or any(
        value.dtype != torch.float32 for value in model.state_dict().values()
    ):
        raise SecondaryTransformerError("model must be a float32 CharacterTransformer")
    try:
        content, _ = transformer_pipeline._serialize_state_dict_npz(model)
    except transformer_pipeline.TransformerPipelineError as exc:
        raise SecondaryTransformerError(str(exc)) from exc
    return content


def load_secondary_transformer_bytes(
    weights_bytes: bytes,
    vocabulary_bytes: bytes,
    *,
    seed: int,
    device: torch.device,
) -> SecondaryTransformerModel:
    """Strictly load caller-supplied bytes without fitting or primary provenance.

    Digests identify supplied bytes only. Binding them to the methods supplement,
    fitted seed, runtime receipt, and input manifests is a later runner obligation.
    CPU supports fixtures; choosing MPS here does not authenticate a research run.
    """
    seed = _secondary_seed(seed)
    if type(weights_bytes) is not bytes or type(vocabulary_bytes) is not bytes:
        raise SecondaryTransformerError("weights and vocabulary must be exact bytes")
    if (
        type(device) is not torch.device
        or device.type not in {"cpu", "mps"}
        or device.index is not None
    ):
        raise SecondaryTransformerError("device must be unindexed CPU or MPS")
    try:
        vocabulary = transformer_inference._load_vocabulary(vocabulary_bytes)
        model = transformer_inference._load_model(
            weights_bytes, vocabulary.size, device
        )
    except transformer_inference.TransformerInferenceError as exc:
        raise SecondaryTransformerError(str(exc)) from exc
    return SecondaryTransformerModel(
        seed=seed,
        vocabulary=vocabulary,
        device=device,
        weights_sha256=sha256(weights_bytes).hexdigest(),
        vocabulary_sha256=sha256(vocabulary_bytes).hexdigest(),
        _model=model,
    )


def score_secondary_transformer_urls(
    loaded: SecondaryTransformerModel, raw_urls: object
) -> tuple[float, ...]:
    """Score all supplied URLs in order, exactly one model call per URL."""
    if type(loaded) is not SecondaryTransformerModel:
        raise SecondaryTransformerError("loaded must be a SecondaryTransformerModel")
    _secondary_seed(loaded.seed)
    try:
        transformer_scoring._validate_transformer_state(
            loaded._model, loaded.vocabulary, loaded.device
        )
        _, tokens, mask = transformer_scoring._prepare_character_inputs(
            loaded.vocabulary, raw_urls
        )
        return transformer_scoring._score_model_prepared(
            loaded._model, loaded.device, tokens, mask, 1
        )
    except transformer_inference.TransformerInferenceError as exc:
        raise SecondaryTransformerError(str(exc)) from exc


def score_secondary_stage1_urls(
    model: fixed_cascade.PortableLogisticL1, raw_urls: object
) -> SecondaryStage1Scores:
    """Score unchanged raw strings via authoritative singleton calls and audits."""
    if (
        isinstance(raw_urls, (str, bytes))
        or not isinstance(raw_urls, Sequence)
        or not raw_urls
    ):
        raise SecondaryTransformerError("raw_urls must be a nonempty ordered sequence")
    urls = tuple(raw_urls)
    try:
        for url in urls:
            extract_url_features(url)
        probabilities, audits = [], []
        for url in urls:
            scores, audit = fixed_cascade.score_logistic_l1_authoritative(model, (url,))
            scores = fixed_cascade._probability_vector(scores, "stage1 probabilities")
            if scores.size != 1:
                raise SecondaryTransformerError(
                    "stage1 singleton must return one probability"
                )
            probabilities.append(float(scores[0]))
            audits.append(audit)
    except (FeatureExtractionError, fixed_cascade.FixedCascadeError) as exc:
        raise SecondaryTransformerError(str(exc)) from exc
    return SecondaryStage1Scores(tuple(probabilities), tuple(audits))


def calibrate_secondary_seed(
    model: fixed_cascade.PortableLogisticL1,
    stage1_scores: object,
    transformer_scores: object,
    labels: object,
) -> dict[str, object]:
    """Keep stage one's historical cutoff; select this seed's threshold and band.

    Stage-one current counts are descriptive, not a replacement historical
    binding. Class totals are checked; exact row/label ordering and validation
    manifest identity remain the caller's responsibility. This secondary result
    does not select a best seed or erase historical seed-42 runtime confounding.
    """
    try:
        if (
            type(model) is not fixed_cascade.PortableLogisticL1
            or model._loader_marker is not fixed_cascade._LOADED_ARTIFACT_MARKER
        ):
            raise SecondaryTransformerError(
                "model must be a loaded PortableLogisticL1 artifact"
            )
        stage1 = fixed_cascade._probability_vector(stage1_scores, "stage1_scores")
        transformer = fixed_cascade._probability_vector(
            transformer_scores, "transformer_scores"
        )
        if stage1.shape != transformer.shape:
            raise SecondaryTransformerError(
                "stage1 and transformer scores must have equal shape"
            )
        labels = fixed_cascade._binary_labels(labels, expected_shape=stage1.shape)
        historical = fixed_cascade._validate_threshold_record(
            model.validation_threshold_record, "historical_threshold"
        )
        current = None
        if historical["status"] == "selected":
            current = fixed_cascade._confusion_counts(
                stage1 >= historical["threshold"], labels
            )
            if any(
                current[key] != historical["counts"][key]
                for key in ("positive", "negative")
            ):
                raise SecondaryTransformerError(
                    "validation class totals differ from historical population"
                )
        transformer_record = baselines.select_validation_threshold(transformer, labels)
        band = fixed_cascade._calibrate_cascade_band(
            stage1, transformer, labels, historical, transformer_record
        )
        return {
            "result_role": "secondary_seed_sensitivity_historical_42_runtime_confounded",
            "stage1": {
                "threshold_source": "fixed_historical_artifact",
                "threshold": historical["threshold"],
                "historical_threshold_record": historical,
                "counts": current,
                "recall": None
                if current is None
                else current["true_positive"] / current["positive"],
                "observed_fpr": None
                if current is None
                else current["false_positive"] / current["negative"],
                "fpr_upper_95": None
                if current is None
                else baselines.clopper_pearson_upper(
                    current["false_positive"], current["negative"]
                ),
            },
            "transformer_threshold": transformer_record,
            "cascade_band": band,
        }
    except (fixed_cascade.FixedCascadeError, baselines.BaselineError) as exc:
        raise SecondaryTransformerError(str(exc)) from exc
