"""No-fit, ordered scoring from an already verified transformer bundle.

These interfaces do not authorize a research-data run or establish equivalence
between numerical backends or batch sizes. The caller must freeze those choices.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch

from . import character_sequence, character_transformer, fixed_cascade
from .transformer_inference import LoadedTransformerCascade, TransformerInferenceError


@dataclass(frozen=True)
class OfflineCascadeScores:
    """Positional outputs; the logical mask is not a count of physical model calls."""

    stage1_probabilities: tuple[float, ...]
    transformer_probabilities: tuple[float, ...]
    probabilities: tuple[float, ...]
    decisions: tuple[int, ...]
    logical_stage2_mask: tuple[bool, ...]
    stage1_scoring_audit: dict[str, object]


def _same_device(actual: torch.device, expected: torch.device) -> bool:
    return actual.type == expected.type and (
        actual.index == expected.index
        or (expected.index is None and actual.index in (None, 0))
    )


def _validate_model(loaded: LoadedTransformerCascade) -> None:
    if type(loaded) is not LoadedTransformerCascade:
        raise TransformerInferenceError("loaded must be a LoadedTransformerCascade")
    _validate_transformer_state(loaded._model, loaded.vocabulary, loaded.device)
    try:
        fixed_cascade._threshold(loaded.stage1_threshold, "stage1_threshold")
        fixed_cascade._threshold(loaded.transformer_threshold, "transformer_threshold")
        half_width = fixed_cascade._finite_number(loaded.half_width, "half_width")
        if half_width < 0:
            raise fixed_cascade.FixedCascadeError("half_width must be nonnegative")
    except fixed_cascade.FixedCascadeError as exc:
        raise TransformerInferenceError(str(exc)) from exc


def _validate_transformer_state(model, vocabulary, device) -> None:
    """Validate numerical state without claiming a primary bundle identity."""
    if (
        type(model) is not character_transformer.CharacterTransformer
        or type(vocabulary) is not character_sequence.CharacterVocabulary
        or model.vocabulary_size != vocabulary.size
    ):
        raise TransformerInferenceError("loaded transformer and vocabulary disagree")
    if not isinstance(device, torch.device) or device.type not in (
        "cpu",
        "mps",
    ):
        raise TransformerInferenceError("loaded transformer device is invalid")
    if any(module.training for module in model.modules()):
        raise TransformerInferenceError(
            "loaded transformer must already be in eval mode"
        )
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise TransformerInferenceError("loaded transformer parameters must be frozen")
    for tensor in model.state_dict().values():
        if tensor.dtype != torch.float32:
            raise TransformerInferenceError(
                "loaded transformer must retain float32 state"
            )
        if not _same_device(tensor.device, device):
            raise TransformerInferenceError(
                "loaded transformer device does not match its state"
            )


def _prepare_inputs(
    loaded: LoadedTransformerCascade, raw_urls: object, batch_size: int
) -> tuple[tuple[str, ...], torch.Tensor, torch.Tensor]:
    _validate_model(loaded)
    if type(batch_size) is not int or batch_size <= 0:
        raise TransformerInferenceError("batch_size must be a positive integer")
    return _prepare_character_inputs(loaded.vocabulary, raw_urls)


def _prepare_character_inputs(
    vocabulary: character_sequence.CharacterVocabulary, raw_urls: object
) -> tuple[tuple[str, ...], torch.Tensor, torch.Tensor]:
    if isinstance(raw_urls, (str, bytes)) or not isinstance(raw_urls, Sequence):
        raise TransformerInferenceError("raw_urls must be a nonempty ordered sequence")
    urls = tuple(raw_urls)
    if not urls:
        raise TransformerInferenceError("raw_urls must be a nonempty ordered sequence")
    shape = (len(urls), character_sequence.MAX_SEQUENCE_LENGTH)
    token_ids = np.empty(shape, dtype=np.int64)
    padding_mask = np.empty(shape, dtype=np.bool_)
    for index, url in enumerate(urls):
        try:
            encoded = character_sequence.encode_character_url(url, vocabulary)
        except character_sequence.CharacterSequenceError as exc:
            raise TransformerInferenceError(
                f"raw_urls[{index}] is invalid under the frozen encoder rules"
            ) from exc
        token_ids[index] = encoded.token_ids
        padding_mask[index] = encoded.padding_mask
    return urls, torch.from_numpy(token_ids), torch.from_numpy(padding_mask)


def _validate_batch_output(
    values: object, *, rows: int, device: torch.device, field: str
) -> torch.Tensor:
    if (
        not isinstance(values, torch.Tensor)
        or values.shape != (rows,)
        or values.dtype != torch.float32
        or not _same_device(values.device, device)
    ):
        raise TransformerInferenceError(
            f"transformer {field} have invalid shape, dtype or device"
        )
    if not bool(torch.isfinite(values).all()):
        raise TransformerInferenceError(f"transformer {field} contain nonfinite values")
    if field == "probabilities" and bool(((values < 0) | (values > 1)).any()):
        raise TransformerInferenceError("transformer probabilities are outside [0, 1]")
    return values


def _score_prepared(
    loaded: LoadedTransformerCascade,
    token_ids: torch.Tensor,
    padding_mask: torch.Tensor,
    batch_size: int,
) -> tuple[float, ...]:
    return _score_model_prepared(
        loaded._model, loaded.device, token_ids, padding_mask, batch_size
    )


def _score_model_prepared(
    model: character_transformer.CharacterTransformer,
    device: torch.device,
    token_ids: torch.Tensor,
    padding_mask: torch.Tensor,
    batch_size: int,
) -> tuple[float, ...]:
    probabilities: list[float] = []
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with (
                torch.inference_mode(),
                torch.autocast(device_type=device.type, enabled=False),
            ):
                # Error mode is process-wide; frozen inference needs no reseeding.
                torch.use_deterministic_algorithms(True, warn_only=False)
                for start in range(0, len(token_ids), batch_size):
                    tokens = token_ids[start : start + batch_size].to(device)
                    mask = padding_mask[start : start + batch_size].to(device)
                    logits = _validate_batch_output(
                        model(tokens, mask),
                        rows=len(tokens),
                        device=device,
                        field="logits",
                    )
                    # Keep validation's float32 sigmoid on the model device.
                    batch = _validate_batch_output(
                        torch.sigmoid(logits),
                        rows=len(tokens),
                        device=device,
                        field="probabilities",
                    )
                    probabilities.extend(
                        batch.detach().cpu().numpy().astype(np.float64).tolist()
                    )
    except (
        RuntimeError,
        Warning,
        character_transformer.TransformerTrainingError,
    ) as exc:
        raise TransformerInferenceError(f"transformer inference failed: {exc}") from exc
    return tuple(probabilities)


def score_transformer_urls(
    loaded: LoadedTransformerCascade,
    raw_urls: object,
    *,
    batch_size: int = character_transformer.VALIDATION_BATCH_SIZE,
) -> tuple[float, ...]:
    """Return probabilities in input order without changing weights or thresholds.

    All rows are validated before the first forward call. This enables PyTorch's
    process-wide deterministic error mode without reseeding any random generator.
    Transformer execution rejects warnings and disables autocast, as validation did.
    Row identifiers and their manifest bindings remain the caller's responsibility.
    """
    _, token_ids, padding_mask = _prepare_inputs(loaded, raw_urls, batch_size)
    return _score_prepared(loaded, token_ids, padding_mask, batch_size)


def score_offline_full_cascade(
    loaded: LoadedTransformerCascade,
    raw_urls: object,
    *,
    batch_size: int = character_transformer.VALIDATION_BATCH_SIZE,
) -> OfflineCascadeScores:
    """Score every row with both models, then apply the frozen inclusive band.

    The mask records logical selection only. Every URL actually receives a
    transformer score; this is not selective serving or evidence of HTTP savings.
    Original strings reach stage one unchanged; only the transformer encoder
    applies its frozen normalization, matching the development scoring path.
    """
    urls, token_ids, padding_mask = _prepare_inputs(loaded, raw_urls, batch_size)
    try:
        stage1, audit = fixed_cascade.score_logistic_l1_authoritative(
            loaded.stage1_model, urls
        )
        validated = fixed_cascade._probability_vector(stage1, "stage-one probabilities")
        if len(validated) != len(urls):
            raise fixed_cascade.FixedCascadeError(
                "stage-one probabilities have incorrect length"
            )
        stage1 = tuple(float(value) for value in validated)
        transformer = _score_prepared(loaded, token_ids, padding_mask, batch_size)
        cascade = fixed_cascade.score_fixed_cascade(
            stage1,
            transformer,
            stage1_threshold=loaded.stage1_threshold,
            transformer_threshold=loaded.transformer_threshold,
            half_width=loaded.half_width,
        )
    except fixed_cascade.FixedCascadeError as exc:
        raise TransformerInferenceError(str(exc)) from exc
    return OfflineCascadeScores(
        stage1_probabilities=stage1,
        transformer_probabilities=transformer,
        probabilities=cascade.probabilities,
        decisions=cascade.decisions,
        logical_stage2_mask=cascade.transformer_invoked,
        stage1_scoring_audit=audit,
    )
