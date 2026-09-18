import warnings
from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from test_transformer_inference import _build_fixture, _load_fixture

from automated_phishing_detection import (
    character_sequence,
    character_transformer,
    fixed_cascade,
    paired_evaluation,
    transformer_scoring,
)
from automated_phishing_detection.phiusiil import canonicalize_url
from automated_phishing_detection.transformer_inference import TransformerInferenceError

URLS = ("https://safe.example/account", "https://signin.example/verify")


@pytest.fixture
def loaded(tmp_path):
    return _load_fixture(_build_fixture(tmp_path))


@pytest.fixture(autouse=True)
def restore_deterministic_mode():
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    yield
    torch.use_deterministic_algorithms(enabled, warn_only=warn_only)


def fail_if_called(*args, **kwargs):
    raise AssertionError("this path must not be called")


def test_known_answer_uses_frozen_model_without_training(loaded, monkeypatch):
    monkeypatch.setattr(
        character_transformer, "fit_character_transformer", fail_if_called
    )
    monkeypatch.setattr(loaded._model, "train", fail_if_called)
    before = {name: value.clone() for name, value in loaded._model.state_dict().items()}
    probabilities = transformer_scoring.score_transformer_urls(loaded, URLS)
    expected = torch.sigmoid(torch.tensor([0.6071698665618896, 0.6393680572509766]))
    assert probabilities == pytest.approx(expected.tolist(), rel=1e-6, abs=1e-6)
    assert type(probabilities) is tuple
    assert all(type(value) is float for value in probabilities)
    for name, value in loaded._model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)


def test_batches_keep_input_order_and_tail_and_use_inference_mode(loaded, monkeypatch):
    urls = tuple(f"https://safe.example/{index}" for index in range(5))
    expected_tokens = [
        character_sequence.encode_character_url(url, loaded.vocabulary).token_ids
        for url in urls
    ]
    observed_tokens = []
    batch_sizes = []

    def forward(token_ids, padding_mask):
        assert torch.is_inference_mode_enabled()
        assert not torch.is_grad_enabled()
        assert token_ids.dtype == torch.int64
        assert padding_mask.dtype == torch.bool
        assert token_ids.device == loaded.device
        batch_sizes.append(len(token_ids))
        start = len(observed_tokens)
        observed_tokens.extend(tuple(row) for row in token_ids.tolist())
        return torch.arange(start, start + len(token_ids), dtype=torch.float32)

    monkeypatch.setattr(loaded._model, "forward", forward)
    probabilities = transformer_scoring.score_transformer_urls(
        loaded, urls, batch_size=2
    )
    assert batch_sizes == [2, 2, 1]
    assert observed_tokens == expected_tokens
    assert probabilities == tuple(
        torch.sigmoid(torch.arange(5, dtype=torch.float32)).tolist()
    )


def test_default_batch_matches_original_validation_size(loaded, monkeypatch):
    batch_sizes = []

    def forward(token_ids, padding_mask):
        batch_sizes.append(len(token_ids))
        return torch.zeros(len(token_ids), dtype=torch.float32)

    monkeypatch.setattr(loaded._model, "forward", forward)
    result = transformer_scoring.score_transformer_urls(loaded, URLS * 257)
    assert batch_sizes == [512, 2]
    assert result == (0.5,) * 514


def test_deterministic_error_mode_is_enabled_without_reseeding(loaded, monkeypatch):
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.manual_seed(17)
    before = torch.random.get_rng_state().clone()

    def forward(token_ids, padding_mask):
        assert torch.are_deterministic_algorithms_enabled()
        assert not torch.is_deterministic_algorithms_warn_only_enabled()
        return torch.zeros(len(token_ids), dtype=torch.float32)

    monkeypatch.setattr(loaded._model, "forward", forward)
    transformer_scoring.score_transformer_urls(loaded, URLS)
    assert torch.equal(before, torch.random.get_rng_state())


@pytest.mark.parametrize("category", [RuntimeWarning, UserWarning, DeprecationWarning])
def test_transformer_warnings_fail_and_restore_caller_filters(
    loaded, monkeypatch, category
):
    def forward(token_ids, padding_mask):
        warnings.warn("synthetic execution warning", category, stacklevel=2)
        return torch.zeros(len(token_ids), dtype=torch.float32)

    monkeypatch.setattr(loaded._model, "forward", forward)
    with warnings.catch_warnings(record=True) as observed:
        warnings.simplefilter("always")
        previous_filters = list(warnings.filters)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            with pytest.raises(
                TransformerInferenceError, match="synthetic execution warning"
            ):
                transformer_scoring.score_transformer_urls(loaded, URLS)
            assert torch.is_autocast_enabled("cpu")
            assert torch.get_autocast_dtype("cpu") == torch.bfloat16
        assert warnings.filters == previous_filters
    assert observed == []


def test_transformer_disables_autocast_and_restores_outer_context(loaded, monkeypatch):
    observed_modes = []
    before = torch.is_autocast_enabled("cpu")
    before_dtype = torch.get_autocast_dtype("cpu")

    def forward(token_ids, padding_mask):
        observed_modes.append(torch.is_autocast_enabled("cpu"))
        left = torch.full((len(token_ids), 2), 0.123456, dtype=torch.float32)
        right = torch.full((2, 1), 0.345678, dtype=torch.float32)
        return (left @ right).squeeze(-1).float()

    monkeypatch.setattr(loaded._model, "forward", forward)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        assert torch.is_autocast_enabled("cpu")
        probabilities = transformer_scoring.score_transformer_urls(loaded, URLS)
        assert torch.is_autocast_enabled("cpu")
        assert torch.get_autocast_dtype("cpu") == torch.bfloat16
    assert observed_modes == [False]
    assert torch.is_autocast_enabled("cpu") == before
    assert torch.get_autocast_dtype("cpu") == before_dtype
    expected_logits = torch.full((2, 2), 0.123456) @ torch.full((2, 1), 0.345678)
    assert probabilities == tuple(torch.sigmoid(expected_logits.squeeze(-1)).tolist())


def test_offline_does_not_apply_transformer_warning_policy_to_stage1(
    loaded, monkeypatch
):
    def authoritative(*args):
        warnings.warn("separate stage-one audit policy", UserWarning, stacklevel=2)
        return (0.1, 0.9), {"synthetic_audit": True}

    monkeypatch.setattr(fixed_cascade, "score_logistic_l1_authoritative", authoritative)
    monkeypatch.setattr(loaded._model, "forward", lambda *args: torch.zeros(2))
    with pytest.warns(UserWarning, match="separate stage-one audit policy"):
        result = transformer_scoring.score_offline_full_cascade(loaded, URLS)
    assert result.stage1_scoring_audit == {"synthetic_audit": True}


def test_existing_encoder_controls_unicode_unknown_tokens_and_truncation(
    loaded, monkeypatch
):
    urls = (
        "https://safe.example/" + "z" * 280 + "0123456789",
        "https://safe.example/\u00e9",
    )
    expected = [
        character_sequence.encode_character_url(url, loaded.vocabulary) for url in urls
    ]
    assert expected[0].length == 256
    assert character_sequence.UNK_ID in expected[0].token_ids

    def forward(token_ids, padding_mask):
        assert token_ids.tolist() == [list(row.token_ids) for row in expected]
        assert padding_mask.tolist() == [list(row.padding_mask) for row in expected]
        return torch.zeros(len(token_ids), dtype=torch.float32)

    monkeypatch.setattr(loaded._model, "forward", forward)
    assert transformer_scoring.score_transformer_urls(loaded, urls) == (0.5, 0.5)


def test_raw_urls_remain_original_for_stage1_and_use_encoder_normalization(
    loaded, monkeypatch
):
    raw_urls = (
        "HTTPS://SAFE.example:443",
        "https://safe.example",
        "https://safe.example/\u00e9",
    )
    normalized_urls = tuple(canonicalize_url(url) for url in raw_urls)
    expected_stage1, _ = fixed_cascade.score_logistic_l1_authoritative(
        loaded.stage1_model, raw_urls
    )
    normalized_stage1, _ = fixed_cascade.score_logistic_l1_authoritative(
        loaded.stage1_model, normalized_urls
    )
    assert expected_stage1[:2] != normalized_stage1[:2]
    original = fixed_cascade.score_logistic_l1_authoritative
    observed = []

    def authoritative(model, urls):
        observed.append(urls)
        return original(model, urls)

    expected = [
        character_sequence.encode_character_url(url, loaded.vocabulary)
        for url in raw_urls
    ]

    def forward(token_ids, padding_mask):
        assert token_ids.tolist() == [list(row.token_ids) for row in expected]
        return torch.zeros(len(token_ids), dtype=torch.float32)

    monkeypatch.setattr(fixed_cascade, "score_logistic_l1_authoritative", authoritative)
    monkeypatch.setattr(loaded._model, "forward", forward)
    result = transformer_scoring.score_offline_full_cascade(loaded, raw_urls=raw_urls)
    assert observed == [raw_urls]
    assert result.stage1_probabilities == expected_stage1
    assert transformer_scoring.score_transformer_urls(loaded, raw_urls=raw_urls) == (
        0.5,
        0.5,
        0.5,
    )


@pytest.mark.parametrize(
    "urls",
    [
        [],
        (),
        None,
        URLS[0],
        set(URLS),
        dict.fromkeys(URLS),
        [None],
        [""],
        ["not a URL"],
        [URLS[0], "https://safe.example/bad path"],
    ],
)
def test_rejects_invalid_inputs_before_any_forward(loaded, monkeypatch, urls):
    monkeypatch.setattr(loaded._model, "forward", fail_if_called)
    with pytest.raises(TransformerInferenceError, match="raw_urls"):
        transformer_scoring.score_transformer_urls(loaded, urls)


@pytest.mark.parametrize("batch_size", [0, -1, True, 1.5, None])
def test_rejects_invalid_batch_size_before_forward(loaded, monkeypatch, batch_size):
    monkeypatch.setattr(loaded._model, "forward", fail_if_called)
    with pytest.raises(TransformerInferenceError, match="batch_size"):
        transformer_scoring.score_transformer_urls(loaded, URLS, batch_size=batch_size)


@pytest.mark.parametrize(
    "change", ["training", "child_training", "grad", "dtype", "device"]
)
def test_rejects_mutated_model_before_forward(loaded, monkeypatch, change):
    if change == "training":
        loaded._model.train()
    elif change == "child_training":
        loaded._model.encoder.layers[0].train()
    elif change == "grad":
        next(loaded._model.parameters()).requires_grad_(True)
    elif change == "dtype":
        loaded._model.double()
    else:
        loaded = replace(loaded, device=torch.device("mps"))
    monkeypatch.setattr(loaded._model, "forward", fail_if_called)
    with pytest.raises(TransformerInferenceError):
        transformer_scoring.score_transformer_urls(loaded, URLS)


@pytest.mark.parametrize(
    "output",
    [
        torch.tensor([float("nan"), 0.0]),
        torch.tensor([float("inf"), 0.0]),
        torch.tensor([-float("inf"), 0.0]),
        torch.tensor([[0.0], [0.0]]),
        torch.tensor([0.0]),
        torch.tensor([0, 0]),
        torch.tensor([0.0, 0.0], dtype=torch.float64),
        [0.0, 0.0],
    ],
)
def test_rejects_invalid_logits(loaded, monkeypatch, output):
    monkeypatch.setattr(loaded._model, "forward", lambda *args: output)
    with pytest.raises(TransformerInferenceError, match="logits"):
        transformer_scoring.score_transformer_urls(loaded, URLS)


def test_rejects_nonfinite_probabilities(loaded, monkeypatch):
    monkeypatch.setattr(loaded._model, "forward", lambda *args: torch.zeros(2))
    monkeypatch.setattr(
        torch, "sigmoid", lambda logits: torch.full_like(logits, float("nan"))
    )
    with pytest.raises(TransformerInferenceError, match="probabilities"):
        transformer_scoring.score_transformer_urls(loaded, URLS)


def test_translates_runtime_failure_without_fallback(loaded, monkeypatch):
    def forward(*args):
        raise RuntimeError("nondeterministic kernel")

    monkeypatch.setattr(loaded._model, "forward", forward)
    with pytest.raises(TransformerInferenceError, match="nondeterministic kernel"):
        transformer_scoring.score_transformer_urls(loaded, URLS)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("stage1_threshold", float("nan")),
        ("transformer_threshold", -0.1),
        ("half_width", -0.1),
        ("half_width", True),
    ],
)
def test_rejects_invalid_fixed_settings_before_forward(
    loaded, monkeypatch, field, value
):
    loaded = replace(loaded, **{field: value})
    monkeypatch.setattr(loaded._model, "forward", fail_if_called)
    with pytest.raises(TransformerInferenceError, match=field):
        transformer_scoring.score_transformer_urls(loaded, URLS)


def test_rejects_unloaded_object():
    with pytest.raises(TransformerInferenceError, match="LoadedTransformerCascade"):
        transformer_scoring.score_transformer_urls(object(), URLS)


def test_offline_cascade_preserves_authoritative_stage1_and_audit(loaded, monkeypatch):
    monkeypatch.setattr(StandardScaler, "fit", fail_if_called)
    monkeypatch.setattr(LogisticRegression, "fit", fail_if_called)
    monkeypatch.setattr(fixed_cascade.PortableLogisticL1, "score_urls", fail_if_called)
    expected_stage1, expected_audit = fixed_cascade.score_logistic_l1_authoritative(
        loaded.stage1_model, URLS
    )
    observed = []
    original = fixed_cascade.score_logistic_l1_authoritative

    def authoritative(model, urls):
        observed.append((model, urls))
        return original(model, urls)

    monkeypatch.setattr(fixed_cascade, "score_logistic_l1_authoritative", authoritative)
    result = transformer_scoring.score_offline_full_cascade(loaded, URLS)
    assert observed == [(loaded.stage1_model, URLS)]
    assert result.stage1_probabilities == expected_stage1
    assert result.stage1_scoring_audit == expected_audit
    assert len(result.transformer_probabilities) == len(URLS)


def test_offline_inclusive_band_is_logical_not_skipped_transformer_work(
    loaded, monkeypatch
):
    loaded = replace(loaded, half_width=0.125)
    urls = tuple(f"https://safe.example/{index}" for index in range(6))
    stage1 = (0.1, 0.375, 0.5, 0.625, 0.9, 0.374)
    logits = torch.tensor([0.0, -2.0, 0.0, 2.0, 0.0, 2.0])
    transformer = tuple(torch.sigmoid(logits).tolist())
    observed_rows = []
    monkeypatch.setattr(
        fixed_cascade, "score_logistic_l1_authoritative", lambda *args: (stage1, {})
    )

    def forward(token_ids, padding_mask):
        start = len(observed_rows)
        observed_rows.extend(token_ids.tolist())
        return logits[start : start + len(token_ids)]

    monkeypatch.setattr(loaded._model, "forward", forward)
    result = transformer_scoring.score_offline_full_cascade(loaded, urls, batch_size=2)
    assert len(observed_rows) == len(urls)
    assert result.transformer_probabilities == transformer
    assert result.logical_stage2_mask == (False, True, True, True, False, False)
    assert result.decisions == (0, 0, 1, 1, 1, 0)
    assert result.probabilities == (stage1[0], *transformer[1:4], *stage1[4:])
    assert loaded.stage1_threshold == 0.5
    assert loaded.transformer_threshold == 0.5
    assert loaded.half_width == 0.125
    with pytest.raises(FrozenInstanceError):
        loaded.stage1_threshold = 0.1
    with pytest.raises(FrozenInstanceError):
        result.decisions = ()


def test_offline_rejects_invalid_rows_before_either_scorer(loaded, monkeypatch):
    monkeypatch.setattr(loaded._model, "forward", fail_if_called)
    monkeypatch.setattr(
        fixed_cascade, "score_logistic_l1_authoritative", fail_if_called
    )
    with pytest.raises(TransformerInferenceError, match="raw_urls"):
        transformer_scoring.score_offline_full_cascade(loaded, [URLS[0], "bad"])


@pytest.mark.parametrize("stage1", [(0.1,), (0.1, np.nan), (0.1, 1.1)])
def test_offline_rejects_invalid_stage1_before_transformer(loaded, monkeypatch, stage1):
    monkeypatch.setattr(loaded._model, "forward", fail_if_called)
    monkeypatch.setattr(
        fixed_cascade, "score_logistic_l1_authoritative", lambda *args: (stage1, {})
    )
    with pytest.raises(TransformerInferenceError, match="stage.one"):
        transformer_scoring.score_offline_full_cascade(loaded, URLS)


def test_synthetic_offline_scores_bind_to_paired_record_ids(loaded):
    """An interface handoff test, not a research endpoint or label assessment."""
    scores = transformer_scoring.score_offline_full_cascade(loaded, URLS)
    domains = ("safe.example", "signin.example")
    records = tuple(
        paired_evaluation.EvaluationRecord(f"synthetic-{index}", domain, 1)
        for index, domain in enumerate(domains)
    )
    stage1_decisions = tuple(
        int(probability >= loaded.stage1_threshold)
        for probability in scores.stage1_probabilities
    )
    candidate = tuple(
        paired_evaluation.BinaryPrediction(record.record_id, decision)
        for record, decision in zip(records, scores.decisions, strict=True)
    )
    reference = tuple(
        paired_evaluation.BinaryPrediction(record.record_id, decision)
        for record, decision in zip(records, stage1_decisions, strict=True)
    )
    result = paired_evaluation.paired_recall_difference(records, candidate, reference)
    assert result.status == "estimated"
    assert result.domain_count == 2
    assert result.positive_count == 2
    assert result.candidate_true_positives == sum(scores.decisions)
    assert result.reference_true_positives == sum(stage1_decisions)
    assert result.estimate == (sum(scores.decisions) - sum(stage1_decisions)) / 2
