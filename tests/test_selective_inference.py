import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from importlib import import_module
from pathlib import Path

import numpy as np
import pytest
import torch
from test_transformer_inference import _build_fixture, _load_fixture

from automated_phishing_detection import (
    fixed_cascade,
    gmm_monitor,
    policy_replay,
    transformer_scoring,
)
from automated_phishing_detection.transformer_inference import TransformerInferenceError

URL = "HTTPS://safe.example:443"


@pytest.fixture
def module():
    return import_module("automated_phishing_detection.selective_inference")


@pytest.fixture
def loaded(tmp_path):
    return replace(
        _load_fixture(_build_fixture(tmp_path)),
        stage1_threshold=0.5,
        transformer_threshold=0.5,
        half_width=0.125,
    )


@pytest.fixture(autouse=True)
def restore_deterministic_mode():
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    yield
    torch.use_deterministic_algorithms(enabled, warn_only=warn_only)


def fixed_stage1(monkeypatch, probability):
    def score(model, raw_urls):
        assert raw_urls == (URL,)
        return (probability,), {"synthetic": True}

    monkeypatch.setattr(fixed_cascade, "score_logistic_l1_authoritative", score)


def test_outside_band_skips_transformer_and_token_preparation(
    module, loaded, monkeypatch
):
    fixed_stage1(monkeypatch, 0.25)

    def forbidden(*args, **kwargs):
        raise AssertionError("transformer path must be skipped")

    monkeypatch.setattr(transformer_scoring, "score_transformer_urls", forbidden)
    with module.SelectiveCascade(loaded, _fixture_cpu=True) as scorer:
        result = scorer.scan(URL)
        assert result.decision == result.fixed_decision == 0
        assert result.transformer_probability is None
        assert result.band_selected is result.drift_override is False
        assert result.logical_stage2_selected is result.transformer_evaluated is False
        assert scorer.counts.transformer_forward_attempts == 0
        assert scorer.counts.successful_transformer_scores == 0
        assert scorer.counts.completed_requests == 1
        assert scorer.counts.failed_requests == 0


@pytest.mark.parametrize(
    "probability,override,selected",
    [
        (0.375, False, True),
        (0.625, False, True),
        (np.nextafter(0.375, 0), False, False),
        (np.nextafter(0.625, 1), False, False),
        (0.25, True, True),
        (0.5, True, True),
    ],
)
def test_exact_band_edges_and_override_invoke_once(
    module, loaded, monkeypatch, probability, override, selected
):
    fixed_stage1(monkeypatch, probability)
    calls = []

    def forward(tokens, mask):
        calls.append(len(tokens))
        assert torch.is_inference_mode_enabled()
        assert torch.get_num_threads() == 1
        return torch.zeros(1, dtype=torch.float32)

    monkeypatch.setattr(loaded._model, "forward", forward)
    with module.SelectiveCascade(loaded, _fixture_cpu=True) as scorer:
        result = scorer.scan(URL, drift_override=override)
        assert result.logical_stage2_selected is selected
        assert result.transformer_evaluated is selected
        assert result.decision == (1 if selected else int(probability >= 0.5))
        assert scorer.counts.transformer_forward_attempts == int(selected)
        assert scorer.counts.successful_transformer_scores == int(selected)
    assert calls == ([1] if selected else [])


@pytest.mark.parametrize("half_width", [0.0, 1.0])
def test_shared_full_and_selective_paths_are_exact_on_synthetic_rows(
    module, loaded, half_width
):
    loaded = replace(loaded, half_width=half_width)
    urls = (URL, "https://other.example/x?q=1", "http://third.example:80/a")
    with module.SelectiveCascade(loaded, _fixture_cpu=True) as scorer:
        full = tuple(scorer.score_all(url) for url in urls)
        selective = tuple(scorer.scan(url) for url in urls)
        assert scorer.counts.transformer_forward_attempts == len(urls) + sum(
            row.band_selected for row in full
        )
        for left, right in zip(full, selective, strict=True):
            assert left.stage1_probability == right.stage1_probability
            assert left.fixed_decision == right.fixed_decision == right.decision
            assert left.band_selected == right.band_selected
            if right.transformer_evaluated:
                assert left.transformer_probability == right.transformer_probability
            else:
                assert right.transformer_probability is None


def test_failed_forward_counts_attempt_not_success_and_removes_hook(
    module, loaded, monkeypatch
):
    fixed_stage1(monkeypatch, 0.5)
    hooks = len(loaded._model._forward_pre_hooks)

    def fail(tokens, mask):
        raise RuntimeError("synthetic forward failure")

    monkeypatch.setattr(loaded._model, "forward", fail)
    with module.SelectiveCascade(loaded, _fixture_cpu=True) as scorer:
        with pytest.raises(
            TransformerInferenceError, match="synthetic forward failure"
        ):
            scorer.scan(URL)
        assert scorer.counts.transformer_forward_attempts == 1
        assert scorer.counts.successful_transformer_scores == 0
        assert scorer.counts.completed_requests == 0
        assert scorer.counts.failed_requests == 1
        assert len(loaded._model._forward_pre_hooks) == hooks


def test_invalid_output_counts_actual_attempt_but_no_success(
    module, loaded, monkeypatch
):
    fixed_stage1(monkeypatch, 0.5)
    monkeypatch.setattr(
        loaded._model, "forward", lambda tokens, mask: torch.tensor([float("nan")])
    )
    with module.SelectiveCascade(loaded, _fixture_cpu=True) as scorer:
        with pytest.raises(TransformerInferenceError, match="nonfinite"):
            scorer.scan(URL)
        assert scorer.counts.transformer_forward_attempts == 1
        assert scorer.counts.successful_transformer_scores == 0
        assert scorer.counts.failed_requests == 1


@pytest.mark.parametrize("value", [None, "not-a-url", 7])
def test_invalid_url_never_produces_a_decision(module, loaded, value):
    with module.SelectiveCascade(loaded, _fixture_cpu=True) as scorer:
        with pytest.raises(
            (TransformerInferenceError, fixed_cascade.FixedCascadeError)
        ):
            scorer.scan(value)
        assert scorer.counts.transformer_forward_attempts == 0
        assert scorer.counts.completed_requests == 0
        assert scorer.counts.failed_requests == 1


@pytest.mark.parametrize("override", [0, 1, "true", None])
def test_override_requires_boolean(module, loaded, override):
    with module.SelectiveCascade(loaded, _fixture_cpu=True) as scorer:
        with pytest.raises(TransformerInferenceError, match="boolean"):
            scorer.scan(URL, drift_override=override)
        assert scorer.counts.transformer_forward_attempts == 0


def test_session_pins_threads_once_and_restores_them(module, loaded):
    before = torch.get_num_threads()
    with module.SelectiveCascade(loaded, _fixture_cpu=True):
        assert torch.get_num_threads() == 1
    assert torch.get_num_threads() == before


@pytest.mark.parametrize("reported", ["pool", "torch"])
def test_session_refuses_thread_limit_not_honored(
    module, loaded, monkeypatch, reported
):
    with monkeypatch.context() as patch:
        if reported == "pool":
            patch.setattr(
                module.threadpoolctl, "threadpool_info", lambda: [{"num_threads": 2}]
            )
        else:
            patch.setattr(torch, "get_num_threads", lambda: 2)
        with pytest.raises(TransformerInferenceError, match="one numerical thread"):
            with module.SelectiveCascade(loaded, _fixture_cpu=True):
                pass
    with module.SelectiveCascade(loaded, _fixture_cpu=True):
        pass


def test_session_is_exclusive_and_owner_thread_only(module, loaded):
    with module.SelectiveCascade(loaded, _fixture_cpu=True) as scorer:
        with pytest.raises(TransformerInferenceError, match="active"):
            with module.SelectiveCascade(loaded, _fixture_cpu=True):
                pass
        with ThreadPoolExecutor(max_workers=1) as pool:
            with pytest.raises(TransformerInferenceError, match="owner thread"):
                pool.submit(scorer.scan, URL).result()
        assert scorer.counts.completed_requests == scorer.counts.failed_requests == 0
    with pytest.raises(TransformerInferenceError, match="open session"):
        scorer.scan(URL)


def test_official_session_refuses_cpu_fixture(module, loaded):
    with pytest.raises(TransformerInferenceError, match="MPS"):
        with module.SelectiveCascade(loaded):
            pass


def test_runtime_failure_precedes_scoring_and_releases_session(
    module, loaded, monkeypatch
):
    original = gmm_monitor._require_runtime

    def wrong_runtime():
        raise gmm_monitor.GMMMonitorError("wrong numerical runtime")

    monkeypatch.setattr(gmm_monitor, "_require_runtime", wrong_runtime)
    with pytest.raises(TransformerInferenceError, match="wrong numerical runtime"):
        with module.SelectiveCascade(loaded, _fixture_cpu=True):
            pass
    monkeypatch.setattr(gmm_monitor, "_require_runtime", original)
    with module.SelectiveCascade(loaded, _fixture_cpu=True) as scorer:
        assert scorer.counts.transformer_forward_attempts == 0


def test_uncaught_request_failure_restores_context_and_releases_ownership(
    module, loaded
):
    before = torch.get_num_threads()
    with pytest.raises(fixed_cascade.FixedCascadeError):
        with module.SelectiveCascade(loaded, _fixture_cpu=True) as scorer:
            scorer.scan("invalid")
    assert torch.get_num_threads() == before
    assert scorer.counts.failed_requests == 1
    with module.SelectiveCascade(loaded, _fixture_cpu=True):
        pass


def test_failure_during_enter_restores_threads_and_releases_ownership(
    module, loaded, monkeypatch
):
    before = torch.get_num_threads()
    original = torch.use_deterministic_algorithms
    calls = []

    def fail_once(*args, **kwargs):
        calls.append(True)
        if len(calls) == 1:
            raise RuntimeError("synthetic configuration failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(torch, "use_deterministic_algorithms", fail_once)
    with pytest.raises(RuntimeError, match="synthetic configuration failure"):
        with module.SelectiveCascade(loaded, _fixture_cpu=True):
            pass
    assert torch.get_num_threads() == before
    with module.SelectiveCascade(loaded, _fixture_cpu=True):
        pass


def test_session_cannot_be_reused_with_old_counters(module, loaded):
    scorer = module.SelectiveCascade(loaded, _fixture_cpu=True)
    with scorer:
        scorer.scan(URL)
    with pytest.raises(TransformerInferenceError, match="used"):
        with scorer:
            pass


@pytest.mark.parametrize("probability", [0.4999999701976776, 0.5, 0.5000000596046448])
def test_transformer_threshold_keeps_exact_neighbor_decisions(
    module, loaded, monkeypatch, probability
):
    fixed_stage1(monkeypatch, 0.5)
    # Isolate the decision operator; real-forward counter tests are separate.
    monkeypatch.setattr(
        transformer_scoring,
        "score_transformer_urls",
        lambda *args, **kwargs: (probability,),
    )
    with module.SelectiveCascade(loaded, _fixture_cpu=True) as scorer:
        result = scorer.scan(URL)
        assert result.decision == result.fixed_decision == int(probability >= 0.5)


def test_cached_policy_and_real_selective_calls_agree_on_future_only_stream(
    module, loaded, monkeypatch
):
    urls = tuple(f"https://sample.example/x?request={index}" for index in range(320))
    monkeypatch.setattr(
        fixed_cascade, "score_logistic_l1_authoritative", lambda *args: ((0.25,), {})
    )
    monkeypatch.setattr(loaded._model, "forward", lambda tokens, mask: torch.zeros(1))
    with module.SelectiveCascade(loaded, _fixture_cpu=True) as scorer:
        full = tuple(scorer.score_all(url) for url in urls)
        paired = tuple(
            policy_replay.PairedProbabilities(
                f"synthetic-{index}",
                row.stage1_probability,
                row.transformer_probability,
            )
            for index, row in enumerate(full)
        )
        nll = tuple(policy_replay.MonitorScore(row.record_id, 1.0) for row in paired)
        replay = policy_replay.replay_policy(
            paired,
            nll,
            stage1_threshold=0.5,
            transformer_threshold=0.5,
            half_width=0.125,
            monitor_boundary=0.5,
        )
        before = scorer.counts.transformer_forward_attempts
        live = tuple(
            scorer.scan(url, drift_override=row.drift_override)
            for url, row in zip(urls, replay.rows, strict=True)
        )
        assert scorer.counts.transformer_forward_attempts - before == 64
        assert tuple(row.decision for row in live) == tuple(
            row.policy_decision for row in replay.rows
        )
        assert all(not row.transformer_evaluated for row in live[:256])
        assert all(row.transformer_evaluated for row in live[256:])


def test_candidate_and_bridge_are_declared_but_not_accepted():
    path = Path(__file__).resolve().parents[1] / "data/evaluation-contract-v1.json"
    contract = json.loads(path.read_text())
    candidate = contract["runtime_candidate"]
    assert candidate["status"] == "specified_pending_development_compatibility"
    assert candidate["device"] == "mps"
    assert candidate["inference_batch_size"] == 1
    assert candidate["threadpoolctl_limits"] == 1
    assert candidate["numpy_blas"] == {"name": "scipy-openblas", "version": "0.3.29"}
    assert candidate["versions"] == {**gmm_monitor._VERSIONS, "torch": "2.7.1"}
    bridge = contract["compatibility_bridge"]
    assert bridge["reference"] == "accelerate_mps_original_validation_batch512"
    assert bridge["candidate"] == "openblas_mps_singleton"
    assert (
        bridge["acceptance"] == "exact_decisions_band_masks_and_separate_stream_alerts"
    )
    assert bridge["reference_counts"] == "accepted_four_model_validation_counts"
    assert bridge["gmm_reference"] == "exact_saved_calibration_and_audit_window_scores"
    assert bridge["retry"] is bridge["protected_evaluation_ready"] is False
    assert contract["protected_evaluation_ready"] is False
