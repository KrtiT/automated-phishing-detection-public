from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError, replace
from importlib import import_module

import numpy as np
import pytest
import torch
from test_transformer_inference import _build_fixture, _load_fixture

from automated_phishing_detection import fixed_cascade, gmm_monitor, policy_replay
from automated_phishing_detection.selective_inference import SelectiveCascade
from automated_phishing_detection.transformer_inference import TransformerInferenceError
from automated_phishing_detection.url_features import extract_url_features

URL = "HTTPS://safe.example:443/%2fA?Q=AbC#Fragment"


@pytest.fixture
def module():
    return import_module("automated_phishing_detection.live_monitor")


@pytest.fixture
def loaded(tmp_path):
    return replace(
        _load_fixture(_build_fixture(tmp_path)),
        stage1_threshold=0.5,
        transformer_threshold=0.5,
        half_width=0.125,
    )


@pytest.fixture
def gmm():
    return {
        "schema_version": 1,
        "contract_id": gmm_monitor.CONTRACT_ID,
        "features": list(gmm_monitor.GMM_FEATURE_NAMES),
        "dtype": "float64",
        "scaler": {
            "mean": [0.0] * 26,
            "scale": [1.0] * 26,
            "variance": [1.0] * 26,
            "n_samples_seen": 20,
            "n_features_in": 26,
        },
        "mixture": {
            "components": 1,
            "covariance_type": "diag",
            "weights": [1.0],
            "means": [[0.0] * 26],
            "variances": [[1.0] * 26],
            "precisions": [[1.0] * 26],
            "precisions_cholesky": [[1.0] * 26],
            "converged": True,
            "n_iter": 1,
            "lower_bound": -1.0,
        },
        "input_hashes": {"fixture": "1" * 64},
    }


def make_monitor(module, scorer, loaded, gmm, boundary=1.0):
    return module.LiveMonitor(
        scorer, stage1_model=loaded.stage1_model, gmm=gmm, boundary=boundary
    )


def synthetic_scores(monkeypatch, loaded, nlls, probabilities=None):
    values = iter(nlls)
    stage1 = iter(probabilities) if probabilities is not None else None
    monkeypatch.setattr(
        fixed_cascade,
        "score_logistic_l1_authoritative",
        lambda model, urls: (
            (next(stage1) if stage1 is not None else 0.25,),
            {"synthetic": True},
        ),
    )
    monkeypatch.setattr(
        gmm_monitor,
        "score_feature_matrix",
        lambda features, artifact: np.asarray([next(values)], dtype=np.float64),
    )
    monkeypatch.setattr(
        loaded._model,
        "forward",
        lambda tokens, mask: torch.zeros(1, dtype=torch.float32),
    )


def test_real_singleton_monitor_keeps_raw_url_and_portable_feature(
    module, loaded, gmm, monkeypatch
):
    portable = loaded.stage1_model.score_urls((URL,))[0]
    features = np.asarray([(*extract_url_features(URL), portable)], dtype=np.float64)
    expected_nll = float(gmm_monitor.score_feature_matrix(features, gmm)[0])
    seen = []
    score_gmm = gmm_monitor.score_feature_matrix

    def observe(features, artifact):
        seen.append(np.asarray(features).copy())
        return score_gmm(features, artifact)

    monkeypatch.setattr(gmm_monitor, "score_feature_matrix", observe)
    monkeypatch.setattr(
        fixed_cascade,
        "score_logistic_l1_authoritative",
        lambda model, urls: ((0.25,), {"synthetic_authoritative": True}),
    )
    with SelectiveCascade(loaded, _fixture_cpu=True) as scorer:
        monitor = make_monitor(module, scorer, loaded, gmm)
        result = monitor.scan_shift(URL)
        assert result.position == 1
        assert result.window is None
        assert result.monitor_nll == expected_nll
        assert result.scores.stage1_probability == 0.25 != portable
        assert len(seen) == 1
        np.testing.assert_array_equal(seen[0], features)
        assert seen[0].shape == (1, 26)
        assert seen[0].dtype == np.float64
        assert monitor.counts == scorer.counts
        assert monitor.counts.transformer_forward_attempts == 0
        with pytest.raises(FrozenInstanceError):
            result.position = 2


def test_unmocked_singleton_scorer_matches_saved_replay(module, loaded, gmm):
    loaded = replace(loaded, half_width=0.0)
    with SelectiveCascade(loaded, _fixture_cpu=True) as scorer:
        saved = scorer.score_all(URL)
        portable = loaded.stage1_model.score_urls((URL,))[0]
        nll = float(
            gmm_monitor.score_feature_matrix(
                [(*extract_url_features(URL), portable)], gmm
            )[0]
        )
    expected = policy_replay.replay_policy(
        [
            policy_replay.PairedProbabilities(
                f"row-{index}", saved.stage1_probability, saved.transformer_probability
            )
            for index in range(257)
        ],
        [policy_replay.MonitorScore(f"row-{index}", nll) for index in range(257)],
        stage1_threshold=loaded.stage1_threshold,
        transformer_threshold=loaded.transformer_threshold,
        half_width=loaded.half_width,
        monitor_boundary=0.0,
    )
    with SelectiveCascade(loaded, _fixture_cpu=True) as scorer:
        monitor = make_monitor(module, scorer, loaded, gmm, 0.0)
        actual = tuple(monitor.scan_shift(URL) for _ in range(257))
        assert tuple(row.monitor_nll for row in actual) == (nll,) * 257
        assert tuple(row.window for row in actual if row.window) == expected.windows
        assert all(
            row.scores.stage1_probability == saved.stage1_probability for row in actual
        )
        assert tuple(row.scores.decision for row in actual) == tuple(
            row.policy_decision for row in expected.rows
        )
        assert (
            actual[-1].scores.transformer_probability == saved.transformer_probability
        )
        assert not any(row.scores.drift_override for row in actual[:256])
        assert actual[-1].scores.drift_override
        assert monitor.counts.transformer_forward_attempts == 1
        assert monitor.counts.successful_transformer_scores == 1


@pytest.mark.parametrize(
    "nlls,boundary",
    [
        ([], 1.0),
        ([2.0] * 255, 1.0),
        ([1.0] * 700, 1.0),
        ([float(np.nextafter(1.0, 2.0))] * 257, 1.0),
        ([2.0] * 256 + [0.0] * 400, 1.9),
        ([2.0] * 320 + [0.0] * 601, 1.9),
        ([0.125, 1e12, -1e12, 0.25] * 240 + [2.0] * 13, 0.1),
    ],
    ids=["empty", "short", "equality", "strict", "expiry", "overlap", "means"],
)
def test_live_complete_trace_exactly_matches_offline_replay(
    module, loaded, gmm, monkeypatch, nlls, boundary
):
    probabilities = tuple(0.5 if index % 7 == 0 else 0.25 for index in range(len(nlls)))
    synthetic_scores(monkeypatch, loaded, nlls, probabilities)
    expected = policy_replay.replay_policy(
        [
            policy_replay.PairedProbabilities(f"row-{index}", probability, 0.5)
            for index, probability in enumerate(probabilities)
        ],
        [
            policy_replay.MonitorScore(f"row-{index}", value)
            for index, value in enumerate(nlls)
        ],
        stage1_threshold=loaded.stage1_threshold,
        transformer_threshold=loaded.transformer_threshold,
        half_width=loaded.half_width,
        monitor_boundary=boundary,
    )
    with SelectiveCascade(loaded, _fixture_cpu=True) as scorer:
        monitor = make_monitor(module, scorer, loaded, gmm, boundary)
        results = tuple(monitor.scan_shift(URL) for _ in nlls)
        assert tuple(result.position for result in results) == tuple(
            range(1, len(nlls) + 1)
        )
        assert tuple(result.monitor_nll for result in results) == tuple(nlls)
        assert (
            tuple(result.window for result in results if result.window)
            == expected.windows
        )
        actual_rows = tuple(
            policy_replay.RoutingTrace(
                f"row-{index}",
                result.scores.fixed_decision,
                result.scores.decision,
                result.scores.band_selected,
                result.scores.drift_override,
                result.scores.logical_stage2_selected,
            )
            for index, result in enumerate(results)
        )
        assert actual_rows == expected.rows
        assert not any(result.scores.drift_override for result in results[:256])
        selected = sum(row.logical_stage2_mask for row in expected.rows)
        assert monitor.counts.transformer_forward_attempts == selected
        assert monitor.counts.successful_transformer_scores == selected
        assert monitor.counts.completed_requests == len(nlls)
        assert monitor.counts.failed_requests == 0


def test_each_reduction_receives_only_the_last_256_nlls(
    module, loaded, gmm, monkeypatch
):
    nlls = tuple(float(index % 37) for index in range(913))
    synthetic_scores(monkeypatch, loaded, nlls)
    received = []
    score_windows = gmm_monitor.window_scores

    def observe(values):
        received.append(tuple(values))
        return score_windows(values)

    monkeypatch.setattr(gmm_monitor, "window_scores", observe)
    with SelectiveCascade(loaded, _fixture_cpu=True) as scorer:
        monitor = make_monitor(module, scorer, loaded, gmm)
        for _ in nlls:
            monitor.scan_shift(URL)
        assert len(monitor._nlls) == monitor._nlls.maxlen == 256
    assert received == [nlls[end - 256 : end] for end in range(256, len(nlls) + 1, 64)]


def test_reset_clears_windows_and_activation_but_not_physical_counters(
    module, loaded, gmm, monkeypatch
):
    synthetic_scores(monkeypatch, loaded, [2.0] * 513)
    with SelectiveCascade(loaded, _fixture_cpu=True) as scorer:
        monitor = make_monitor(module, scorer, loaded, gmm)
        warmup = tuple(monitor.scan_shift(URL) for _ in range(257))
        assert warmup[-1].scores.drift_override
        before = monitor.counts
        monitor.reset_monitor()
        assert monitor.counts == before
        measured = tuple(monitor.scan_shift(URL) for _ in range(256))
        assert measured[0].position == 1
        assert not any(row.scores.drift_override for row in measured)
        assert all(row.window is None for row in measured[:-1])
        assert measured[-1].window == policy_replay.WindowTrace(1, 256, 2.0, True)
        assert monitor.counts.transformer_forward_attempts == 1
        assert monitor.counts.completed_requests == 513


@pytest.mark.parametrize("stage", ["scorer", "portable", "gmm", "window"])
def test_failed_computation_permanently_poisons_monitor(
    module, loaded, gmm, monkeypatch, stage
):
    synthetic_scores(monkeypatch, loaded, [2.0] * 300)
    with SelectiveCascade(loaded, _fixture_cpu=True) as scorer:
        monitor = make_monitor(module, scorer, loaded, gmm)
        for _ in range(255):
            monitor.scan_shift(URL)

        def fail(*args, **kwargs):
            raise RuntimeError("synthetic failure")

        target, name = {
            "scorer": (scorer, "scan"),
            "portable": (fixed_cascade.PortableLogisticL1, "score_urls"),
            "gmm": (gmm_monitor, "score_feature_matrix"),
            "window": (gmm_monitor, "window_scores"),
        }[stage]
        with monkeypatch.context() as failure:
            failure.setattr(target, name, fail)
            with pytest.raises(RuntimeError, match="synthetic failure"):
                monitor.scan_shift(URL)
        after_failure = monitor.counts
        with pytest.raises(module.LiveMonitorError, match="poisoned"):
            monitor.scan_shift(URL)
        with pytest.raises(module.LiveMonitorError, match="poisoned"):
            monitor.reset_monitor()
        assert monitor.counts == after_failure


@pytest.mark.parametrize("bad_nll", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_monitor_score_poisons_before_any_window(
    module, loaded, gmm, monkeypatch, bad_nll
):
    synthetic_scores(monkeypatch, loaded, [bad_nll, 2.0])
    with SelectiveCascade(loaded, _fixture_cpu=True) as scorer:
        monitor = make_monitor(module, scorer, loaded, gmm)
        with pytest.raises((module.LiveMonitorError, gmm_monitor.GMMMonitorError)):
            monitor.scan_shift(URL)
        with pytest.raises(module.LiveMonitorError, match="poisoned"):
            monitor.scan_shift(URL)


def test_alerted_physical_forward_failure_cannot_be_retried(
    module, loaded, gmm, monkeypatch
):
    synthetic_scores(monkeypatch, loaded, [2.0] * 300)

    def fail(tokens, mask):
        raise RuntimeError("synthetic forward failure")

    monkeypatch.setattr(loaded._model, "forward", fail)
    with SelectiveCascade(loaded, _fixture_cpu=True) as scorer:
        monitor = make_monitor(module, scorer, loaded, gmm)
        for _ in range(256):
            monitor.scan_shift(URL)
        with pytest.raises(
            TransformerInferenceError, match="synthetic forward failure"
        ):
            monitor.scan_shift(URL)
        assert monitor.counts.transformer_forward_attempts == 1
        assert monitor.counts.successful_transformer_scores == 0
        assert monitor.counts.completed_requests == 256
        assert monitor.counts.failed_requests == 1
        before = monitor.counts
        with pytest.raises(module.LiveMonitorError, match="poisoned"):
            monitor.scan_shift(URL)
        with pytest.raises(module.LiveMonitorError, match="poisoned"):
            monitor.reset_monitor()
        assert monitor.counts == before


def test_monitor_owns_a_detached_copy_of_gmm_state(module, loaded, gmm):
    with SelectiveCascade(loaded, _fixture_cpu=True) as scorer:
        monitor = make_monitor(module, scorer, loaded, gmm)
        before = monitor.scan_shift(URL)
        gmm["mixture"]["means"][0][0] = 1000.0
        after = monitor.scan_shift(URL)
        assert after.monitor_nll == before.monitor_nll


@pytest.mark.parametrize("operation", ["scan_shift", "reset_monitor"])
def test_original_scorer_owner_required_without_poisoning_rejected_call(
    module, loaded, gmm, monkeypatch, operation
):
    synthetic_scores(monkeypatch, loaded, [2.0])
    with SelectiveCascade(loaded, _fixture_cpu=True) as scorer:
        monitor = make_monitor(module, scorer, loaded, gmm)
        with ThreadPoolExecutor(max_workers=1) as pool:
            method = getattr(monitor, operation)
            future = (
                pool.submit(method, URL)
                if operation == "scan_shift"
                else pool.submit(method)
            )
            with pytest.raises(TransformerInferenceError, match="owner thread"):
                future.result()
        assert monitor.scan_shift(URL).position == 1


def test_monitor_cannot_outlive_scorer_session(module, loaded, gmm):
    scorer = SelectiveCascade(loaded, _fixture_cpu=True)
    with pytest.raises(TransformerInferenceError, match="open session"):
        make_monitor(module, scorer, loaded, gmm)
    with scorer:
        monitor = make_monitor(module, scorer, loaded, gmm)
    for operation in (lambda: monitor.scan_shift(URL), monitor.reset_monitor):
        with pytest.raises(TransformerInferenceError, match="open session"):
            operation()


@pytest.mark.parametrize("boundary", [None, True, "1", float("nan"), float("inf")])
def test_boundary_must_be_finite_numeric(module, loaded, gmm, boundary):
    with SelectiveCascade(loaded, _fixture_cpu=True) as scorer:
        with pytest.raises(module.LiveMonitorError, match="boundary"):
            make_monitor(module, scorer, loaded, gmm, boundary)
