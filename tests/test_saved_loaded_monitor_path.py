"""Reuse one numerical-thread guard with caller-verified loaded replay models."""

import pytest
import threadpoolctl

from automated_phishing_detection import saved_evidence


def verify(rows, models):
    checker = getattr(saved_evidence, "_verify_loaded_monitor_path", None)
    assert callable(checker), "missing shared loaded-monitor verifier"
    return checker(rows, *models)


@pytest.mark.parametrize("fail", [False, True])
def test_loaded_path_reuses_exact_models_and_restores_threads(monkeypatch, fail):
    models = (object(), object(), object())
    rows = ({"record": {"raw_url": "https://invented.test/"}},)
    observations = []
    before = threadpoolctl.threadpool_info()

    def observe(*arguments):
        observations.append(arguments)
        assert all(pool["num_threads"] == 1 for pool in threadpoolctl.threadpool_info())
        if fail:
            raise saved_evidence.SavedEvidenceError("invented_failure")

    def forbidden(*args):
        pytest.fail("loaded monitor verification attempted another artifact load")

    monkeypatch.setattr(saved_evidence, "_replay_models", forbidden)
    monkeypatch.setattr(saved_evidence, "_verify_monitor_rows", observe)
    if fail:
        with pytest.raises(saved_evidence.SavedEvidenceError, match="invented_failure"):
            verify(rows, models)
    else:
        verify(rows, models)
    assert observations == [(rows, *models)]
    assert threadpoolctl.threadpool_info() == before


@pytest.mark.parametrize("pools", [[], [{"num_threads": 2}]])
def test_loaded_path_rejects_unconfirmed_thread_limit(monkeypatch, pools):
    monkeypatch.setattr(saved_evidence.threadpoolctl, "threadpool_info", lambda: pools)
    with pytest.raises(saved_evidence.SavedEvidenceError, match="one numerical thread"):
        verify((), (object(), object(), object()))
