"""Bind external drift before numerical ownership and restore on all exits."""

from dataclasses import FrozenInstanceError

import pytest
from test_bound_runtime import fixture_evaluation_composition


@pytest.fixture
def runtime():
    from automated_phishing_detection import bound_external_runtime

    return bound_external_runtime


def external_fixture(runtime, monkeypatch, tmp_path, *, load_fails=False):
    fixture = fixture_evaluation_composition(runtime, monkeypatch, tmp_path)
    fixture.drift_paths, fixture.drift = object(), object()

    def load(binding, paths, models):
        assert binding is fixture.binding
        assert paths is fixture.drift_paths
        assert models is fixture.models
        assert not fixture.state.owner_active
        fixture.events.append("drift_load")
        if load_fails:
            raise ValueError("drift fixture rejected")
        return fixture.drift

    assert hasattr(runtime, "open_bound_external_session"), "missing external session"
    monkeypatch.setattr(runtime, "load_bound_drift", load)
    return fixture


@pytest.mark.parametrize("consumer_fails", [False, True])
def test_external_state_binds_before_owner_and_rechecks_after_restoration(
    runtime, monkeypatch, tmp_path, consumer_fails
):
    fixture = external_fixture(runtime, monkeypatch, tmp_path)

    def consume():
        with runtime.open_bound_external_session(
            fixture.binding,
            fixture.primary_paths,
            fixture.secondary_paths,
            fixture.drift_paths,
        ) as session:
            assert session.evaluation.primary.models is fixture.models
            assert session.evaluation.primary.scorer is fixture.scorer
            assert session.evaluation.secondary is fixture.secondary
            assert session.drift is fixture.drift and fixture.state.owner_active
            with pytest.raises(FrozenInstanceError):
                session.drift = object()
            fixture.events.append("consumer")
            if consumer_fails:
                raise RuntimeError("consumer failed")

    if consumer_fails:
        with pytest.raises(RuntimeError, match="consumer failed"):
            consume()
    else:
        consume()
    assert fixture.events == [
        "check",
        "primary_load",
        "secondary_load",
        "drift_load",
        "check",
        "owner_enter",
        "consumer",
        "owner_exit",
        "check",
    ]


def test_external_drift_rejection_does_not_enter_owner(runtime, monkeypatch, tmp_path):
    fixture = external_fixture(runtime, monkeypatch, tmp_path, load_fails=True)
    with (
        pytest.raises(ValueError, match="drift fixture rejected"),
        runtime.open_bound_external_session(
            fixture.binding,
            fixture.primary_paths,
            fixture.secondary_paths,
            fixture.drift_paths,
        ),
    ):
        pytest.fail("yielded rejected drift state")
    assert fixture.events == ["check", "primary_load", "secondary_load", "drift_load"]


def test_external_owner_restores_before_final_recheck_rejects(
    runtime, monkeypatch, tmp_path
):
    fixture = external_fixture(runtime, monkeypatch, tmp_path)
    check = runtime.recheck_binding

    def changed(binding):
        check(binding)
        if fixture.events.count("check") == 3:
            raise ValueError("final checkout changed")

    monkeypatch.setattr(runtime, "recheck_binding", changed)
    with (
        pytest.raises(ValueError, match="final checkout changed"),
        runtime.open_bound_external_session(
            fixture.binding,
            fixture.primary_paths,
            fixture.secondary_paths,
            fixture.drift_paths,
        ),
    ):
        assert fixture.state.owner_active
    assert not fixture.state.owner_active
    assert fixture.events[-2:] == ["owner_exit", "check"]
