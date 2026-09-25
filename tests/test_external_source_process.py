"""Same-parent composition keeps both access gates closed before any run."""

import importlib
import importlib.util
from types import SimpleNamespace

import pytest


def module():
    name = "automated_phishing_detection.external_source_process"
    assert importlib.util.find_spec(name), "missing same-parent source composition"
    return importlib.import_module(name)


def test_same_parent_source_process_api_exists():
    module()


@pytest.mark.parametrize("binding_ready", [False, True])
def test_each_gate_stops_before_any_supplied_path_or_worker(monkeypatch, binding_ready):
    api = module()
    binding = SimpleNamespace(protected_evaluation_ready=binding_ready)
    profile = SimpleNamespace(protected_evaluation_ready=not binding_ready)
    calls = []
    monkeypatch.setattr(api, "bind_execution", lambda *args, **kwargs: binding)
    monkeypatch.setattr(
        api,
        "resolve_external_source_profile",
        lambda unused: calls.append("profile") or profile,
    )

    def forbidden(*args, **kwargs):
        pytest.fail("incomplete access profile reached a supplied path or worker")

    monkeypatch.setattr(api, "_run_observed_sources", forbidden)
    with pytest.raises(api.ExternalSourceExecutionError, match="pre_access_freeze"):
        api.run_internal_external_process(
            object(),
            expected_revision="unopened",
            expected_contract_sha256="closed",
            internal_paths=object(),
            external_paths=object(),
        )
    assert calls == (["profile"] if binding_ready else [])


def test_private_pair_uses_observed_internal_snapshot_without_reopening(monkeypatch):
    api, events = module(), []
    binding, internal_paths, external_paths = object(), object(), object()
    internal, handoff, external = object(), object(), object()
    monkeypatch.setattr(
        api, "resolve_external_source_profile", lambda value: events.append("profile")
    )

    def internal_run(actual_binding, paths):
        assert actual_binding is binding and paths is internal_paths
        events.append("internal")
        return internal

    def build(actual):
        assert actual is internal
        events.append("handoff")
        return handoff

    def external_run(actual_binding, paths, actual_handoff):
        assert actual_binding is binding and paths is external_paths
        assert actual_handoff is handoff
        events.append("external")
        return external

    monkeypatch.setattr(api.source_runner, "_run_observed_internal", internal_run)
    monkeypatch.setattr(api, "build_internal_handoff", build)
    monkeypatch.setattr(api, "_run_observed_external", external_run)
    result = api._run_observed_sources(binding, internal_paths, external_paths)
    assert result.internal is internal and result.external is external
    assert events == ["profile", "internal", "handoff", "external"]


@pytest.mark.parametrize("stage", ["profile", "internal", "handoff", "external"])
def test_pair_stops_at_first_failure_without_repeating_predecessors(monkeypatch, stage):
    api = module()
    error = KeyboardInterrupt("original")
    events = []
    internal = object()

    def operation(name, result):
        def execute(*args, **kwargs):
            events.append(name)
            if name == stage:
                raise error
            return result

        return execute

    monkeypatch.setattr(
        api, "resolve_external_source_profile", operation("profile", None)
    )
    monkeypatch.setattr(
        api.source_runner, "_run_observed_internal", operation("internal", internal)
    )
    monkeypatch.setattr(api, "build_internal_handoff", operation("handoff", object()))
    monkeypatch.setattr(api, "_run_observed_external", operation("external", object()))
    with pytest.raises(KeyboardInterrupt) as rejected:
        api._run_observed_sources(object(), object(), object())
    assert rejected.value is error
    order = ["profile", "internal", "handoff", "external"]
    assert events == order[: order.index(stage) + 1]
    if stage in ("handoff", "external"):
        assert error.source_internal is internal
