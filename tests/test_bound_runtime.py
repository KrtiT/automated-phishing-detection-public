"""Compose authenticated loading with the inference owner's lifecycle."""

import importlib.util
import json
import threading
from contextlib import contextmanager
from dataclasses import FrozenInstanceError, fields
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_http_replay import loopback_server
from test_selective_service import SyntheticScorer

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def runtime():
    assert (ROOT / "src/automated_phishing_detection/bound_runtime.py").is_file(), (
        "missing bound runtime composition"
    )
    from automated_phishing_detection import bound_runtime

    return bound_runtime


def fixture_composition(runtime, monkeypatch, tmp_path):
    events = []
    binding = SimpleNamespace(
        root=tmp_path, revision="a" * 40, contract_sha256="b" * 64
    )
    paths = object()
    model = SimpleNamespace(stage1_model=object())

    def check(value):
        assert value is binding
        events.append(("check", threading.get_ident()))

    def load(root, supplied_paths):
        assert root == tmp_path and supplied_paths is paths
        events.append(("load", threading.get_ident()))
        return SimpleNamespace(cascade=model, gmm={}, monitor_boundary=0.0)

    @contextmanager
    def scorer(loaded):
        assert loaded is model
        events.append(("enter", threading.get_ident()))
        try:
            yield SyntheticScorer()
        finally:
            events.append(("exit", threading.get_ident()))

    monkeypatch.setattr(runtime, "recheck_binding", check)
    monkeypatch.setattr(runtime, "load_bound_models", load)
    monkeypatch.setattr(runtime, "SelectiveCascade", scorer)
    return binding, paths, events


def test_bound_evaluation_session_is_immutable(runtime):
    primary = runtime.BoundSession(SimpleNamespace(), object())
    secondary = object()

    session = runtime.BoundEvaluationSession(primary, secondary)

    assert [field.name for field in fields(session)] == ["primary", "secondary"]
    assert session.primary is primary
    assert session.secondary is secondary
    with pytest.raises(FrozenInstanceError):
        session.secondary = object()


def fixture_evaluation_composition(runtime, monkeypatch, tmp_path):
    events = []
    binding = SimpleNamespace(root=tmp_path)
    primary_paths = object()
    secondary_paths = object()
    cascade = object()
    models = SimpleNamespace(cascade=cascade)
    scorer = object()
    secondary = object()
    state = SimpleNamespace(owner_active=False)

    def check(value):
        assert value is binding
        assert not state.owner_active, "runtime probe ran inside numerical owner"
        events.append("check")

    def load_primary(root, supplied_paths):
        assert root == tmp_path and supplied_paths is primary_paths
        assert not state.owner_active
        events.append("primary_load")
        return models

    @contextmanager
    def open_owner(loaded):
        assert loaded is cascade
        state.owner_active = True
        events.append("owner_enter")
        try:
            yield scorer
        finally:
            state.owner_active = False
            events.append("owner_exit")

    def load(root, supplied_paths, supplied_primary):
        assert root == tmp_path
        assert supplied_paths is secondary_paths
        assert supplied_primary is cascade
        assert not state.owner_active, "secondary load ran inside numerical owner"
        events.append("secondary_load")
        return secondary

    monkeypatch.setattr(runtime, "recheck_binding", check)
    monkeypatch.setattr(runtime, "load_bound_models", load_primary)
    monkeypatch.setattr(runtime, "SelectiveCascade", open_owner)
    monkeypatch.setattr(runtime, "load_bound_secondary", load)
    return SimpleNamespace(
        binding=binding,
        primary_paths=primary_paths,
        secondary_paths=secondary_paths,
        models=models,
        scorer=scorer,
        secondary=secondary,
        events=events,
        state=state,
    )


@pytest.mark.parametrize("consumer_fails", [False, True])
def test_bound_evaluation_rechecks_outside_owner_and_after_restoration(
    runtime, monkeypatch, tmp_path, consumer_fails
):
    fixture = fixture_evaluation_composition(runtime, monkeypatch, tmp_path)

    def consume():
        with runtime.open_bound_evaluation_session(
            fixture.binding, fixture.primary_paths, fixture.secondary_paths
        ) as session:
            assert type(session) is runtime.BoundEvaluationSession
            assert type(session.primary) is runtime.BoundSession
            assert session.primary.models is fixture.models
            assert session.primary.scorer is fixture.scorer
            assert session.secondary is fixture.secondary
            assert fixture.state.owner_active
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
        "check",
        "owner_enter",
        "consumer",
        "owner_exit",
        "check",
    ]
    assert not fixture.state.owner_active


def test_failed_evaluation_secondary_load_never_enters_owner(
    runtime, monkeypatch, tmp_path
):
    fixture = fixture_evaluation_composition(runtime, monkeypatch, tmp_path)

    def reject(*args):
        fixture.events.append("secondary_rejected")
        raise ValueError("secondary artifact mismatch")

    monkeypatch.setattr(runtime, "load_bound_secondary", reject)
    with pytest.raises(ValueError, match="secondary artifact mismatch"):
        with runtime.open_bound_evaluation_session(
            fixture.binding, fixture.primary_paths, fixture.secondary_paths
        ):
            pytest.fail("invalid secondary binding reached consumer")

    assert fixture.events == ["check", "primary_load", "secondary_rejected"]
    assert not fixture.state.owner_active


def test_changed_evaluation_binding_after_both_loads_never_enters_owner(
    runtime, monkeypatch, tmp_path
):
    fixture = fixture_evaluation_composition(runtime, monkeypatch, tmp_path)
    original = runtime.recheck_binding
    checks = 0

    def check(binding):
        nonlocal checks
        checks += 1
        original(binding)
        if checks == 2:
            raise ValueError("binding changed during secondary loading")

    monkeypatch.setattr(runtime, "recheck_binding", check)
    with pytest.raises(ValueError, match="during secondary loading"):
        with runtime.open_bound_evaluation_session(
            fixture.binding, fixture.primary_paths, fixture.secondary_paths
        ):
            pytest.fail("stale binding reached consumer")

    assert fixture.events == ["check", "primary_load", "secondary_load", "check"]
    assert not fixture.state.owner_active


def test_bound_loading_and_rechecks_run_on_service_owner(
    runtime, monkeypatch, tmp_path
):
    binding, paths, events = fixture_composition(runtime, monkeypatch, tmp_path)
    app = runtime.create_bound_app(binding, paths)
    assert events == []  # Creating an app on the event-loop thread must not load.
    with loopback_server(app):
        assert app.state.owner.healthy
    assert [name for name, _ in events] == [
        "check",
        "load",
        "check",
        "enter",
        "exit",
        "check",
    ]
    assert len({thread for _, thread in events}) == 1
    assert events[0][1] != threading.get_ident()


def test_failed_initial_binding_never_opens_artifacts(runtime, monkeypatch, tmp_path):
    binding, paths, events = fixture_composition(runtime, monkeypatch, tmp_path)

    def reject(value):
        raise ValueError("synthetic changed revision")

    monkeypatch.setattr(runtime, "recheck_binding", reject)
    with pytest.raises(ValueError, match="changed revision"):
        with runtime.open_bound_session(binding, paths):
            pytest.fail("unbound session opened")
    assert events == []


def test_changed_binding_after_load_never_enters_numerical_context(
    runtime, monkeypatch, tmp_path
):
    binding, paths, events = fixture_composition(runtime, monkeypatch, tmp_path)
    checks = 0

    def check(value):
        nonlocal checks
        checks += 1
        if checks == 2:
            raise ValueError("synthetic source changed during loading")

    monkeypatch.setattr(runtime, "recheck_binding", check)
    with pytest.raises(ValueError, match="during loading"):
        with runtime.open_bound_session(binding, paths):
            pytest.fail("stale session opened")
    assert [name for name, _ in events] == ["load"]


def test_context_failure_still_exits_and_rechecks_binding(
    runtime, monkeypatch, tmp_path
):
    binding, paths, events = fixture_composition(runtime, monkeypatch, tmp_path)
    with pytest.raises(RuntimeError, match="consumer failed"):
        with runtime.open_bound_session(binding, paths) as session:
            assert isinstance(session.scorer, SyntheticScorer)
            raise RuntimeError("consumer failed")
    assert [name for name, _ in events][-2:] == ["exit", "check"]


def test_shift_workload_cannot_be_mislabeled_as_implemented_http(
    runtime, monkeypatch, tmp_path
):
    binding, paths, events = fixture_composition(runtime, monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="workload"):
        runtime.create_bound_app(binding, paths, workload="shift_period")
    assert events == []


def test_bound_shift_app_constructs_monitor_inside_bound_owner(
    runtime, monkeypatch, tmp_path
):
    from test_shift_service import SyntheticMonitor, make_plan

    from automated_phishing_detection import shift_schema

    binding, paths, events = fixture_composition(runtime, monkeypatch, tmp_path)
    assert hasattr(runtime, "create_bound_shift_app"), "missing bound shift composition"

    def monitor(scorer, **kwargs):
        assert isinstance(scorer, SyntheticScorer)
        assert kwargs["gmm"] == {} and kwargs["boundary"] == 0.0
        events.append(("monitor", threading.get_ident()))
        return SyntheticMonitor()

    monkeypatch.setattr(runtime, "LiveMonitor", monitor)
    app = runtime.create_bound_shift_app(binding, paths, make_plan(shift_schema))
    assert events == []
    with loopback_server(app):
        assert app.state.owner.healthy
    assert [name for name, _ in events] == [
        "check",
        "load",
        "check",
        "enter",
        "monitor",
        "exit",
        "check",
    ]
    assert len({thread for _, thread in events}) == 1


@pytest.fixture
def command():
    path = ROOT / "scripts/verify_execution_binding.py"
    assert path.is_file(), "missing metadata-only preflight command"
    spec = importlib.util.spec_from_file_location("verify_execution_binding", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_metadata_command_has_no_artifact_or_dataset_arguments(command):
    options = {action.dest for action in command.parser()._actions}
    assert options == {
        "help",
        "repo_root",
        "expected_revision",
        "expected_contract_sha256",
    }


def test_metadata_command_reports_preflight_not_measurement(
    command, monkeypatch, capsys, tmp_path
):
    calls = []

    def bind(root, **kwargs):
        calls.append((root, kwargs))
        return SimpleNamespace(
            revision="a" * 40,
            contract_sha256="b" * 64,
            source_hashes=(("src/synthetic.py", "c" * 64),),
            runtime_json='{"synthetic":true}',
        )

    monkeypatch.setattr(command, "bind_execution", bind)
    assert (
        command.main(
            [
                "--repo-root",
                str(tmp_path),
                "--expected-revision",
                "a" * 40,
                "--expected-contract-sha256",
                "b" * 64,
            ]
        )
        == 0
    )
    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert calls == [
        (
            tmp_path,
            {
                "expected_revision": "a" * 40,
                "expected_contract_sha256": "b" * 64,
            },
        )
    ]
    assert result == {
        "schema_version": 1,
        "status": "verified_preflight",
        "protected_evaluation_ready": False,
        "research_measurements_run": False,
        "revision": "a" * 40,
        "contract_sha256": "b" * 64,
        "source_file_count": 1,
        "runtime": {"synthetic": True},
    }
    assert str(tmp_path) not in captured.out
    assert captured.err == ""


def test_metadata_command_fails_without_success_output(
    command, monkeypatch, capsys, tmp_path
):
    def reject(*args, **kwargs):
        raise command.ExecutionPreflightError("synthetic identity mismatch")

    monkeypatch.setattr(command, "bind_execution", reject)
    assert (
        command.main(
            [
                "--repo-root",
                str(tmp_path),
                "--expected-revision",
                "a" * 40,
                "--expected-contract-sha256",
                "b" * 64,
            ]
        )
        == 1
    )
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "Execution preflight failed: synthetic identity mismatch\n"
