"""Invented worker composition; fixture lineage grants no execution authority."""

import importlib
import importlib.util
import json
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import pytest
from external_completion_fixtures import external_completion_case
from saved_external_scorer_fixtures import FixturePrimaryScorer

from automated_phishing_detection._checkpoint_codec import canonical_bytes
from automated_phishing_detection.execution_preflight import ExecutionBinding


@pytest.fixture
def runner_api():
    name = "automated_phishing_detection.external_source_runner"
    assert importlib.util.find_spec(name), "missing external source worker composition"
    return importlib.import_module(name)


def fresh_session(case):
    session = case.bundle.session
    primary = session.evaluation.primary
    return replace(
        session,
        evaluation=replace(
            session.evaluation,
            primary=replace(
                primary, scorer=FixturePrimaryScorer(primary.models.cascade)
            ),
        ),
    )


@contextmanager
def session_owner(case, *args):
    assert args == (
        case.binding,
        case.paths.artifacts,
        case.paths.secondary_artifacts,
        case.paths.drift_artifacts,
    )
    case.events.append("session_open")
    case.active = True
    try:
        yield case.session
    finally:
        case.active = False
        case.events.append("session_closed")


def configured_case(runner_api, tmp_path, monkeypatch, count=5):
    case = external_completion_case(tmp_path / "fixture", monkeypatch, count)
    case.paths = replace(
        case.paths,
        attempt=tmp_path / "attempt",
        public_summary=tmp_path / "public.json",
    )
    case.paths.archive.write_bytes(case.archivebytes)
    case.paths.suffix_rules.write_bytes(case.suffix)
    case.events, case.active, case.session = [], False, fresh_session(case)
    monkeypatch.setattr(
        runner_api.body, "resolve_external_source_profile", lambda binding: case.profile
    )
    monkeypatch.setattr(
        runner_api,
        "open_bound_external_session",
        lambda *args: session_owner(case, *args),
    )

    def checked(binding):
        assert binding is case.binding and not case.active
        case.events.append("binding_checked")

    monkeypatch.setattr(runner_api, "recheck_binding", checked)
    return case


@pytest.fixture
def runner_case(runner_api, tmp_path, monkeypatch):
    return configured_case(runner_api, tmp_path, monkeypatch)


def run(api, case, **changes):
    return api._run_bound_external(
        case.binding,
        changes.get("paths", case.paths),
        handoff=changes.get("handoff", case.handoff),
    )


def test_public_gate_stops_before_all_supplied_inputs(runner_api, monkeypatch):
    binding = ExecutionBinding(Path("/invented"), "c" * 40, "d" * 64, (), "{}")
    calls = []
    expected = {
        "expected_revision": binding.revision,
        "expected_contract_sha256": binding.contract_sha256,
    }

    def bound(root, **kwargs):
        assert root == binding.root and kwargs == expected
        calls.append("bound")
        return binding

    def forbidden(*args, **kwargs):
        pytest.fail("closed gate inspected supplied transport or protected paths")

    monkeypatch.setattr(runner_api, "bind_execution", bound)
    monkeypatch.setattr(runner_api, "read_internal_handoff_transport", forbidden)
    monkeypatch.setattr(runner_api, "_run_bound_external", forbidden)
    with pytest.raises(runner_api.ExternalSourceExecutionError):
        runner_api.run_external_evaluation(
            binding.root,
            **expected,
            paths=object(),
            internal_transport=object(),
            expected_handoff_sha256=object(),
        )
    assert calls == ["bound"]


@pytest.mark.parametrize("count", [0, 1, 255, 256, 319, 320])
def test_real_producer_retains_and_publishes_exact_thirty_six_outputs(
    runner_api, tmp_path, monkeypatch, count
):
    case = configured_case(runner_api, tmp_path, monkeypatch, count)
    result = run(runner_api, case)
    assert result == case.paths.public_summary.absolute()
    public = json.loads(result.read_bytes())
    assert canonical_bytes(public["composition"]) == canonical_bytes(
        case.produced.public_summary
    )
    assert public["protected_evaluation_authorized"] is False
    assert len(public["private_sha256"]) == 36
    for name, expected in case.produced.private_outputs.items():
        assert (case.paths.attempt / "checkpoints" / name).read_bytes() == expected
        assert (case.paths.attempt / "evidence" / name).read_bytes() == expected
    assert len(case.session.evaluation.primary.scorer.urls) == count
    assert case.events == ["session_open", "session_closed", "binding_checked"]
    assert not (case.paths.attempt / "external-failure.json").exists()
