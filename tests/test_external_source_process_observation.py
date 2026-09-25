"""Actual no-op children test observation only, not producer lineage."""

import tempfile
from dataclasses import replace

import pytest
from external_completion_fixtures import external_completion_case
from test_external_source_completion import bind_fixture
from test_external_source_completion import module as completion_module
from test_external_source_process import module


def context(tmp_path, monkeypatch):
    api = module()
    case = external_completion_case(tmp_path, monkeypatch)
    bind_fixture(completion_module(), case, monkeypatch)
    monkeypatch.setattr(
        api, "resolve_external_source_profile", lambda unused: case.profile
    )
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr(api, "_worker_command", lambda *args: case.command)
    return api, case


def test_actual_observed_zero_accepts_once_after_saved_and_transport_checks(
    tmp_path, monkeypatch
):
    api, case = context(tmp_path, monkeypatch)
    original = api.observe_worker
    calls = []

    def observed(command):
        calls.append(command)
        return original(command)

    monkeypatch.setattr(api, "observe_worker", observed)
    result = api._run_observed_external(case.binding, case.paths, case.handoff)
    assert calls == [case.command]
    assert result.worker.exit.exit_observed is True
    assert result.worker.exit.exit_code == 0
    assert result.public_summary == case.expectedpublic


@pytest.mark.parametrize("exit_code", [7, 17])
def test_existing_marker_and_real_nonzero_exit_never_reach_saved_verification(
    tmp_path, monkeypatch, exit_code
):
    api, case = context(tmp_path, monkeypatch)
    case.command = (*case.command[:2], f"raise SystemExit({exit_code})")

    def forbidden(*args, **kwargs):
        pytest.fail("nonzero worker reached saved completion")

    monkeypatch.setattr(api, "verify_external_completion_snapshot", forbidden)
    with pytest.raises(Exception) as rejected:
        api._run_observed_external(case.binding, case.paths, case.handoff)
    failure = rejected.value.external_failure
    assert case.paths.public_summary.is_file()
    assert failure.worker.exit.exit_observed is True
    assert failure.worker.exit.exit_code == exit_code
    assert failure.handoff is case.handoff
    assert failure.candidate_snapshot is None


@pytest.mark.parametrize("kind", ["unknown", "command_mismatch"])
def test_invalid_actual_observation_is_not_repaired_from_marker(
    tmp_path, monkeypatch, kind
):
    api, case = context(tmp_path, monkeypatch)
    worker = case.worker
    if kind == "unknown":
        worker = replace(
            worker, exit=replace(worker.exit, exit_observed=False, exit_code=None)
        )
    else:
        worker = replace(worker, command_sha256="f" * 64)
    monkeypatch.setattr(api, "observe_worker", lambda unused: worker)
    monkeypatch.setattr(
        api,
        "verify_external_completion_snapshot",
        lambda *args, **kwargs: pytest.fail("invalid observation reached completion"),
    )
    with pytest.raises(Exception) as rejected:
        api._run_observed_external(case.binding, case.paths, case.handoff)
    assert rejected.value.external_failure.worker is worker


def test_transport_mutation_after_saved_verification_prevents_acceptance(
    tmp_path, monkeypatch
):
    api, case = context(tmp_path, monkeypatch)
    original = api.verify_external_completion_snapshot
    snapshots = []

    def changed(*args, **kwargs):
        result = original(*args, **kwargs)
        snapshots.append(result)
        transports = list(tmp_path.glob("*/internal-source-overlap.json"))
        assert len(transports) == 1
        transports[0].write_bytes(b"replaced after verification")
        return result

    monkeypatch.setattr(api, "verify_external_completion_snapshot", changed)
    with pytest.raises(Exception) as rejected:
        api._run_observed_external(case.binding, case.paths, case.handoff)
    failure = rejected.value.external_failure
    assert failure.stage == "transport_finalization"
    assert failure.worker.exit.exit_code == 0
    assert failure.candidate_snapshot is snapshots[0]


@pytest.mark.parametrize("location", ["observation", "verification"])
def test_original_interruptions_keep_existing_progress_and_observations(
    tmp_path, monkeypatch, location
):
    api, case = context(tmp_path, monkeypatch)
    error = KeyboardInterrupt("private interruption")
    error.progress = b"existing worker progress"

    def interrupted(*args, **kwargs):
        raise error

    operation = (
        "observe_worker"
        if location == "observation"
        else "verify_external_completion_snapshot"
    )
    monkeypatch.setattr(api, operation, interrupted)
    with pytest.raises(KeyboardInterrupt) as rejected:
        api._run_observed_external(case.binding, case.paths, case.handoff)
    assert rejected.value is error
    assert error.progress == b"existing worker progress"
    failure = error.external_failure
    assert failure.worker_progress == error.progress
    assert (failure.worker is None) == (location == "observation")
