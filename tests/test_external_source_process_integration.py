"""Real invented producer children publish the evidence their parent verifies."""

import json
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import pytest
from external_completion_fixtures import _paths, external_completion_case
from test_external_source_completion import bind_fixture
from test_external_source_completion import module as completion_module
from test_external_source_process import module


def child_case(tmp_path, monkeypatch, count, mode):
    api = module()
    case = external_completion_case(tmp_path / "parent", monkeypatch, count)
    child_root = tmp_path / "child"
    case.paths = replace(
        _paths(child_root / "fixture"),
        attempt=child_root / "attempt",
        public_summary=child_root / "public.json",
    )
    bind_fixture(completion_module(), case, monkeypatch)
    monkeypatch.setattr(
        api, "resolve_external_source_profile", lambda unused: case.profile
    )
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr(api, "_worker_command", _command(child_root, count, mode))
    return api, case


def _command(child_root, count, mode):
    test_root = str(Path(__file__).parent)
    code = (
        f"import sys;sys.path.insert(0,{test_root!r});"
        "from external_source_worker_fixtures import run_child;"
        "run_child(sys.argv[1],sys.argv[2],sys.argv[3],int(sys.argv[4]),sys.argv[5])"
    )

    def command(binding, paths, transport):
        return (
            sys.executable,
            "-c",
            code,
            str(child_root),
            str(transport.directory),
            transport.expected_handoff_sha256,
            str(count),
            mode,
        )

    return command


@pytest.mark.parametrize("count", [0, 320])
def test_actual_producer_child_is_observed_and_reconstructed_once(
    tmp_path, monkeypatch, count
):
    api, case = child_case(tmp_path, monkeypatch, count, "success")
    assert not case.paths.attempt.exists()
    result = api._run_observed_external(case.binding, case.paths, case.handoff)
    assert result.worker.exit.exit_observed is True
    assert result.worker.exit.exit_code == 0
    assert len(result.snapshot.rows) == count
    assert (
        result.snapshot.payload("attempt/evidence/all-scores.jsonl")
        == (case.filesinputs["all-scores.jsonl"])
    )
    public = result.public_summary
    assert public["protected_evaluation_authorized"] is False
    assert public["composition"]["protected_evaluation_authorized"] is False


def test_child_publication_then_nonzero_exit_never_becomes_accepted(
    tmp_path, monkeypatch
):
    api, case = child_case(tmp_path, monkeypatch, 5, "nonzero")
    monkeypatch.setattr(
        api,
        "verify_external_completion_snapshot",
        lambda *args, **kwargs: pytest.fail("nonzero producer reached acceptance"),
    )
    with pytest.raises(Exception) as rejected:
        api._run_observed_external(case.binding, case.paths, case.handoff)
    assert case.paths.public_summary.is_file()
    assert rejected.value.external_failure.worker.exit.exit_code == 17
    assert rejected.value.external_failure.candidate_snapshot is None


def test_child_teardown_failure_preserves_completed_science_without_publication(
    tmp_path, monkeypatch
):
    api, case = child_case(tmp_path, monkeypatch, 5, "teardown")
    monkeypatch.setattr(
        api,
        "verify_external_completion_snapshot",
        lambda *args, **kwargs: pytest.fail("failed producer reached acceptance"),
    )
    with pytest.raises(Exception) as rejected:
        api._run_observed_external(case.binding, case.paths, case.handoff)
    assert not case.paths.public_summary.exists()
    assert rejected.value.external_failure.worker.exit.exit_code != 0
    failure = json.loads((case.paths.attempt / "external-failure.json").read_bytes())
    assert failure["cleanup_failed"] is True
    assert failure["completed"] is not None
    assert len(failure["completed"]["private_outputs_base64"]) == 30
