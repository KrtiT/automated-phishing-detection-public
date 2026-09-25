import json

import pytest
from operational_cell_io_fixtures import IDENTITY, holder, module, setup, write_working

from automated_phishing_detection._operational_cell_protocol import (
    SNAPSHOT_NAMES,
    WORKING_NAMES,
)


def test_hold_before_launch_complete_and_publish_exact36(tmp_path, monkeypatch):
    case = setup(tmp_path, monkeypatch)
    with holder(case) as completer:
        assert completer.working is completer.candidate is None
        write_working(case)
        result = completer.complete(**case.options)
        assert completer.candidate is result
        assert set(dict(result.payloads)) == set(SNAPSHOT_NAMES)
        assert tuple(name for name, _ in completer.working.payloads) == WORKING_NAMES
    assert json.loads(case.public.read_bytes()) == {"status": "invented_only"}
    assert [call[0] for call in case.calls] == ["working", "published"]
    assert case.calls[0][2]["attempt"] is case.attempt
    assert case.calls[0][2]["expected_identity"] == IDENTITY


def test_original_working_buffers_are_not_reread_after_publication(
    tmp_path, monkeypatch
):
    case = setup(tmp_path, monkeypatch)
    original, reads = module().storage.read_file, []

    def read(directory, name, initial):
        reads.append((directory.path, name))
        return original(directory, name, initial)

    monkeypatch.setattr(module().storage, "read_file", read)
    with holder(case) as completer:
        write_working(case)
        completer.complete(**case.options)
    assert len(reads) == len(set(reads)) == 36
    assert {name for path, name in reads if path == case.attempt.directory} == set(
        WORKING_NAMES
    ) | {"finalize.claim", "outcome.json"}


def test_completion_and_publication_are_not_retryable(tmp_path, monkeypatch):
    case = setup(tmp_path, monkeypatch)
    with holder(case) as completer:
        write_working(case)
        completer.complete(**case.options)
        with pytest.raises(module().OperationalCellCompletionError):
            completer.complete(**case.options)
    assert len(case.calls) == 2


def test_normal_exit_without_completion_rejects(tmp_path, monkeypatch):
    case = setup(tmp_path, monkeypatch)
    with pytest.raises(module().OperationalCellCompletionError):
        with holder(case):
            pass
    assert not case.public.exists()


def test_failed_scientific_verification_preserves_working_and_never_publishes(
    tmp_path, monkeypatch
):
    case = setup(tmp_path, monkeypatch)

    def reject(*arguments, **keywords):
        raise ValueError("invented private rejection")

    monkeypatch.setattr(module(), "_verify_working", reject)
    with pytest.raises(module().OperationalCellCompletionError):
        with holder(case) as completer:
            write_working(case)
            completer.complete(**case.options)
    assert set(path.name for path in case.attempt.directory.iterdir()) == set(
        WORKING_NAMES
    )
    assert not case.public.exists()
    assert not completer.publishing
    assert completer.working is completer.candidate is None


def test_post_publication_error_retains_candidate_and_prevents_retry(
    tmp_path, monkeypatch
):
    case = setup(tmp_path, monkeypatch)
    original = module().receipt.publish_completion

    def reject(*arguments, **keywords):
        original(*arguments, **keywords)
        raise OSError("after public rename")

    monkeypatch.setattr(module().receipt, "publish_completion", reject)
    with pytest.raises(module().OperationalCellCompletionError):
        with holder(case) as completer:
            write_working(case)
            completer.complete(**case.options)
    assert case.public.exists()
    assert completer.publishing
    assert completer.working is not None
    assert completer.candidate is None
    with pytest.raises(module().OperationalCellCompletionError):
        completer.complete(**case.options)


def test_parent_identity_must_match_actual_attempt_before_launch(tmp_path, monkeypatch):
    case = setup(tmp_path, monkeypatch)
    entered = []
    with pytest.raises(module().OperationalCellCompletionError):
        with module().hold_operational_cell(
            case.attempt, case.public, expected_identity={"kind": "different"}
        ):
            entered.append(True)
    assert not entered
