import pytest
from operational_cell_io_fixtures import holder, module, setup, write_working


@pytest.mark.parametrize("original", [KeyboardInterrupt("first"), SystemExit(0)])
@pytest.mark.parametrize("later", [OSError("later"), KeyboardInterrupt("later")])
def test_original_interruption_survives_final_check_failure(
    tmp_path, monkeypatch, original, later
):
    case = setup(tmp_path, monkeypatch)

    def fail(*arguments):
        raise later

    with pytest.raises(BaseException) as caught:
        with holder(case):
            monkeypatch.setattr(module().storage.HeldCellFiles, "check", fail)
            raise original
    assert caught.value is original


def test_late_interruption_keeps_original_progress_and_real_candidates(
    tmp_path, monkeypatch
):
    case = setup(tmp_path, monkeypatch)
    original, later = ValueError("body"), KeyboardInterrupt("later")
    original.progress = b"original immutable process bytes"

    def fail(*arguments):
        raise later

    with pytest.raises(KeyboardInterrupt) as caught:
        with holder(case) as completer:
            write_working(case)
            candidate = completer.complete(**case.options)
            monkeypatch.setattr(module().storage.HeldCellFiles, "check", fail)
            raise original
    assert caught.value is later
    assert caught.value.progress is original.progress
    assert completer.candidate is candidate and completer.working is not None


def test_failure_with_partial_child_files_does_not_require_complete_inventory(
    tmp_path, monkeypatch
):
    case = setup(tmp_path, monkeypatch)
    original = ValueError("child failed")
    with pytest.raises(ValueError) as caught:
        with holder(case) as completer:
            (case.attempt.directory / "run.json").write_bytes(b"partial")
            raise original
    assert caught.value is original
    assert (case.attempt.directory / "run.json").read_bytes() == b"partial"
    assert completer.working is completer.candidate is None
    assert not completer.publishing


def test_failed_published_verification_retains_working_only(tmp_path, monkeypatch):
    case = setup(tmp_path, monkeypatch)

    def reject(*arguments, **keywords):
        raise ValueError("invalid published record")

    monkeypatch.setattr(module(), "_verify_published", reject)
    with pytest.raises(module().OperationalCellCompletionError):
        with holder(case) as completer:
            write_working(case)
            completer.complete(**case.options)
    assert case.public.exists()
    assert completer.working is not None and completer.candidate is None
    assert completer.publishing


def test_closed_holder_cannot_start_completion(tmp_path, monkeypatch):
    case = setup(tmp_path, monkeypatch)
    with pytest.raises(module().OperationalCellCompletionError):
        with holder(case) as completer:
            pass
    with pytest.raises(module().OperationalCellCompletionError):
        completer.complete(**case.options)
    assert not case.calls
