"""Independent real-file lifecycle review with explicitly stubbed science."""

import signal

import pytest
from operational_input_signal_fixtures import assert_closed, watch_open
from study_series_completion_fixtures import holder, module, setup, write_working


def forbidden(*arguments, **keywords):
    pytest.fail("fresh completion invoked a legacy verifier or fabricated observation")


@pytest.mark.parametrize("selected", ["attempt", "evidence", "run.json", "public.json"])
@pytest.mark.parametrize("boundary", ["open", "close"])
def test_real_signal_resource_boundaries_remain_owned_and_closed(
    tmp_path, monkeypatch, selected, boundary
):
    case = setup(tmp_path, monkeypatch)
    with monkeypatch.context() as guard:
        descriptors = watch_open(
            guard,
            selected,
            interrupt=boundary == "open",
            before_close=boundary == "close",
        )
        with pytest.raises(KeyboardInterrupt):
            with holder(case) as completer:
                write_working(case)
                completer.complete(**case.options)
    assert descriptors
    assert_closed(descriptors)
    assert (case.attempt.directory / "reservation.json").exists()


def test_new_completion_never_invokes_legacy_verifiers_or_builds_observation(
    tmp_path, monkeypatch
):
    case = setup(tmp_path, monkeypatch)
    current = module()
    for name in ("_verify_working", "_verify_published", "_build_public"):
        monkeypatch.setattr(current.original, name, forbidden)
    for name in ("complete", "_publish"):
        monkeypatch.setattr(current.original._Completer, name, forbidden)
    monkeypatch.setattr(current.ProcessObservation, "__init__", forbidden)
    with holder(case) as completer:
        write_working(case)
        candidate = completer.complete(**case.options)
    assert candidate is completer.candidate
    assert [call[0] for call in case.calls] == ["working", "published"]
    assert case.calls[0][2]["observation"] is case.options["observation"]


def test_untyped_observation_cannot_read_working_files_or_retry(tmp_path, monkeypatch):
    case = setup(tmp_path, monkeypatch)
    monkeypatch.setattr(module().storage.HeldCellFiles, "read_working", forbidden)
    with pytest.raises(module().OperationalCellCompletionError):
        with holder(case) as completer:
            write_working(case)
            with pytest.raises(module().OperationalCellCompletionError):
                completer.complete(**(case.options | {"observation": None}))
            with pytest.raises(module().OperationalCellCompletionError):
                completer.complete(**case.options)
    assert completer.working is completer.candidate is None
    assert not completer.publishing and not case.calls


def test_yielded_parent_body_is_not_inside_deferred_interrupt_region(
    tmp_path, monkeypatch
):
    case = setup(tmp_path, monkeypatch)
    reached = []
    with pytest.raises(KeyboardInterrupt):
        with holder(case) as completer:
            signal.raise_signal(signal.SIGINT)
            reached.append("interruption was deferred")
    assert not reached and not case.calls
    assert completer.working is completer.candidate is None
    with pytest.raises(module().OperationalCellCompletionError):
        completer.complete(**case.options)
