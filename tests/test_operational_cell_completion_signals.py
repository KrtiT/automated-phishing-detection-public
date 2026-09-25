import signal

import pytest
from operational_cell_io_fixtures import holder, module, setup, write_working
from operational_input_signal_fixtures import (
    assert_closed,
    interrupt_fdopen,
    watch_open,
)


@pytest.mark.parametrize("selected", ["attempt", "evidence", "run.json", "public.json"])
@pytest.mark.parametrize("boundary", ["open", "close"])
def test_real_signal_acquisition_and_close_preserve_owned_resources(
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


def test_actual_signal_after_working_fdopen_closes_file(tmp_path, monkeypatch):
    case = setup(tmp_path, monkeypatch)
    with pytest.raises(module().OperationalCellCompletionError):
        with holder(case) as completer:
            write_working(case)
            with monkeypatch.context() as guard:
                descriptors = interrupt_fdopen(guard)
                with pytest.raises(KeyboardInterrupt):
                    completer.complete(**case.options)
            assert_closed(descriptors)
            with pytest.raises(module().OperationalCellCompletionError):
                completer.complete(**case.options)


@pytest.mark.parametrize("phase", ["working", "published"])
def test_pure_validation_remains_immediately_interruptible(
    tmp_path, monkeypatch, phase
):
    case = setup(tmp_path, monkeypatch)
    reached = []

    def interrupt(*arguments, **keywords):
        signal.raise_signal(signal.SIGINT)
        reached.append("signal was deferred")

    monkeypatch.setattr(module(), f"_verify_{phase}", interrupt)
    with pytest.raises(KeyboardInterrupt):
        with holder(case) as completer:
            write_working(case)
            completer.complete(**case.options)
    assert not reached
    assert completer.publishing is (phase == "published")


def test_yielded_process_body_remains_immediately_interruptible(tmp_path, monkeypatch):
    case = setup(tmp_path, monkeypatch)
    reached = []
    with pytest.raises(KeyboardInterrupt):
        with holder(case):
            signal.raise_signal(signal.SIGINT)
            reached.append("signal was deferred")
    assert not reached
    assert not case.calls
