"""Real interruption boundaries retain root observations and partial files."""

import asyncio
import os
import signal
from contextlib import contextmanager

import pytest
import study_runner_fixtures as fixtures
from study_run_record_fixtures import prepared

from automated_phishing_detection import execution_receipt as receipt

__all__ = ["prepared"]


def test_reservation_assignment_precedes_deferred_signal(
    tmp_path, prepared, monkeypatch
):
    case = fixtures.setup(tmp_path, prepared, monkeypatch)
    original = receipt.reserve_attempt
    obtained = []

    def reserve(*arguments, **keywords):
        result = original(*arguments, **keywords)
        obtained.append(result)
        os.kill(os.getpid(), signal.SIGINT)
        return result

    monkeypatch.setattr(receipt, "reserve_attempt", reserve)
    with pytest.raises(KeyboardInterrupt) as caught:
        fixtures.execute(case)
    assert caught.value.study_failure.attempt is obtained[0]
    assert "prepare" not in case.events


def test_late_holder_signal_rejects_without_overwriting_published_hold(
    tmp_path, prepared, monkeypatch
):
    case = fixtures.setup(tmp_path, prepared, monkeypatch)
    original = case.body.held_preparation

    @contextmanager
    def late(*arguments):
        with original(*arguments) as retained:
            yield retained
        os.kill(os.getpid(), signal.SIGINT)

    def forbidden(*arguments, **keywords):
        pytest.fail("attempted replacement failure after publication")

    monkeypatch.setattr(case.body, "held_preparation", late)
    monkeypatch.setattr(receipt, "record_failure", forbidden)
    with pytest.raises(KeyboardInterrupt) as caught:
        fixtures.execute(case)
    assert caught.value.study_failure.publishing
    assert caught.value.study_failure.candidate is not None
    assert case.paths.public_summary.exists()


@pytest.mark.parametrize(
    "first",
    [KeyboardInterrupt("first"), SystemExit(19), asyncio.CancelledError("first")],
)
def test_first_interruption_survives_failure_persistence(
    tmp_path, prepared, monkeypatch, first
):
    case = fixtures.setup(tmp_path, prepared, monkeypatch)

    def fail(*arguments, **keywords):
        raise first

    def later(*arguments, **keywords):
        os.kill(os.getpid(), signal.SIGINT)

    monkeypatch.setattr(case.body, "_run_bound_preparation", fail)
    monkeypatch.setattr(receipt, "record_failure", later)

    async def catch_inside_task():
        with pytest.raises(BaseException) as caught:
            await case.module._run_bound_study(
                case.binding, case.profile, paths=case.paths, deadlines=case.deadlines
            )
        return caught.value

    caught = asyncio.run(catch_inside_task())
    assert caught is first
    assert caught.study_failure.preparation is None


def test_existing_study_association_survives_later_cleanup_signal(
    tmp_path, prepared, monkeypatch
):
    case = fixtures.setup(tmp_path, prepared, monkeypatch)
    original = ValueError("earlier")
    original.study_failure = object()
    original.preparation_progress = b"original progress"
    holder = case.module.hold_study_root

    def fail(*arguments):
        raise original

    @contextmanager
    def late(*arguments, **keywords):
        try:
            with holder(*arguments, **keywords) as retained:
                yield retained
        finally:
            os.kill(os.getpid(), signal.SIGINT)

    monkeypatch.setattr(case.body, "_run_bound_preparation", fail)
    monkeypatch.setattr(case.module, "hold_study_root", late)
    with pytest.raises(KeyboardInterrupt) as caught:
        fixtures.execute(case)
    assert caught.value.study_failure is original.study_failure
    assert caught.value.preparation_progress == original.preparation_progress


def test_cancelled_root_context_survives_python_task_boundary(
    tmp_path, prepared, monkeypatch
):
    case = fixtures.setup(tmp_path, prepared, monkeypatch)
    first = asyncio.CancelledError("original")

    def fail(*arguments):
        raise first

    monkeypatch.setattr(case.body, "_run_bound_preparation", fail)
    with pytest.raises(asyncio.CancelledError) as caught:
        fixtures.execute(case)
    assert case.module.study_failure(caught.value) is first.study_failure
