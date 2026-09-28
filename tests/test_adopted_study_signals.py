"""First interruption and permanent publication guards also apply to adopted roots."""

import asyncio
import os
import signal
from contextlib import contextmanager

import adopted_study_fixtures as fixtures
import pytest
from study_run_record_fixtures import prepared

from automated_phishing_detection import execution_receipt as receipt

__all__ = ["prepared"]


@pytest.mark.parametrize(
    "first",
    [KeyboardInterrupt("first"), SystemExit(23), asyncio.CancelledError("first")],
)
def test_first_interruption_keeps_authorization_and_scientific_progress(
    tmp_path, prepared, monkeypatch, first
):
    case = fixtures.setup(tmp_path, prepared, monkeypatch)

    def fail(*arguments):
        raise first

    def later(*arguments, **keywords):
        raise KeyboardInterrupt("later")

    monkeypatch.setattr(case.body, "_run_bound_preparation", fail)
    monkeypatch.setattr(receipt, "record_failure", later)

    async def capture():
        with pytest.raises(BaseException) as caught:
            await case.module._run_adopted_bound(case.authorization)
        return caught.value

    caught = asyncio.run(capture())
    assert caught is first
    assert (
        case.module.adopted_study_failure(caught).scientific.attempt
        is caught.study_failure.attempt
    )


def test_late_holder_interrupt_keeps_publication_and_ledger(
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
        pytest.fail("attempted replacement publication")

    monkeypatch.setattr(case.body, "held_preparation", late)
    monkeypatch.setattr(receipt, "record_failure", forbidden)
    with pytest.raises(KeyboardInterrupt) as caught:
        fixtures.execute(case)
    retained = case.module.adopted_study_failure(caught.value)
    assert retained.scientific.publishing
    assert retained.scientific.candidate is not None
    assert case.paths.public_summary.exists()


def test_late_ledger_snapshot_interrupt_preserves_first_interruption(
    tmp_path, prepared, monkeypatch
):
    case = fixtures.setup(tmp_path, prepared, monkeypatch)
    first, later = KeyboardInterrupt("first"), KeyboardInterrupt("later")

    def fail(*arguments):
        raise first

    def fail_snapshot(*arguments):
        raise later

    monkeypatch.setattr(case.body, "_run_bound_preparation", fail)
    monkeypatch.setattr(
        fixtures.api("_adopted_study_ledger").AdmissionLedger, "snapshot", fail_snapshot
    )
    with pytest.raises(KeyboardInterrupt) as caught:
        fixtures.execute(case)
    assert caught.value is first
