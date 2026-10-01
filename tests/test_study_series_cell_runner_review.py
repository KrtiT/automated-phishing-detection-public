"""Independent actual-fact retention and all-holder acceptance ordering checks."""

import base64
import json
from contextlib import contextmanager

import pytest
from study_series_cell_runner_fixtures import (
    candidates,
    execute,
    inputs,
    install_observer,
    manifests,
    series_case,
    setup,
)
from study_series_ledger_fixtures import commands, snapshot
from test_study_series_cell_runner import api

from automated_phishing_detection._study_history_snapshot_records import digest

__all__ = ["candidates", "manifests", "series_case"]


def tracked_manager(manager, events, name, *, fail=False):
    @contextmanager
    def held(*args, **kwargs):
        events.append(f"{name}_opened")
        try:
            with manager(*args, **kwargs) as value:
                yield value
        finally:
            events.append(f"{name}_closed")
            if fail:
                raise OSError("invented later holder failure")

    return held


def track_holders(case, monkeypatch, *, fail=None):
    for module, attribute, name in (
        (api(), "retain_operational_cell_inputs", "retention"),
        (api().context.storage, "hold", "inputs"),
        (api(), "hold_series_cell", "completion"),
        (api(), "held_attempt_writer", "writer"),
    ):
        monkeypatch.setattr(
            module,
            attribute,
            tracked_manager(
                getattr(module, attribute),
                case.events,
                name,
                fail=name == fail,
            ),
        )

    def recheck(public):
        case.events.append("public_recheck")

    monkeypatch.setattr(api(), "recheck_series_public_execution", recheck)


def test_acceptance_waits_for_every_holder_and_final_public_check(
    tmp_path, series_case, monkeypatch
):
    case = setup(tmp_path, series_case, monkeypatch)
    install_observer(case, monkeypatch)
    track_holders(case, monkeypatch)
    accept = case.ledger.accept_cell

    def accepted(*args):
        expected = [
            "writer_closed",
            "completion_closed",
            "inputs_closed",
            "retention_closed",
        ]
        closed = [event for event in case.events if event.endswith("_closed")]
        assert closed == expected and case.events[-1] == "public_recheck"
        case.events.append("accepted")
        return accept(*args)

    monkeypatch.setattr(case.ledger, "accept_cell", accepted)
    with inputs(case):
        execute(case)
    assert case.events[-1] == "accepted"


@pytest.mark.parametrize("holder", ("completion", "inputs", "retention"))
def test_each_late_holder_failure_retains_full_stopped_candidate_without_republishing(
    tmp_path, series_case, monkeypatch, holder
):
    case = setup(tmp_path, series_case, monkeypatch)
    install_observer(case, monkeypatch)
    track_holders(case, monkeypatch, fail=holder)
    monkeypatch.setattr(
        case.ledger, "accept_cell", lambda *args: pytest.fail("early acceptance")
    )
    with inputs(case), pytest.raises(ValueError) as caught:
        execute(case)
    saved = snapshot(case.ledger)
    stopped = saved["cells"][0]
    assert_complete_stop(case, caught.value.operational_failure, stopped)
    assert all(cell["status"] == "unattempted" for cell in saved["cells"][1:])
    assert all(not entry["accepted"] for entry in saved["admissions"])
    with pytest.raises(ValueError):
        case.ledger.issue("service", commands(case.selected)[0])


def assert_complete_stop(case, failure, stopped):
    assert stopped["status"] == "stopped" and stopped["stage"] == "finalization"
    assert stopped["publishing"] is True and stopped["holders_closed"] is False
    assert stopped["reservation_sha256"] == case.selected.attempt.reservation_sha256
    assert (
        base64.b64decode(stopped["descriptor_bytes"])
        == case.selected.selected.descriptor_bytes
    )
    assert base64.b64decode(stopped["binding_bytes"]) == case.selected.binding
    assert base64.b64decode(stopped["observation_bytes"]) == case.observation.record
    assert (
        base64.b64decode(stopped["pair_intent_bytes"])
        == (case.selected.attempt.directory / "process-pair-intent.json").read_bytes()
    )
    assert failure.candidate is not None and failure.observation is case.observation
    assert stopped["snapshot_sha256"] == {
        name: digest(content) for name, content in failure.candidate.payloads
    }
    assert len(stopped["snapshot_sha256"]) == 36
    assert stopped["progress_bytes"] is None
    outcome = json.loads(
        (case.selected.attempt.directory / "outcome.json").read_bytes()
    )
    assert outcome["status"] == "completion_prepared"


@pytest.mark.parametrize("interrupted", (False, True))
def test_observer_progress_survives_writer_failure_with_no_fabricated_observation(
    tmp_path, series_case, monkeypatch, interrupted
):
    from automated_phishing_detection._operational_process_records import Observations

    case = setup(tmp_path, series_case, monkeypatch)
    original = (
        KeyboardInterrupt() if interrupted else ValueError("invented observer stop")
    )
    track_holders(case, monkeypatch, fail="writer")

    async def observe(attempt, **options):
        progress = Observations(attempt, options["writer"])
        progress.claim(
            options["service_command"], options["client_command"], options["deadlines"]
        )
        progress.fail("invented_process_stop")
        original.progress = progress.snapshot()
        raise original

    monkeypatch.setattr(api(), "observe_series_operational_children", observe)
    with inputs(case), pytest.raises(type(original)) as caught:
        execute(case)
    stopped = snapshot(case.ledger)["cells"][0]
    assert base64.b64decode(stopped["progress_bytes"]) == original.progress
    assert stopped["observation_bytes"] is None and stopped["snapshot_sha256"] is None
    assert stopped["pair_intent_bytes"] is not None
    if interrupted:
        assert caught.value is original
