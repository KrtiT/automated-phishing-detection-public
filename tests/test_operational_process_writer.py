"""The cell writer shares owned supervision without transient staging entries."""

import asyncio
import inspect
import json
import os

import pytest
from test_operational_process import assert_reaped, inputs, operational

from automated_phishing_detection._operational_attempt_io import held_attempt_writer
from automated_phishing_detection._operational_cell_protocol import WORKING_NAMES

__all__ = ["operational"]

CHILD_NAMES = {
    "service-role.json",
    "client-role.json",
    "service-ready.json",
    "service-cleanup.json",
    "warmup.json",
    "measured.json",
    "run.json",
}
PARENT_NAMES = tuple(
    name for name in WORKING_NAMES if name not in CHILD_NAMES | {"reservation.json"}
)


def pair_options(options):
    return {
        "service_command": options["service_command"],
        "client_command": options["client_command"],
        "deadlines": {
            name: options[f"{name}_timeout_seconds"]
            for name in ("startup", "shutdown", "terminate", "kill")
        },
    }


def observe(api, attempt, options, writer):
    assert hasattr(api, "_observe_pair_with_writer"), "missing private cell writer seam"
    return asyncio.run(
        api._observe_pair_with_writer(attempt, **pair_options(options), writer=writer)
    )


def test_cell_writer_keeps_exact_owned_process_behavior_without_staging(
    operational, tmp_path
):
    attempt, options = inputs(tmp_path)
    for role in ("service", "client"):
        executable, flag, code, mode = options[f"{role}_command"]
        options[f"{role}_command"] = (
            executable,
            flag,
            "import os; os.umask(0o077)\n" + code,
            mode,
        )
    retained = []
    with held_attempt_writer(attempt, names=PARENT_NAMES) as writer:

        def retain(attempt_value, name, content):
            assert attempt_value is attempt
            writer.retain(name, content)
            retained.append(name)
            assert set(os.listdir(attempt.directory)) <= set(WORKING_NAMES)

        observed = observe(operational, attempt, options, retain)
    assert set(retained) == set(PARENT_NAMES)
    assert len(retained) == len(PARENT_NAMES)
    assert observed.record == (attempt.directory / "process-pair.json").read_bytes()
    progress = json.loads(observed.record)
    assert progress["status"] == "observed" and progress["research_accepted"] is False
    assert_reaped(progress)


@pytest.mark.parametrize("error", [KeyboardInterrupt("first"), SystemExit(17)])
def test_interrupted_initial_claim_retains_only_actual_progress(
    operational, tmp_path, error
):
    attempt, options = inputs(tmp_path)

    def retain(attempt_value, name, content):
        assert name == "process-pair-intent.json"
        operational._record(attempt_value, name, content)
        raise error

    with pytest.raises(type(error)) as caught:
        observe(operational, attempt, options, retain)
    assert caught.value is error
    progress = json.loads(operational.process_progress(caught.value))
    assert progress["status"] == "failed"
    assert progress["service"]["pid"] is progress["client"]["pid"] is None
    assert progress["service"]["exit_code"] is progress["client"]["exit_code"] is None
    assert set(os.listdir(attempt.directory)) == {
        "reservation.json",
        "process-pair-intent.json",
    }


@pytest.mark.parametrize("bad", [None, False, b"writer", 1])
def test_invalid_writer_cannot_consume_attempt(operational, tmp_path, bad):
    attempt, options = inputs(tmp_path)
    with pytest.raises(operational.OperationalProcessError):
        observe(operational, attempt, options, bad)
    assert os.listdir(attempt.directory) == ["reservation.json"]


def test_public_supervisor_has_no_writer_override(operational):
    parameters = inspect.signature(operational.observe_process_pair).parameters
    assert "writer" not in parameters and "_writer" not in parameters
