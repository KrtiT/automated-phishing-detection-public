import asyncio
import inspect
import json

import pytest
from study_admission_fixtures import frame, module
from test_operational_process import inputs
from test_operational_process_writer import pair_options

from automated_phishing_detection import operational_process as operational


def options_with_admission(options):
    for role in ("service", "client"):
        executable, flag, code, mode = options[f"{role}_command"]
        prefix = (
            "import os, sys\n"
            "from automated_phishing_detection._study_admission import "
            "consume_child_admission\n"
            "arguments=(sys.executable, '-c', sys.argv[2], sys.argv[1], sys.argv[2])\n"
            f"admission=consume_child_admission({role!r}, arguments)\n"
        )
        combined = prefix + code + "\nadmission.check()\nadmission.close()\n"
        options[f"{role}_command"] = executable, flag, combined, mode, combined
    return options


def observe(attempt, options, admissions):
    parameters = inspect.signature(operational._observe_pair_with_writer).parameters
    assert "study_admissions" in parameters, "missing private study pair admission"
    return asyncio.run(
        operational._observe_pair_with_writer(
            attempt,
            **pair_options(options),
            writer=operational._record,
            study_admissions=admissions,
        )
    )


def test_owned_pair_keeps_both_writers_through_actual_service_cleanup(tmp_path):
    attempt, options = inputs(tmp_path)
    admissions = {}

    def issue(role, command):
        admissions[role] = module().ParentAdmission(frame(command, role))
        return admissions[role]

    observed = observe(attempt, options_with_admission(options), issue)
    progress = json.loads(observed.record)
    assert progress["status"] == "observed"
    assert set(admissions) == {"service", "client"}
    for role, admission in admissions.items():
        assert admission.pid == progress[role]["pid"]
        assert admission.exit_observed is True
        assert admission.exit_code == 0
        assert admission.pipe.descriptors == set()


@pytest.mark.parametrize("role", ("service", "client"))
def test_launch_callback_failure_retains_actual_children_and_cleans_writers(
    tmp_path, role
):
    attempt, options = inputs(tmp_path)
    admissions = {}

    def failure(pid):
        raise ValueError("ledger failed after launch")

    def issue(actual_role, command):
        admission = module().ParentAdmission(
            frame(command, actual_role),
            on_launched=failure if role == actual_role else None,
        )
        admissions[actual_role] = admission
        return admission

    with pytest.raises(operational.OperationalProcessError) as caught:
        observe(attempt, options_with_admission(options), issue)
    progress = json.loads(caught.value.progress)
    for actual_role, admission in admissions.items():
        assert admission.pid == progress[actual_role]["pid"]
        assert admission.exit_observed is True
        assert admission.pipe.descriptors == set()


def test_public_process_pair_has_no_admission_override():
    assert (
        "study_admissions"
        not in inspect.signature(operational.observe_process_pair).parameters
    )


@pytest.mark.parametrize("role", ("service", "client"))
@pytest.mark.parametrize("event", ("launched", "observed"))
def test_pair_callback_interrupt_preserves_error_and_actual_observations(
    tmp_path, role, event
):
    attempt, options = inputs(tmp_path)
    admissions, failure = {}, KeyboardInterrupt("first")

    def interrupt(*unused):
        raise failure

    def issue(actual_role, command):
        callbacks = {f"on_{event}": interrupt} if actual_role == role else {}
        admission = module().ParentAdmission(frame(command, actual_role), **callbacks)
        admissions[actual_role] = admission
        return admission

    with pytest.raises(KeyboardInterrupt) as caught:
        observe(attempt, options_with_admission(options), issue)
    assert caught.value is failure
    progress = json.loads(operational.process_progress(caught.value))
    for actual_role, admission in admissions.items():
        assert admission.pid == progress[actual_role]["pid"]
        assert admission.exit_observed is True
        assert admission.pipe.descriptors == set()


def test_observation_interrupt_survives_later_finish_failure(tmp_path, monkeypatch):
    attempt, options = inputs(tmp_path)
    failure = KeyboardInterrupt("first")

    def interrupt(*unused):
        raise failure

    def issue(role, command):
        return module().ParentAdmission(
            frame(command, role), on_observed=interrupt if role == "service" else None
        )

    def fail_finish(*unused):
        raise ValueError("later diagnostic failure")

    monkeypatch.setattr(operational.Observations, "finish", fail_finish)
    with pytest.raises(KeyboardInterrupt) as caught:
        observe(attempt, options_with_admission(options), issue)
    assert caught.value is failure
    progress = json.loads(operational.process_progress(caught.value))
    assert progress["service"]["exit_observed"] is True
    assert progress["client"]["exit_observed"] is True
