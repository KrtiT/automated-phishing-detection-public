"""Independent real synthetic process checks at the new admission seam."""

import json
import os
import signal
import subprocess

import pytest
from study_series_admission_fixtures import api, frame
from study_series_process_fixtures import (
    admitted_options,
    assert_closed,
    assert_resources_closed,
    issuer,
    observe,
    progress,
    resources,
)
from test_operational_process import inputs

from automated_phishing_detection import _study_series_process_children as children
from automated_phishing_detection import operational_process as original


def test_externally_reaped_admitted_client_keeps_exit_unknown(tmp_path, monkeypatch):
    attempt, options = inputs(tmp_path)
    options = admitted_options(options)
    admissions, popen = {}, subprocess.Popen

    def launch(command, **keywords):
        process = popen(command, **keywords)
        if command == options["client_command"]:
            os.waitpid(process.pid, 0)
            process.returncode = 0
        return process

    monkeypatch.setattr(subprocess, "Popen", launch)
    with pytest.raises(original.OperationalProcessError) as caught:
        observe(attempt, options, issuer(admissions))
    observed = progress(caught.value)
    assert observed["failure"] == "client_exit_unobserved"
    assert observed["client"]["exit_observed"] is False
    assert observed["client"]["exit_code"] is None
    assert admissions["client"].exit_observed is False
    assert admissions["client"].exit_code is None
    assert observed["service"]["exit_code"] == 0
    for role, admission in admissions.items():
        assert admission.pipe.descriptors == set()
        with pytest.raises(ProcessLookupError):
            os.kill(observed[role]["pid"], 0)


@pytest.mark.parametrize("first", (KeyboardInterrupt("first"), SystemExit(31)))
@pytest.mark.parametrize("later", (KeyboardInterrupt("later"), SystemExit(32)))
def test_first_observation_callback_interrupt_survives_service_callback(
    tmp_path, first, later
):
    attempt, options = inputs(tmp_path)
    admissions, callbacks = {}, []

    def issue(role, command):
        def fail(known, code):
            callbacks.append((role, known, code))
            raise first if role == "client" else later

        admissions[role] = api().SeriesParentAdmission(
            frame(command, role), on_observed=fail
        )
        return admissions[role]

    with pytest.raises(type(first)) as caught:
        observe(attempt, admitted_options(options), issue)
    assert caught.value is first
    assert callbacks == [("client", True, 0), ("service", True, 0)]
    observed = progress(caught.value)
    assert observed["status"] == "failed"
    assert observed["research_accepted"] is False
    assert_closed(admissions, observed)


def signal_after(monkeypatch, owner, name):
    function = getattr(owner, name)

    def interrupted(*arguments, **keywords):
        result = function(*arguments, **keywords)
        signal.raise_signal(signal.SIGINT)
        return result

    monkeypatch.setattr(owner, name, interrupted)


@pytest.mark.parametrize("boundary", ("validation", "enter"))
def test_signal_before_launch_keeps_series_pipe_owned(tmp_path, monkeypatch, boundary):
    attempt, options = inputs(tmp_path)
    admissions, captured = {}, resources(monkeypatch)
    owner, name = (
        (children, "validate_series_launch")
        if boundary == "validation"
        else (api().SeriesParentAdmission, "__enter__")
    )
    signal_after(monkeypatch, owner, name)
    with pytest.raises(KeyboardInterrupt) as caught:
        observe(attempt, admitted_options(options), issuer(admissions))
    observed = progress(caught.value)
    assert observed["failure"] == "parent_interrupted"
    assert set(admissions) == {"service"}
    assert admissions["service"].entered is True
    assert observed["service"]["pid"] is observed["client"]["pid"] is None
    assert_closed(admissions, observed)
    assert_resources_closed(captured)


def test_inherited_admission_and_authority_environment_is_scrubbed(
    tmp_path, monkeypatch
):
    attempt, options = inputs(tmp_path)
    admissions, environments, popen = {}, [], subprocess.Popen
    for name in (
        "APD_STUDY_ADMISSION_FD",
        "APD_STUDY_SERIES_ADMISSION_FD",
        "APD_INVENTED_AUTHORITY",
        "APD_LISTENER_FD",
    ):
        monkeypatch.setenv(name, "999999")

    def launch(command, **keywords):
        environments.append(keywords["env"])
        return popen(command, **keywords)

    monkeypatch.setattr(subprocess, "Popen", launch)
    result = observe(attempt, admitted_options(options), issuer(admissions))
    assert len(environments) == 2
    for environment in environments:
        assert "APD_STUDY_ADMISSION_FD" not in environment
        assert "APD_INVENTED_AUTHORITY" not in environment
        assert environment["APD_STUDY_SERIES_ADMISSION_FD"] != "999999"
    assert "APD_LISTENER_FD" not in environments[-1]
    assert_closed(admissions, json.loads(result.record))
