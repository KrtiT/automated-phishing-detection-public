"""Synthetic admitted failures retain actual exits and close owned resources."""

import json
import os
import signal

import pytest
from study_series_process_fixtures import (
    admitted_options,
    assert_closed,
    assert_resources_closed,
    assert_writers_live,
    issuer,
    observe,
    progress,
    resources,
)
from test_operational_process import inputs
from test_operational_process_writer import PARENT_NAMES

from automated_phishing_detection import operational_process as original
from automated_phishing_detection._operational_attempt_io import held_attempt_writer
from automated_phishing_detection._operational_cell_protocol import WORKING_NAMES


@pytest.mark.parametrize(
    "mode", ("startup_death", "wrong_ready", "missing_ready", "startup_hang")
)
def test_startup_failure_never_admits_or_launches_client(tmp_path, monkeypatch, mode):
    attempt, options = inputs(tmp_path, service=mode)
    if mode == "startup_hang":
        options.update(startup_timeout_seconds=0.05, shutdown_timeout_seconds=0.05)
    admissions, captured = {}, resources(monkeypatch)
    with pytest.raises(original.OperationalProcessError) as caught:
        observe(attempt, admitted_options(options), issuer(admissions))
    observed = progress(caught.value)
    assert observed["status"] == "failed"
    assert observed["research_accepted"] is False
    assert set(admissions) == {"service"}
    assert observed["client"]["pid"] is observed["client"]["exit_code"] is None
    assert not (attempt.directory / "client-intent.json").exists()
    assert_closed(admissions, observed)
    assert_resources_closed(captured)


@pytest.mark.parametrize("mode,code", (("nonzero", 17), ("signal", -signal.SIGTERM)))
def test_client_failure_preserves_checkpoint_and_graceful_service_exit(
    tmp_path, mode, code
):
    attempt, options = inputs(tmp_path, client=mode)
    admissions = {}
    with pytest.raises(original.OperationalProcessError) as caught:
        observe(attempt, admitted_options(options), issuer(admissions))
    observed = progress(caught.value)
    assert observed["client"]["exit_code"] == code
    assert observed["service"]["exit_code"] == 0
    assert observed["service"]["forced"] is False
    assert observed["cleanup_sha256"] is not None
    assert (attempt.directory / "measured.json").read_bytes() == b'{"fixture": true}\n'
    assert_closed(admissions, observed)


@pytest.mark.parametrize(
    "mode", ("cleanup_nonzero", "wrong_cleanup", "ignore_stop", "term_zero")
)
def test_unclean_service_shutdown_is_never_observed_success(tmp_path, mode):
    attempt, options = inputs(tmp_path, service=mode)
    options["shutdown_timeout_seconds"] = 0.1
    admissions = {}
    with pytest.raises(original.OperationalProcessError) as caught:
        observe(attempt, admitted_options(options), issuer(admissions))
    observed = progress(caught.value)
    assert observed["status"] == "failed"
    assert observed["client"]["exit_code"] == 0
    if mode in ("ignore_stop", "term_zero"):
        assert observed["service"]["forced"] is True
        assert observed["service"]["signals"][0] == signal.SIGTERM
    assert_closed(admissions, observed)


def test_service_death_reaps_the_admitted_client(tmp_path):
    attempt, options = inputs(tmp_path, service="service_dies", client="block")
    admissions = {}
    with pytest.raises(original.OperationalProcessError) as caught:
        observe(attempt, admitted_options(options), issuer(admissions))
    observed = progress(caught.value)
    assert observed["service"]["exit_code"] == 12
    assert observed["client"]["forced"] is True
    assert_closed(admissions, observed)


@pytest.mark.parametrize(
    "failed_name",
    (
        "service-started.json",
        "client-started.json",
        "service-stop.json",
        "service-process.json",
        "process-pair.json",
    ),
)
def test_failed_record_is_not_retried_and_does_not_leak(
    tmp_path, monkeypatch, failed_name
):
    attempt, options = inputs(tmp_path)
    admissions, writes, captured = {}, [], resources(monkeypatch)

    def write(receipt, name, content):
        writes.append(name)
        if name == failed_name:
            raise OSError("private write failure")
        original._record(receipt, name, content)

    with pytest.raises(original.OperationalProcessError) as caught:
        observe(attempt, admitted_options(options), issuer(admissions), write)
    observed = progress(caught.value)
    assert writes.count(failed_name) == 1
    assert failed_name in observed["record_failures"]
    assert "private" not in str(caught.value)
    assert_closed(admissions, observed)
    assert_resources_closed(captured)


@pytest.mark.parametrize("role", ("service", "client"))
@pytest.mark.parametrize(
    "error", (OSError("private issuer"), KeyboardInterrupt("first"))
)
def test_issuer_failure_retains_only_actual_process_exits(tmp_path, role, error):
    attempt, options = inputs(tmp_path)
    admissions = {}
    create = issuer(admissions)

    def issue(actual_role, command):
        if actual_role == role:
            raise error
        return create(actual_role, command)

    expected = (
        KeyboardInterrupt
        if isinstance(error, KeyboardInterrupt)
        else original.OperationalProcessError
    )
    with pytest.raises(expected) as caught:
        observe(attempt, admitted_options(options), issue)
    observed = progress(caught.value)
    if isinstance(error, KeyboardInterrupt):
        assert caught.value is error
    assert observed[role]["pid"] is observed[role]["exit_code"] is None
    assert_closed(admissions, observed)


def test_held_writer_preserves_exact_names_and_closes_every_descriptor(
    tmp_path, monkeypatch
):
    attempt, options = inputs(tmp_path)
    admissions, names, captured = {}, [], resources(monkeypatch)
    with held_attempt_writer(attempt, names=PARENT_NAMES) as writer:
        descriptor = writer.directory.descriptor

        def retain(receipt, name, content):
            assert receipt is attempt
            if name == "service-process.json":
                assert_writers_live(admissions)
            writer.retain(name, content)
            names.append(name)
            assert set(os.listdir(attempt.directory)) <= set(WORKING_NAMES)

        observed = observe(
            attempt, admitted_options(options), issuer(admissions), retain
        )
        writer.check()
    assert len(names) == len(set(names)) and set(names) == set(PARENT_NAMES)
    assert observed.record == (attempt.directory / "process-pair.json").read_bytes()
    assert_closed(admissions, json.loads(observed.record))
    assert_resources_closed(captured)
    with pytest.raises(OSError):
        os.fstat(descriptor)
