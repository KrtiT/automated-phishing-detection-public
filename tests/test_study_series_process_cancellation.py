"""Cancellation and late cleanup faults retain original interruptions and exits."""

import asyncio
import os
import signal
import subprocess

import pytest
from study_series_process_fixtures import (
    admitted_options,
    assert_closed,
    assert_resources_closed,
    cancel_pair,
    issuer,
    observe,
    progress,
    resources,
)
from test_operational_process import inputs

from automated_phishing_detection import operational_process as original


@pytest.mark.parametrize("phase", ("startup", "client"))
@pytest.mark.parametrize("repeated", (False, True))
def test_cancellation_preserves_admitted_children_and_closes_writers(
    tmp_path, monkeypatch, phase, repeated
):
    service = "startup_hang" if phase == "startup" else "slow_cleanup"
    attempt, options = inputs(tmp_path, service=service, client="block")
    options["shutdown_timeout_seconds"] = 0.05 if phase == "startup" else 1.0
    trigger = "service-started.json" if phase == "startup" else "measured.json"
    admissions, captured = {}, resources(monkeypatch)
    observed = asyncio.run(
        asyncio.wait_for(
            cancel_pair(
                attempt,
                admitted_options(options),
                issuer(admissions),
                trigger,
                repeated,
            ),
            timeout=10,
        )
    )
    assert observed["failure"] == "parent_cancelled"
    assert observed["status"] == "failed"
    assert set(admissions) == (
        {"service"} if phase == "startup" else {"service", "client"}
    )
    if phase == "client":
        assert observed["service"]["exit_code"] == 0
        assert observed["service"]["forced"] is False
    assert_closed(admissions, observed)
    assert_resources_closed(captured)


def _late_failure(monkeypatch, phase, later, first=None):
    run, finish = original._run, original._finish_cleanup

    def close():
        raise later

    async def running(children, *arguments):
        if phase == "descriptor":
            children.stack.callback(close)
        await run(children, *arguments)
        if first is not None:
            raise first

    async def cleanup(*arguments):
        result = await finish(*arguments)
        if phase == "finish":
            raise later
        return result

    monkeypatch.setattr(original, "_run", running)
    monkeypatch.setattr(original, "_finish_cleanup", cleanup)


@pytest.mark.parametrize("phase", ("finish", "descriptor"))
@pytest.mark.parametrize("first", (KeyboardInterrupt("first"), SystemExit(17)))
@pytest.mark.parametrize("later", (OSError("later"), KeyboardInterrupt("later")))
def test_late_cleanup_failure_preserves_first_interrupt(
    tmp_path, monkeypatch, phase, first, later
):
    attempt, options = inputs(tmp_path)
    admissions, captured = {}, resources(monkeypatch)
    _late_failure(monkeypatch, phase, later, first)
    with pytest.raises(type(first)) as caught:
        observe(attempt, admitted_options(options), issuer(admissions))
    assert caught.value is first
    observed = progress(caught.value)
    assert observed["service"]["exit_code"] == observed["client"]["exit_code"] == 0
    assert observed["status"] == "failed"
    assert_closed(admissions, observed)
    assert_resources_closed(captured)


@pytest.mark.parametrize("phase", ("finish", "descriptor"))
@pytest.mark.parametrize("later", (OSError("later"), KeyboardInterrupt("later")))
def test_late_failure_does_not_replace_original_cancellation(
    tmp_path, monkeypatch, phase, later
):
    attempt, options = inputs(tmp_path, client="block")
    admissions, captured = {}, resources(monkeypatch)
    _late_failure(monkeypatch, phase, later)
    observed = asyncio.run(
        asyncio.wait_for(
            cancel_pair(
                attempt,
                admitted_options(options),
                issuer(admissions),
                "measured.json",
            ),
            timeout=10,
        )
    )
    assert observed["failure"] == "parent_cancelled"
    assert observed["service"]["exit_code"] == 0
    assert_closed(admissions, observed)
    assert_resources_closed(captured)


@pytest.mark.parametrize("first", (KeyboardInterrupt("claim"), SystemExit(23)))
def test_claim_interruption_preserves_object_and_actual_empty_progress(tmp_path, first):
    attempt, options = inputs(tmp_path)
    admissions = {}

    def write(receipt, name, content):
        original._record(receipt, name, content)
        raise first

    with pytest.raises(type(first)) as caught:
        observe(attempt, admitted_options(options), issuer(admissions), write)
    assert caught.value is first
    observed = progress(caught.value)
    assert observed["service"]["pid"] is observed["client"]["pid"] is None
    assert not admissions
    assert set(os.listdir(attempt.directory)) == {
        "reservation.json",
        "process-pair-intent.json",
    }


def test_real_sigint_at_popen_return_keeps_child_owned(tmp_path, monkeypatch):
    attempt, options = inputs(tmp_path)
    admissions, spawned, captured = {}, [], resources(monkeypatch)
    popen = subprocess.Popen

    def launch(*arguments, **keywords):
        process = popen(*arguments, **keywords)
        spawned.append(process)
        signal.raise_signal(signal.SIGINT)
        return process

    monkeypatch.setattr(subprocess, "Popen", launch)
    try:
        with pytest.raises(KeyboardInterrupt) as caught:
            observe(attempt, admitted_options(options), issuer(admissions))
        observed = progress(caught.value)
        assert len(spawned) == 1
        assert observed["service"]["pid"] == spawned[0].pid
        assert_closed(admissions, observed)
        assert_resources_closed(captured)
    finally:
        for process in spawned:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)
