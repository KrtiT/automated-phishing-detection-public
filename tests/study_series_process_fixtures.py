"""Invented process receipts and synthetic admitted service/client pairs."""

import asyncio
import importlib
import importlib.util
import json
import os
from types import SimpleNamespace

import pytest
from study_series_admission_fixtures import api, frame
from test_operational_process import assert_reaped
from test_operational_process_writer import pair_options

from automated_phishing_detection import operational_process as original


def module():
    name = "automated_phishing_detection.study_series_process"
    assert importlib.util.find_spec(name), "missing series process observer"
    return importlib.import_module(name)


def admitted_options(options):
    for role in ("service", "client"):
        executable, flag, code, mode = options[f"{role}_command"]
        prefix = (
            "import os, sys\n"
            "os.umask(0o077)\n"
            "from automated_phishing_detection._study_series_admission import "
            "consume_series_admission\n"
            "arguments=(sys.executable, '-c', sys.argv[2], sys.argv[1], sys.argv[2])\n"
            f"admission=consume_series_admission({role!r}, arguments)\n"
        )
        combined = prefix + code + "\nadmission.check()\nadmission.close()\n"
        options[f"{role}_command"] = executable, flag, combined, mode, combined
    return options


def observe(attempt, options, admissions, writer=original._record):
    return asyncio.run(
        module().observe_series_operational_children(
            attempt,
            **pair_options(options),
            writer=writer,
            series_admissions=admissions,
        )
    )


def issuer(admissions):
    def issue(role, command):
        admissions[role] = api().SeriesParentAdmission(frame(command, role))
        return admissions[role]

    return issue


def progress(error):
    return json.loads(original.process_progress(error))


def assert_closed(admissions, observed):
    for role, admission in admissions.items():
        assert admission.pipe.descriptors == set()
        assert admission.pid == observed[role]["pid"]
        if admission.pid is not None:
            assert admission.exit_observed == observed[role]["exit_observed"]
            assert admission.exit_code == observed[role]["exit_code"]
    assert_reaped(observed)


def assert_writers_live(admissions):
    assert set(admissions) == {"service", "client"}
    for admission in admissions.values():
        assert admission.pid is not None
        assert len(admission.pipe.descriptors) == 1
        for descriptor in admission.pipe.descriptors:
            os.fstat(descriptor)


def resources(monkeypatch):
    captured = SimpleNamespace(children=[], streams=[], descriptors=set())
    owner = module().SeriesOwnedChildren
    original_enter, original_stream = owner.__enter__, owner._stream

    def enter(children):
        result = original_enter(children)
        captured.children.append(children)
        captured.descriptors.update(children.descriptors)
        captured.descriptors.add(children.listener.fileno())
        return result

    def stream(children):
        result = original_stream(children)
        captured.streams.append(result)
        return result

    monkeypatch.setattr(owner, "__enter__", enter)
    monkeypatch.setattr(owner, "_stream", stream)
    return captured


def assert_resources_closed(captured):
    assert all(stream.closed for stream in captured.streams)
    assert all(not children.descriptors for children in captured.children)
    for descriptor in captured.descriptors:
        with pytest.raises(OSError):
            os.fstat(descriptor)


async def cancel_pair(attempt, options, issue, trigger, repeated=False):
    task = asyncio.create_task(
        module().observe_series_operational_children(
            attempt,
            **pair_options(options),
            writer=original._record,
            series_admissions=issue,
        )
    )
    while not (attempt.directory / trigger).exists():
        assert not task.done()
        await asyncio.sleep(0.005)
    task.cancel()
    if repeated:
        await asyncio.sleep(0.02)
        task.cancel()
    with pytest.raises(asyncio.CancelledError) as caught:
        await task
    assert task.cancelled()
    return progress(caught.value)
