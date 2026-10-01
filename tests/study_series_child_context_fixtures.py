"""Invented public binding and held live pipe, with execution explicitly isolated."""

import os
import sys
from contextlib import contextmanager
from dataclasses import replace
from importlib import import_module
from importlib.util import find_spec
from types import SimpleNamespace

from study_series_admission_fixtures import frame
from study_series_execution_fixtures import bind

from automated_phishing_detection._study_series_child_commands import (
    series_child_command,
)


def api():
    name = "automated_phishing_detection._study_series_child_context"
    assert find_spec(name), "missing series child context integration"
    return import_module(name)


def setup(public_case, monkeypatch, role="client", *, allow_execution=True):
    module, events = api(), []
    binding = bind(public_case)
    command = series_child_command(
        binding, role, cell_ordinal=73, cell_binding_sha256="a" * 64
    )
    arguments = SimpleNamespace(
        role=role,
        repo_root=binding.base.root,
        expected_revision=binding.base.revision,
        expected_profile_sha256=binding.profile_sha256,
        envelope=binding.envelope_path,
        expected_envelope_sha256=binding.envelope_sha256,
        cell_ordinal=73,
        expected_binding_sha256="a" * 64,
    )
    selected = frame(
        command,
        role,
        parent_pid=os.getppid(),
        profile_sha256=binding.profile_sha256,
        envelope_sha256=binding.envelope_sha256,
    )
    case = SimpleNamespace(**locals())
    install(case, monkeypatch, allow_execution)
    return case


def install(case, monkeypatch, allow_execution):
    module = case.module
    monkeypatch.setattr(sys, "argv", list(case.command[1:]))
    install_environment(case, monkeypatch)
    install_admission(case, monkeypatch)
    monkeypatch.setattr(
        module,
        "bind_series_public_execution",
        lambda *args, **kwargs: (case.events.append("bind"), case.binding)[1],
    )
    monkeypatch.setattr(
        module,
        "recheck_series_public_execution",
        lambda value: case.events.append("public_recheck"),
    )
    if allow_execution:
        monkeypatch.setattr(
            module,
            "_require_execution_policy",
            lambda binding: case.events.append("synthetic_policy_gate"),
        )
    monkeypatch.setattr(module, "hold_series_child_prefix", prefix(case.events))


def install_environment(case, monkeypatch):
    for name in tuple(os.environ):
        if name.startswith("APD_"):
            monkeypatch.delenv(name)
    for name, value in environment(case.role).items():
        monkeypatch.setenv(name, value)


def install_admission(case, monkeypatch):
    case.child = SimpleNamespace(
        frame=case.selected,
        check=lambda: case.events.append("live"),
        close=lambda: case.events.append("close"),
    )
    monkeypatch.setattr(
        case.module,
        "consume_series_admission",
        lambda *args, **kwargs: (case.events.append("consume"), case.child)[1],
    )


def environment(role):
    values = {
        "APD_STUDY_SERIES_ADMISSION_FD": "7",
        "APD_BASE_URL": "http://127.0.0.1:12345",
        "APD_ATTEMPT_DIRECTORY": "/invented/attempt",
        "APD_RESERVATION_SHA256": "b" * 64,
    }
    if role == "service":
        values.update(APD_LISTENER_FD="8", APD_STOP_FD="9", APD_READY_FD="10")
    return values


def prefix(events):
    @contextmanager
    def hold(*args):
        events.append("prefix")
        try:
            yield (("invented", b"prefix"),)
        finally:
            events.append("prefix_closed")

    return hold


def change_frame(case, **changes):
    case.child.frame = replace(case.child.frame, **changes)
