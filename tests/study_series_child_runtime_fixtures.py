"""Explicit no-model, no-request stand-ins beneath unchanged role orchestration."""

import json
from contextlib import contextmanager
from types import SimpleNamespace

from operational_owner_fixtures import models

from automated_phishing_detection import operational_runtime as original
from automated_phishing_detection.operational_cell_inputs import RestoredOperationalCell


def owner(case, monkeypatch):
    selected = models()
    primary = json.loads(case.selected.metadata)["primary"]
    thresholds = primary["thresholds"]
    selected.artifact_hashes = tuple(sorted(primary["artifact_hashes"].items()))
    selected.length_only.validation_threshold_record["threshold"] = thresholds[
        "length_only"
    ]
    selected.cascade.stage1_threshold = thresholds["logistic_l1"]
    selected.cascade.transformer_threshold = thresholds["transformer"]
    selected.cascade.half_width = thresholds["half_width"]
    selected.monitor_boundary = thresholds["monitor_boundary"]

    @contextmanager
    def session(binding, paths):
        assert binding is case.auth.base
        case.events.append("session_open")
        try:
            yield SimpleNamespace(models=selected)
        finally:
            case.events.append("session_closed")

    monkeypatch.setattr(original, "open_bound_session", session)
    monkeypatch.setattr(original, "_scorer", lambda *args: object())
    inspect_owner(case, monkeypatch)


def inspect_owner(case, monkeypatch):
    original_owner = original._owner

    @contextmanager
    def observed(binding, paths, inputs, role_context, retain):
        assert type(inputs) is RestoredOperationalCell
        assert inputs.binding_bytes == case.selected.binding
        with original_owner(binding, paths, inputs, role_context, retain) as scorer:
            yield scorer

    monkeypatch.setattr(original, "_owner", observed)


async def zero_request_service(app, listener, stop_fd, *, retain):
    async with app.router.lifespan_context(app):
        retain("service-ready.json", b"invented ready")
    retain("service-cleanup.json", b"invented clean")
