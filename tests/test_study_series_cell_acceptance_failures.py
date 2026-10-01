"""Observed failure markers cannot be normalized into current-series success."""

import json
from dataclasses import replace
from hashlib import sha256
from types import SimpleNamespace

import pytest
from study_series_acceptance_fixtures import api, arguments, forbid, payloads, working
from study_series_cell_fixtures import candidates, fresh_cell, manifests, series_case

from automated_phishing_detection import _study_history_cell_science as science
from automated_phishing_detection._checkpoint_codec import canonical_bytes
from automated_phishing_detection._operational_process_records import (
    ProcessObservation,
    _bytes,
)

__all__ = ["candidates", "fresh_cell", "manifests", "series_case"]


def _reject(fresh_cell, monkeypatch, pair, values=None):
    values = dict(payloads(fresh_cell)) if values is None else values
    values["process-pair.json"] = _bytes(pair)
    observation = ProcessObservation(values["process-pair.json"])
    monkeypatch.setattr(science, "summary", forbid)
    with pytest.raises(api().SeriesCellAcceptanceError, match="^invalid_series_cell$"):
        working(fresh_cell, values=tuple(values.items()), observation=observation)


@pytest.mark.parametrize(
    "field,value",
    (
        ("status", "failed"),
        ("research_accepted", True),
        ("stop_sent", False),
        ("failure", "private-source-url"),
        ("record_failures", ["run.json"]),
        ("schema_version", True),
        ("extra", True),
        ("readiness_sha256", "0" * 64),
        ("cleanup_sha256", "0" * 64),
    ),
)
def test_failed_or_incomplete_observation_never_reaches_math(
    fresh_cell, monkeypatch, field, value
):
    pair = json.loads(fresh_cell.values["attempt/process-pair.json"])
    pair[field] = value
    _reject(fresh_cell, monkeypatch, pair)


@pytest.mark.parametrize("role", ("service", "client"))
@pytest.mark.parametrize(
    "field,value",
    (
        ("forced", True),
        ("signals", [15]),
        ("exit_observed", False),
        ("exit_code", 17),
        ("exit_code", None),
        ("exit_code", False),
        ("pid", True),
        ("pid", 0),
        ("stdout_sha256", "bad"),
        ("stderr_sha256", "A" * 64),
    ),
)
def test_each_role_requires_owned_clean_zero_exit(
    fresh_cell, monkeypatch, role, field, value
):
    pair = json.loads(fresh_cell.values["attempt/process-pair.json"])
    pair[role][field] = value
    values = dict(payloads(fresh_cell))
    values[f"{role}-process.json"] = _bytes(pair[role])
    _reject(fresh_cell, monkeypatch, pair, values)


def test_fully_rejoined_same_service_client_pid_is_rejected(fresh_cell, monkeypatch):
    values = dict(payloads(fresh_cell))
    pair = json.loads(values["process-pair.json"])
    pair["client"]["pid"] = pair["service"]["pid"]
    values["client-process.json"] = _bytes(pair["client"])
    values["client-started.json"] = _bytes({"pid": pair["client"]["pid"]})
    role = json.loads(values["client-role.json"])
    role["pid"] = pair["client"]["pid"]
    values["client-role.json"] = canonical_bytes(role)
    _reject(fresh_cell, monkeypatch, pair, values)


@pytest.mark.parametrize(
    "name,field,value",
    (
        ("service-ready.json", "pid", 999),
        ("service-ready.json", "host", "localhost"),
        ("service-ready.json", "port", True),
        ("service-ready.json", "port", 54322),
        ("service-cleanup.json", "status", "dirty"),
        ("service-cleanup.json", "workload", "other"),
    ),
)
def test_rehashed_lifecycle_still_binds_the_exact_service(
    fresh_cell, monkeypatch, name, field, value
):
    values = dict(payloads(fresh_cell))
    lifecycle = json.loads(values[name]) | {field: value}
    values[name] = _bytes(lifecycle)
    pair = json.loads(values["process-pair.json"])
    digest_name = (
        "readiness_sha256" if name == "service-ready.json" else "cleanup_sha256"
    )
    pair[digest_name] = sha256(values[name]).hexdigest()
    _reject(fresh_cell, monkeypatch, pair, values)


@pytest.mark.parametrize("role", ("service", "client"))
@pytest.mark.parametrize(
    "field,value",
    (
        ("binding_sha256", "0" * 64),
        ("command_sha256", "0" * 64),
        ("base_url", "http://127.0.0.1:54322"),
        ("workload", "other"),
    ),
)
def test_role_record_must_match_binding_command_and_workload(
    fresh_cell, role, field, value
):
    values = dict(payloads(fresh_cell))
    name = f"{role}-role.json"
    values[name] = canonical_bytes(json.loads(values[name]) | {field: value})
    with pytest.raises(api().SeriesCellAcceptanceError):
        working(fresh_cell, values=tuple(values.items()))


@pytest.mark.parametrize("name", ("startup", "shutdown", "terminate", "kill"))
def test_rejoined_other_deadlines_cannot_replace_frozen_limits(
    fresh_cell, monkeypatch, name
):
    values = dict(payloads(fresh_cell))
    deadlines = arguments(fresh_cell)["expected_deadlines"]
    deadlines[name] += 1
    intent = json.loads(values["process-pair-intent.json"])
    intent["deadlines"] = deadlines
    values["process-pair-intent.json"] = _bytes(intent)
    monkeypatch.setattr(science, "summary", forbid)
    with pytest.raises(api().SeriesCellAcceptanceError):
        working(fresh_cell, values=tuple(values.items()), expected_deadlines=deadlines)


@pytest.mark.parametrize("kind", ("namespace", "bytes", "mutable", "stale"))
def test_observation_must_be_actual_supplied_exact_type_and_bytes(fresh_cell, kind):
    content = fresh_cell.values["attempt/process-pair.json"]
    variants = {
        "namespace": SimpleNamespace(record=content),
        "bytes": content,
        "mutable": ProcessObservation(bytearray(content)),
        "stale": ProcessObservation(content + b"\n"),
    }
    with pytest.raises(api().SeriesCellAcceptanceError):
        working(fresh_cell, observation=variants[kind])


@pytest.mark.parametrize("kind", ("wrong_digest", "wrong_directory", "not_attempt"))
def test_attempt_must_match_original_reserved_context(fresh_cell, monkeypatch, kind):
    attempt = arguments(fresh_cell)["attempt"]
    variants = {
        "wrong_digest": replace(attempt, reservation_sha256="0" * 64),
        "wrong_directory": replace(attempt, directory=attempt.directory / "other"),
        "not_attempt": SimpleNamespace(**vars(attempt)),
    }
    monkeypatch.setattr(science, "summary", forbid)
    with pytest.raises(api().SeriesCellAcceptanceError):
        working(fresh_cell, attempt=variants[kind])
