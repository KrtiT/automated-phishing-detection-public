"""Authenticate outer completion links using invented retained bytes only."""

import importlib
import importlib.util
import json
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace

import pytest
import test_external_source_records as record_fixtures

from automated_phishing_detection import _external_source_records as records
from automated_phishing_detection import execution_receipt
from automated_phishing_detection._external_completion_files import ExternalFileSnapshot

inputs = record_fixtures.inputs
published = record_fixtures.published
runner = record_fixtures.runner
verifier = record_fixtures.verifier
observed_worker = record_fixtures.observed_worker
completion = record_fixtures.completion
handoff_api = record_fixtures.handoff_api
records_case = record_fixtures.records_case
digest = record_fixtures.digest


def api():
    name = "automated_phishing_detection._external_completion_records"
    assert importlib.util.find_spec(name), "missing pure external completion records"
    return importlib.import_module(name)


def encoded(value):
    return execution_receipt._json_bytes(value, "invented")


def outcome(public_bytes, reservation_sha256, outputs):
    return encoded(
        {
            "schema_version": 1,
            "status": "completion_prepared",
            "reservation_sha256": reservation_sha256,
            "public_summary_sha256": digest(public_bytes),
            "private_sha256": {
                name: digest(content) for name, content in outputs.items()
            },
        }
    )


@pytest.fixture
def completion_case(records_case):
    case = records_case
    attempt = Path("/invented/external-attempt")
    identity = records.external_identity(case.binding, case.profile, case.handoff)
    reservation = encoded(
        {
            "schema_version": 1,
            "status": "reserved",
            "directory": str(attempt),
            "identity": identity,
        }
    )
    reservation_hash = digest(reservation)
    public = records.build_external_public(
        case.binding,
        case.profile,
        identity,
        reservation_hash,
        case.outputs,
        case.composition,
    )
    contents = _file_contents(reservation, public, case.outputs)
    return SimpleNamespace(
        records=case,
        attempt=attempt,
        identity=identity,
        public=public,
        reservation_hash=reservation_hash,
        contents=contents,
    )


def _file_contents(reservation, public, outputs):
    reservation_hash = digest(reservation)
    return {
        "attempt/reservation.json": reservation,
        "attempt/finalize.claim": encoded(
            {
                "schema_version": 1,
                "reservation_sha256": reservation_hash,
                "operation": "completion",
            }
        ),
        "attempt/outcome.json": outcome(encoded(public), reservation_hash, outputs),
        "public-summary.json": encoded(public),
        **{
            f"attempt/{directory}/{name}": content
            for directory in ("checkpoints", "evidence")
            for name, content in outputs.items()
        },
    }


def snapshot(case, **changes):
    return ExternalFileSnapshot(tuple((case.contents | changes).items()))


def authenticate(case, *, files=None, **changes):
    arguments = {
        "binding": case.records.binding,
        "profile": case.records.profile,
        "handoff": case.records.handoff,
    }
    return api().authenticate_external_records(
        snapshot(case) if files is None else files,
        case.attempt,
        **(arguments | changes),
    )


def relink(case, contents):
    reservation_hash = digest(contents["attempt/reservation.json"])
    claim = json.loads(contents["attempt/finalize.claim"])
    claim["reservation_sha256"] = reservation_hash
    contents["attempt/finalize.claim"] = encoded(claim)
    public = json.loads(contents["public-summary.json"])
    public["execution"]["reservation_sha256"] = reservation_hash
    outputs = {
        name: contents[f"attempt/evidence/{name}"] for name in case.records.outputs
    }
    public["checkpoint_sha256"] = public["private_sha256"] = {
        name: digest(value) for name, value in outputs.items()
    }
    contents["public-summary.json"] = encoded(public)
    contents["attempt/outcome.json"] = outcome(
        contents["public-summary.json"], reservation_hash, outputs
    )
    return ExternalFileSnapshot(tuple(contents.items()))


def test_exact_retained_snapshot_returns_fresh_frozen_record_context(completion_case):
    case = completion_case
    result = authenticate(case)
    assert type(result) is api().ExternalCompletionRecords
    assert result.identity == case.identity
    assert result.reservation_sha256 == case.reservation_hash
    assert result.public == case.public
    assert result.private_outputs == case.records.outputs
    assert len(snapshot(case).payloads) == 76
    with pytest.raises(FrozenInstanceError):
        result.reservation_sha256 = "changed"
    result.identity["revision"] = "changed"
    result.public["composition"]["invented_nested"]["unchanged"].clear()
    result.private_outputs.clear()
    assert authenticate(case).public == case.public
    assert "publisher-source.json" not in repr(result)


def test_logical_tuple_order_does_not_change_byte_identity(completion_case):
    files = ExternalFileSnapshot(tuple(reversed(snapshot(completion_case).payloads)))
    assert authenticate(completion_case, files=files) == authenticate(completion_case)
