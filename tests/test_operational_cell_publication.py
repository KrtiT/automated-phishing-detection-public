"""Final receipt verification is relative to this parent's retained working bytes."""

import json
from hashlib import sha256

import pytest
from operational_cell_acceptance_fixtures import http_case, verify
from operational_input_fixtures import candidates, case, manifests
from test_operational_cell_identity import api

from automated_phishing_detection import execution_receipt as receipt
from automated_phishing_detection._operational_cell_protocol import (
    PROTOCOL,
    SNAPSHOT_NAMES,
)

__all__ = ["candidates", "case", "manifests"]


def _finalization(working, public):
    reservation = working.reservation_sha256
    claim = {
        "schema_version": 1,
        "reservation_sha256": reservation,
        "operation": "completion",
    }
    outcome = {
        "schema_version": 1,
        "reservation_sha256": reservation,
        "status": "completion_prepared",
        "private_sha256": {
            name: sha256(content).hexdigest()
            for name, content in working.private_outputs.items()
        },
        "public_summary_sha256": sha256(public).hexdigest(),
    }
    return {
        "attempt/finalize.claim": receipt._json_bytes(claim, "fixture"),
        "attempt/outcome.json": receipt._json_bytes(outcome, "fixture"),
        "public-summary.json": public,
    }


def published(working, public):
    values = {f"attempt/{name}": content for name, content in working.payloads}
    values.update(
        {
            f"attempt/evidence/{name}": content
            for name, content in working.private_outputs.items()
        }
    )
    values.update(_finalization(working, public))
    return values


@pytest.fixture(scope="module")
def completed(case):
    working = verify(api(), http_case(api(), case))
    summary = api().build_cell_public(
        working, reservation_sha256=working.reservation_sha256
    )
    content = receipt._json_bytes(summary, "fixture")
    return working, content, published(working, content)


def test_closed_public_projection_and_exact36_snapshot(completed):
    working, content, values = completed
    public = json.loads(content)
    assert set(public) == {
        "schema_version",
        "protocol",
        "status",
        "execution",
        "cell",
        "summary",
        "private_sha256",
    }
    assert public["protocol"] == PROTOCOL
    assert public["status"] == "operational_evidence_published"
    assert public["execution"]["reservation_sha256"] == working.reservation_sha256
    assert public["summary"] == working.summary
    assert set(values) == set(SNAPSHOT_NAMES)
    result = api().verify_published_cell(
        tuple(values.items()), working=working, expected_public_bytes=content
    )
    assert result.payloads == tuple(values.items())
    assert result.inputs is working.inputs and result.accepted is working.accepted
    assert result.summary_bytes == working.summary_bytes
    result.run.after_measured.admitted_requests = 0
    assert result.run.after_measured.admitted_requests == 11000
    assert b"raw_url" not in content and b'"pid"' not in content


@pytest.mark.parametrize("name", SNAPSHOT_NAMES)
def test_every_published_payload_is_bound_to_parent_bytes(completed, name):
    working, content, values = completed
    changed = values | {name: values[name] + b"\n"}
    with pytest.raises(api().OperationalCellAcceptanceError):
        api().verify_published_cell(
            tuple(changed.items()), working=working, expected_public_bytes=content
        )


def test_caller_cannot_replace_expected_public_or_reservation(completed):
    working, content, values = completed
    changed = json.loads(content)
    changed["summary"]["request_errors"] = 0
    replacement = receipt._json_bytes(changed, "fixture")
    with pytest.raises(api().OperationalCellAcceptanceError):
        api().verify_published_cell(
            tuple((values | {"public-summary.json": replacement}).items()),
            working=working,
            expected_public_bytes=replacement,
        )
    with pytest.raises(api().OperationalCellAcceptanceError):
        api().build_cell_public(working, reservation_sha256="0" * 64)


@pytest.mark.parametrize("kind", ["missing", "extra", "duplicate", "list"])
def test_published_inventory_is_closed(completed, kind):
    working, content, values = completed
    pairs = tuple(values.items())
    variants = {
        "missing": pairs[:-1],
        "extra": (*pairs, ("extra", b"x")),
        "duplicate": (*pairs, pairs[0]),
        "list": list(pairs),
    }
    with pytest.raises(api().OperationalCellAcceptanceError):
        api().verify_published_cell(
            variants[kind], working=working, expected_public_bytes=content
        )
