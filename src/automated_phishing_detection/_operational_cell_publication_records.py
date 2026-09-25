"""Exact existing receipt byte projections; no publisher or filesystem access."""

from hashlib import sha256

from . import _operational_input_schema as schema
from . import execution_receipt as receipt


def inventory(payloads, expected):
    schema.require(type(payloads) is tuple)
    values = {}
    for member in payloads:
        schema.require(type(member) is tuple and len(member) == 2)
        name, content = member
        schema.require(
            type(name) is str and type(content) is bytes and name not in values
        )
        values[name] = content
    schema.keys(values, expected)
    return values


def verify_reservation(content, attempt, identity):
    schema.require(type(attempt) is receipt.Attempt)
    schema.digest(attempt.reservation_sha256)
    schema.require(attempt.directory.is_absolute())
    schema.require(sha256(content).hexdigest() == attempt.reservation_sha256)
    expected = {
        "schema_version": 1,
        "status": "reserved",
        "directory": str(attempt.directory),
        "identity": identity,
    }
    schema.require(content == receipt._json_bytes(expected, "reservation"))


def private_hashes(working):
    return {
        name: sha256(content).hexdigest()
        for name, content in working.private_outputs.items()
    }


def verify_completion(values, working, public_bytes):
    for name, content in working.payloads:
        schema.require(values[f"attempt/{name}"] == content)
    for name, content in working.private_outputs.items():
        schema.require(values[f"attempt/evidence/{name}"] == content)
    reservation = working.reservation_sha256
    claim = {
        "schema_version": 1,
        "reservation_sha256": reservation,
        "operation": "completion",
    }
    outcome = {
        "schema_version": 1,
        "status": "completion_prepared",
        "reservation_sha256": reservation,
        "public_summary_sha256": sha256(public_bytes).hexdigest(),
        "private_sha256": private_hashes(working),
    }
    schema.require(
        values["attempt/finalize.claim"] == receipt._json_bytes(claim, "claim")
    )
    schema.require(
        values["attempt/outcome.json"] == receipt._json_bytes(outcome, "outcome")
    )
    schema.require(values["public-summary.json"] == public_bytes)
