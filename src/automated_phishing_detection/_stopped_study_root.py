"""Exact stopped-root inventory and failed accounting joins using only bytes."""

from hashlib import sha256

from . import _adopted_study_records as records
from . import _study_root_records as retention
from . import _study_run_public as original
from . import _study_run_schema as schema
from . import execution_receipt as receipt
from ._adopted_study_intent import authenticate_intent, validate_source_profile
from ._adopted_study_verification import _reservation, _snapshot

MEMBERS = frozenset(
    {f"attempt/{name}" for name in retention.ORDER}
    | {"attempt/reservation.json", "attempt/finalize.claim", "attempt/outcome.json"}
)


def authenticate_snapshot(snapshot, expected):
    payloads = _snapshot(snapshot)
    schema.keys(payloads, MEMBERS)
    schema.keys(expected, MEMBERS)
    hashes = {}
    for name, content in payloads.items():
        schema.operational.digest(expected[name])
        hashes[name] = sha256(content).hexdigest()
        schema.require(hashes[name] == expected[name])
    return payloads, tuple(sorted(hashes.items()))


def _finalization(payloads, reservation):
    common = {"schema_version": 1, "reservation_sha256": reservation}
    claim = common | {"operation": "failure"}
    outcome = common | {
        "status": "failed",
        "stage": "cell_execution",
        "error_type": "cancelled",
    }
    for name, value in (("finalize.claim", claim), ("outcome.json", outcome)):
        schema.require(
            payloads[f"attempt/{name}"] == receipt._json_bytes(value, "stopped_root")
        )


def _accounting(content, execution):
    accounting = records.unwrap(
        content, execution, {"scientific_accounting_bytes", "authorization_ledger"}
    )
    scientific = schema.record(
        records.decoded(accounting["scientific_accounting_bytes"]),
        records.scientific_execution(execution),
        original._ACCOUNTING,
    )
    schema.require(accounting["status"] == scientific["status"] == "failed")
    schema.require(scientific["stage"] == "cell_execution")
    schema.require(
        scientific["internal_status"] == scientific["external_status"] == "accepted"
    )
    schema.accounting(scientific)
    cells = scientific["cells"]
    count = sum(cell["status"] == "accepted" for cell in cells)
    schema.require(1 <= count <= 124)
    schema.require(all(cell["retention"] == "compact" for cell in cells[:count]))
    schema.require(cells[count]["status"] == "stopped")
    schema.require(all(cell["status"] == "unattempted" for cell in cells[count + 1 :]))
    return accounting, scientific, tuple(range(1, count + 1))


def authenticate_root(payloads, reservation_pin, profile_pin, envelope_pin):
    reservation, execution = _reservation(
        payloads["attempt/reservation.json"], reservation_pin
    )
    contents = {name: payloads[f"attempt/{name}"] for name in retention.ORDER}
    profile = authenticate_intent(
        contents["study-intent.json"], execution, profile_pin, envelope_pin
    )
    schema.require(reservation["directory"] == profile["paths"]["attempt"])
    _finalization(payloads, reservation_pin)
    scientific_execution = records.scientific_execution(execution)
    barrier = original._barrier(
        contents["prediction-barrier.json"], scientific_execution, True
    )
    original._sources(contents["source-results.json"], scientific_execution, barrier)
    validate_source_profile(contents["source-results.json"], profile)
    accounting, scientific, ordinals = _accounting(
        contents["study-accounting.json"], execution
    )
    return execution, contents, accounting, scientific, ordinals
