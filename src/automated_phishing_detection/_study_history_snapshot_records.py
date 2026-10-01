"""Authenticate opaque snapshot inventories and exact publication receipts."""

from hashlib import sha256

from . import _study_run_schema as schema
from . import execution_receipt as receipt
from ._operational_cell_publication_records import inventory


def digest(content):
    return sha256(content).hexdigest()


def hashes(values):
    return {name: digest(content) for name, content in values.items()}


def authenticate(payloads, names, expected, recorded):
    values = inventory(payloads, names)
    schema.keys(expected, names)
    schema.keys(recorded, names)
    for name, content in values.items():
        schema.operational.digest(expected[name])
        schema.operational.digest(recorded[name])
        schema.require(digest(content) == expected[name] == recorded[name])
    return values


def receipt_value(content):
    value = schema.operational.loads(content, canonical=False)
    schema.require(content == receipt._json_bytes(value, "history_snapshot"))
    return value


def reservation(values, path, execution):
    content = values["attempt/reservation.json"]
    schema.require(digest(content) == execution["reservation_sha256"])
    identity = {
        name: value for name, value in execution.items() if name != "reservation_sha256"
    }
    expected = dict(
        schema_version=1, status="reserved", directory=path, identity=identity
    )
    schema.same(receipt_value(content), expected)


def outputs(values, names, directory):
    private = {}
    for name in names:
        content = values[f"attempt/evidence/{name}"]
        schema.require(values[f"attempt/{directory}{name}"] == content)
        private[name] = content
    return private


def finalization(values, execution, private):
    common = dict(schema_version=1, reservation_sha256=execution["reservation_sha256"])
    expected = {
        "attempt/finalize.claim": common | {"operation": "completion"},
        "attempt/outcome.json": common
        | {
            "status": "completion_prepared",
            "public_summary_sha256": digest(values["public-summary.json"]),
            "private_sha256": hashes(private),
        },
    }
    for name, value in expected.items():
        schema.same(receipt_value(values[name]), value)


def public_record(values, execution, private, names, status, version):
    public = receipt_value(values["public-summary.json"])
    schema.keys(public, names)
    schema.require(type(public["schema_version"]) is int)
    schema.require(public["schema_version"] == version and public["status"] == status)
    schema.same(public["execution"], execution)
    schema.same(public["private_sha256"], hashes(private))
    return public
