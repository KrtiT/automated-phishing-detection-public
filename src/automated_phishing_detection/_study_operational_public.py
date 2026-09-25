"""Closed public-cell links for compact retained records, without I/O."""

from dataclasses import asdict
from hashlib import sha256

from . import _operational_input_schema as schema
from . import execution_receipt as receipt
from ._operational_cell_protocol import PRIVATE_NAMES, PROTOCOL

_FIELDS = frozenset(
    (
        "schema_version",
        "protocol",
        "status",
        "execution",
        "cell",
        "summary",
        "private_sha256",
    )
)


def _identity(execution, record, described, bound):
    constants = {
        "kind": "operational_cell",
        "protocol": PROTOCOL,
        "root_reservation_sha256": described["root_reservation_sha256"],
        "descriptor_sha256": sha256(record.descriptor_bytes).hexdigest(),
        "reservation_sha256": bound["cell_reservation_sha256"],
    }
    schema.keys(
        execution, {*constants, *schema.EXECUTION_FIELDS, "operational_profile_sha256"}
    )
    schema.same({key: execution[key] for key in constants}, constants)
    schema.shared_execution({key: execution[key] for key in schema.EXECUTION_FIELDS})
    schema.digest(execution["operational_profile_sha256"])


def validate(record, described, bound, retained):
    schema.require(type(record.public_bytes) is bytes)
    schema.require(
        sha256(record.public_bytes).hexdigest() == retained["public-summary.json"]
    )
    value = schema.loads(record.public_bytes, canonical=False)
    schema.require(record.public_bytes == receipt._json_bytes(value, "cell_public"))
    schema.keys(value, _FIELDS)
    schema.require(
        type(value["schema_version"]) is int and value["schema_version"] == 1
    )
    schema.require(
        value["protocol"] == PROTOCOL
        and value["status"] == "operational_evidence_published"
    )
    schema.same(value["cell"], asdict(record.cell))
    schema.require(type(value["summary"]) is dict)
    schema.keys(value["private_sha256"], PRIVATE_NAMES)
    for name in PRIVATE_NAMES:
        schema.require(
            value["private_sha256"][name]
            == retained[f"attempt/{name}"]
            == retained[f"attempt/evidence/{name}"]
        )
    _identity(value["execution"], record, described, bound)
    return value
