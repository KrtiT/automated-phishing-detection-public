"""Exact historical cell byte inventories, independent pins and old receipts."""

from dataclasses import asdict
from pathlib import PurePosixPath

from . import _operational_input_schema as schema
from . import _study_history_snapshot_records as records
from ._checkpoint_codec import canonical_bytes
from ._operational_cell_protocol import PRIVATE_NAMES, PROTOCOL, SNAPSHOT_NAMES

PUBLIC_NAMES = {
    "schema_version",
    "protocol",
    "status",
    "execution",
    "cell",
    "summary",
    "private_sha256",
}


def authenticate(payloads, expected, buffers, pins):
    values = records.authenticate(payloads, SNAPSHOT_NAMES, expected, expected)
    for content, pin in zip(buffers, pins[:2], strict=True):
        schema.require(type(content) is bytes)
        schema.digest(pin)
        schema.require(records.digest(content) == pin)
    schema.digest(pins[2])
    schema.require(records.digest(values["attempt/reservation.json"]) == pins[2])
    return values


def _execution(inputs, reservation):
    metadata = schema.loads(inputs.accepted_bytes)
    return {
        "kind": "operational_cell",
        "protocol": PROTOCOL,
        **metadata["execution"],
        "operational_profile_sha256": metadata["operational_profile_sha256"],
        "root_reservation_sha256": metadata["root_reservation_sha256"],
        "descriptor_sha256": records.digest(inputs.descriptor_bytes),
        "reservation_sha256": reservation,
    }


def publication(values, inputs, reservation, directory):
    schema.require(type(directory) is str and "\0" not in directory)
    path = PurePosixPath(directory)
    schema.require(path.is_absolute() and ".." not in path.parts)
    schema.require(str(path) == directory)
    execution = _execution(inputs, reservation)
    records.reservation(values, directory, execution)
    private = records.outputs(values, PRIVATE_NAMES, "")
    public = records.public_record(
        values, execution, private, PUBLIC_NAMES, "operational_evidence_published", 1
    )
    schema.require(public["protocol"] == PROTOCOL)
    schema.same(public["cell"], asdict(inputs.cell))
    records.finalization(values, execution, private)
    return private, public


def verify_summary(public, summary):
    schema.require(canonical_bytes(public["summary"]) == summary)
