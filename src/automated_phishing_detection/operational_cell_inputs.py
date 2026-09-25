"""Restore fixed operational cells from retained bytes, never original sources.

These records establish consistency only. Actual process observation, public
profile admission, protected access, and acceptance remain parent responsibilities.
"""

import json
from dataclasses import asdict, dataclass, field
from hashlib import sha256

from . import _operational_cell_records as records
from . import _operational_input_schema as schema
from ._checkpoint_codec import canonical_bytes
from .evaluation_producer import ManifestOutcome
from .external_replay_codec import decode_external_replay_manifest
from .http_replay import ReplayRequest
from .operational_inputs import AcceptedOperationalInputs
from .operational_schedule import OperationalCell, validate_cell
from .phishvn import _json_bytes
from .replay_manifest_codec import encode_replay_manifest

OperationalInputError = schema.OperationalInputError


class OperationalCapacityError(ValueError):
    def __init__(self, cell, *, manifest_outcome=None, available=None):
        self.cell, self.manifest_outcome = cell, manifest_outcome
        self.required = 1000 if manifest_outcome is None else manifest_outcome.required
        self.available = (
            available if manifest_outcome is None else manifest_outcome.available
        )
        super().__init__("insufficient_operational_capacity")


@dataclass(frozen=True)
class CellDescriptorPayloads:
    descriptor_bytes: bytes = field(repr=False)
    manifest_bytes: bytes = field(repr=False)
    requests: tuple[ReplayRequest, ...] = field(repr=False)


@dataclass(frozen=True)
class RestoredOperationalCell:
    accepted_bytes: bytes = field(repr=False)
    descriptor_bytes: bytes = field(repr=False)
    binding_bytes: bytes = field(repr=False)
    manifest_bytes: bytes = field(repr=False)
    cell: OperationalCell
    requests: tuple[ReplayRequest, ...] = field(repr=False)

    @property
    def manifest_sha256(self):
        return sha256(self.manifest_bytes).hexdigest()

    @property
    def binding_sha256(self):
        return sha256(self.binding_bytes).hexdigest()

    @property
    def primary(self):
        return json.loads(self.accepted_bytes)["primary"]

    @property
    def execution(self):
        return json.loads(self.accepted_bytes)["execution"]

    @property
    def operational_profile_sha256(self):
        return json.loads(self.accepted_bytes)["operational_profile_sha256"]


def _external_rows(accepted, cell, metadata):
    rows = accepted.external.snapshot.rows
    schema.require(type(rows) is tuple)
    content = accepted.external.snapshot.payload("attempt/evidence/retained-test.jsonl")
    expected = metadata["external"]["snapshot_sha256"][
        "attempt/evidence/retained-test.jsonl"
    ]
    schema.require(type(content) is bytes and sha256(content).hexdigest() == expected)
    if len(rows) < 1000:
        schema.require(
            content == b"".join(_json_bytes(asdict(row.record)) for row in rows)
        )
        raise OperationalCapacityError(cell, available=len(rows))
    restored = decode_external_replay_manifest(
        content, expected_sha256=sha256(content).hexdigest()
    )
    schema.require(tuple(row.record for row in rows) == restored)
    return content, restored


def _select(accepted, cell, metadata):
    if cell.workload == "shift_period":
        return _external_rows(accepted, cell, metadata)
    outcome = accepted.internal.snapshot.manifests[cell.prevalence_basis_points]
    schema.require(type(outcome) is ManifestOutcome)
    if outcome.status == "insufficient_capacity":
        schema.require(
            outcome.manifest is None
            and type(outcome.required) is int
            and type(outcome.available) is int
        )
        schema.require(
            type(outcome.insufficient_label) is int
            and outcome.insufficient_label in (0, 1)
            and 0 <= outcome.available < outcome.required
        )
        quota = cell.prevalence_basis_points
        if outcome.insufficient_label == 0:
            quota = 10000 - quota
        schema.require(outcome.required == quota)
        raise OperationalCapacityError(cell, manifest_outcome=outcome)
    schema.require(
        outcome.status == "prepared"
        and outcome.insufficient_label is outcome.required is outcome.available is None
    )
    content = encode_replay_manifest(outcome.manifest)
    schema.require(
        outcome.manifest.prevalence_basis_points == cell.prevalence_basis_points
    )
    return content, outcome.manifest.records


def build_cell_descriptor(accepted, cell) -> CellDescriptorPayloads:
    """Select exact retained capacity, never resample or substitute a cell."""
    try:
        schema.require(type(accepted) is AcceptedOperationalInputs)
        validate_cell(cell)
        metadata = schema.loads(accepted.metadata_bytes)
        schema.validate_metadata(metadata)
        content, rows = _select(accepted, cell, metadata)
        descriptor = records.encode_descriptor(
            accepted.metadata_bytes, metadata, cell, content, rows
        )
        return CellDescriptorPayloads(
            descriptor,
            content,
            tuple(ReplayRequest(row.record_id, row.raw_url) for row in rows),
        )
    except OperationalCapacityError:
        raise
    except Exception:
        raise OperationalInputError("invalid_operational_inputs") from None


def bind_cell_descriptor(descriptor_bytes, *, cell_reservation_sha256) -> bytes:
    """Bind an actual returned reservation after the descriptor already exists."""
    try:
        records.descriptor(schema.loads(descriptor_bytes))
        schema.digest(cell_reservation_sha256)
        return canonical_bytes(
            {
                "schema_version": 1,
                "kind": "operational-cell-binding-v1",
                "descriptor_sha256": sha256(descriptor_bytes).hexdigest(),
                "cell_reservation_sha256": cell_reservation_sha256,
            }
        )
    except Exception:
        raise OperationalInputError("invalid_operational_inputs") from None


def restore_cell_inputs(
    accepted_bytes,
    descriptor_bytes,
    binding_bytes,
    manifest_bytes,
    *,
    expected_binding_sha256,
    expected_cell_reservation_sha256,
) -> RestoredOperationalCell:
    """Authenticate every buffer before its parse; retain exact immutable bytes."""
    try:
        cell, described, accepted = records.authenticate(
            accepted_bytes,
            descriptor_bytes,
            binding_bytes,
            expected_binding_sha256=expected_binding_sha256,
            expected_cell_reservation_sha256=expected_cell_reservation_sha256,
        )
        rows = records.restored_rows(manifest_bytes, cell, described, accepted)
        return RestoredOperationalCell(
            accepted_bytes,
            descriptor_bytes,
            binding_bytes,
            manifest_bytes,
            cell,
            tuple(ReplayRequest(row.record_id, row.raw_url) for row in rows),
        )
    except Exception:
        raise OperationalInputError("invalid_operational_inputs") from None
