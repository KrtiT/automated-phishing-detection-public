"""Closed prefix and acyclic new-cell joins without historical science or IO."""

import os
from pathlib import Path

from . import _operational_cell_records as cells
from . import _operational_input_schema as schema
from . import _study_execution_schema as paths
from . import execution_receipt as receipt
from ._checkpoint_codec import canonical_bytes
from ._study_history_snapshot_records import digest
from ._study_series_cell_records import identity
from .operational_schedule import cell_for_ordinal, validate_cell


def parent(ledger):
    schema.require(
        type(ledger._parent_pid) is int and ledger._parent_pid == os.getpid()
    )


def reservation(attempt, expected_directory, expected_identity):
    schema.require(type(attempt) is receipt.Attempt)
    schema.require(type(attempt.directory) is type(Path()))
    schema.require(attempt.directory == paths.lexical_path(str(expected_directory)))
    content = receipt._json_bytes(
        {
            "schema_version": 1,
            "status": "reserved",
            "directory": str(attempt.directory),
            "identity": expected_identity,
        },
        "reservation",
    )
    schema.require(digest(content) == attempt.reservation_sha256)
    return content


def _prefix(public, series, segment, intent, imported, metadata):
    from . import study_series_prefix as prefix

    profile = schema.loads(public.profile_bytes)
    reservation(
        series, profile["paths"]["series_attempt"], prefix.series_identity(public)
    )
    reservation(
        segment,
        profile["paths"]["segment_attempt"],
        prefix.segment_identity(public, series),
    )
    declared, accepted = schema.loads(imported), schema.loads(metadata)
    expected = prefix.history_import_bytes(
        public,
        series,
        segment,
        metadata,
        origin_metadata_bytes=canonical_bytes(accepted["origin"]),
        imported_prefix_length=declared["imported_prefix_length"],
        origin_accounting_sha256=declared["origin_accounting_sha256"],
    )
    schema.require(imported == expected)
    schema.require(
        intent
        == prefix.segment_intent_bytes(public, series, segment, imported, metadata)
    )
    return profile


def initialize(public, series, segment, intent, imported, metadata):
    from .study_series_execution import SeriesPublicBinding

    schema.require(type(public) is SeriesPublicBinding)
    for content in (intent, imported, metadata):
        schema.require(type(content) is bytes)
    profile = _prefix(public, series, segment, intent, imported, metadata)
    return {
        "_public": public,
        "_series": series,
        "_segment": segment,
        "_profile": profile,
        "_intent": intent,
        "_import": imported,
        "_metadata": metadata,
        "_parent_pid": os.getpid(),
        "_entries": [],
        "_commands": [],
        "_accepted": [],
        "_current": None,
        "_stopped": None,
        "_closed": False,
    }


def next_cell(ledger):
    return cell_for_ordinal(
        ledger._profile["segment"]["start_ordinal"] + len(ledger._accepted)
    )


def cell_context(ledger, cell, attempt, descriptor, binding):
    schema.require(validate_cell(cell) == next_cell(ledger))
    described = schema.loads(descriptor)
    schema.require(cells.descriptor(described) == cell)
    schema.require(described["accepted_inputs_sha256"] == digest(ledger._metadata))
    schema.require(
        described["root_reservation_sha256"] == ledger._segment.reservation_sha256
    )
    directory = (
        Path(ledger._profile["paths"]["cells_dir"]) / f"cell-{cell.ordinal:03d}-attempt"
    )
    expected = identity(ledger._metadata, descriptor, digest(ledger._metadata))
    reservation(attempt, directory, expected)
    bound = schema.loads(binding)
    cells.binding(bound)
    schema.require(bound["descriptor_sha256"] == digest(descriptor))
    schema.require(bound["cell_reservation_sha256"] == attempt.reservation_sha256)
    return dict(
        cell=cell, attempt=attempt, descriptor_bytes=descriptor, binding_bytes=binding
    )


def start(ledger, cell, attempt, descriptor, binding):
    schema.require(not ledger._closed and ledger._current is None)
    ledger._current = cell_context(ledger, cell, attempt, descriptor, binding)


def partial(ledger, current):
    cell, attempt = current["cell"], current["attempt"]
    described, bound = current["descriptor_bytes"], current["binding_bytes"]
    schema.require(cell == next_cell(ledger))
    if described is not None:
        value = schema.loads(described)
        schema.require(cells.descriptor(value) == cell)
        schema.require(value["accepted_inputs_sha256"] == digest(ledger._metadata))
        schema.require(
            value["root_reservation_sha256"] == ledger._segment.reservation_sha256
        )
    if attempt is not None:
        schema.require(described is not None)
        directory = (
            Path(ledger._profile["paths"]["cells_dir"])
            / f"cell-{cell.ordinal:03d}-attempt"
        )
        reservation(
            attempt,
            directory,
            identity(ledger._metadata, described, digest(ledger._metadata)),
        )
    if bound is not None:
        cell_context(ledger, cell, attempt, described, bound)
    return None if attempt is None else attempt.reservation_sha256
