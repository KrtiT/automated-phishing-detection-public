"""Fixed next-cell selection and held input joins, never independent authority."""

import json
from pathlib import Path

from . import _operational_cell_runner_context as original
from . import _operational_input_files as storage
from . import _study_preparation_files as files
from . import _study_series_child_commands as commands
from . import _study_series_ledger_context as ledger_context
from . import operational_input_transport as transport
from ._study_execution_paths import _group
from ._study_history_snapshot_records import digest
from .bound_models import ArtifactPaths
from .operational_cell_runner import OperationalCellPaths
from .operational_schedule import validate_cell
from .study_series_inputs import (
    build_series_cell_descriptor,
    restore_series_cell_inputs,
)
from .study_series_ledger import SeriesAdmissionLedger

require = original.require


def require_final_policy(public):
    require(
        json.loads(public.policy_bytes)["status"] == "specified_for_explicit_adoption"
    )


def validate(state):
    require(type(state.admissions) is SeriesAdmissionLedger)
    ledger_context.parent(state.admissions)
    require(state.admissions._public is state.public)
    require(type(state.metadata_bytes) is bytes)
    require(state.admissions._metadata == state.metadata_bytes)
    require(not state.admissions._closed and state.admissions._current is None)
    require(validate_cell(state.cell) == ledger_context.next_cell(state.admissions))
    state.ledger_checked = True
    profile = commands._profile(state.public)
    require_final_policy(state.public)
    state.paths = cell_paths(profile, state.cell.ordinal)
    mapping = {
        name: commands.schema.lexical_path(value)
        for name, value in profile["origin"]["profile"]["paths"].items()
    }
    original._paths(state.paths, _group(mapping, ArtifactPaths), state.public.base.root)


def cell_paths(profile, ordinal):
    paths = profile["paths"]
    cells = Path(paths["cells_dir"])
    prefix = f"cell-{ordinal:03d}"
    return OperationalCellPaths(
        Path(paths["historical_inputs_dir"]),
        cells / f"{prefix}-inputs",
        cells / f"{prefix}-attempt",
        cells / f"{prefix}-summary.json",
    )


def select(state):
    return build_series_cell_descriptor(
        state.metadata_bytes,
        state.public.profile_bytes,
        state.internal_snapshot,
        state.external_snapshot,
        state.cell.ordinal,
        expected_metadata_sha256=digest(state.metadata_bytes),
        expected_profile_sha256=state.public.profile_sha256,
    )


def restore(state, held):
    payloads = transport._contents(held, digest(state.binding_bytes))
    require(payloads["accepted-inputs.json"] == state.metadata_bytes)
    require(payloads["descriptor.json"] == state.selected.descriptor_bytes)
    require(payloads["binding.json"] == state.binding_bytes)
    require(payloads["manifest"] == state.selected.manifest_bytes)
    restored = restore_series_cell_inputs(
        state.metadata_bytes,
        state.public.profile_bytes,
        state.internal_snapshot,
        state.external_snapshot,
        payloads["descriptor.json"],
        payloads["binding.json"],
        payloads["manifest"],
        expected_metadata_sha256=digest(state.metadata_bytes),
        expected_profile_sha256=state.public.profile_sha256,
        expected_binding_sha256=digest(state.binding_bytes),
        expected_cell_reservation_sha256=state.attempt.reservation_sha256,
    )
    require(restored.computational.requests == state.selected.requests)
    files.deferred(storage.check_all, held)
    return restored
