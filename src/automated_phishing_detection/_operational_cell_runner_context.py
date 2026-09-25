"""Validate fixture-only cell context before consuming an attempt."""

import math
from dataclasses import dataclass, fields
from pathlib import Path

from . import _operational_input_schema as schema
from . import _operational_profile as profiles
from . import execution_receipt as receipt
from ._checkpoint_codec import canonical_bytes
from .bound_models import ArtifactPaths
from .execution_preflight import ExecutionBinding
from .operational_inputs import AcceptedOperationalInputs
from .operational_schedule import validate_cell


class OperationalCellExecutionError(ValueError):
    """Symbolic failure without private inputs or diagnostic values."""


@dataclass(frozen=True)
class OperationalCellPaths:
    accepted_inputs_directory: Path
    cell_input_directory: Path
    attempt: Path
    public_summary: Path


def require(condition):
    if not condition:
        raise OperationalCellExecutionError("invalid_operational_cell_execution")


def _path(value, root):
    require(isinstance(value, Path) and value.is_absolute())
    require(".." not in value.parts and "\0" not in str(value))
    require(not value.is_relative_to(root) and not root.is_relative_to(value))
    return value


def _paths(paths, artifacts, root):
    require(type(paths) is OperationalCellPaths and type(artifacts) is ArtifactPaths)
    outputs = tuple(
        _path(getattr(paths, member.name), root) for member in fields(paths)
    )
    models = tuple(
        _path(getattr(artifacts, member.name), root) for member in fields(artifacts)
    )
    for position, first in enumerate(outputs):
        for second in (*outputs[position + 1 :], *models):
            require(
                not first.is_relative_to(second) and not second.is_relative_to(first)
            )


def validate(binding, profile, accepted, cell, paths, artifacts, deadlines):
    require(type(binding) is ExecutionBinding)
    require(type(profile) is profiles.CandidateOperationalProfile)
    expected = profiles._projection(binding)
    require(type(profile.canonical_bytes) is bytes)
    require(profile.canonical_bytes == canonical_bytes(expected))
    require(type(accepted) is AcceptedOperationalInputs)
    metadata = schema.loads(accepted.metadata_bytes)
    schema.validate_metadata(metadata)
    schema.same(metadata["execution"], expected["execution"])
    require(metadata["operational_profile_sha256"] == profile.profile_sha256)
    validate_cell(cell)
    require(
        type(deadlines) is dict
        and set(deadlines) == {"startup", "shutdown", "terminate", "kill"}
    )
    require(
        all(
            type(value) in (int, float) and math.isfinite(value) and value > 0
            for value in deadlines.values()
        )
    )
    _paths(paths, artifacts, binding.root)


def absent_outputs(paths):
    for destination in (
        paths.cell_input_directory,
        paths.attempt,
        paths.public_summary,
    ):
        with receipt._directory(destination.parent) as directory:
            receipt._require_absent(directory, destination.name)


def same_inputs(inputs, accepted, selected, binding_bytes):
    require(inputs.accepted_bytes == accepted.metadata_bytes)
    require(inputs.descriptor_bytes == selected.descriptor_bytes)
    require(inputs.manifest_bytes == selected.manifest_bytes)
    require(inputs.binding_bytes == binding_bytes)
    require(inputs.requests == selected.requests)
