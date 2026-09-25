"""Pure fixed operational argv; constructing commands does not authorize a run."""

import sys
from dataclasses import fields
from pathlib import Path

from . import _operational_cell_protocol as protocol
from . import _operational_profile as profiles
from ._checkpoint_codec import canonical_bytes
from ._external_records_validation import digest
from .bound_models import ArtifactPaths
from .execution_preflight import ExecutionBinding


class OperationalCommandError(ValueError):
    """Commands cannot be bound to the supplied closed input identities."""


def _require(condition):
    if not condition:
        raise OperationalCommandError("invalid_operational_commands")


def _path(value):
    _require(isinstance(value, Path) and value.is_absolute())
    _require(".." not in value.parts and "\0" not in str(value))
    return str(value)


def _arguments(names, values):
    return tuple(item for pair in zip(names, values, strict=True) for item in pair)


def _common(binding, profile, accepted, cell, expected):
    _require(type(binding) is ExecutionBinding)
    _require(type(profile) is profiles.CandidateOperationalProfile)
    _require(type(profile.canonical_bytes) is bytes)
    _require(profile.canonical_bytes == canonical_bytes(profiles._projection(binding)))
    digest(expected)
    return (
        _path(binding.root),
        binding.revision,
        binding.contract_sha256,
        profile.profile_sha256,
        _path(accepted),
        _path(cell),
        expected,
    )


def _commands(root, common, extra):
    executable = sys.executable
    _require(type(executable) is str and bool(executable) and "\0" not in executable)
    _require(Path(executable).is_absolute())
    return (
        (
            executable,
            _path(root / protocol.SERVICE_SCRIPT),
            *_arguments(protocol.SERVICE_ARGUMENTS, (*common, *extra)),
        ),
        (
            executable,
            _path(root / protocol.CLIENT_SCRIPT),
            *_arguments(protocol.COMMON_ARGUMENTS, common),
        ),
    )


def build_cell_commands(
    binding,
    profile,
    *,
    accepted_inputs_directory,
    cell_input_directory,
    expected_binding_sha256,
    artifacts,
):
    """Use the actual Python executable and fixed scripts, with no caller overrides."""
    try:
        _require(type(artifacts) is ArtifactPaths)
        common = _common(
            binding,
            profile,
            accepted_inputs_directory,
            cell_input_directory,
            expected_binding_sha256,
        )
        extra = tuple(
            _path(getattr(artifacts, member.name)) for member in fields(artifacts)
        )
        return _commands(binding.root, common, extra)
    except Exception:
        raise OperationalCommandError("invalid_operational_commands") from None
