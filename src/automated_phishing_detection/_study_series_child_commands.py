"""Fixed series child argv; constructing it grants no access or execution."""

import json
import sys
from pathlib import Path

from . import _study_execution_schema as schema
from .study_series_adoption import validate_series_adoption_header
from .study_series_execution import SeriesPublicBinding

SCRIPT = "scripts/run_study_series_child.py"


def _profile(binding):
    schema.require(type(binding) is SeriesPublicBinding)
    validate_series_adoption_header(
        binding.policy_bytes,
        binding.profile_bytes,
        binding.envelope_bytes,
        expected_profile_sha256=binding.profile_sha256,
        expected_envelope_sha256=binding.envelope_sha256,
    )
    value = json.loads(binding.profile_bytes)
    schema.require(
        binding.base.root == schema.lexical_path(value["paths"]["repo_root"])
    )
    schema.require(binding.base.revision == value["execution"]["revision"])
    schema.require(
        binding.base.contract_sha256 == value["execution"]["contract_sha256"]
    )
    return value


def _common(binding, role):
    schema.require(type(sys.executable) is str and bool(sys.executable))
    schema.require(Path(sys.executable).is_absolute() and "\0" not in sys.executable)
    envelope = schema.lexical_path(str(binding.envelope_path))
    return (
        sys.executable,
        str(binding.base.root / SCRIPT),
        "--role",
        role,
        "--repo-root",
        str(binding.base.root),
        "--expected-revision",
        binding.base.revision,
        "--expected-profile-sha256",
        binding.profile_sha256,
        "--envelope",
        str(envelope),
        "--expected-envelope-sha256",
        binding.envelope_sha256,
    )


def series_child_command(binding, role, *, cell_ordinal, cell_binding_sha256):
    """Select only the fixed service/client command for a declared fresh cell."""
    try:
        profile = _profile(binding)
        schema.require(type(role) is str and role in ("service", "client"))
        schema.require(type(cell_ordinal) is int)
        schema.require(profile["segment"]["start_ordinal"] <= cell_ordinal <= 125)
        schema.digest(cell_binding_sha256)
        return (
            *_common(binding, role),
            "--cell-ordinal",
            str(cell_ordinal),
            "--expected-binding-sha256",
            cell_binding_sha256,
        )
    except Exception:
        raise ValueError("invalid_series_child_command") from None
