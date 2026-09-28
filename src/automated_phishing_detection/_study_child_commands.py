"""Fixed study-only child commands; constructing argv grants no access."""

import re
import sys
from pathlib import Path

SCRIPT = "scripts/run_study_child.py"


def require(condition):
    if not condition:
        raise ValueError("invalid_study_child_command")


def _path(value):
    require(isinstance(value, Path) and value.is_absolute())
    require(".." not in value.parts and "\0" not in str(value))
    return str(value)


def _digest(value, length=64):
    require(type(value) is str and re.fullmatch(rf"[0-9a-f]{{{length}}}", value))
    return value


def _common(authorization, role):
    require(type(sys.executable) is str and bool(sys.executable))
    require(Path(sys.executable).is_absolute() and "\0" not in sys.executable)
    return (
        sys.executable,
        _path(authorization.base.root / SCRIPT),
        "--role",
        role,
        "--repo-root",
        _path(authorization.base.root),
        "--expected-revision",
        _digest(authorization.base.revision, 40),
        "--envelope",
        _path(authorization.envelope_path),
        "--expected-envelope-sha256",
        _digest(authorization.envelope_sha256),
    )


def internal_command(authorization):
    return _common(authorization, "internal")


def external_command(authorization, transport):
    return (
        *_common(authorization, "external"),
        "--internal-transport",
        _path(transport.directory),
        "--expected-handoff-sha256",
        _digest(transport.expected_handoff_sha256),
    )


def cell_command(authorization, role, ordinal, expected_binding_sha256):
    require(role in ("service", "client"))
    require(type(ordinal) is int and 1 <= ordinal <= 125)
    return (
        *_common(authorization, role),
        "--cell-ordinal",
        str(ordinal),
        "--expected-binding-sha256",
        _digest(expected_binding_sha256),
    )
