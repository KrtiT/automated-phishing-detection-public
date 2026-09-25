"""Closed child declarations; only the observing parent establishes process proof."""

import re
from dataclasses import dataclass, field

from ._checkpoint_codec import canonical_bytes
from ._process_support import command_hash
from .operational_schedule import validate_cell


class OperationalRoleError(ValueError):
    """A role declaration differs from the caller's independently retained context."""


def _require(condition):
    if not condition:
        raise OperationalRoleError("invalid_operational_role")


@dataclass(frozen=True)
class OperationalRoleContext:
    pid: int
    command: tuple[str, ...] = field(repr=False)
    base_url: str

    def __post_init__(self):
        _context(self)


def _context(context):
    _require(type(context) is OperationalRoleContext)
    _require(type(context.pid) is int and context.pid > 0)
    _require(type(context.command) is tuple and bool(context.command))
    _require(
        all(
            type(value) is str and value and "\0" not in value
            for value in context.command
        )
    )
    _require(type(context.base_url) is str)
    matched = re.fullmatch(r"http://127\.0\.0\.1:([1-9][0-9]{0,4})", context.base_url)
    _require(matched is not None and int(matched.group(1)) <= 65535)


def _common(inputs, context, role):
    from .operational_cell_inputs import RestoredOperationalCell

    _require(type(inputs) is RestoredOperationalCell)
    _context(context)
    cell = validate_cell(inputs.cell)
    digest = inputs.binding_sha256
    _require(type(digest) is str and re.fullmatch(r"[0-9a-f]{64}", digest) is not None)
    return {
        "schema_version": 1,
        "protocol": "operational-role-v1",
        "role": role,
        "binding_sha256": digest,
        "pid": context.pid,
        "command_sha256": command_hash(context.command),
        "base_url": context.base_url,
        "workload": cell.workload,
    }


def build_service_role(
    inputs, context: OperationalRoleContext, *, primary: dict
) -> bytes:
    """Join already-loaded primary identity without loading or scoring anything."""
    from ._operational_input_schema import validate_primary

    try:
        value = _common(inputs, context, "service")
        expected = inputs.primary
        validate_primary(expected)
        validate_primary(primary)
        _require(canonical_bytes(primary) == canonical_bytes(expected))
        return canonical_bytes(value | primary)
    except Exception:
        raise OperationalRoleError("invalid_operational_role") from None


def build_client_role(inputs, context: OperationalRoleContext) -> bytes:
    """Declare request-client context, never a client-side model load."""
    try:
        return canonical_bytes(_common(inputs, context, "client"))
    except Exception:
        raise OperationalRoleError("invalid_operational_role") from None


def verify_role_record(
    content: bytes, *, inputs, context: OperationalRoleContext, role: str
) -> None:
    """Compare closed bytes to supplied expectations without asserting ownership."""
    try:
        _require(type(content) is bytes)
        _require(type(role) is str and role in ("service", "client"))
        expected = (
            build_service_role(inputs, context, primary=inputs.primary)
            if role == "service"
            else build_client_role(inputs, context)
        )
        _require(content == expected)
    except Exception:
        raise OperationalRoleError("invalid_operational_role") from None
