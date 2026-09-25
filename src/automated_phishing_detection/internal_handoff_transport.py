"""Private retained delivery within the parent observing both workers.

Retained bytes and caller-supplied digests establish consistency, not independent
process authority. Keep this context alive through owned exit and verification;
neither transport files nor this constructible value authorize protected access.
The 0700 directory and 0600 files remain after descriptor closure, including on
failure. Separate cleanup outside execution is required; no pathname is deleted.
"""

from contextlib import contextmanager
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path

from . import _internal_transport_io as transport_io
from ._exception_cleanup import preserve_cleanup
from ._internal_handoff_validation import digest, require
from .internal_external_handoff import InternalHandoffPayloads, verify_internal_handoff


class InternalTransportError(ValueError):
    """Symbolic transport failure without private paths or payload contents."""


@dataclass(frozen=True)
class InternalHandoffTransport:
    directory: Path = field(repr=False)
    payloads: InternalHandoffPayloads = field(repr=False)
    expected_handoff_sha256: str


def _validated(payloads):
    try:
        require(type(payloads) is InternalHandoffPayloads)
        expected = sha256(payloads.handoff_bytes).hexdigest()
        verify_internal_handoff(
            payloads.handoff_bytes,
            payloads.overlap_bytes,
            expected_handoff_sha256=expected,
        )
        return expected
    except Exception:
        raise InternalTransportError("invalid_internal_handoff_transport") from None


def _operation(function, *arguments):
    try:
        return function(*arguments)
    except Exception:
        raise InternalTransportError("invalid_internal_handoff_transport") from None


@contextmanager
def retain_internal_handoff(payloads: InternalHandoffPayloads):
    """Retain private files and exact parent bytes; close descriptors, not paths."""
    expected = _validated(payloads)
    owner = transport_io.RetainedFiles()
    with preserve_cleanup(lambda: _operation(owner.close)):
        _operation(owner.create, (payloads.handoff_bytes, payloads.overlap_bytes))
        yield InternalHandoffTransport(owner.path, payloads, expected)


def read_internal_handoff_transport(
    directory: Path, *, expected_handoff_sha256: str
) -> InternalHandoffPayloads:
    """Read each private file once and check the independent hash before parsing."""
    try:
        digest(expected_handoff_sha256)
        with transport_io.read_snapshot(directory) as contents:
            payloads = InternalHandoffPayloads(
                *(contents[name] for name in transport_io.NAMES)
            )
            verify_internal_handoff(
                payloads.handoff_bytes,
                payloads.overlap_bytes,
                expected_handoff_sha256=expected_handoff_sha256,
            )
            return payloads
    except Exception:
        raise InternalTransportError("invalid_internal_handoff_transport") from None
