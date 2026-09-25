"""Byte-only consistency handoff within the parent that observed the worker.

Serialized exit fields and caller-constructed dataclasses are not independent
process authority. The observing parent must retain and supply the expected
identity; this format neither accepts historical execution nor authorizes access.
"""

from dataclasses import asdict, dataclass
from hashlib import sha256

from . import _internal_handoff_validation as validation
from . import execution_receipt, source_checkpoint_verification
from ._checkpoint_codec import canonical_bytes
from ._internal_handoff_validation import InternalHandoffError
from ._owned_process_exit import OwnedProcessExit
from .internal_process_handoff import ObservedInternalCompletion
from .internal_source_handoff import VerifiedInternalSnapshot
from .owned_worker import WorkerObservation

__all__ = [
    "InternalHandoffError",
    "InternalHandoffPayloads",
    "build_internal_handoff",
    "verify_internal_handoff",
]


@dataclass(frozen=True)
class InternalHandoffPayloads:
    handoff_bytes: bytes
    overlap_bytes: bytes


def _snapshot_payloads(completion: ObservedInternalCompletion) -> dict[str, bytes]:
    validation.require(type(completion) is ObservedInternalCompletion)
    validation.require(type(completion.worker) is WorkerObservation)
    validation.require(type(completion.worker.exit) is OwnedProcessExit)
    validation.require(type(completion.snapshot) is VerifiedInternalSnapshot)
    payloads = completion.snapshot.payloads
    validation.require(type(payloads) is tuple)
    for member in payloads:
        validation.require(type(member) is tuple and len(member) == 2)
        validation.require(type(member[0]) is str and type(member[1]) is bytes)
    contents = dict(payloads)
    validation.require(len(contents) == len(payloads))
    validation.keys(contents, validation.SNAPSHOT_NAMES)
    validation.require(type(completion.snapshot.overlap_domains) is frozenset)
    validation.require(
        all(type(domain) is str for domain in completion.snapshot.overlap_domains)
    )
    return contents


def _build(completion: ObservedInternalCompletion) -> InternalHandoffPayloads:
    contents = _snapshot_payloads(completion)
    public = source_checkpoint_verification._json(contents["public-summary.json"])
    validation.require(
        contents["public-summary.json"]
        == execution_receipt._json_bytes(public, "internal_handoff")
    )
    handoff = {
        "schema_version": 1,
        "kind": "same-parent-internal-handoff-v1",
        "execution": public["execution"],
        "worker": asdict(completion.worker),
        "snapshot_sha256": {
            name: sha256(content).hexdigest() for name, content in contents.items()
        },
    }
    validation.envelope(handoff)
    overlap = contents[validation.OVERLAP_NAME]
    validation.require(
        validation.overlap(overlap, handoff) == completion.snapshot.overlap_domains
    )
    return InternalHandoffPayloads(canonical_bytes(handoff), overlap)


def build_internal_handoff(
    completion: ObservedInternalCompletion,
) -> InternalHandoffPayloads:
    """Project retained bytes without reopening any path or serializing population.

    Exact dataclass types enforce shape, not authority: a caller-created completion
    is never proof of supervision. Use only the actual observing parent's result.
    """
    try:
        return _build(completion)
    except Exception:
        raise InternalHandoffError("invalid_internal_handoff") from None


def verify_internal_handoff(
    handoff_bytes: bytes,
    overlap_bytes: bytes,
    *,
    expected_handoff_sha256: str,
) -> frozenset[str]:
    """Check consistency against the observing parent's retained expected identity.

    Parent-declared exit fields are not independent process proof. Authenticate
    before parsing, retaining all original valid domains, including quarantine.
    """
    validation.require(type(handoff_bytes) is bytes and type(overlap_bytes) is bytes)
    validation.digest(expected_handoff_sha256)
    validation.require(
        sha256(handoff_bytes).hexdigest() == expected_handoff_sha256,
        "handoff_identity_mismatch",
    )
    try:
        handoff = validation.loads(handoff_bytes)
        validation.envelope(handoff)
        return validation.overlap(overlap_bytes, handoff)
    except Exception:
        raise InternalHandoffError("invalid_internal_handoff") from None
