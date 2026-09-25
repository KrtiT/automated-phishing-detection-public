"""Retain actual same-parent source results without re-reading or reconstructing.

Caller-constructed observations and these metadata bytes are not portable process
proof. Public-profile and protected-access gates remain the parent's responsibility.
"""

from dataclasses import asdict, dataclass, field
from hashlib import sha256

from . import _operational_input_schema as schema
from . import saved_evidence
from ._checkpoint_codec import canonical_bytes
from ._external_completion_records import _LOGICAL_NAMES
from ._external_records_validation import pins
from ._external_source_profile import _execution
from ._owned_process_exit import OwnedProcessExit
from .execution_preflight import ExecutionBinding
from .external_source_handoff import (
    ObservedExternalCompletion,
    VerifiedExternalSnapshot,
)
from .internal_external_handoff import build_internal_handoff
from .internal_process_handoff import ObservedInternalCompletion
from .owned_worker import WorkerObservation

OperationalInputError = schema.OperationalInputError


@dataclass(frozen=True)
class AcceptedOperationalInputs:
    metadata_bytes: bytes = field(repr=False)
    internal: ObservedInternalCompletion = field(repr=False)
    external: ObservedExternalCompletion = field(repr=False)


def _external(completion):
    schema.require(type(completion) is ObservedExternalCompletion)
    schema.require(type(completion.snapshot) is VerifiedExternalSnapshot)
    schema.require(
        type(completion.worker) is WorkerObservation
        and type(completion.worker.exit) is OwnedProcessExit
    )
    schema.require(type(completion.snapshot.payloads) is tuple)
    contents = {}
    for member in completion.snapshot.payloads:
        schema.require(type(member) is tuple and len(member) == 2)
        name, content = member
        schema.require(
            type(name) is str and type(content) is bytes and name not in contents
        )
        contents[name] = content
    schema.keys(contents, _LOGICAL_NAMES)
    public = schema.loads(contents["public-summary.json"], canonical=False)
    return contents, {
        "execution": public["execution"],
        "worker": asdict(completion.worker),
        "snapshot_sha256": {
            name: sha256(content).hexdigest() for name, content in contents.items()
        },
    }


def _primary(first, second):
    projections = []
    for content in (first, second):
        binding = schema.loads(content)
        saved_evidence._validate_binding_core(binding)
        value = {name: binding[name] for name in ("artifact_hashes", "thresholds")}
        schema.validate_primary(value)
        projections.append(value)
    schema.same(*projections)
    return projections[0]


def _profile(second, external, binding, internal_execution):
    profile_bytes = second.snapshot.profile_bytes
    schema.require(type(profile_bytes) is bytes)
    schema.require(
        sha256(profile_bytes).hexdigest()
        == external["execution"]["source_profile_sha256"]
    )
    bound_pins = pins(binding)
    schema.require(
        internal_execution["preparation_summary_sha256"]
        == bound_pins["reports/phiusiil-preparation-summary.json"]
    )
    execution = _execution(binding, bound_pins)
    schema.same(schema.loads(profile_bytes)["execution"], execution)
    return execution


def _build(first, second, binding, reservation, profile):
    schema.require(type(binding) is ExecutionBinding)
    handoff = build_internal_handoff(first)
    external_contents, external = _external(second)
    for name, expected in (
        ("internal-source-handoff.json", handoff.handoff_bytes),
        ("internal-source-overlap.json", handoff.overlap_bytes),
    ):
        schema.require(external_contents[f"attempt/evidence/{name}"] == expected)
    result = {
        "schema_version": 1,
        "kind": "same-parent-operational-inputs-v1",
        "root_reservation_sha256": reservation,
        "execution": _profile(
            second, external, binding, first.public_summary["execution"]
        ),
        "operational_profile_sha256": profile,
        "primary": _primary(
            first.snapshot.payload("attempt/evidence/bindings.json"),
            external_contents["attempt/evidence/bindings.json"],
        ),
        "internal": schema.loads(handoff.handoff_bytes),
        "external": external,
    }
    schema.validate_metadata(result)
    return AcceptedOperationalInputs(canonical_bytes(result), first, second)


def build_accepted_inputs(
    internal, external, *, binding, root_reservation_sha256, operational_profile_sha256
) -> AcceptedOperationalInputs:
    """Project accepted bytes only; profile arguments do not grant access."""
    try:
        return _build(
            internal,
            external,
            binding,
            root_reservation_sha256,
            operational_profile_sha256,
        )
    except Exception:
        raise OperationalInputError("invalid_operational_inputs") from None
