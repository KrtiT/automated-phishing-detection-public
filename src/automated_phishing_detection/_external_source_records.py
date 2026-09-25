"""Outer external records preserve candidate and parent-declared provenance.

These byte-only projections do not authenticate a process or authorize access.
Callers supply the binding, resolved profile and observing parent's retained bytes.
"""

import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from . import _external_records_validation as validation
from . import execution_receipt
from ._checkpoint_codec import canonical_bytes
from ._external_checkpoint_protocol import PROTOCOL
from ._external_source_profile import CandidateExternalProfile
from ._prepared_external_records import PreparedExternalRunPaths, preparation_links
from .bound_drift import DriftArtifactPaths
from .bound_models import ArtifactPaths
from .bound_secondary import SecondaryArtifactPaths
from .execution_preflight import ExecutionBinding
from .internal_external_handoff import InternalHandoffPayloads, verify_internal_handoff

ExternalSourceExecutionError = validation.ExternalSourceExecutionError
__all__ = [
    "ExternalRunPaths",
    "PreparedExternalRunPaths",
    "ExternalSourceExecutionError",
    "external_identity",
    "build_external_public",
]


@dataclass(frozen=True)
class ExternalRunPaths:
    archive: Path
    suffix_rules: Path
    artifacts: ArtifactPaths
    secondary_artifacts: SecondaryArtifactPaths
    drift_artifacts: DriftArtifactPaths
    attempt: Path
    public_summary: Path


def _internal(profile, projection, pins, handoff):
    validation.require(type(handoff) is InternalHandoffPayloads)
    validation.require(type(handoff.handoff_bytes) is bytes)
    verify_internal_handoff(
        handoff.handoff_bytes,
        handoff.overlap_bytes,
        expected_handoff_sha256=sha256(handoff.handoff_bytes).hexdigest(),
    )
    internal = json.loads(handoff.handoff_bytes)["execution"]
    expected = projection["execution"] | {
        "suffix_rules_sha256": profile.suffix_rules_sha256,
        "preparation_summary_sha256": pins[validation.PREPARATION],
    }
    validation.require(
        canonical_bytes({name: internal[name] for name in expected})
        == canonical_bytes(expected)
    )
    return internal


def _identity(binding, profile, handoff, preparation):
    projection, pins = validation.context(binding, profile)
    internal = _internal(profile, projection, pins, handoff)
    archive = profile.archive_pins
    result = {
        "kind": "external_evaluation",
        "source_interface": "publisher_archive_reconstruction_v1",
        "checkpoint_protocol": PROTOCOL,
        "source_profile_sha256": profile.profile_sha256,
        **projection["execution"],
        "archive_sha256": archive.archive_sha256,
        "archive_size_bytes": archive.archive_size_bytes,
        "suffix_rules_sha256": profile.suffix_rules_sha256,
        "internal_handoff_sha256": sha256(handoff.handoff_bytes).hexdigest(),
        "internal_overlap_sha256": sha256(handoff.overlap_bytes).hexdigest(),
        "internal_reservation_sha256": internal["reservation_sha256"],
    }
    if preparation is not None:
        result.update(
            preparation_links(binding, profile, handoff, internal, preparation)
        )
    else:
        validation.require(
            internal["source_interface"] == "original_csv_reconstruction_v1"
        )
    return result


def external_identity(
    binding: ExecutionBinding,
    profile: CandidateExternalProfile,
    handoff: InternalHandoffPayloads,
    *,
    preparation=None,
) -> dict:
    """Bind consistent retained context without inferring parent observation."""
    try:
        return _identity(binding, profile, handoff, preparation)
    except Exception:
        raise ExternalSourceExecutionError(
            "invalid_external_source_execution"
        ) from None


def _public_record(binding, profile, identity, reservation, outputs, composition):
    hashes = validation.hashes(outputs)
    return {
        "schema_version": 1,
        "status": "external_evidence_published",
        "source_binding": (
            "authenticated_retained_preparation_with_parent_declared_internal_handoff"
            if identity["source_interface"] == "retained_study_preparation_v1"
            else "authenticated_publisher_with_parent_declared_internal_handoff"
        ),
        "protected_evaluation_authorized": binding.protected_evaluation_ready,
        "execution": identity | {"reservation_sha256": reservation},
        "source_profile": profile.projection(),
        "publisher": validation.loads(outputs["publisher-summary.json"]),
        "composition": validation.composition(composition, outputs),
        "checkpoint_sha256": hashes.copy(),
        "private_sha256": hashes,
    }


def _build(
    binding, profile, identity, reservation, private_outputs, composition, preparation
):
    outputs = validation.snapshot(private_outputs)
    validation.digest(reservation)
    validation.require(type(identity) is dict)
    handoff = InternalHandoffPayloads(
        outputs["internal-source-handoff.json"], outputs["internal-source-overlap.json"]
    )
    expected = external_identity(binding, profile, handoff, preparation=preparation)
    validation.require(canonical_bytes(identity) == canonical_bytes(expected))
    result = _public_record(
        binding, profile, expected, reservation, outputs, composition
    )
    return json.loads(execution_receipt._json_bytes(result, "external_public"))


def build_external_public(
    binding: ExecutionBinding,
    profile: CandidateExternalProfile,
    identity: dict,
    reservation_sha256: str,
    private_outputs: dict[str, bytes],
    composition: dict,
    *,
    preparation=None,
) -> dict:
    """Project exact private hashes and unchanged canonical composition aggregates."""
    try:
        return _build(
            binding,
            profile,
            identity,
            reservation_sha256,
            private_outputs,
            composition,
            preparation,
        )
    except Exception:
        raise ExternalSourceExecutionError(
            "invalid_external_source_execution"
        ) from None
