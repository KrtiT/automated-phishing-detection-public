"""Reconstruct six provenance payloads without source access or execution authority.

Expected parent identities establish same-parent consistency only. Saved bytes
cannot independently establish archive completeness or an observed process exit.
"""

from hashlib import sha256

from . import _external_provenance_payloads as payloads
from ._phishvn_archive import PhishVNSourcePins
from .phishvn import PreparedExternal
from .saved_phishvn_source import restore_phishvn_source

PROVENANCE_NAMES = payloads.PROVENANCE_NAMES


class ExternalSourceProvenanceError(ValueError):
    """A fixed-symbol provenance rejection without private values or diagnostics."""


def _build(decoded, prepared, suffix, handoff, overlap, execution, reservation):
    restored, provenance = payloads.decoder_inputs(decoded)
    provenance.update(
        {
            "suffix-rules.dat": suffix,
            "internal-source-handoff.json": handoff,
            "internal-source-overlap.json": overlap,
        }
    )
    provenance = payloads.snapshot(
        provenance, PROVENANCE_NAMES - {"external-source-reconstruction.json"}
    )
    identity, reservation = payloads.context(execution, reservation)
    reconstructed = payloads.prepare(restored, provenance, sha256(handoff).hexdigest())
    payloads.require(
        payloads.prepared_outputs(prepared) == payloads.prepared_outputs(reconstructed)
    )
    provenance["external-source-reconstruction.json"] = payloads.receipt(
        provenance, restored, reconstructed, identity, reservation
    )
    return provenance


def build_external_provenance(
    decoded,
    prepared,
    *,
    suffix_rules: bytes,
    internal_handoff: bytes,
    internal_overlap: bytes,
    execution: dict,
    reservation_sha256: str,
) -> dict[str, bytes]:
    """Bind consistent supplied views; infer no physical source authentication."""
    try:
        return _build(
            decoded,
            prepared,
            suffix_rules,
            internal_handoff,
            internal_overlap,
            execution,
            reservation_sha256,
        )
    except Exception:
        raise ExternalSourceProvenanceError(
            "invalid_external_source_provenance"
        ) from None


def _expected_inputs(provenance, handoff, overlap, suffix_hash):
    payloads.require(type(handoff) is bytes and type(overlap) is bytes)
    payloads.require(provenance["internal-source-handoff.json"] == handoff)
    payloads.require(provenance["internal-source-overlap.json"] == overlap)
    payloads.require(
        type(suffix_hash) is str
        and sha256(provenance["suffix-rules.dat"]).hexdigest() == suffix_hash
    )


def _verify(
    provenance, outputs, pins, handoff, overlap, execution, reservation, suffix_hash
):
    provenance = payloads.snapshot(provenance, PROVENANCE_NAMES)
    outputs = payloads.snapshot(outputs, payloads.PREPARED_NAMES)
    _expected_inputs(provenance, handoff, overlap, suffix_hash)
    identity, reservation = payloads.context(execution, reservation)
    payloads.verify_receipt_inputs(provenance, outputs, identity, reservation)
    decoded = restore_phishvn_source(
        provenance["publisher-source.json"],
        provenance["publisher-summary.json"],
        pins=pins,
    )
    prepared = payloads.prepare(decoded, provenance, sha256(handoff).hexdigest())
    payloads.require(outputs == payloads.prepared_outputs(prepared))
    expected = payloads.receipt(provenance, decoded, prepared, identity, reservation)
    payloads.require(provenance["external-source-reconstruction.json"] == expected)
    return prepared


def verify_external_provenance(
    provenance: dict[str, bytes],
    prepared_outputs: dict[str, bytes],
    *,
    pins: PhishVNSourcePins,
    expected_handoff: bytes,
    expected_overlap: bytes,
    execution: dict,
    reservation_sha256: str,
    suffix_rules_sha256: str,
) -> PreparedExternal:
    """Reconstruct against independent parent expectations and exact saved bytes."""
    try:
        return _verify(
            provenance,
            prepared_outputs,
            pins,
            expected_handoff,
            expected_overlap,
            execution,
            reservation_sha256,
            suffix_rules_sha256,
        )
    except Exception:
        raise ExternalSourceProvenanceError(
            "invalid_external_source_provenance"
        ) from None
