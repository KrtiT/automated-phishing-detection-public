"""Restore saved external science without process, custody or access authority.

Independent expectations must come from the authenticated original context.
Preparation ancestry and physical eligibility remain separate requirements.
No model, original source or filesystem path is opened by this pure verifier.
"""

from . import _operational_input_schema as schema
from . import _study_history_external_records as records
from ._checkpoint_codec import canonical_bytes
from ._external_completion_files import ExternalFileSnapshot
from ._external_provenance_payloads import PREPARED_NAMES, PROVENANCE_NAMES
from ._phishvn_archive import PhishVNSourcePins
from ._saved_external_bindings import PRIVATE_OUTPUTS
from .external_source_handoff import freeze_external_snapshot
from .external_source_provenance import verify_external_provenance
from .saved_external_evidence import reconstruct_external_evidence


class HistoricalExternalScienceError(ValueError):
    """Symbolic historical reconstruction rejection without private diagnostics."""


def _provenance(outputs, execution, handoff_bytes, overlap_bytes):
    identity = {
        name: value for name, value in execution.items() if name != "reservation_sha256"
    }
    verify_external_provenance(
        {name: outputs[name] for name in PROVENANCE_NAMES},
        {name: outputs[name] for name in PREPARED_NAMES},
        pins=PhishVNSourcePins(
            execution["archive_sha256"], execution["archive_size_bytes"]
        ),
        expected_handoff=handoff_bytes,
        expected_overlap=overlap_bytes,
        execution=identity,
        reservation_sha256=execution["reservation_sha256"],
        suffix_rules_sha256=execution["suffix_rules_sha256"],
    )


def _restore(
    values,
    pins,
    execution,
    source_pins,
    profile_bytes,
    handoff_bytes,
    overlap_bytes,
    directory,
):
    schema.require(type(execution) is dict)
    execution = schema.loads(canonical_bytes(execution))
    original = records.handoff(execution, pins, handoff_bytes, overlap_bytes)
    candidate = records.profile(profile_bytes, execution)
    public, outputs = records.publication(values, execution, candidate, directory)
    records.public_sources(outputs, source_pins, execution, original)
    _provenance(outputs, execution, handoff_bytes, overlap_bytes)
    replay = reconstruct_external_evidence(
        {name: outputs[name] for name in PRIVATE_OUTPUTS},
        canonical_bytes(public["composition"]),
    )
    return freeze_external_snapshot(
        ExternalFileSnapshot(tuple(sorted(values.items()))), profile_bytes, replay
    )


def verify_historical_external_science(
    payloads,
    *,
    expected_snapshot_sha256,
    expected_execution,
    expected_source_sha256,
    expected_profile_bytes,
    expected_handoff_bytes,
    expected_overlap_bytes,
    expected_attempt_directory,
):
    """Verify complete original byte joins and unchanged provenance/science kernels."""
    try:
        values = records.snapshot(payloads, expected_snapshot_sha256)
        return _restore(
            values,
            expected_snapshot_sha256,
            expected_execution,
            expected_source_sha256,
            expected_profile_bytes,
            expected_handoff_bytes,
            expected_overlap_bytes,
            expected_attempt_directory,
        )
    except Exception:
        raise HistoricalExternalScienceError(
            "invalid_historical_external_science"
        ) from None
