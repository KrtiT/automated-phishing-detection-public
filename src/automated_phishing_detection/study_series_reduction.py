"""Unchanged full125 mathematics across original and fresh execution contexts.

Source and cell science must already be verified by their pure full verifiers.
This adapter checks supplied immutable consistency, not process ownership,
custody, access, physical eligibility or live acceptance. Caller construction
confers none of those facts. Mixed-session disclosure remains separate.
"""

from dataclasses import asdict

from . import _study_series_reduction_records as records
from ._checkpoint_codec import canonical_bytes
from .http_replay import primary_http_summary, reference_invocations
from .operational_summary import summarize_operational_runs
from .study_evidence import reduce_study_evidence
from .study_reduction import ReducedStudyBytes


class SeriesReductionError(ValueError):
    """The complete mixed-origin scientific inputs are inconsistent."""


def _reduce(values, internal, external):
    runs = tuple(value.run for value in values)
    reference = reference_invocations(runs[0])
    http = primary_http_summary(runs[20:25])
    operational = summarize_operational_runs(runs)
    records.match_summaries(operational, values)
    study = reduce_study_evidence(
        internal=internal.population,
        external=external.replay.evidence,
        reference=reference,
        http=http,
    )
    return ReducedStudyBytes(
        canonical_bytes(operational), canonical_bytes(asdict(study))
    )


def reduce_series_science(
    historical_prefix,
    fresh_suffix,
    *,
    selected_metadata_bytes,
    current_context_bytes,
    profile_bytes,
    expected_profile_sha256,
    expected_selected_metadata_sha256,
    expected_current_context_sha256,
    internal_snapshot,
    external_snapshot,
) -> ReducedStudyBytes:
    """Reduce every frozen ordinal once, without reauthenticating publications."""
    try:
        values = records.prepare(
            historical_prefix,
            fresh_suffix,
            (selected_metadata_bytes, current_context_bytes, profile_bytes),
            (
                expected_selected_metadata_sha256,
                expected_current_context_sha256,
                expected_profile_sha256,
            ),
            internal_snapshot,
            external_snapshot,
        )
        return _reduce(values, internal_snapshot, external_snapshot)
    except Exception:
        raise SeriesReductionError("invalid_series_reduction") from None
