"""Prepare authenticated caller-held buffers without models or filesystem access.

These helpers grant no source-access, scoring or continuation authority. The
coordinator retains decoder bytes before invoking external preparation.
"""

from hashlib import sha256

from . import (
    evaluation_producer,
    phishvn,
    phishvn_source,
    protocol_preflight,
    source_overlap,
)
from ._external_inputs import validate_prepared_external
from ._external_provenance_payloads import decoder_inputs
from ._phishvn_archive import PhishVNSourcePins
from .phiusiil import _validate_sha256

_INTERNAL_SOURCE_FIELDS = frozenset(
    {
        "expected_sha256",
        "source_csv_sha256",
        "suffix_rules_sha256",
        "expected_row_count",
        "expected_domain_count",
        "expected_class_counts",
    }
)


class StudyPreparationInputError(ValueError):
    """Symbolic preparation rejection without private source values."""


def _internal_claims(pins, source):
    if (
        type(pins) is not source_overlap.SourceOverlapPins
        or type(source) is not dict
        or set(source) != _INTERNAL_SOURCE_FIELDS
        or source["source_csv_sha256"] != pins.source_csv_sha256
        or source["suffix_rules_sha256"] != pins.suffix_rules_sha256
    ):
        raise StudyPreparationInputError("invalid_internal_inputs")


def prepare_internal_inputs(
    csv_bytes: bytes,
    suffix_bytes: bytes,
    source_spec_bytes: bytes,
    preparation_summary_bytes: bytes,
    *,
    pins: source_overlap.SourceOverlapPins,
    source: dict,
) -> tuple[source_overlap.ReconstructedSource, evaluation_producer.PreparedInternal]:
    """Keep the original reconstruction and strict two-class partition policy."""
    try:
        _internal_claims(pins, source)
        reconstructed = source_overlap.reconstruct_source_overlap(
            csv_bytes,
            suffix_bytes,
            source_spec_bytes,
            preparation_summary_bytes,
            pins=pins,
        )
        rules = protocol_preflight.parse_suffix_rules(suffix_bytes.decode("utf-8"))
        prepared = evaluation_producer.parse_internal_partition(
            reconstructed.group_test_bytes, suffix_rules=rules, **source
        )
        return reconstructed, prepared
    except Exception:
        raise StudyPreparationInputError("invalid_internal_inputs") from None


def decode_external_inputs(
    archive_bytes: bytes,
    suffix_bytes: bytes,
    *,
    archive_pins: PhishVNSourcePins,
    suffix_rules_sha256: str,
) -> phishvn_source.DecodedPhishVNSource:
    """Authenticate PSL before decoding; return before any external preparation."""
    try:
        _validate_sha256(suffix_rules_sha256, "suffix_rules_sha256")
        if type(suffix_bytes) is not bytes or sha256(suffix_bytes).hexdigest() != (
            suffix_rules_sha256
        ):
            raise StudyPreparationInputError("invalid_external_source")
        return phishvn_source.decode_phishvn_archive(archive_bytes, pins=archive_pins)
    except Exception:
        raise StudyPreparationInputError("invalid_external_source") from None


def prepare_external_inputs(
    decoded: phishvn_source.DecodedPhishVNSource,
    suffix_bytes: bytes,
    *,
    overlap_domains: frozenset[str],
) -> phishvn.PreparedExternal:
    """Apply unchanged preparation to the retained decoder view and complete overlap."""
    try:
        if type(suffix_bytes) is not bytes:
            raise StudyPreparationInputError("invalid_external_inputs")
        restored, unused = decoder_inputs(decoded)
        prepared = phishvn.prepare_external_rows(
            restored.rows,
            published_split_counts=restored.published_split_counts,
            test_split="test",
            suffix_rules=protocol_preflight.parse_suffix_rules(
                suffix_bytes.decode("utf-8")
            ),
            phiusiil_domains=overlap_domains,
        )
        validate_prepared_external(prepared)
        return prepared
    except Exception:
        raise StudyPreparationInputError("invalid_external_inputs") from None
