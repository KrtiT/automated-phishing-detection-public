"""Validate prepared record consistency without authenticating publisher claims."""

from hashlib import sha256

from . import (
    evaluation_stream,
    fixed_cascade,
    phishvn,
    phiusiil,
    proposed_label_contract,
)
from .protocol_preflight import normalize_hostname

_QUARANTINE_REASONS = frozenset(
    {
        "missing_or_invalid_published_id",
        "missing_phishvn_mapping_field",
        "undefined_phishvn_mapping",
        "invalid_published_split",
        "invalid_or_missing_url",
        "canonical_url_invalid_mapping",
        "canonical_url_conflicting_mapping",
        "canonical_url_missing_stable_id",
        "canonical_url_duplicate_same_mapping",
        "domain_crosses_published_splits",
        "phiusiil_domain_overlap",
        "domain_invalid_published_split",
    }
)


class ExternalInputError(ValueError):
    """Symbolic rejection of a prepared snapshot before inference."""


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise ExternalInputError(reason)


def _mapping(row: phishvn.PreparedExternalRow) -> None:
    decision = proposed_label_contract.evaluate_phishvn_policy(
        row.source_group, row.confidence_tier, row.source_class
    )
    expected_role = {
        "primary_external": "gold" if decision.is_phishing == 1 else "certified",
        "secondary_or_sensitivity": "secondary",
        "reference_negative_control": "tranco",
    }.get(decision.evidence_role)
    _require(
        row.role == expected_role
        and expected_role is not None
        and row.is_phishing == decision.is_phishing
        and phishvn._stable_string(row.record_id)
        and row.source_split == row.published_split == "test",
        "invalid_prepared_mapping",
    )
    evaluation_stream._validate_outcome(
        evaluation_stream.OutcomeMetadata(
            row.record_id, row.registrable_domain, row.is_phishing, row.role
        )
    )


def _url_identity(row: phishvn.PreparedExternalRow) -> None:
    canonical = phiusiil.canonicalize_url(row.raw_url)
    host = normalize_hostname(canonical)
    _require(
        sha256(canonical.encode("utf-8")).hexdigest() == row.canonical_url_sha256
        and (
            host == row.registrable_domain
            or host.endswith("." + row.registrable_domain)
        ),
        "invalid_prepared_url_identity",
    )


def _quarantine_identity(row: phishvn.ExternalQuarantine) -> None:
    reasons = row.reason_codes
    _require(
        type(reasons) is tuple
        and bool(reasons)
        and all(
            type(reason) is str and reason in _QUARANTINE_REASONS for reason in reasons
        )
        and reasons == tuple(sorted(set(reasons))),
        "invalid_quarantine_reasons",
    )
    _require(
        (row.published_id is None or phishvn._stable_string(row.published_id))
        and (row.published_id is None)
        == ("missing_or_invalid_published_id" in reasons),
        "invalid_quarantine_identity",
    )
    _require(
        (row.canonical_url_sha256 is None) == ("invalid_or_missing_url" in reasons),
        "invalid_quarantine_url_identity",
    )
    if row.canonical_url_sha256 is not None:
        fixed_cascade._lowercase_sha256(row.canonical_url_sha256, "quarantine_url_hash")


def _retained_records(rows):
    identities, canonical, positions = set(), set(), set()
    previous = 0
    for row in rows:
        _mapping(row)
        _url_identity(row)
        _require(
            type(row.file_position) is int and previous < row.file_position,
            "invalid_retained_order",
        )
        _require(
            row.record_id not in identities
            and row.canonical_url_sha256 not in canonical,
            "duplicate_retained_identity",
        )
        identities.add(row.record_id)
        canonical.add(row.canonical_url_sha256)
        positions.add(row.file_position)
        previous = row.file_position
    return identities, positions


def _retained_positions(rows, counts):
    identities, positions = _retained_records(rows)
    _require(
        all(position <= counts["test"] for position in positions),
        "invalid_retained_order",
    )
    return identities, positions


def _quarantine_positions(rows, counts, identities, test_positions):
    previous = None
    for row in rows:
        _quarantine_identity(row)
        _require(
            type(row.source_split) is str
            and row.source_split in counts
            and type(row.file_position) is int
            and 1 <= row.file_position <= counts[row.source_split],
            "invalid_quarantine_position",
        )
        coordinate = (row.source_split, row.file_position)
        _require(previous is None or previous < coordinate, "invalid_quarantine_order")
        previous = coordinate
        if row.published_id is not None:
            _require(row.published_id not in identities, "duplicate_prepared_identity")
            identities.add(row.published_id)
        if row.source_split == "test":
            _require(row.file_position not in test_positions, "duplicate_test_position")
            test_positions.add(row.file_position)


def validate_positions(
    prepared: phishvn.PreparedExternal, counts: dict[str, int]
) -> None:
    identities, test_positions = _retained_positions(prepared.retained, counts)
    _quarantine_positions(prepared.quarantine, counts, identities, test_positions)
    _require(
        test_positions == set(range(1, counts["test"] + 1)), "incomplete_test_inventory"
    )
