"""Mechanical preparation of normalized external rows, exercised on fixtures.

This is not a parser for any verified PhishVN file schema. No file is opened and
no numeric label encoding, source spelling, or published split name is guessed.
The caller supplies normalized policy values and a complete split inventory;
matching that inventory does not authenticate its completeness against a source.
The official schema, license, source/version hashes, normalization mapping, PSL,
PhiUSIIL overlap set and staged execution freeze still require separate binding.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from hashlib import sha256

from . import phiusiil, proposed_label_contract, protocol_preflight
from .evaluation_stream import OutcomeMetadata
from .proposed_label_contract import LabelDecision

_ROLES = ("gold", "certified", "secondary", "tranco")


class ExternalPreparationError(ValueError):
    """Normalized inputs lack a consistent declared identity or split inventory."""


@dataclass(frozen=True)
class NormalizedExternalRow:
    published_id: object
    source_split: str
    file_position: int
    raw_url: object
    source_group: object
    source_class: object
    confidence_tier: object
    published_split: object


@dataclass(frozen=True)
class PreparedExternalRow:
    record_id: str
    source_split: str
    file_position: int
    raw_url: str
    canonical_url_sha256: str
    registrable_domain: str
    is_phishing: int | None
    role: str
    source_group: str
    source_class: str
    confidence_tier: str
    published_split: str


@dataclass(frozen=True)
class ExternalQuarantine:
    published_id: str | None
    source_split: str
    file_position: int
    canonical_url_sha256: str | None
    reason_codes: tuple[str, ...]


@dataclass(frozen=True)
class PreparedExternal:
    retained: tuple[PreparedExternalRow, ...]
    quarantine: tuple[ExternalQuarantine, ...]
    private_outputs: dict[str, bytes]
    public_summary: dict

    @property
    def metadata(self) -> tuple[OutcomeMetadata, ...]:
        return tuple(
            OutcomeMetadata(
                row.record_id, row.registrable_domain, row.is_phishing, row.role
            )
            for row in self.retained
        )


@dataclass(frozen=True)
class _Candidate:
    row: NormalizedExternalRow
    published_id: str | None
    canonical_url: str | None
    canonical_url_sha256: str | None
    domain: str | None
    decision: LabelDecision


def _stable_string(value: object) -> bool:
    return (
        type(value) is str
        and bool(value)
        and all(
            character.isprintable() and not character.isspace() for character in value
        )
    )


def _inventory(rows, counts, test_split):
    if not isinstance(counts, Mapping):
        raise ExternalPreparationError("declare the complete published split inventory")
    declared = dict(counts)
    if (
        not _stable_string(test_split)
        or test_split not in declared
        or len(declared) < 2
        or any(not _stable_string(name) for name in declared)
        or any(type(count) is not int or count < 0 for count in declared.values())
    ):
        raise ExternalPreparationError(
            "inventory must include test and non-test splits"
        )
    if type(rows) not in (tuple, list):
        raise ExternalPreparationError("rows must be a materialized tuple or list")
    positions = {name: set() for name in declared}
    identifiers = set()
    for row in rows:
        if type(row) is not NormalizedExternalRow:
            raise ExternalPreparationError("each row must be a NormalizedExternalRow")
        if type(row.source_split) is not str or row.source_split not in declared:
            raise ExternalPreparationError("source split is absent from inventory")
        if (
            type(row.file_position) is not int
            or not 1 <= row.file_position <= declared[row.source_split]
            or row.file_position in positions[row.source_split]
        ):
            raise ExternalPreparationError(
                "source file positions are invalid or ambiguous"
            )
        positions[row.source_split].add(row.file_position)
        if _stable_string(row.published_id):
            if row.published_id in identifiers:
                raise ExternalPreparationError("duplicate published ID is ambiguous")
            identifiers.add(row.published_id)
    if any(len(positions[name]) != count for name, count in declared.items()):
        raise ExternalPreparationError("rows do not cover the declared split inventory")
    return declared, tuple(
        sorted(rows, key=lambda row: (row.source_split, row.file_position))
    )


def _validate_domains(domains, rules):
    if type(rules) is not protocol_preflight.SuffixRules:
        raise ExternalPreparationError("suffix rules must be parsed SuffixRules")
    if type(domains) is not frozenset:
        raise ExternalPreparationError("PhiUSIIL domains must be an explicit frozenset")
    for domain in domains:
        try:
            normalized = protocol_preflight._ascii_domain(domain)
            protocol_preflight._reject_ip_literal(normalized)
            registrable = protocol_preflight.registrable_domain(normalized, rules)
        except (ValueError, TypeError, AttributeError) as exc:
            raise ExternalPreparationError("invalid PhiUSIIL domain claim") from exc
        if normalized != domain or registrable != domain:
            raise ExternalPreparationError(
                "PhiUSIIL domain claim is not canonical registrable"
            )


def _candidate(row, rules, reasons):
    identity = row.published_id if _stable_string(row.published_id) else None
    if identity is None:
        reasons.add("missing_or_invalid_published_id")
    decision = proposed_label_contract.evaluate_phishvn_policy(
        row.source_group, row.confidence_tier, row.source_class
    )
    if decision.disposition == "quarantine":
        reasons.add(decision.reason_code)
    if type(row.published_split) is not str or row.published_split != row.source_split:
        reasons.add("invalid_published_split")
    try:
        canonical = phiusiil.canonicalize_url(row.raw_url)
        domain = protocol_preflight.registrable_domain_for_url(canonical, rules)
        canonical_hash = sha256(canonical.encode("utf-8")).hexdigest()
    except (ValueError, UnicodeError):
        canonical, canonical_hash, domain = None, None, None
        reasons.add("invalid_or_missing_url")
    return _Candidate(row, identity, canonical, canonical_hash, domain, decision)


def _quarantine_groups(candidates, reasons, phiusiil_domains):
    by_canonical, by_domain = defaultdict(list), defaultdict(list)
    for index, candidate in enumerate(candidates):
        if candidate.canonical_url is not None:
            by_canonical[candidate.canonical_url].append(index)
        if candidate.domain is not None:
            by_domain[candidate.domain].append(index)

    for indices in by_canonical.values():
        group = [candidates[index] for index in indices]
        group_reasons = set()
        if any(item.decision.disposition == "quarantine" for item in group):
            group_reasons.add("canonical_url_invalid_mapping")
        mappings = {
            (item.decision.disposition, item.decision.is_phishing)
            for item in group
            if item.decision.disposition != "quarantine"
        }
        if len(mappings) > 1:
            group_reasons.add("canonical_url_conflicting_mapping")
        if len(group) > 1 and any(item.published_id is None for item in group):
            group_reasons.add("canonical_url_missing_stable_id")
        for index in indices:
            reasons[index].update(group_reasons)
        if not group_reasons and len(group) > 1:
            winner = min(indices, key=lambda index: candidates[index].published_id)
            for index in indices:
                if index != winner:
                    reasons[index].add("canonical_url_duplicate_same_mapping")

    # Domain exclusions use every located row, not only mapping-eligible test rows.
    for domain, indices in by_domain.items():
        group_reasons = set()
        if len({candidates[index].row.source_split for index in indices}) > 1:
            group_reasons.add("domain_crosses_published_splits")
        if domain in phiusiil_domains:
            group_reasons.add("phiusiil_domain_overlap")
        if any("invalid_published_split" in reasons[index] for index in indices):
            group_reasons.add("domain_invalid_published_split")
        for index in indices:
            reasons[index].update(group_reasons)


def _retained(candidate):
    decision, row = candidate.decision, candidate.row
    if decision.evidence_role == "primary_external":
        role = "gold" if decision.is_phishing == 1 else "certified"
    elif decision.evidence_role == "reference_negative_control":
        role = "tranco"
    elif decision.evidence_role == "secondary_or_sensitivity":
        role = "secondary"
    else:
        raise ExternalPreparationError("eligible row lacks a defined outcome role")
    return PreparedExternalRow(
        candidate.published_id,
        row.source_split,
        row.file_position,
        row.raw_url,
        candidate.canonical_url_sha256,
        candidate.domain,
        decision.is_phishing,
        role,
        row.source_group,
        row.source_class,
        row.confidence_tier,
        row.published_split,
    )


def _json_bytes(value):
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("ascii")


def prepare_external_rows(
    rows: tuple[NormalizedExternalRow, ...],
    *,
    published_split_counts: Mapping[str, int],
    test_split: str,
    suffix_rules: protocol_preflight.SuffixRules,
    phiusiil_domains: frozenset[str],
) -> PreparedExternal:
    """Apply existing mappings and quarantine before selecting the test stream.

    source_split/file_position are caller-declared file coordinates. They allow
    a missing publisher split value to be quarantined without guessing where
    that row came from. Every declared split must have positions 1..count exactly
    once. An inventory with only the test split is insufficient for this kernel.
    Same-label duplicate selection does not prefer gold or another evidence tier.
    Reason counts are incidences and can exceed the number of quarantined rows.
    """
    declared, ordered = _inventory(rows, published_split_counts, test_split)
    _validate_domains(phiusiil_domains, suffix_rules)
    reasons = [set() for _ in ordered]
    candidates = tuple(
        _candidate(row, suffix_rules, reasons[index])
        for index, row in enumerate(ordered)
    )
    _quarantine_groups(candidates, reasons, phiusiil_domains)
    retained, quarantine = [], []
    valid_non_test = 0
    for candidate, row_reasons in zip(candidates, reasons, strict=True):
        row = candidate.row
        if row_reasons:
            quarantine.append(
                ExternalQuarantine(
                    candidate.published_id,
                    row.source_split,
                    row.file_position,
                    candidate.canonical_url_sha256,
                    tuple(sorted(row_reasons)),
                )
            )
        elif row.source_split == test_split:
            retained.append(_retained(candidate))
        else:
            valid_non_test += 1
    retained.sort(key=lambda row: row.file_position)
    role_counts = Counter(row.role for row in retained)
    reason_counts = Counter(reason for row in quarantine for reason in row.reason_codes)
    private = {
        "retained-test.jsonl": b"".join(_json_bytes(asdict(row)) for row in retained),
        "quarantine.jsonl": b"".join(_json_bytes(asdict(row)) for row in quarantine),
        "inventory.json": _json_bytes(
            {
                "declared_split_counts": declared,
                "test_split": test_split,
                "inventory_binding": "caller_supplied_claim_only",
            }
        ),
    }
    summary = {
        "schema_version": 1,
        "scope": "normalized_inputs_only",
        "schema_verified": False,
        "protected_evaluation_authorized": False,
        "inventory_binding": "caller_supplied_claim_only",
        "declared_split_count": len(declared),
        "input_row_count": len(ordered),
        "input_test_rows": declared[test_split],
        "retained_test_rows": len(retained),
        "valid_non_test_rows": valid_non_test,
        "quarantined_rows": len(quarantine),
        "quarantined_test_rows": sum(
            row.source_split == test_split for row in quarantine
        ),
        "retained_test_domain_count": len({row.registrable_domain for row in retained}),
        "role_counts": {role: role_counts[role] for role in _ROLES},
        "quarantine_reason_counts": dict(sorted(reason_counts.items())),
        "quarantine_count_basis": "reason_incidences_not_disjoint_categories",
        "private_sha256": {
            name: sha256(content).hexdigest() for name, content in private.items()
        },
    }
    return PreparedExternal(tuple(retained), tuple(quarantine), private, summary)
