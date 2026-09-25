"""Reconstruct accepted preparation and complete source overlap from saved buffers.

Expected pins are caller-supplied claims until an execution wrapper authenticates
them. The original CSV contains protected holdout records; this byte-only module
does not authorize access, open files, fit models, or score records.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from hashlib import sha256

from . import phiusiil, protocol_preflight

_ALGORITHM_ID = "phiusiil-source-overlap-v1"
_SCOPE = "original_valid_domains_before_quarantine"


class SourceOverlapError(ValueError):
    """A symbolic rejection without source values or underlying parser messages."""


@dataclass(frozen=True)
class SourceOverlapPins:
    source_csv_sha256: str
    suffix_rules_sha256: str
    source_spec_sha256: str
    preparation_summary_sha256: str


@dataclass(frozen=True)
class ReconstructedSource:
    group_test_bytes: bytes = field(repr=False)
    overlap_domains: frozenset[str] = field(repr=False)
    private_outputs: dict[str, bytes] = field(repr=False)
    public_summary: dict


def _require(condition, reason):
    if not condition:
        raise SourceOverlapError(reason)


def _reject_constant(value):
    raise SourceOverlapError("invalid_public_json")


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


def _authenticate_inputs(buffers, pins):
    _require(type(pins) is SourceOverlapPins, "invalid_source_pins")
    for name, digest in asdict(pins).items():
        phiusiil._validate_sha256(digest, name)
    for role, content, digest in buffers:
        _require(type(content) is bytes, f"invalid_{role}_bytes")
        _require(sha256(content).hexdigest() == digest, f"{role}_hash_mismatch")


def _public_binding(source_spec_bytes, preparation_summary_bytes, pins):
    source = phiusiil._load_source_spec(source_spec_bytes)
    report = json.loads(
        preparation_summary_bytes,
        object_pairs_hook=phiusiil._object_without_duplicate_keys,
        parse_constant=_reject_constant,
    )
    _require(
        source["phiusiil"]["csv_sha256"] == pins.source_csv_sha256
        and source["public_suffix_list"]["sha256"] == pins.suffix_rules_sha256,
        "source_binding_mismatch",
    )
    _require(
        type(report) is dict
        and report.get("source_spec_sha256") == pins.source_spec_sha256
        and phiusiil._matches_exactly(report.get("declared_sources"), source),
        "preparation_source_binding_mismatch",
    )
    return source, report


def _membership(rows, source_hash, rules):
    membership = []
    domains = set()
    for row in rows:
        record_id = phiusiil.record_id_for_row(source_hash, row.ordinal)
        try:
            canonical = phiusiil.canonicalize_url(row.raw_url)
            domain = protocol_preflight.registrable_domain(
                protocol_preflight.normalize_hostname(canonical), rules
            )
        except (phiusiil.PreparationError, protocol_preflight.PreflightError):
            canonical_hash, domain = None, None
            status = "invalid_url_or_domain"
        else:
            canonical_hash = sha256(canonical.encode("utf-8")).hexdigest()
            domains.add(domain)
            status = "valid_url_and_domain"
        membership.append(
            {
                "source_ordinal": row.ordinal,
                "record_id": record_id,
                "canonical_url_sha256": canonical_hash,
                "registrable_domain": domain,
                "status": status,
            }
        )
    return tuple(membership), frozenset(domains)


def _reconstruct_preparation(rows, rules, source, report, pins):
    resolution = phiusiil.resolve_rows(
        rows, csv_sha256=pins.source_csv_sha256, suffix_rules=rules
    )
    assigned = phiusiil.assign_splits(resolution.retained)
    outputs = phiusiil._private_output_contents(assigned, resolution)
    hashes = {name: sha256(content).hexdigest() for name, content in outputs.items()}
    checksum_content = "".join(
        f"{hashes[name]}  {name}\n" for name in sorted(hashes)
    ).encode("ascii")
    hashes["SHA256SUMS"] = sha256(checksum_content).hexdigest()
    expected = phiusiil._build_summary(
        assigned, resolution, source, pins.source_spec_sha256, hashes
    )
    _require(phiusiil._matches_exactly(report, expected), "preparation_mismatch")
    return assigned, resolution, outputs, hashes


def reconstruct_source_overlap(
    csv_bytes: bytes,
    suffix_rules_bytes: bytes,
    source_spec_bytes: bytes,
    preparation_summary_bytes: bytes,
    *,
    pins: SourceOverlapPins,
) -> ReconstructedSource:
    """Authenticate all buffers before parsing; preserve accepted partition bytes.

    Membership includes every original valid URL/domain before label, duplicate,
    conflict or split exclusions. The generated group-test bytes can enter the
    single frozen scorer pass without reopening a raw partition. Every prior
    preparation output hash and all public counts must match exactly.
    """
    try:
        _require(type(pins) is SourceOverlapPins, "invalid_source_pins")
        _authenticate_inputs(
            (
                ("source_csv", csv_bytes, pins.source_csv_sha256),
                ("suffix_rules", suffix_rules_bytes, pins.suffix_rules_sha256),
                ("source_spec", source_spec_bytes, pins.source_spec_sha256),
                (
                    "preparation_summary",
                    preparation_summary_bytes,
                    pins.preparation_summary_sha256,
                ),
            ),
            pins,
        )
        source, report = _public_binding(
            source_spec_bytes, preparation_summary_bytes, pins
        )
        rules = protocol_preflight.parse_suffix_rules(
            suffix_rules_bytes.decode("utf-8")
        )
        rows = phiusiil._parse_csv_rows(csv_bytes)
        membership, domains = _membership(rows, pins.source_csv_sha256, rules)
        assigned, resolution, outputs, hashes = _reconstruct_preparation(
            rows, rules, source, report, pins
        )
        retained_domains = {row.registrable_domain for row in assigned}
        valid_ids = {
            row["record_id"]
            for row in membership
            if row["status"] == "valid_url_and_domain"
        }
        counts = {
            "input_rows": len(rows),
            "valid_url_domain_rows": len(valid_ids),
            "invalid_url_domain_rows": len(rows) - len(valid_ids),
            "original_valid_domains": len(domains),
            "retained_domains": len(retained_domains),
            "valid_quarantined_rows": sum(
                row.record_id in valid_ids for row in resolution.quarantine
            ),
            "quarantined_only_domains": len(domains - retained_domains),
        }
        identity = {
            "schema_version": 1,
            "algorithm_id": _ALGORITHM_ID,
            "scope": _SCOPE,
            "input_hashes": asdict(pins),
            "reconstructed_output_sha256": hashes,
        }
        private = {
            "source-overlap.json": _json_bytes(
                identity | {"rows": membership, "domains": sorted(domains)}
            )
        }
        summary = identity | {
            "source_binding": "caller_supplied_pins_only",
            "protected_evaluation_authorized": False,
            "counts": counts,
            "private_sha256": {
                name: sha256(content).hexdigest() for name, content in private.items()
            },
        }
        return ReconstructedSource(
            outputs["group_test.jsonl"], domains, private, summary
        )
    except SourceOverlapError:
        raise
    except (ValueError, TypeError, KeyError, OverflowError, RecursionError, csv.Error):
        raise SourceOverlapError("source_reconstruction_failed") from None
