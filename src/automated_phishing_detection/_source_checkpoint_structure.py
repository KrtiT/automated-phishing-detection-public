"""Shared original-overlap structure; retained claims are not original-source proof."""

import json
from dataclasses import asdict
from hashlib import sha256

from . import (
    execution_receipt,
    fixed_cascade,
    phiusiil,
    protocol_preflight,
    source_overlap,
)
from .evaluation_producer import _json_bytes
from .source_checkpoints import CHECKPOINT_NAMES, _hashes

_MEMBERSHIP_FIELDS = {
    "source_ordinal",
    "record_id",
    "canonical_url_sha256",
    "registrable_domain",
    "status",
}


def _require(condition):
    if not condition:
        raise ValueError("source_checkpoint_mismatch")


def _json(content):
    return json.loads(
        content,
        object_pairs_hook=fixed_cascade._object_without_duplicate_keys,
        parse_constant=fixed_cascade._reject_json_constant,
    )


def _same(first, second):
    return _json_bytes(first) == _json_bytes(second)


def _member(row, identity, ordinal):
    _require(type(row) is dict and set(row) == _MEMBERSHIP_FIELDS)
    _require(type(row["source_ordinal"]) is int and row["source_ordinal"] == ordinal)
    _require(
        row["record_id"]
        == phiusiil.record_id_for_row(identity["source_csv_sha256"], ordinal)
    )
    if row["status"] == "invalid_url_or_domain":
        _require(
            row["canonical_url_sha256"] is None and row["registrable_domain"] is None
        )
        return None
    _require(row["status"] == "valid_url_and_domain")
    domain, digest = row["registrable_domain"], row["canonical_url_sha256"]
    _require(type(domain) is str and domain.isascii())
    _require(protocol_preflight.normalize_hostname("https://" + domain) == domain)
    _require(
        type(digest) is str and execution_receipt._SHA256.fullmatch(digest) is not None
    )
    return digest, domain


def _membership(rows, identity, expected_count):
    _require(type(rows) is list and len(rows) == expected_count)
    valid, canonical_domains = {}, {}
    for ordinal, row in enumerate(rows, 1):
        member = _member(row, identity, ordinal)
        if member is not None:
            digest, domain = member
            _require(canonical_domains.setdefault(digest, domain) == domain)
            valid[row["record_id"]] = member
    return valid


def _common(identity, report):
    return {
        "schema_version": 1,
        "algorithm_id": source_overlap._ALGORITHM_ID,
        "scope": source_overlap._SCOPE,
        "input_hashes": {
            name: identity[name]
            for name in (
                "source_csv_sha256",
                "suffix_rules_sha256",
                "source_spec_sha256",
                "preparation_summary_sha256",
            )
        },
        "reconstructed_output_sha256": report["output_hashes"],
    }


def _manifest(contents, identity, report):
    manifest = _json(contents["source-overlap.json"])
    common = _common(identity, report)
    _require(
        type(manifest) is dict and set(manifest) == set(common) | {"rows", "domains"}
    )
    _require(contents["source-overlap.json"] == _json_bytes(manifest))
    _require(_same({name: manifest[name] for name in common}, common))
    valid = _membership(
        manifest["rows"], identity, report["overall_counts"]["input_rows"]
    )
    domains = sorted({domain for unused, domain in valid.values()})
    _require(_same(manifest["domains"], domains))
    groups = {digest for digest, unused in valid.values()}
    _require(len(groups) == report["overall_counts"]["canonicalized_url_groups"])
    return common, valid, domains


def _summary(common, valid, domains, report, overlap_hash):
    overall = report["overall_counts"]
    counts = {
        "input_rows": overall["input_rows"],
        "valid_url_domain_rows": len(valid),
        "invalid_url_domain_rows": overall["input_rows"] - len(valid),
        "original_valid_domains": len(domains),
        "retained_domains": overall["retained_domains"],
        "valid_quarantined_rows": len(valid) - overall["retained_rows"],
        "quarantined_only_domains": len(domains) - overall["retained_domains"],
    }
    _require(all(type(value) is int and value >= 0 for value in counts.values()))
    _require(
        counts["invalid_url_domain_rows"]
        == report["quarantine_reason_counts"]["invalid_or_missing_url"]
    )
    return common | {
        "source_binding": "caller_supplied_pins_only",
        "protected_evaluation_authorized": False,
        "counts": counts,
        "private_sha256": {"source-overlap.json": overlap_hash},
    }


def _prepared_partition(partition, internal, valid):
    _require(sha256(partition).hexdigest() == internal.partition_sha256)
    _require(
        partition == b"".join(_json_bytes(asdict(row)) for row in internal.records)
    )
    for row in internal.records:
        _require(
            valid.get(row.record_id)
            == (row.canonical_url_sha256, row.registrable_domain)
        )


def verify_internal_preparation_structure(
    contents, *, internal, preparation_report, execution, reservation_sha256
):
    _require(type(contents) is dict and set(contents) == CHECKPOINT_NAMES)
    _require(all(type(content) is bytes for content in contents.values()))
    hashes = _hashes(contents)
    common, valid, domains = _manifest(contents, execution, preparation_report)
    summary = _summary(
        common, valid, domains, preparation_report, hashes["source-overlap.json"]
    )
    expected = {
        "schema_version": 1,
        "execution": execution,
        "reservation_sha256": reservation_sha256,
        "reconstruction": summary,
        "checkpoint_sha256": {
            name: digest
            for name, digest in hashes.items()
            if name != "source-reconstruction.json"
        },
    }
    _require(contents["source-reconstruction.json"] == _json_bytes(expected))
    _prepared_partition(contents["group_test.jsonl"], internal, valid)
    return frozenset(domains)
