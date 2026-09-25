"""Link restored external records to retained publisher coordinates without preparing."""

from hashlib import sha256

from . import phishvn, phiusiil, protocol_preflight
from ._retained_preparation_records import load, require
from ._saved_external_inputs import restore_preparation

PREPARED_NAMES = (
    "retained-test.jsonl",
    "quarantine.jsonl",
    "inventory.json",
    "preparation-summary.json",
)


def restore_external(outputs):
    return restore_preparation({name: outputs[name] for name in PREPARED_NAMES})


def _retained(row, source, rules, domains):
    require(row.record_id == source.published_id and row.raw_url == source.raw_url)
    for field in ("source_group", "source_class", "confidence_tier", "published_split"):
        require(getattr(row, field) == getattr(source, field))
    require(
        protocol_preflight.registrable_domain_for_url(row.raw_url, rules)
        == row.registrable_domain
    )
    require(row.registrable_domain not in domains)


def _quarantined(row, source, rules):
    identifier = (
        source.published_id if phishvn._stable_string(source.published_id) else None
    )
    require(row.published_id == identifier)
    try:
        canonical = phiusiil.canonicalize_url(source.raw_url)
        protocol_preflight.registrable_domain_for_url(canonical, rules)
        digest = sha256(canonical.encode("utf-8")).hexdigest()
    except (ValueError, UnicodeError):
        digest = None
    require(row.canonical_url_sha256 == digest)


def validate_external(outputs, publisher, rules, domains):
    prepared = restore_external(outputs)
    inventory = load(outputs["inventory.json"])
    require(inventory["declared_split_counts"] == publisher.published_split_counts)
    originals = {(row.source_split, row.file_position): row for row in publisher.rows}
    for row in prepared.retained:
        _retained(row, originals[(row.source_split, row.file_position)], rules, domains)
    for row in prepared.quarantine:
        _quarantined(row, originals[(row.source_split, row.file_position)], rules)
    return prepared
