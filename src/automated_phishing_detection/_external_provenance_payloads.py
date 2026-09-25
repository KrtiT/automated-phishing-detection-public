"""Closed provenance payloads and pure preparation from retained source bytes."""

import json
import re
from dataclasses import asdict
from hashlib import sha256

from . import phishvn, protocol_preflight
from ._checkpoint_codec import canonical_bytes
from ._external_inputs import validate_prepared_external
from ._phishvn_archive import PhishVNSourcePins
from .internal_external_handoff import verify_internal_handoff
from .phishvn_source import DecodedPhishVNSource
from .saved_phishvn_source import restore_phishvn_source

PROVENANCE_NAMES = frozenset(
    {
        "publisher-source.json",
        "publisher-summary.json",
        "suffix-rules.dat",
        "internal-source-overlap.json",
        "internal-source-handoff.json",
        "external-source-reconstruction.json",
    }
)
PREPARED_NAMES = frozenset(
    {
        "retained-test.jsonl",
        "quarantine.jsonl",
        "inventory.json",
        "preparation-summary.json",
    }
)


def require(condition):
    if not condition:
        raise ValueError("invalid_external_source_provenance")


def snapshot(outputs, names):
    require(type(outputs) is dict)
    retained = outputs.copy()
    require(set(retained) == names)
    require(
        all(
            type(name) is str and type(content) is bytes
            for name, content in retained.items()
        )
    )
    return retained


def hashes(outputs):
    return {name: sha256(content).hexdigest() for name, content in outputs.items()}


def provenance_hashes(provenance):
    return hashes(
        {
            name: content
            for name, content in provenance.items()
            if name != "external-source-reconstruction.json"
        }
    )


def verify_receipt_inputs(provenance, outputs, execution, reservation_sha256):
    content = provenance["external-source-reconstruction.json"]
    envelope = json.loads(content)
    require(type(envelope) is dict)
    counts = envelope["counts"]
    require(
        type(counts) is dict
        and set(counts) == {"published_split_counts", "mapping_counts", "preparation"}
    )
    expected = {
        "schema_version": 1,
        "execution": execution,
        "reservation_sha256": reservation_sha256,
        "provenance_sha256": provenance_hashes(provenance),
        "prepared_sha256": hashes(outputs),
        "counts": counts,
    }
    require(canonical_bytes(expected) == content)


def prepared_outputs(prepared):
    validate_prepared_external(prepared)
    return snapshot(
        prepared.private_outputs, PREPARED_NAMES - {"preparation-summary.json"}
    ) | {"preparation-summary.json": canonical_bytes(prepared.public_summary)}


def context(execution, reservation_sha256):
    require(type(execution) is dict)
    require(
        type(reservation_sha256) is str
        and re.fullmatch(r"[0-9a-f]{64}", reservation_sha256) is not None
    )
    return json.loads(canonical_bytes(execution)), reservation_sha256


def decoder_inputs(decoded):
    require(type(decoded) is DecodedPhishVNSource)
    private = snapshot(decoded.private_outputs, {"publisher-source.json"})
    public = canonical_bytes(decoded.public_summary)
    pins = PhishVNSourcePins(**decoded.public_summary["input_archive"])
    restored = restore_phishvn_source(
        private["publisher-source.json"], public, pins=pins
    )
    require(type(decoded.rows) is tuple)
    require(all(type(row) is phishvn.NormalizedExternalRow for row in decoded.rows))
    require(
        canonical_bytes([asdict(row) for row in decoded.rows])
        == canonical_bytes([asdict(row) for row in restored.rows])
    )
    require(type(decoded.published_split_counts) is dict)
    require(
        canonical_bytes(decoded.published_split_counts)
        == canonical_bytes(restored.published_split_counts)
    )
    return restored, private | {"publisher-summary.json": public}


def prepare(decoded, provenance, expected_handoff_sha256):
    handoff = provenance["internal-source-handoff.json"]
    domains = verify_internal_handoff(
        handoff,
        provenance["internal-source-overlap.json"],
        expected_handoff_sha256=expected_handoff_sha256,
    )
    suffix = provenance["suffix-rules.dat"]
    require(
        sha256(suffix).hexdigest()
        == json.loads(handoff)["execution"]["suffix_rules_sha256"]
    )
    prepared = phishvn.prepare_external_rows(
        decoded.rows,
        published_split_counts=decoded.published_split_counts,
        test_split="test",
        suffix_rules=protocol_preflight.parse_suffix_rules(suffix.decode("utf-8")),
        phiusiil_domains=domains,
    )
    validate_prepared_external(prepared)
    return prepared


def receipt(provenance, decoded, prepared, execution, reservation_sha256):
    return canonical_bytes(
        {
            "schema_version": 1,
            "execution": execution,
            "reservation_sha256": reservation_sha256,
            "provenance_sha256": provenance_hashes(provenance),
            "prepared_sha256": hashes(prepared_outputs(prepared)),
            "counts": {
                "published_split_counts": decoded.published_split_counts,
                "mapping_counts": decoded.public_summary["mapping_counts"],
                "preparation": prepared.public_summary,
            },
        }
    )
