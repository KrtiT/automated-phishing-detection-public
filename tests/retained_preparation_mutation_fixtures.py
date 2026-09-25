"""Rehash invented derived records without creating independent source authority."""

import json
from dataclasses import asdict, replace
from hashlib import sha256

from retained_study_preparation_fixtures import changed

from automated_phishing_detection import _external_inputs, phishvn
from automated_phishing_detection._checkpoint_codec import canonical_bytes
from automated_phishing_detection.study_feasibility import (
    assess_preparation_feasibility,
)


def overlap_candidate(case, manifest):
    content = canonical_bytes(manifest)
    receipt = json.loads(case.snapshot.payload("source-reconstruction.json"))
    digest = sha256(content).hexdigest()
    receipt["checkpoint_sha256"]["source-overlap.json"] = digest
    receipt["reconstruction"]["private_sha256"]["source-overlap.json"] = digest
    return changed(
        case,
        {
            "source-overlap.json": content,
            "source-reconstruction.json": canonical_bytes(receipt),
        },
        repin=True,
    )


def _external_private(retained, quarantine, inventory):
    return {
        "retained-test.jsonl": b"".join(
            canonical_bytes(asdict(row)) for row in retained
        ),
        "quarantine.jsonl": b"".join(
            canonical_bytes(asdict(row)) for row in quarantine
        ),
        "inventory.json": canonical_bytes(inventory),
    }


def external_candidate(case, *, retained=None, quarantine=None, counts=None):
    prepared = case.external
    retained = prepared.retained if retained is None else retained
    quarantine = prepared.quarantine if quarantine is None else quarantine
    inventory = json.loads(prepared.private_outputs["inventory.json"])
    if counts is not None:
        inventory["declared_split_counts"] = counts
    private = _external_private(retained, quarantine, inventory)
    candidate = replace(
        prepared, retained=retained, quarantine=quarantine, private_outputs=private
    )
    summary = dict(prepared.public_summary)
    summary.update(
        _external_inputs._summary_counts(candidate, inventory["declared_split_counts"])
    )
    summary["private_sha256"] = {
        name: sha256(value).hexdigest() for name, value in private.items()
    }
    candidate = replace(candidate, public_summary=summary)
    _external_inputs.validate_prepared_external(candidate)
    return preparation_candidate(case, candidate)


def preparation_candidate(case, prepared):
    return changed(
        case,
        prepared.private_outputs
        | {
            "preparation-summary.json": canonical_bytes(prepared.public_summary),
            "feasibility.json": assess_preparation_feasibility(
                case.internal.records, prepared
            ),
        },
        repin=True,
    )


def changed_retained(case, **changes):
    first, *remaining = case.external.retained
    return external_candidate(case, retained=(replace(first, **changes), *remaining))


def normalized_preparation(case, *, rows=None, domains=None):
    from automated_phishing_detection.protocol_preflight import parse_suffix_rules

    prepared = phishvn.prepare_external_rows(
        case.publisher.rows if rows is None else rows,
        published_split_counts=case.publisher.published_split_counts,
        test_split="test",
        suffix_rules=parse_suffix_rules(
            case.source.buffers["suffix_rules_bytes"].decode()
        ),
        phiusiil_domains=case.reconstructed.overlap_domains
        if domains is None
        else domains,
    )
    return preparation_candidate(case, prepared)
