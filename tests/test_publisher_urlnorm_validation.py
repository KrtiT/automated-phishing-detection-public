"""Reject mixed representations and rehashed invented publisher forgeries."""

from dataclasses import replace
from hashlib import sha256

import pytest
from test_publisher_urlnorm import encoded, module, restore, sample

from automated_phishing_detection import phishvn_source, saved_phishvn_source


def rehash(source, summary):
    summary["private_sha256"]["publisher-source.json"] = sha256(
        encoded(source)
    ).hexdigest()


@pytest.mark.parametrize("field", ["parent_source_sha256", "parent_summary_sha256"])
@pytest.mark.parametrize("target", ["source", "summary", "both"])
def test_both_parent_hashes_are_checked_even_after_rehashing_derived_source(
    field, target
):
    _, _, source, summary, pins = sample()
    for document in (
        [source, summary]
        if target == "both"
        else [source if target == "source" else summary]
    ):
        document[field] = "a" * 64
    rehash(source, summary)
    with pytest.raises(phishvn_source.PhishVNSourceError):
        restore(source, summary, pins)


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", 1),
        ("schema_version", True),
        ("schema_version", 2.0),
        ("algorithm_id", "phishvn-publisher-decoder-v1"),
        ("representation", "url"),
        ("parent_source_sha256", "F" * 64),
        ("parent_summary_sha256", None),
        ("protected_evaluation_authorized", True),
    ],
)
def test_derived_identity_is_closed_and_typed(field, value):
    _, _, source, summary, pins = sample()
    source[field] = summary[field] = value
    rehash(source, summary)
    with pytest.raises(phishvn_source.PhishVNSourceError):
        restore(source, summary, pins)


@pytest.mark.parametrize(
    "field", ["raw", "normalized", "header", "mapping", "position", "counts", "member"]
)
def test_rehashed_source_and_summary_tampering_cannot_rewrite_retained_ancestry(field):
    _, _, source, summary, pins = sample()
    if field in ("raw", "normalized"):
        header = source["headers"][source["rows"][0]["source_member"]]
        source["rows"][0]["cells"][
            header.index("url" if field == "raw" else "url_norm")
        ] = "https://forged.example/"
    elif field == "header":
        source["headers"]["data/splits/url_train.csv"].reverse()
    elif field == "mapping":
        source["rows"][0]["normalization_status"] = "undefined_mapping"
    elif field == "position":
        source["rows"][0]["file_position"] = 2
    elif field == "counts":
        summary["published_split_counts"]["test"] += 1
    else:
        source["member_inventory"]["LICENSE"]["sha256"] = "b" * 64
    rehash(source, summary)
    with pytest.raises(phishvn_source.PhishVNSourceError):
        restore(source, summary, pins)


@pytest.mark.parametrize("target", ["source", "summary"])
@pytest.mark.parametrize("operation", ["extra", "missing", "noncanonical", "nonbytes"])
def test_derived_documents_require_exact_canonical_closed_records(target, operation):
    _, _, source, summary, pins = sample()
    document = source if target == "source" else summary
    if operation == "extra":
        document["private-canary"] = "unexpected"
    elif operation == "missing":
        del document["parent_source_sha256"]
    rehash(source, summary)
    arguments = {"source": encoded(source), "summary": encoded(summary)}
    if operation == "noncanonical":
        arguments[target] += b" "
    elif operation == "nonbytes":
        arguments[target] = bytearray(arguments[target])
    with pytest.raises(phishvn_source.PhishVNSourceError) as rejected:
        module().restore_derived_publisher(
            arguments["source"], arguments["summary"], pins=pins
        )
    assert "private-canary" not in str(rejected.value)


@pytest.mark.parametrize("derived_source", [True, False])
def test_mixed_v1_and_v2_documents_reject_at_generic_dispatch(derived_source):
    parent, _, source, summary, pins = sample()
    original_source = parent.private_outputs["publisher-source.json"]
    original_summary = encoded(parent.public_summary)
    with pytest.raises(phishvn_source.PhishVNSourceError):
        saved_phishvn_source.restore_phishvn_source(
            encoded(source) if derived_source else original_source,
            original_summary if derived_source else encoded(summary),
            pins=pins,
        )


def test_derived_parent_cannot_be_amended_again():
    _, derived, _, _, _ = sample()
    with pytest.raises(phishvn_source.PhishVNSourceError):
        module().derive_publisher_url_norm(derived)


@pytest.mark.parametrize("changes", ["rows", "counts", "private", "summary"])
def test_parent_object_must_match_its_authenticated_original_bytes(changes):
    parent, _, _, _, _ = sample()
    if changes == "rows":
        parent = replace(
            parent, rows=(replace(parent.rows[0], raw_url="forged"), *parent.rows[1:])
        )
    elif changes == "counts":
        parent = replace(parent, published_split_counts={"test": 9})
    elif changes == "private":
        parent = replace(
            parent, private_outputs=parent.private_outputs | {"extra": b"forged"}
        )
    else:
        parent = replace(
            parent, public_summary=parent.public_summary | {"extra": "forged"}
        )
    with pytest.raises(phishvn_source.PhishVNSourceError):
        module().derive_publisher_url_norm(parent)
