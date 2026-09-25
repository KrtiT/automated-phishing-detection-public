"""Publisher decoding precedes external preparation and permits durable retention."""

from copy import deepcopy
from dataclasses import replace
from hashlib import sha256

import pytest
from phishvn_source_fixtures import bundle, members, record
from test_study_preparation_inputs import api as api

from automated_phishing_detection import phishvn, phishvn_source
from automated_phishing_detection._phishvn_archive import PhishVNSourcePins

SUFFIX = b"example\n"


def decode_external(api, case, **changes):
    values = {
        "archive_pins": PhishVNSourcePins(**case.pins),
        "suffix_rules_sha256": sha256(SUFFIX).hexdigest(),
    } | changes
    return api.decode_external_inputs(case.content, SUFFIX, **values)


def test_external_preserves_roles_raw_order_and_complete_overlap(api):
    rows = [
        record("gold", url="HTTPS://gold.example:443/path"),
        record("certified", source="tinnhiem_web", label="benign"),
        record("secondary", tier="silver"),
        record("control", source="tranco", tier="silver", label="benign"),
        record("overlap", url="https://conflict.example/path"),
    ]
    decoded = decode_external(api, bundle(members(rows)))
    prepared = api.prepare_external_inputs(
        decoded, SUFFIX, overlap_domains=frozenset({"conflict.example"})
    )
    assert type(decoded) is phishvn_source.DecodedPhishVNSource
    assert type(prepared) is phishvn.PreparedExternal
    assert [row.role for row in prepared.retained] == [
        "gold",
        "certified",
        "secondary",
        "tranco",
    ]
    assert [row.is_phishing for row in prepared.retained] == [1, 0, 1, None]
    assert [row.file_position for row in prepared.retained] == [1, 2, 3, 4]
    assert prepared.retained[0].raw_url == rows[0]["url"]
    assert prepared.quarantine[0].reason_codes == ("phiusiil_domain_overlap",)


def test_decoded_bytes_are_available_before_preparation_failure(api, monkeypatch):
    def fail(*args, **kwargs):
        raise ValueError("private-secret-canary")

    monkeypatch.setattr(phishvn, "prepare_external_rows", fail)
    decoded = decode_external(api, bundle())
    retained = dict(decoded.private_outputs)
    public = deepcopy(decoded.public_summary)
    with pytest.raises(
        api.StudyPreparationInputError, match="^invalid_external_inputs$"
    ):
        api.prepare_external_inputs(decoded, SUFFIX, overlap_domains=frozenset())
    assert retained == decoded.private_outputs
    assert public == decoded.public_summary


@pytest.mark.parametrize("digest", ["0" * 64, "F" * 64, "a", None, 1])
def test_psl_hash_is_checked_before_archive_decoder(api, monkeypatch, digest):
    case = bundle()

    def forbidden(*args, **kwargs):
        pytest.fail("archive decoder preceded PSL authentication")

    monkeypatch.setattr(phishvn_source, "decode_phishvn_archive", forbidden)
    with pytest.raises(
        api.StudyPreparationInputError, match="^invalid_external_source$"
    ):
        decode_external(api, case, suffix_rules_sha256=digest)


def test_archive_hash_is_checked_before_member_tables(api, monkeypatch):
    case = bundle()

    def forbidden(*args, **kwargs):
        pytest.fail("parsed unauthenticated archive")

    monkeypatch.setattr(phishvn_source, "_tables", forbidden)
    with pytest.raises(
        api.StudyPreparationInputError, match="^invalid_external_source$"
    ):
        decode_external(
            api, case, archive_pins=PhishVNSourcePins("0" * 64, len(case.content))
        )


@pytest.mark.parametrize(
    "member", ["rows", "published_split_counts", "private_outputs"]
)
def test_external_rejects_changed_decoded_views(api, member):
    decoded = decode_external(api, bundle())
    changes = {
        "rows": (),
        "published_split_counts": {"train": 0, "val": 0, "test": 0},
        "private_outputs": {"publisher-source.json": b"private-secret-canary"},
    }
    changed = replace(decoded, **{member: changes[member]})
    with pytest.raises(
        api.StudyPreparationInputError, match="^invalid_external_inputs$"
    ):
        api.prepare_external_inputs(changed, SUFFIX, overlap_domains=frozenset())


@pytest.mark.parametrize("domains", [None, set(), ("example",), frozenset({"EXAMPLE"})])
def test_external_keeps_existing_overlap_validation(api, domains):
    decoded = decode_external(api, bundle())
    with pytest.raises(
        api.StudyPreparationInputError, match="^invalid_external_inputs$"
    ):
        api.prepare_external_inputs(decoded, SUFFIX, overlap_domains=domains)


def test_external_empty_retention_is_valid_not_a_synthetic_population(api):
    decoded = decode_external(
        api, bundle(members([record("invalid", url="bare.example")]))
    )
    prepared = api.prepare_external_inputs(decoded, SUFFIX, overlap_domains=frozenset())
    assert prepared.retained == ()
    assert len(prepared.quarantine) == 1
    assert prepared.public_summary["retained_test_rows"] == 0
