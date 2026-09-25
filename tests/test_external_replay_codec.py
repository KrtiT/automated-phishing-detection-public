from dataclasses import FrozenInstanceError, replace
from hashlib import sha256

import pytest
from external_replay_codec_fixtures import decode, encode, module, sample

from automated_phishing_detection.phiusiil import canonicalize_url

__all__ = ["sample"]


@pytest.mark.parametrize("count", [1000, 1001])
def test_roundtrip_preserves_all_roles_and_complete_order(sample, count):
    records = sample.retained[:count]
    content = encode(records)
    restored = decode(content)
    assert restored == records and type(restored) is tuple
    assert encode(restored) == content
    assert {row.role for row in restored} == {
        "gold",
        "certified",
        "secondary",
        "tranco",
    }
    assert any(row.is_phishing is None for row in restored)
    assert all(row.is_phishing is None for row in restored if row.role == "tranco")


def test_original_preparation_hash_convention_is_used(sample):
    content = sample.private_outputs["retained-test.jsonl"]
    expected = sample.public_summary["private_sha256"]["retained-test.jsonl"]
    assert encode(decode(content, expected)) == content


def test_noncontiguous_positions_do_not_invent_quarantine(sample):
    records = tuple(
        replace(row, file_position=row.file_position * 3) for row in sample.retained
    )
    assert decode(encode(records)) == records


def test_distinct_urls_can_share_one_domain(sample):
    records = tuple(
        replace(
            row,
            raw_url=f"https://shared.example.com/{position}",
            registrable_domain="example.com",
            canonical_url_sha256=sha256(
                f"https://shared.example.com/{position}".encode()
            ).hexdigest(),
        )
        for position, row in enumerate(sample.retained)
    )
    assert decode(encode(records)) == records


def test_raw_unicode_spelling_remains_byte_exact(sample):
    first, *remaining = sample.retained
    raw_url = "HTTPS://Host.Example.Com:00443/café?encoded=%ab#fragment"
    first = replace(
        first,
        raw_url=raw_url,
        registrable_domain="example.com",
        canonical_url_sha256=sha256(canonicalize_url(raw_url).encode()).hexdigest(),
    )
    records = (first, *remaining)
    content = encode(records)
    assert content.isascii() and b"caf\\u00e9" in content
    assert decode(content) == records


def test_restored_records_are_immutable_and_fresh(sample):
    content = encode(sample.retained)
    first, second = decode(content), decode(content)
    assert first == second and first[0] is not second[0]
    with pytest.raises(FrozenInstanceError):
        first[0].raw_url = "https://replacement.example/"


@pytest.mark.parametrize("count", [0, 1, 999])
def test_short_stream_cannot_be_an_operational_manifest(sample, count):
    api = module()
    with pytest.raises(api.ExternalReplayCodecError):
        decode(encode(sample.retained[:count]))


@pytest.mark.parametrize(
    "digest", ["f" * 64, "F" * 64, "short", "", True, 1, b"a" * 64]
)
def test_independent_hash_must_match_exact_lowercase_digest(sample, digest):
    api = module()
    with pytest.raises(api.ExternalReplayCodecError):
        decode(encode(sample.retained), digest)
