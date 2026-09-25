import json
from dataclasses import replace
from hashlib import sha256

import pytest
from test_replay_manifest_codec import candidates, decode, encode, manifests, module

from automated_phishing_detection import evaluation_manifest

__all__ = ["candidates", "manifests"]


@pytest.mark.parametrize("prevalence", [True, 100.0, "100", None, 1, 10])
def test_decoder_requires_exact_expected_prevalence(manifests, prevalence):
    api = module()
    content = api.encode_replay_manifest(manifests[100])
    with pytest.raises(api.ReplayManifestCodecError):
        decode(content, prevalence)


@pytest.mark.parametrize(
    "mutation",
    [
        "newline",
        "duplicate_key",
        "ascii_escape",
        "float_version",
        "row_scalar",
        "null_records",
    ],
)
def test_decoder_requires_exact_original_encoding(manifests, mutation):
    api = module()
    content = api.encode_replay_manifest(manifests[100])
    if mutation == "newline":
        content += b"\n"
    elif mutation == "duplicate_key":
        content = content.replace(
            b'"schema_version":1', b'"schema_version":1,"schema_version":1'
        )
    elif mutation == "ascii_escape":
        content = content.replace(b"\xc3\xa9", b"\\u00e9")
    else:
        value = json.loads(content)
        if mutation == "float_version":
            value["schema_version"] = 1.0
        elif mutation == "row_scalar":
            value["records"][0] = 7
        else:
            value["records"] = None
        content = encode(value)
    with pytest.raises(api.ReplayManifestCodecError):
        decode(content)


def test_decoder_preserves_many_urls_on_one_domain(manifests):
    api, original = module(), manifests[100]
    records = tuple(
        replace(
            row,
            raw_url=f"https://shared.example/{position}",
            registrable_domain="shared.example",
            canonical_url_sha256=sha256(
                f"https://shared.example/{position}".encode()
            ).hexdigest(),
        )
        for position, row in enumerate(original.records)
    )
    manifest = replace(
        original,
        records=records,
        sha256=evaluation_manifest._manifest_hash(100, records),
    )
    assert decode(api.encode_replay_manifest(manifest)) == manifest


@pytest.mark.parametrize("interruption", [KeyboardInterrupt(), SystemExit(4)])
def test_decoder_preserves_parser_interruption(monkeypatch, interruption):
    module()

    def interrupted(*_arguments, **_keywords):
        raise interruption

    monkeypatch.setattr(json, "loads", interrupted)
    with pytest.raises(type(interruption)) as captured:
        decode(b"{}")
    assert captured.value is interruption
