import importlib
import importlib.util
import json
from dataclasses import replace
from hashlib import sha256

import pytest
from test_evaluation_manifest import GOLDENS, candidates, manifests

from automated_phishing_detection import evaluation_manifest

__all__ = ["candidates", "manifests"]


def module():
    name = "automated_phishing_detection.replay_manifest_codec"
    assert importlib.util.find_spec(name), "missing retained replay manifest codec"
    return importlib.import_module(name)


def encode(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def decode(content, prevalence=100, digest=None):
    return module().decode_replay_manifest(
        content,
        expected_sha256=sha256(content).hexdigest() if digest is None else digest,
        expected_prevalence_basis_points=prevalence,
    )


@pytest.mark.parametrize("prevalence", [10, 100, 500])
def test_roundtrip_preserves_existing_full_manifest_hash(manifests, prevalence):
    api, manifest = module(), manifests[prevalence]
    content = api.encode_replay_manifest(manifest)
    assert sha256(content).hexdigest() == GOLDENS[prevalence][2] == manifest.sha256
    assert not content.endswith(b"\n") and b"caf\xc3\xa9" in content
    restored = decode(content, prevalence)
    assert restored == manifest
    assert (
        restored.records == manifest.records
        and restored.warmup_records == manifest.records[:1000]
    )


def test_hash_rejection_precedes_json_decode(monkeypatch):
    api = module()
    monkeypatch.setattr(
        json, "loads", lambda *_args, **_kwargs: pytest.fail("unbound bytes parsed")
    )
    with pytest.raises(api.ReplayManifestCodecError):
        decode(b"not JSON", digest="f" * 64)


def test_codec_never_resamples_or_sorts(manifests, monkeypatch):
    api, manifest = module(), manifests[100]

    def forbidden(*_args, **_kwargs):
        pytest.fail("retained codec invoked selection")

    for name in ("build_manifest", "_permutation", "_validated_candidates"):
        monkeypatch.setattr(evaluation_manifest, name, forbidden)
    monkeypatch.setattr(evaluation_manifest.np.random, "Generator", forbidden)
    assert decode(api.encode_replay_manifest(manifest)) == manifest


@pytest.mark.parametrize(
    "field, value",
    [
        ("sha256", "f" * 64),
        ("prevalence_basis_points", True),
        ("prevalence_basis_points", 10),
        ("records", ()),
    ],
)
def test_encoder_rejects_inconsistent_manifest(manifests, field, value):
    api = module()
    with pytest.raises(api.ReplayManifestCodecError):
        api.encode_replay_manifest(replace(manifests[100], **{field: value}))


@pytest.mark.parametrize(
    "mutation",
    [
        "version_bool",
        "algorithm",
        "extra",
        "prevalence",
        "records_short",
        "records_extra",
        "row_extra",
        "row_missing",
        "label_bool",
        "label_float",
        "label_change",
        "split",
        "duplicate",
        "source",
        "hash",
        "domain",
        "raw_url",
    ],
)
def test_decoder_rejects_rehashed_invalid_population(manifests, mutation):
    api = module()
    value = json.loads(api.encode_replay_manifest(manifests[100]))
    row = value["records"][0]
    changes = {
        "label_bool": ("is_phishing", True),
        "label_float": ("is_phishing", 0.0),
        "label_change": ("is_phishing", 1 - row["is_phishing"]),
        "split": ("split", "train"),
        "hash": ("canonical_url_sha256", "f" * 64),
        "domain": ("registrable_domain", "wrong.example"),
        "raw_url": ("raw_url", "invalid"),
    }
    if mutation in changes:
        name, changed = changes[mutation]
        row[name] = changed
    elif mutation == "row_extra":
        row["extra"] = 1
    elif mutation == "row_missing":
        del row["raw_url"]
    elif mutation == "duplicate":
        value["records"][1] = row.copy()
    elif mutation == "source":
        row["record_id"] = row["record_id"].replace("a" * 64, "b" * 64)
    else:
        mutate_envelope(value, mutation)
    with pytest.raises(api.ReplayManifestCodecError):
        decode(encode(value))


def mutate_envelope(value, mutation):
    if mutation == "records_short":
        value["records"].pop()
    elif mutation == "records_extra":
        value["records"].append(value["records"][0])
    else:
        field, changed = {
            "version_bool": ("schema_version", True),
            "algorithm": ("algorithm_id", "other"),
            "extra": ("extra", 1),
            "prevalence": ("prevalence_basis_points", 10),
        }[mutation]
        value[field] = changed
