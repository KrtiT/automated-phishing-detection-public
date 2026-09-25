import asyncio
import builtins
import importlib
import os
import random
import socket
from hashlib import sha256
from pathlib import Path

import pytest
from external_replay_codec_fixtures import decode, encode, module, sample

__all__ = ["sample"]

_FORBIDDEN = {
    "phiusiil": ("_parse_csv_rows", "resolve_rows", "assign_splits"),
    "phishvn_source": ("decode_phishvn_archive", "_csv_table"),
    "phishvn": ("prepare_external_rows",),
    "evaluation_manifest": ("build_manifest", "_permutation"),
    "evaluation_producer": ("produce_internal_evidence",),
    "external_producer": ("produce_external_evidence",),
    "external_primary": ("score_external_primary",),
    "transformer_inference": ("load_transformer_cascade_bundle", "_load_model"),
    "_saved_external_inputs": ("restore_preparation",),
    "_external_input_records": ("_retained_positions", "validate_positions"),
}


def test_restore_uses_only_retained_rows_not_other_pipeline_stages(sample, monkeypatch):
    api, content = module(), encode(sample.retained)
    owners = {
        name: importlib.import_module(f"automated_phishing_detection.{name}")
        for name in _FORBIDDEN
    }

    def forbidden(*args, **kwargs):
        pytest.fail("codec invoked I/O, whole preparation, sampling, or science")

    with monkeypatch.context() as guard:
        for owner, names in _FORBIDDEN.items():
            for name in names:
                guard.setattr(owners[owner], name, forbidden)
        for owner, name in (
            (builtins, "open"),
            (os, "open"),
            (Path, "open"),
            (socket, "socket"),
            (random, "Random"),
        ):
            guard.setattr(owner, name, forbidden)
        restored = api.decode_external_replay_manifest(
            content, expected_sha256=sha256(content).hexdigest()
        )
    assert restored == sample.retained


def stage_target(api, stage):
    if stage in ("_rows", "_fields"):
        return api.saved, stage
    return api.records, stage


@pytest.mark.parametrize(
    "stage", ["_rows", "_fields", "_retained_records", "_mapping", "_url_identity"]
)
@pytest.mark.parametrize(
    "kind", [KeyboardInterrupt, SystemExit, asyncio.CancelledError]
)
def test_exact_interruptions_survive_each_stage(sample, monkeypatch, stage, kind):
    api, content = module(), encode(sample.retained)
    original = kind("private-secret-canary")

    def interrupted(*args, **kwargs):
        raise original

    owner, name = stage_target(api, stage)
    monkeypatch.setattr(owner, name, interrupted)
    with pytest.raises(kind) as caught:
        decode(content)
    assert caught.value is original


@pytest.mark.parametrize(
    "stage", ["_rows", "_fields", "_retained_records", "_mapping", "_url_identity"]
)
def test_ordinary_stage_failures_are_symbolic(sample, monkeypatch, stage):
    api, content = module(), encode(sample.retained)

    def failed(*args, **kwargs):
        raise OSError("private-secret-canary")

    owner, name = stage_target(api, stage)
    monkeypatch.setattr(owner, name, failed)
    with pytest.raises(api.ExternalReplayCodecError) as caught:
        decode(content)
    assert str(caught.value) == "invalid_retained_external_replay_manifest"
    assert caught.value.__suppress_context__ is True


@pytest.mark.parametrize("digest", ["f" * 64, "F" * 64, True, None])
def test_expected_digest_is_checked_before_any_parser(sample, monkeypatch, digest):
    api, observed = module(), []

    def parser(*args):
        observed.append(args)
        raise AssertionError("unbound bytes parsed")

    monkeypatch.setattr(api.saved, "_rows", parser)
    with pytest.raises(api.ExternalReplayCodecError):
        api.decode_external_replay_manifest(
            b"private-malformed", expected_sha256=digest
        )
    assert observed == []


@pytest.mark.parametrize("kind", [bytearray, memoryview, str])
def test_mutable_or_nonbytes_content_is_rejected_before_parsing(sample, kind):
    api, content = module(), encode(sample.retained)
    candidate = content.decode() if kind is str else kind(content)
    with pytest.raises(api.ExternalReplayCodecError):
        api.decode_external_replay_manifest(
            candidate, expected_sha256=sha256(content).hexdigest()
        )


@pytest.mark.parametrize("kind", [str, bytes])
def test_exact_scalar_types_reject_subclasses(sample, kind):
    api, content = module(), encode(sample.retained)
    subtype = type("Subclass", (kind,), {})
    digest = sha256(content).hexdigest()
    if kind is bytes:
        content = subtype(content)
    else:
        digest = subtype(digest)
    with pytest.raises(api.ExternalReplayCodecError):
        api.decode_external_replay_manifest(content, expected_sha256=digest)


def test_original_preparation_retains_declared_test_count_upper_bound(sample):
    api = module()
    assert decode(encode(sample.retained)) == sample.retained
    with pytest.raises(api.records.ExternalInputError, match="invalid_retained_order"):
        api.records._retained_positions(sample.retained, {"test": 1000})


def test_same_bytes_remain_bound_to_parent_digest(sample):
    api, content = module(), encode(sample.retained)
    with pytest.raises(api.ExternalReplayCodecError):
        decode(content.replace(b'"row-0"', b'"forged"', 1), sha256(content).hexdigest())
