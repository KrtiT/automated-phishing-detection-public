"""No I/O, scoring, sampling, mutable-buffer coercion or interruption replacement."""

import asyncio
import builtins
import os
from pathlib import Path

import pytest
from phishvn_source_fixtures import bundle
from test_study_preparation_inputs import api as api
from test_study_preparation_inputs import internal_case as internal_case
from test_study_preparation_inputs import prepare_internal
from test_study_preparation_inputs_external import SUFFIX, decode_external

from automated_phishing_detection import (
    evaluation_manifest,
    evaluation_producer,
    external_producer,
    phishvn,
    phishvn_source,
    source_overlap,
)


def test_helpers_do_not_open_paths_score_or_sample(api, internal_case, monkeypatch):
    case = bundle()

    def forbidden(*args, **kwargs):
        pytest.fail("preparation attempted I/O, scoring or sampling")

    with monkeypatch.context() as guard:
        guard.setattr(builtins, "open", forbidden)
        guard.setattr(os, "open", forbidden)
        guard.setattr(Path, "open", forbidden)
        guard.setattr(evaluation_producer, "produce_internal_evidence", forbidden)
        guard.setattr(external_producer, "produce_external_evidence", forbidden)
        guard.setattr(evaluation_manifest, "build_manifest", forbidden)
        reconstructed, prepared = prepare_internal(api, internal_case)
        decoded = decode_external(api, case)
        external = api.prepare_external_inputs(
            decoded, SUFFIX, overlap_domains=reconstructed.overlap_domains
        )
    assert prepared.records
    assert len(external.retained) == 1


def invoke(api, case, decoded, stage):
    if stage in ("reconstruct", "partition"):
        return prepare_internal(api, case)
    if stage == "decode":
        return decode_external(api, bundle())
    return api.prepare_external_inputs(decoded, SUFFIX, overlap_domains=frozenset())


def patch_stage(api, monkeypatch, stage, replacement):
    module, attribute = {
        "reconstruct": (source_overlap, "reconstruct_source_overlap"),
        "partition": (evaluation_producer, "parse_internal_partition"),
        "decode": (phishvn_source, "decode_phishvn_archive"),
        "restore": (api, "decoder_inputs"),
        "prepare": (phishvn, "prepare_external_rows"),
        "validate": (api, "validate_prepared_external"),
    }[stage]
    monkeypatch.setattr(module, attribute, replacement)


@pytest.mark.parametrize(
    "stage", ["reconstruct", "partition", "decode", "restore", "prepare", "validate"]
)
@pytest.mark.parametrize(
    "kind", [KeyboardInterrupt, SystemExit, asyncio.CancelledError]
)
def test_exact_interruption_survives_every_stage(
    api, internal_case, monkeypatch, stage, kind
):
    decoded = decode_external(api, bundle())
    original = kind("private-secret-canary")

    def interrupt(*args, **kwargs):
        raise original

    patch_stage(api, monkeypatch, stage, interrupt)
    with pytest.raises(kind) as caught:
        invoke(api, internal_case, decoded, stage)
    assert caught.value is original


@pytest.mark.parametrize(
    "stage", ["reconstruct", "partition", "decode", "restore", "prepare", "validate"]
)
def test_ordinary_failures_have_only_fixed_symbols(
    api, internal_case, monkeypatch, stage
):
    decoded = decode_external(api, bundle())

    def fail(*args, **kwargs):
        raise OSError("private-secret-canary")

    patch_stage(api, monkeypatch, stage, fail)
    with pytest.raises(api.StudyPreparationInputError) as caught:
        invoke(api, internal_case, decoded, stage)
    assert str(caught.value) in {
        "invalid_internal_inputs",
        "invalid_external_source",
        "invalid_external_inputs",
    }
    assert caught.value.__suppress_context__ is True


@pytest.mark.parametrize(
    "field",
    [
        "csv_bytes",
        "suffix_rules_bytes",
        "source_spec_bytes",
        "preparation_summary_bytes",
    ],
)
def test_internal_rejects_mutable_buffers(api, internal_case, field):
    changed = bytearray(internal_case.buffers[field])
    with pytest.raises(api.StudyPreparationInputError):
        prepare_internal(api, internal_case, **{field: changed})


def test_external_rejects_mutable_archive_and_psl(api):
    case = bundle()
    original = case.content
    case.content = bytearray(original)
    with pytest.raises(api.StudyPreparationInputError):
        decode_external(api, case)
    case.content = original
    decoded = decode_external(api, case)
    with pytest.raises(api.StudyPreparationInputError):
        api.prepare_external_inputs(
            decoded, bytearray(SUFFIX), overlap_domains=frozenset()
        )
