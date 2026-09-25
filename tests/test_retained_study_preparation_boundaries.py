"""Restoration and fresh projections perform retained-byte consistency only."""

import asyncio
import builtins
import importlib
import os
import random
import socket
from pathlib import Path

import pytest
from retained_study_preparation_fixtures import api, restore, retained_case

__all__ = ["api", "retained_case"]

_FORBIDDEN = {
    "phiusiil": (
        "_parse_csv_rows",
        "resolve_rows",
        "assign_splits",
        "prepare_phiusiil",
    ),
    "source_overlap": ("reconstruct_source_overlap",),
    "phishvn_source": ("decode_phishvn_archive", "_csv_table", "_tables"),
    "phishvn": ("prepare_external_rows",),
    "evaluation_manifest": ("build_manifest",),
    "evaluation_producer": ("produce_internal_evidence", "score_primary_url"),
    "external_producer": ("produce_external_evidence",),
    "external_primary": ("score_external_primary",),
    "transformer_inference": ("load_transformer_cascade_bundle", "_load_model"),
    "length_inference": (
        "load_length_only_artifact",
        "score_length_only_authoritative",
    ),
    "secondary_tabular": ("load_secondary_model_bytes", "_fit", "_score_state"),
    "secondary_transformer": (
        "load_secondary_transformer_bytes",
        "fit_secondary_transformer",
        "score_secondary_transformer_urls",
    ),
}


def forbidden(*args, **kwargs):
    pytest.fail(
        "retained restoration attempted original I/O, preparation, RNG or science"
    )


def test_restore_and_views_do_not_read_prepare_sample_or_score(
    api, retained_case, monkeypatch
):
    owners = {
        name: importlib.import_module(f"automated_phishing_detection.{name}")
        for name in _FORBIDDEN
    }
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
        restored = restore(api, retained_case)
        assert restored.publisher == retained_case.publisher
        assert restored.external == retained_case.external
        assert restored.reconstructed_internal == retained_case.reconstructed
        assert restored.internal == retained_case.internal
        assert restored.execution == retained_case.identity
        assert restored.feasibility["shortages"]


def target(api, stage):
    return {
        "authenticate": (api.records, "authenticate"),
        "public_buffers": (api.records, "public_inputs"),
        "partition": (api.evaluation_producer, "parse_internal_partition"),
        "structure": (api, "verify_internal_preparation_structure"),
        "publisher": (api, "restore_phishvn_source"),
        "external": (api.external, "validate_external"),
        "feasibility": (api, "assess_preparation_feasibility"),
    }[stage]


@pytest.mark.parametrize(
    "stage",
    [
        "authenticate",
        "public_buffers",
        "partition",
        "structure",
        "publisher",
        "external",
        "feasibility",
    ],
)
@pytest.mark.parametrize(
    "kind", [KeyboardInterrupt, SystemExit, asyncio.CancelledError]
)
def test_exact_interruptions_survive(api, retained_case, monkeypatch, stage, kind):
    original = kind("private-secret-canary")

    def interrupt(*args, **kwargs):
        raise original

    owner, name = target(api, stage)
    monkeypatch.setattr(owner, name, interrupt)
    with pytest.raises(kind) as caught:
        restore(api, retained_case)
    assert caught.value is original


@pytest.mark.parametrize(
    "stage",
    [
        "authenticate",
        "public_buffers",
        "partition",
        "structure",
        "publisher",
        "external",
        "feasibility",
    ],
)
def test_ordinary_errors_are_symbolic(api, retained_case, monkeypatch, stage):
    def fail(*args, **kwargs):
        raise OSError("private-secret-canary")

    owner, name = target(api, stage)
    monkeypatch.setattr(owner, name, fail)
    with pytest.raises(api.StudyPreparationRestoreError) as caught:
        restore(api, retained_case)
    assert str(caught.value) == "invalid_retained_study_preparation"
    assert caught.value.__suppress_context__ is True


def test_completion_is_hashed_before_its_parser(api, retained_case, monkeypatch):
    original, observed = api.records.load, []

    def observe(content):
        observed.append(content)
        return original(content)

    monkeypatch.setattr(api.records, "load", observe)
    with pytest.raises(api.StudyPreparationRestoreError):
        restore(api, retained_case, expected_completion_sha256="0" * 64)
    assert retained_case.snapshot.payload("preparation-complete.json") not in observed


@pytest.mark.parametrize("field", ["source_spec_bytes", "preparation_summary_bytes"])
def test_public_hashes_are_checked_before_any_public_parser(
    api, retained_case, monkeypatch, field
):
    observed = []

    def observe(*args):
        observed.append(args)
        raise AssertionError("public parser reached")

    monkeypatch.setattr(api.records.phiusiil, "_load_source_spec", observe)
    monkeypatch.setattr(api.records.baselines, "_validate_preparation_summary", observe)
    with pytest.raises(api.StudyPreparationRestoreError):
        restore(api, retained_case, **{field: b"malformed-private-canary"})
    assert observed == []
