"""Corrupt inputs are processing failures, never empty populations."""

import builtins
import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from study_feasibility_fixtures import api, external_rows, internal_rows

from automated_phishing_detection import evaluation_manifest


def rejected(internal, external):
    with pytest.raises(
        api().StudyFeasibilityError, match="^invalid_preparation_feasibility$"
    ):
        api().assess_preparation_feasibility(internal, external)


@pytest.mark.parametrize("value", [None, [], {}, "", 0, True])
def test_internal_requires_exact_materialized_tuple(value):
    rejected(value, external_rows())


@pytest.mark.parametrize(
    "field,value",
    [
        ("is_phishing", True),
        ("is_phishing", 1.0),
        ("is_phishing", None),
        ("split", "validation"),
        ("record_id", "private identifier"),
        ("canonical_url_sha256", "b" * 64),
        ("registrable_domain", "other.test"),
        ("raw_url", "private invalid URL"),
    ],
)
def test_internal_record_validation_precedes_capacity(field, value):
    rows = internal_rows()
    rejected((replace(rows[0], **{field: value}), *rows[1:]), external_rows())


def test_duplicate_and_mixed_source_records_do_not_inflate_capacity():
    rows = internal_rows()
    rejected((rows[0], *rows), external_rows())
    changed = replace(rows[0], record_id=rows[0].record_id.replace("a" * 64, "b" * 64))
    rejected((changed, *rows[1:]), external_rows())


@pytest.mark.parametrize(
    "field,value",
    [
        ("retained_test_rows", 0),
        ("input_test_rows", True),
        ("quarantined_test_rows", 2),
        ("role_counts", {"gold": 0}),
        ("retained_test_domain_count", 0),
    ],
)
def test_declared_external_count_mismatch_is_not_population_absence(field, value):
    external = external_rows()
    rejected(
        internal_rows(),
        replace(external, public_summary=external.public_summary | {field: value}),
    )


def test_removing_retained_rows_or_bytes_does_not_establish_empty_population():
    external = external_rows()
    rejected((), replace(external, retained=()))
    rejected((), replace(external, private_outputs={}))
    rejected((), None)


@pytest.mark.parametrize(
    "field,value",
    [
        ("is_phishing", None),
        ("is_phishing", True),
        ("role", "certified"),
        ("source_group", "unverified"),
        ("registrable_domain", "wrong.test"),
    ],
)
def test_external_rich_rows_cannot_be_relabeled_for_capacity(field, value):
    external = external_rows()
    rows = (replace(external.retained[0], **{field: value}), *external.retained[1:])
    rejected((), replace(external, retained=rows))


def test_projection_has_no_source_io_rng_sampling_or_scoring(monkeypatch):
    module = api()
    internal, external = internal_rows(), external_rows()
    expected = module.assess_preparation_feasibility(internal, external)

    def forbidden(*args, **kwargs):
        pytest.fail("feasibility projection crossed its preparation-only boundary")

    for target, name in (
        (builtins, "open"),
        (Path, "open"),
        (os, "open"),
        (np.random, "Generator"),
        (np.random, "default_rng"),
        (evaluation_manifest, "_permutation"),
        (evaluation_manifest, "build_manifest"),
    ):
        monkeypatch.setattr(target, name, forbidden)
    assert module.assess_preparation_feasibility(internal, external) == expected


@pytest.mark.parametrize("error_type", [KeyboardInterrupt, SystemExit])
def test_nonexception_validation_interruptions_propagate_unchanged(
    monkeypatch, error_type
):
    module = api()
    interruption = error_type("invented")

    def interrupt(*args):
        raise interruption

    monkeypatch.setattr(module, "_validated_candidates", interrupt)
    with pytest.raises(error_type) as caught:
        module.assess_preparation_feasibility((), external_rows(()))
    assert caught.value is interruption
