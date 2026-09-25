"""Whole-study composition rejects inconsistent paths before private access."""

import builtins
from dataclasses import replace
from pathlib import Path
from unittest.mock import ANY

import pytest
import study_run_context_fixtures as fixtures
from operational_profile_fixtures import profile_case, resolve

__all__ = ["profile_case"]


def context(profile_case):
    module = fixtures.api()
    return module, profile_case.binding, resolve(profile_case), fixtures.paths(module)


def test_study_context_joins_fresh_preparation_and_common_model_paths(profile_case):
    module, binding, profile, paths = context(profile_case)
    assert (
        module.validate_context(binding, profile, paths, fixtures.deadlines()) is None
    )
    assert (
        paths.internal.preparation
        == paths.external.preparation
        == paths.preparation.attempt
    )


@pytest.mark.parametrize("member", ["preparation", "internal", "external"])
def test_study_context_rejects_wrong_typed_path_groups(profile_case, member):
    module, binding, profile, paths = context(profile_case)
    with pytest.raises(module.StudyRunError):
        module.validate_context(
            binding, profile, replace(paths, **{member: object()}), fixtures.deadlines()
        )


@pytest.mark.parametrize("member", ["internal", "external"])
def test_study_context_rejects_historical_or_different_preparation(
    profile_case, member
):
    module, binding, profile, paths = context(profile_case)
    changed = replace(
        getattr(paths, member), preparation=Path("/invented/old-preparation")
    )
    with pytest.raises(module.StudyRunError):
        module.validate_context(
            binding, profile, replace(paths, **{member: changed}), fixtures.deadlines()
        )


@pytest.mark.parametrize("member", ["internal", "external"])
def test_worker_preparation_link_requires_an_actual_path(profile_case, member):
    module, binding, profile, paths = context(profile_case)
    changed = replace(getattr(paths, member), preparation=ANY)
    with pytest.raises(module.StudyRunError):
        module.validate_context(
            binding, profile, replace(paths, **{member: changed}), fixtures.deadlines()
        )


@pytest.mark.parametrize("member", ["artifacts", "secondary_artifacts"])
def test_study_context_rejects_different_worker_artifacts(profile_case, member):
    module, binding, profile, paths = context(profile_case)
    original = getattr(paths.external, member)
    attribute = "gmm" if member == "artifacts" else "formatting"
    changed = replace(original, **{attribute: Path("/invented/other")})
    paths = replace(paths, external=replace(paths.external, **{member: changed}))
    with pytest.raises(module.StudyRunError):
        module.validate_context(binding, profile, paths, fixtures.deadlines())


@pytest.mark.parametrize("value", [None, {}, {"startup": 1}, [], True])
def test_study_context_requires_four_explicit_deadlines(profile_case, value):
    module, binding, profile, paths = context(profile_case)
    with pytest.raises(module.StudyRunError):
        module.validate_context(binding, profile, paths, value)


@pytest.mark.parametrize("value", [False, 0, -1, float("inf"), float("nan"), "1"])
def test_study_context_rejects_invalid_deadline_value(profile_case, value):
    module, binding, profile, paths = context(profile_case)
    with pytest.raises(module.StudyRunError):
        module.validate_context(
            binding, profile, paths, fixtures.deadlines() | {"kill": value}
        )


@pytest.mark.parametrize(
    "value",
    [
        Path("relative"),
        Path("/invented/../other"),
        Path("/bad\0path"),
        "/invented/string",
    ],
)
def test_study_context_rejects_nonabsolute_or_aliased_output(profile_case, value):
    module, binding, profile, paths = context(profile_case)
    with pytest.raises(module.StudyRunError):
        module.validate_context(
            binding, profile, replace(paths, attempt=value), fixtures.deadlines()
        )


@pytest.mark.parametrize(
    "location",
    [
        "checkout",
        "checkout-parent",
        "preparation",
        "source",
        "model",
        "cells",
        "public",
    ],
)
def test_study_context_rejects_output_collision_or_overlap(profile_case, location):
    module, binding, profile, paths = context(profile_case)
    collision = {
        "checkout": binding.root / "attempt",
        "checkout-parent": binding.root.parent,
        "preparation": paths.preparation.attempt / "root",
        "source": paths.preparation.source_csv,
        "model": paths.internal.artifacts.transformer_bundle / "root",
        "cells": paths.cells_directory,
        "public": paths.public_summary,
    }[location]
    with pytest.raises(module.StudyRunError):
        module.validate_context(
            binding, profile, replace(paths, attempt=collision), fixtures.deadlines()
        )


def test_study_context_is_pure_and_does_not_authorize(profile_case, monkeypatch):
    module, binding, profile, paths = context(profile_case)

    def forbidden(*arguments, **keywords):
        pytest.fail("pure context inspected a filesystem path")

    for owner, name in ((builtins, "open"), (Path, "stat"), (Path, "resolve")):
        monkeypatch.setattr(owner, name, forbidden)
    module.validate_context(binding, profile, paths, fixtures.deadlines())
    assert profile.protected_evaluation_ready is False


@pytest.mark.parametrize("parent", [False, True])
def test_study_rejects_models_that_the_cell_boundary_cannot_accept(
    profile_case, parent
):
    module, binding, profile, paths = context(profile_case)
    location = binding.root.parent if parent else binding.root / "gmm.json"
    artifacts = replace(paths.internal.artifacts, gmm=location)
    paths = replace(
        paths,
        internal=replace(paths.internal, artifacts=artifacts),
        external=replace(paths.external, artifacts=artifacts),
    )
    with pytest.raises(module.StudyRunError):
        module.validate_context(binding, profile, paths, fixtures.deadlines())


@pytest.mark.parametrize(
    "changed", ["binding", "profile", "mutable-profile", "forged-profile", "paths"]
)
def test_study_context_rejects_wrong_or_forged_context(profile_case, changed):
    module, binding, profile, paths = context(profile_case)
    if changed == "binding":
        binding = object()
    elif changed == "profile":
        profile = object()
    elif changed == "mutable-profile":
        profile = replace(profile, canonical_bytes=bytearray(profile.canonical_bytes))
    elif changed == "forged-profile":
        profile = replace(profile, canonical_bytes=b"{}\n")
    else:
        paths = object()
    with pytest.raises(module.StudyRunError):
        module.validate_context(binding, profile, paths, fixtures.deadlines())
