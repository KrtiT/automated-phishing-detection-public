"""Historical hold readers accept invented receipts, never new source authority."""

import importlib
import importlib.util
from dataclasses import replace
from pathlib import Path

import pytest
from retained_study_hold_fixtures import (
    REPORT,
    SOURCE,
    changed_profile,
    execution_case,
    manager,
    prepared,
    prior_hold,
)

__all__ = ["execution_case", "prepared", "prior_hold"]


def test_retained_hold_reader_has_no_raw_acquisition_entrypoint():
    name = "automated_phishing_detection.retained_study_hold"
    assert importlib.util.find_spec(name), (
        "missing authenticated historical hold reader"
    )
    module = importlib.import_module(name)
    assert callable(module.hold_prior_study_hold)
    assert not hasattr(module, "run_study_preparation")


def test_prior_hold_authenticates_exact_metadata_and_historical_identity(prior_hold):
    with manager(prior_hold) as held:
        assert held.profile == prior_hold.profile
        assert held.barrier == prior_hold.barrier
        assert held.expected_identity == prior_hold.identity | {
            "revision": prior_hold.profile["execution"]["revision"],
            "source_profile_sha256": prior_hold.profile["components"]["external"],
        }
        assert held.source_spec_bytes == prior_hold.buffers[SOURCE]
        assert held.preparation_summary_bytes == prior_hold.buffers[REPORT]


@pytest.mark.parametrize(
    "pin",
    (
        "prior_profile_sha256",
        "prior_envelope_sha256",
        "prior_root_reservation_sha256",
        "prior_public_summary_sha256",
        "prior_preparation_reservation_sha256",
        "prior_preparation_complete_sha256",
    ),
)
def test_wrong_independent_historical_pin_rejects(prior_hold, pin):
    with (
        pytest.raises(ValueError),
        manager(prior_hold, prior_hold.continuation | {pin: "0" * 64}),
    ):
        pytest.fail("wrong historical pin was accepted")


def test_rehashed_different_embedded_old_profile_rejects(prior_hold):
    with pytest.raises(ValueError), manager(prior_hold, changed_profile(prior_hold)):
        pytest.fail("different historical profile was accepted")


def test_changed_current_public_scope_rejects(prior_hold):
    pins = dict(prior_hold.binding.source_hashes) | {REPORT: "0" * 64}
    prior_hold.binding = replace(
        prior_hold.binding, source_hashes=tuple(sorted(pins.items()))
    )
    with pytest.raises(ValueError), manager(prior_hold):
        pytest.fail("changed current public scope was accepted")


@pytest.mark.parametrize("target", ("attempt", "evidence", "public"))
def test_historical_file_mutation_during_body_rejects_on_exit(prior_hold, target):
    root = prior_hold.authorization.paths.attempt
    path = {
        "attempt": root / "study-intent.json",
        "evidence": root / "evidence/study-intent.json",
        "public": prior_hold.authorization.paths.public_summary,
    }[target]
    with pytest.raises(ValueError):
        with manager(prior_hold):
            path.write_bytes(path.read_bytes() + b" ")


@pytest.mark.parametrize(
    "mutation", ("extra", "missing", "hardlink", "symlink", "mode")
)
def test_unsafe_or_partial_historical_root_rejects(prior_hold, mutation):
    root = prior_hold.authorization.paths.attempt
    path = root / "study-intent.json"
    if mutation == "extra":
        (root / "unexpected.json").write_bytes(b"{}")
    elif mutation == "missing":
        path.unlink()
    elif mutation == "hardlink":
        (root.parent / "linked.json").hardlink_to(path)
    elif mutation == "symlink":
        content = path.read_bytes()
        path.unlink()
        alternative = root.parent / "alternative.json"
        alternative.write_bytes(content)
        path.symlink_to(alternative)
    else:
        path.chmod(0o644)
    with pytest.raises(ValueError), manager(prior_hold):
        pytest.fail("unsafe or partial root was accepted")


def test_missing_old_raw_paths_are_never_opened(prior_hold):
    for name in ("source-csv", "suffix-rules", "archive"):
        path = Path(prior_hold.profile["paths"][name])
        assert not path.exists()
    with manager(prior_hold) as held:
        assert held.barrier["status"] == "whole_study_hold"
