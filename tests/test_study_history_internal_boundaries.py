"""Historical science reconstructs once without live custody or filesystem IO."""

import builtins
import os
import subprocess
from pathlib import Path

import pytest
from study_history_internal_fixtures import (
    historical_internal,
    inputs,
    published,
    runner,
    verify,
)
from test_stopped_study_authorization_review import _guard_observations
from test_stopped_study_cell_review import _guard_live_types

from automated_phishing_detection import _study_history_internal_science as science
from automated_phishing_detection import (
    saved_evidence,
    source_completion,
    source_runner,
)

__all__ = ["historical_internal", "inputs", "published", "runner"]


def forbidden(*args, **kwargs):
    pytest.fail("historical science attempted IO or live custody verification")


def test_historical_science_has_no_io_or_observed_owner(
    historical_internal, monkeypatch
):
    _guard_observations(monkeypatch)
    _guard_live_types(monkeypatch)
    for module, name in (
        (builtins, "open"),
        (os, "open"),
        (Path, "open"),
        (subprocess, "Popen"),
        (source_runner, "_public_sources"),
        (source_runner, "recheck_binding"),
        (source_completion, "_verify_outputs"),
        (source_completion, "verify_internal_completion_snapshot"),
    ):
        monkeypatch.setattr(module, name, forbidden)
    assert verify(historical_internal).records


def _spy(original, calls, name):
    def wrapped(*args, **kwargs):
        calls.append(name)
        return original(*args, **kwargs)

    return wrapped


def test_unchanged_kernels_each_run_once(historical_internal, monkeypatch):
    calls = []
    for module, name in (
        (science, "verify_source_checkpoints"),
        (science, "verify_scientific_checkpoints"),
        (saved_evidence, "reconstruct_internal_evidence_and_population"),
    ):
        monkeypatch.setattr(module, name, _spy(getattr(module, name), calls, name))
    verify(historical_internal)
    assert calls == [
        "verify_source_checkpoints",
        "verify_scientific_checkpoints",
        "reconstruct_internal_evidence_and_population",
    ]
