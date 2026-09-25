"""Private retained snapshots must not be aliases at any path component."""

import os
from dataclasses import replace

import pytest
from test_bound_drift import drift as drift
from test_bound_drift import prepared as prepared


def test_private_hardlink_rejected(drift, prepared, tmp_path):
    os.link(prepared.paths.training_reference, tmp_path / "alias")
    with pytest.raises(drift.BoundDriftError, match="^invalid_bound_drift_evidence$"):
        drift.load_bound_drift(prepared.binding, prepared.paths, prepared.models)
    assert "retained_reference" not in prepared.events


def test_private_symlinked_parent_rejected(drift, prepared, tmp_path):
    parent = tmp_path / "real-directory"
    parent.mkdir()
    target = parent / "reference"
    prepared.paths.training_reference.rename(target)
    alias = tmp_path / "directory-alias"
    alias.symlink_to(parent, target_is_directory=True)
    paths = replace(prepared.paths, training_reference=alias / "reference")
    with pytest.raises(drift.BoundDriftError, match="^invalid_bound_drift_evidence$"):
        drift.load_bound_drift(prepared.binding, paths, prepared.models)
    assert "retained_reference" not in prepared.events
