"""Invented source-only inputs; no research artifacts are read or scored."""

import importlib
import importlib.util
from hashlib import sha256
from types import SimpleNamespace

import pytest
from phishvn_source_fixtures import bundle, members, record
from test_source_runner import inputs, runner

from automated_phishing_detection._checkpoint_codec import canonical_bytes
from automated_phishing_detection._external_source_profile import (
    CandidateExternalProfile,
    _execution,
)

__all__ = ["inputs", "runner", "preparation_api", "preparation_case"]


@pytest.fixture
def preparation_api():
    name = "automated_phishing_detection.study_preparation_runner"
    assert importlib.util.find_spec(name), "missing closed study preparation runner"
    return importlib.import_module(name)


def profile(binding, suffix, archive):
    return CandidateExternalProfile(
        canonical_bytes(
            {
                "schema_version": 1,
                "profile_id": "external-source-candidate-v1",
                "status": "specified_closed_candidate",
                "protected_evaluation_ready": False,
                "protected_evaluation_authorized": False,
                "execution": _execution(binding, dict(binding.source_hashes)),
                "publisher": {"expected_format": archive.pins},
                "public_suffix_list": {"sha256": sha256(suffix).hexdigest()},
            }
        )
    )


@pytest.fixture
def preparation_case(preparation_api, inputs, tmp_path, monkeypatch):
    binding, old_paths, session, events = inputs
    archive = bundle(members([record("external", url="https://external.com/a")]))
    archive_path = tmp_path / "invented.zip"
    archive_path.write_bytes(archive.content)
    candidate = profile(binding, old_paths.suffix_rules.read_bytes(), archive)
    paths = preparation_api.StudyPreparationPaths(
        old_paths.source_csv, old_paths.suffix_rules, archive_path, old_paths.attempt
    )
    monkeypatch.setattr(
        preparation_api.body, "resolve_external_source_profile", lambda _: candidate
    )
    monkeypatch.setattr(
        preparation_api, "recheck_binding", lambda _: events.append("binding_checked")
    )
    return SimpleNamespace(
        binding=binding, paths=paths, session=session, events=events, profile=candidate
    )
