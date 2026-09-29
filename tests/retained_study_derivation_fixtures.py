"""Temporary invented preparation receipts and absent original input paths."""

import importlib
import json
from dataclasses import replace
from hashlib import sha256
from types import SimpleNamespace

import pytest
from study_preparation_runner_fixtures import (
    inputs,
    preparation_api,
    preparation_case,
    runner,
)

from automated_phishing_detection import retained_study_preparation
from automated_phishing_detection._checkpoint_codec import canonical_bytes
from automated_phishing_detection._external_source_profile import (
    CandidateExternalProfile,
)

__all__ = [
    "inputs",
    "preparation_api",
    "preparation_case",
    "runner",
    "derivation_case",
]


def api():
    return importlib.import_module(
        "automated_phishing_detection.retained_study_derivation"
    )


def _prior(case, preparation_api):
    snapshot = preparation_api._run_bound_preparation(case.binding, case.paths)
    complete = snapshot.payload("preparation-complete.json")
    return retained_study_preparation.restore_study_preparation(
        snapshot,
        expected_identity=json.loads(complete)["execution"],
        expected_reservation_sha256=snapshot.reservation_sha256,
        expected_completion_sha256=sha256(complete).hexdigest(),
        source_spec_bytes=(case.binding.root / "data/sources.json").read_bytes(),
        preparation_summary_bytes=(
            case.binding.root / "reports/phiusiil-preparation-summary.json"
        ).read_bytes(),
    )


def continuation(case, prior):
    profile = {
        "execution": {"revision": case.binding.revision},
        "components": {"external": case.profile.profile_sha256},
        "paths": {
            name: str(path)
            for name, path in zip(
                ("source-csv", "suffix-rules", "archive"),
                (case.paths.source_csv, case.paths.suffix_rules, case.paths.archive),
            )
        },
    }
    return {
        "representation": "publisher_url_norm_v1",
        "prior_profile": profile,
        "prior_profile_sha256": sha256(canonical_bytes(profile)).hexdigest(),
        "prior_envelope_sha256": "a" * 64,
        "prior_root_reservation_sha256": "b" * 64,
        "prior_public_summary_sha256": "c" * 64,
        "prior_preparation_reservation_sha256": prior.reservation_sha256,
        "prior_preparation_complete_sha256": prior.completion_sha256,
        "publisher_source_sha256": sha256(
            prior.payload("publisher-source.json")
        ).hexdigest(),
        "publisher_summary_sha256": sha256(
            prior.payload("publisher-summary.json")
        ).hexdigest(),
        "diagnostic_sha256": "d" * 64,
    }


@pytest.fixture
def derivation_case(preparation_case, preparation_api, monkeypatch):
    case = preparation_case
    prior = _prior(case, preparation_api)
    declared = continuation(case, prior)
    binding = replace(case.binding, revision="f" * 40)
    projection = case.profile.projection()
    projection["execution"]["revision"] = binding.revision
    profile = CandidateExternalProfile(canonical_bytes(projection))
    monkeypatch.setattr(
        preparation_api.body, "resolve_external_source_profile", lambda unused: profile
    )
    paths = replace(case.paths, attempt=case.paths.attempt.parent / "derived")
    for path in (paths.source_csv, paths.suffix_rules, paths.archive):
        path.unlink()
    return SimpleNamespace(
        binding=binding,
        paths=paths,
        prior=prior,
        continuation=declared,
        case=case,
    )


def run(case, **changes):
    return api().run_retained_study_preparation(
        case.binding,
        case.paths,
        **(
            {"prior_preparation": case.prior, "continuation": case.continuation}
            | changes
        ),
    )
