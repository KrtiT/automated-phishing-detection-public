"""Invented on-disk adopted holds with independent public identity expectations."""

import importlib
import json
from dataclasses import replace
from hashlib import sha256
from types import SimpleNamespace

import pytest
from study_execution_fixtures import execution_case
from study_run_record_fixtures import prepared
from test_adopted_study_verification import snapshot

from automated_phishing_detection import _external_source_profile as external
from automated_phishing_detection import _operational_profile as operational
from automated_phishing_detection import execution_preflight as preflight
from automated_phishing_detection._checkpoint_codec import canonical_bytes

__all__ = ["execution_case", "prepared", "prior_hold"]
SOURCE = "data/sources.json"
REPORT = "reports/phiusiil-preparation-summary.json"


def api():
    return importlib.import_module("automated_phishing_detection.retained_study_hold")


def _public_context(case, prepared, monkeypatch):
    source = preflight._read_regular(case.root, SOURCE)
    report = b"invented public preparation report\n"
    pins = dict(case.base.source_hashes) | {REPORT: sha256(report).hexdigest()}
    case.base = replace(case.base, source_hashes=tuple(sorted(pins.items())))
    case.external = external.resolve_external_source_profile(case.base)
    case.operational = operational.resolve_operational_profile(case.base)
    auth, retained = snapshot(case, prepared)
    current = replace(case.base, revision="c" * 40)
    old = json.loads(retained.payload("attempt/reservation.json"))["identity"]
    identity = prepared.preparation.execution | {
        "revision": current.revision,
        "execution_contract_sha256": old["execution_contract_sha256"],
        "runtime_sha256": old["runtime_sha256"],
        "source_spec_sha256": old["source_spec_sha256"],
        "preparation_summary_sha256": pins[REPORT],
        "source_profile_sha256": "9" * 64,
    }
    buffers = {SOURCE: source, REPORT: report}
    monkeypatch.setattr(
        api(),
        "bound_preparation_context",
        lambda binding: (identity, {}, buffers, None),
    )
    return auth, retained, current, identity, buffers


def _continuation(auth, retained, profile, barrier):
    return {
        "representation": "publisher_url_norm_v1",
        "prior_profile": profile,
        "prior_profile_sha256": auth.profile_sha256,
        "prior_envelope_sha256": auth.envelope_sha256,
        "prior_root_reservation_sha256": retained.reservation_sha256,
        "prior_public_summary_sha256": sha256(
            retained.payload("public-summary.json")
        ).hexdigest(),
        "prior_preparation_reservation_sha256": barrier[
            "study_preparation_reservation_sha256"
        ],
        "prior_preparation_complete_sha256": barrier[
            "study_preparation_complete_sha256"
        ],
        "publisher_source_sha256": "7" * 64,
        "publisher_summary_sha256": "8" * 64,
        "diagnostic_sha256": "6" * 64,
    }


@pytest.fixture
def prior_hold(execution_case, prepared, monkeypatch):
    auth, retained, current, identity, buffers = _public_context(
        execution_case, prepared, monkeypatch
    )
    profile = json.loads(auth.profile_bytes)
    barrier = json.loads(retained.payload("attempt/prediction-barrier.json"))
    return SimpleNamespace(
        authorization=auth,
        retained=retained,
        binding=current,
        continuation=_continuation(auth, retained, profile, barrier),
        identity=identity,
        buffers=buffers,
        barrier=barrier,
        profile=profile,
    )


def manager(case, continuation=None):
    return api().hold_prior_study_hold(
        case.binding, case.continuation if continuation is None else continuation
    )


def changed_profile(case):
    profile = json.loads(canonical_bytes(case.profile))
    profile["session"]["session_id"] = "other-session"
    return case.continuation | {
        "prior_profile": profile,
        "prior_profile_sha256": sha256(canonical_bytes(profile)).hexdigest(),
    }
