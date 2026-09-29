"""Invented branch-only preparation/child observations, never scientific proof."""

import json
from contextlib import contextmanager
from dataclasses import replace
from hashlib import sha256
from importlib import import_module
from importlib.util import find_spec
from pathlib import Path
from types import SimpleNamespace

import adopted_study_fixtures as adopted

from automated_phishing_detection._checkpoint_codec import canonical_bytes
from automated_phishing_detection._study_preparation_records import (
    PreparedStudySnapshot,
)
from automated_phishing_detection._study_urlnorm_scope import PIN_FIELDS


def api():
    name = "automated_phishing_detection._study_urlnorm_preparation"
    assert find_spec(name), "missing retained-only preparation integration"
    return import_module(name)


def continuation(case, prior):
    profile = {
        "paths": {"preparation-attempt": str(case.paths.attempt.parent / "prior")}
    }
    result = {name: sha256(name.encode()).hexdigest() for name in PIN_FIELDS}
    return result | {
        "representation": "publisher_url_norm_v1",
        "prior_profile": profile,
        "prior_profile_sha256": sha256(canonical_bytes(profile)).hexdigest(),
        "prior_preparation_reservation_sha256": prior.reservation_sha256,
        "prior_preparation_complete_sha256": prior.completion_sha256,
        "publisher_source_sha256": sha256(
            prior.payload("publisher-source.json")
        ).hexdigest(),
        "publisher_summary_sha256": sha256(
            prior.payload("publisher-summary.json")
        ).hexdigest(),
    }


def repin(case, complete):
    payloads = dict(case.retained.payloads)
    payloads["preparation-complete.json"] = canonical_bytes(complete)
    case.retained = replace(
        case.retained,
        payloads=tuple(payloads.items()),
        completion_sha256=sha256(payloads["preparation-complete.json"]).hexdigest(),
    )
    case.fresh = PreparedStudySnapshot(
        case.retained.reservation_sha256, case.retained.payloads
    )


def setup(tmp_path, prepared, monkeypatch, preparation=None):
    module = api()
    case = adopted.setup(tmp_path, prepared, monkeypatch, preparation)
    case.wrapper, case.prior = module, prepared.preparation
    case.continuation = continuation(case, case.prior)
    profile = {
        "profile_id": "study-urlnorm-profile-v1",
        "continuation": case.continuation,
    }
    case.authorization.profile_bytes = canonical_bytes(profile)
    case.authorization.profile_sha256 = sha256(
        case.authorization.profile_bytes
    ).hexdigest()
    complete = json.loads(case.retained.payload("preparation-complete.json"))
    complete.update(
        schema_version=2,
        protocol="study-preparation-derived-v1",
        derivation={
            name: value
            for name, value in case.continuation.items()
            if name != "prior_profile"
        },
    )
    repin(case, complete)
    case.old_barrier = barrier(case.prior)
    install(case, monkeypatch)
    return case


def barrier(prior):
    return {
        "feasibility": prior.feasibility,
        "feasibility_sha256": sha256(prior.payload("feasibility.json")).hexdigest(),
    }


def old_root(case):
    @contextmanager
    def hold(binding, declared):
        assert binding is case.binding and declared == case.continuation
        case.events.append("hold_prior_root")
        try:
            yield SimpleNamespace(
                profile=declared["prior_profile"],
                barrier=case.old_barrier,
                expected_identity=case.prior.execution,
                source_spec_bytes=b"invented source",
                preparation_summary_bytes=b"invented summary",
            )
        finally:
            case.events.append("release_prior_root")

    return hold


def old_preparation(case):
    @contextmanager
    def hold(path, **expected):
        assert path == Path(
            case.continuation["prior_profile"]["paths"]["preparation-attempt"]
        )
        assert expected == {
            "expected_identity": case.prior.execution,
            "expected_reservation_sha256": case.continuation[
                "prior_preparation_reservation_sha256"
            ],
            "expected_completion_sha256": case.continuation[
                "prior_preparation_complete_sha256"
            ],
            "source_spec_bytes": b"invented source",
            "preparation_summary_bytes": b"invented summary",
        }
        case.events.append("hold_prior_preparation")
        try:
            yield case.prior
        finally:
            case.events.append("release_prior_preparation")

    return hold


def derive(case):
    def run(binding, paths, *, prior_preparation, continuation):
        assert binding is case.binding and paths is case.paths.preparation
        assert prior_preparation is case.prior and continuation == case.continuation
        assert "hold_prior_preparation" in case.events
        assert "release_prior_preparation" not in case.events
        case.events.append("derive")
        return case.fresh

    return run


def forbidden(*arguments, **keywords):
    raise AssertionError("amended root called original-source preparation")


def install(case, monkeypatch):
    monkeypatch.setattr(case.wrapper, "hold_prior_study_hold", old_root(case))
    monkeypatch.setattr(case.wrapper, "hold_study_preparation", old_preparation(case))
    monkeypatch.setattr(case.wrapper, "run_retained_study_preparation", derive(case))
    monkeypatch.setattr(case.wrapper, "held_preparation", case.body.held_preparation)
    monkeypatch.setattr(case.body, "_run_bound_preparation", forbidden)


def failure(case, caught):
    result = case.module.adopted_study_failure(caught)
    assert result is not None
    assert all(slot.status == "unattempted" for slot in result.scientific.cells)
    assert json.loads(result.authorization_ledger)["admissions"] == []
    assert not case.paths.public_summary.exists()
    assert not (case.paths.attempt / "prediction-barrier.json").exists()
    return result


def substitute(case, monkeypatch, field, value):
    if field in ("schema_version", "protocol"):
        complete = json.loads(case.retained.payload("preparation-complete.json"))
        repin(case, complete | {field: value})
        return
    changed = replace(case.retained, **{field: value})
    original = case.wrapper.held_preparation

    @contextmanager
    def hold(*arguments):
        with original(*arguments):
            yield changed

    monkeypatch.setattr(case.wrapper, "held_preparation", hold)
