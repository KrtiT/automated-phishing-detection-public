"""Retained continuation rejects lineage substitution before fresh reservation."""

import json
from dataclasses import replace
from hashlib import sha256

import pytest
from retained_study_derivation_fixtures import (
    derivation_case,
    inputs,
    preparation_api,
    preparation_case,
    run,
    runner,
)

from automated_phishing_detection import study_preparation_inputs
from automated_phishing_detection._checkpoint_codec import canonical_bytes

__all__ = ["derivation_case", "inputs", "preparation_api", "preparation_case", "runner"]


@pytest.mark.parametrize(
    "name",
    ("suffix-rules.dat", "group_test.jsonl", "source-overlap.json", "feasibility.json"),
)
def test_mutated_retained_payload_rejects_before_reservation(derivation_case, name):
    prior = replace(
        derivation_case.prior,
        payloads=tuple(
            (key, content + b" " if key == name else content)
            for key, content in derivation_case.prior.payloads
        ),
    )
    with pytest.raises(ValueError):
        run(derivation_case, prior_preparation=prior)
    assert not derivation_case.paths.attempt.exists()


@pytest.mark.parametrize(
    "field,value",
    (
        ("representation", "raw_url"),
        ("extra", True),
        ("diagnostic_sha256", "not-a-digest"),
        ("prior_profile_sha256", "0" * 64),
    ),
)
def test_unbound_continuation_rejects_before_reservation(derivation_case, field, value):
    with pytest.raises(ValueError):
        run(derivation_case, continuation=derivation_case.continuation | {field: value})
    assert not derivation_case.paths.attempt.exists()


def test_changed_raw_locator_rejects_even_without_opening_it(derivation_case):
    derivation_case.paths = replace(
        derivation_case.paths,
        archive=derivation_case.paths.archive.parent / "different-archive.zip",
    )
    with pytest.raises(ValueError):
        run(derivation_case)
    assert not derivation_case.paths.attempt.exists()


def test_different_historical_component_rejects_even_when_rehashed(derivation_case):
    declared = json.loads(canonical_bytes(derivation_case.continuation))
    declared["prior_profile"]["components"]["external"] = "0" * 64
    declared["prior_profile_sha256"] = sha256(
        canonical_bytes(declared["prior_profile"])
    ).hexdigest()
    with pytest.raises(ValueError):
        run(derivation_case, continuation=declared)
    assert not derivation_case.paths.attempt.exists()


def test_preparation_kernel_and_feasibility_are_invoked_once(
    derivation_case, monkeypatch
):
    from automated_phishing_detection import _study_preparation_body as body

    external, assess = study_preparation_inputs.prepare_external_inputs, body.assess
    calls = []

    def prepare(*arguments, **keywords):
        calls.append("external")
        return external(*arguments, **keywords)

    def feasibility(*arguments):
        calls.append("feasibility")
        return assess(*arguments)

    monkeypatch.setattr(study_preparation_inputs, "prepare_external_inputs", prepare)
    monkeypatch.setattr(body, "assess", feasibility)
    run(derivation_case)
    assert calls == ["external", "feasibility"]


def test_derivation_interruption_keeps_owned_partial_receipt(
    derivation_case, monkeypatch
):
    original = KeyboardInterrupt("invented interruption")

    def interrupt(*arguments, **keywords):
        raise original

    monkeypatch.setattr(study_preparation_inputs, "prepare_external_inputs", interrupt)
    with pytest.raises(KeyboardInterrupt) as caught:
        run(derivation_case)
    assert caught.value is original
    assert type(caught.value.preparation_progress) is bytes
    outcome = json.loads((derivation_case.paths.attempt / "outcome.json").read_bytes())
    assert outcome["status"] == "failed"
