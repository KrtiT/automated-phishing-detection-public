"""Reject changed historical phase state and noncanonical checkpoint bytes."""

import json
from dataclasses import replace

import pytest
from http_run_checkpoints_fixtures import api, checkpoint_case

from automated_phishing_detection._checkpoint_codec import canonical_bytes


@pytest.fixture(scope="module")
def sample():
    with pytest.MonkeyPatch.context() as patch:
        return checkpoint_case(patch)


def reject(sample, warmup=None, measured=None, run=None):
    original, first, second = sample
    with pytest.raises(api().HttpRunCodecError, match="^invalid_http_run_checkpoints$"):
        api().verify_http_checkpoints(
            first if warmup is None else warmup,
            second if measured is None else measured,
            run=original if run is None else run,
        )


@pytest.mark.parametrize("phase", ["warmup", "measured"])
@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", True),
        ("stage", "complete"),
        ("manifest_sha256", "b" * 64),
        ("prevalence_basis_points", 10),
        ("concurrency", 128),
        ("run_index", 2),
        ("workload", "transformer_only"),
        ("measured_drain_ms", 10001.0),
    ],
)
def test_changed_identity_or_later_drain_duration_rejects(sample, phase, field, value):
    saved = json.loads(sample[1 if phase == "warmup" else 2])
    saved[field] = value
    reject(sample, **{phase: canonical_bytes(saved)})


@pytest.mark.parametrize("phase", ["warmup", "measured"])
def test_later_final_counters_cannot_be_added(sample, phase):
    saved = json.loads(sample[1 if phase == "warmup" else 2])
    saved["after_measured"] = sample[0].after_measured.model_dump()
    reject(sample, **{phase: canonical_bytes(saved)})


@pytest.mark.parametrize("phase", ["warmup", "measured"])
@pytest.mark.parametrize("change", ["missing", "extra", "started", "outcome"])
def test_exact_fields_flags_and_outcomes_reject_mutation(sample, phase, change):
    saved = json.loads(sample[1 if phase == "warmup" else 2])
    if change == "missing":
        del saved["initial"]
    elif change == "extra":
        saved["unexpected"] = None
    elif change == "started":
        saved["warmup_started"][0] = 1
    else:
        saved["warmup"][0]["response"]["probability"] = 0.5
    reject(sample, **{phase: canonical_bytes(saved)})


@pytest.mark.parametrize("field", ["after_warmup", "measured_elapsed_ms"])
def test_warmup_cannot_claim_later_measured_stage_state(sample, field):
    saved = json.loads(sample[1])
    saved[field] = json.loads(sample[2])[field]
    reject(sample, warmup=canonical_bytes(saved))


@pytest.mark.parametrize("field", ["after_warmup", "measured_elapsed_ms"])
def test_measured_stage_cannot_omit_existing_state(sample, field):
    saved = json.loads(sample[2])
    saved[field] = None
    reject(sample, measured=canonical_bytes(saved))


@pytest.mark.parametrize("phase", ["warmup", "measured"])
@pytest.mark.parametrize("change", ["pretty", "truncated", "duplicate", "bytearray"])
def test_noncanonical_or_inexact_byte_type_rejects(sample, phase, change):
    content = sample[1 if phase == "warmup" else 2]
    changed = {
        "pretty": lambda: json.dumps(json.loads(content), indent=2).encode(),
        "truncated": lambda: content[:-1],
        "duplicate": lambda: b'{"schema_version":1,' + content[1:],
        "bytearray": lambda: bytearray(content),
    }[change]()
    reject(sample, **{phase: changed})


@pytest.mark.parametrize("field", ["warmup", "measured"])
def test_shortened_run_cannot_authorize_shortened_checkpoints(sample, field):
    run = sample[0]
    reject(sample, run=replace(run, **{field: getattr(run, field)[:-1]}))


@pytest.mark.parametrize("interruption", [KeyboardInterrupt(), SystemExit(7)])
def test_original_interruption_survives_validation(sample, monkeypatch, interruption):
    module = api()

    def interrupted(*args):
        raise interruption

    monkeypatch.setattr(module, "encode_http_run", interrupted)
    with pytest.raises(type(interruption)) as caught:
        module.verify_http_checkpoints(*sample[1:], run=sample[0])
    assert caught.value is interruption


def test_verifier_never_opens_files_or_samples(sample, monkeypatch):
    import builtins
    import random

    import numpy as np

    module = api()

    def forbidden(*args, **kwargs):
        pytest.fail("checkpoint verifier attempted I/O or sampling")

    monkeypatch.setattr(builtins, "open", forbidden)
    monkeypatch.setattr(random, "Random", forbidden)
    monkeypatch.setattr(np.random, "default_rng", forbidden)
    assert module.verify_http_checkpoints(*sample[1:], run=sample[0]) is None
