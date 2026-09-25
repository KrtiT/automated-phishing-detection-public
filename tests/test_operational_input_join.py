"""Reject divergent retained identities without opening or recomputing sources."""

from dataclasses import replace
from pathlib import Path

import pytest
from operational_input_fixtures import (
    build,
    candidates,
    case,
    change_payload,
    manifests,
)
from test_operational_inputs import module

__all__ = ["candidates", "manifests", "case"]


def altered(case, observation):
    return module().build_accepted_inputs(
        case.internal,
        observation,
        binding=case.binding,
        root_reservation_sha256=case.reservation,
        operational_profile_sha256=case.profile,
    )


@pytest.mark.parametrize(
    "field",
    [
        "revision",
        "runtime_sha256",
        "source_spec_sha256",
        "execution_contract_sha256",
        "study_preparation_reservation_sha256",
        "study_preparation_complete_sha256",
        "internal_handoff_sha256",
        "internal_overlap_sha256",
        "internal_reservation_sha256",
        "suffix_rules_sha256",
        "source_interface",
        "checkpoint_protocol",
        "archive_size_bytes",
    ],
)
def test_rejects_each_external_execution_join(case, field):
    def mutate(value):
        value["execution"][field] = True if field == "archive_size_bytes" else "0" * 64
        return value

    changed = change_payload(case.external, "public-summary.json", mutate)
    with pytest.raises(ValueError, match="invalid_operational_inputs"):
        altered(case, changed)


@pytest.mark.parametrize(
    "field",
    [
        "cascade.json",
        "gmm.json",
        "length-only.json",
        "logistic-l1.json",
        "transformer-weights.npz",
        "transformer.json",
        "vocabulary.json",
    ],
)
def test_every_primary_hash_is_compared(case, field):
    def mutate(value):
        value["artifact_hashes"][field] = "0" * 64
        return value

    with pytest.raises(ValueError):
        altered(
            case,
            change_payload(case.external, "attempt/evidence/bindings.json", mutate),
        )


@pytest.mark.parametrize(
    "field",
    ["length_only", "logistic_l1", "transformer", "half_width", "monitor_boundary"],
)
def test_every_operating_point_is_compared(case, field):
    def mutate(value):
        value["thresholds"][field] = 0.125
        return value

    with pytest.raises(ValueError):
        altered(
            case,
            change_payload(case.external, "attempt/evidence/bindings.json", mutate),
        )


def test_rejects_profile_bytes_and_worker_exit_substitution(case):
    changed = replace(
        case.external, snapshot=replace(case.external.snapshot, profile_bytes=b"{}\n")
    )
    with pytest.raises(ValueError):
        altered(case, changed)
    for exit_code in (False, 0.0, 17, None):
        worker = replace(
            case.external.worker,
            exit=replace(case.external.worker.exit, exit_code=exit_code),
        )
        with pytest.raises(ValueError):
            altered(case, replace(case.external, worker=worker))


def test_preparation_summary_must_match_independent_execution_binding(case):
    pins = dict(case.binding.source_hashes)
    pins["reports/phiusiil-preparation-summary.json"] = "0" * 64
    changed = replace(case.binding, source_hashes=tuple(sorted(pins.items())))
    with pytest.raises(ValueError, match="invalid_operational_inputs"):
        build(module(), case, binding=changed)


def test_projection_does_not_read_model_source_or_repeat_reconstruction(
    case, monkeypatch
):
    from automated_phishing_detection import (
        bound_models,
        evaluation_manifest,
        phiusiil,
        saved_evidence,
        saved_external_evidence,
    )

    def forbidden(*args, **kwargs):
        pytest.fail("operational projection repeated source/model/scientific work")

    for name in ("read_bytes", "read_text", "open", "stat"):
        monkeypatch.setattr(Path, name, forbidden)
    monkeypatch.setattr(bound_models, "load_bound_models", forbidden)
    monkeypatch.setattr(phiusiil, "_parse_csv_rows", forbidden)
    monkeypatch.setattr(evaluation_manifest, "build_manifest", forbidden)
    monkeypatch.setattr(saved_evidence, "_replay_models", forbidden)
    monkeypatch.setattr(
        saved_external_evidence, "reconstruct_external_evidence", forbidden
    )
    assert build(module(), case).internal is case.internal
