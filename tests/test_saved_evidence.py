"""Independent saved-byte reconstruction over invented evidence only."""

import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_evaluation_producer import parse, synthetic_session

from automated_phishing_detection import (
    bound_models,
    bound_secondary,
    evaluation_producer,
    source_runner,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def reconstructor():
    from automated_phishing_detection import saved_evidence

    return saved_evidence


def expect_synthetic_binding(reconstructor, monkeypatch, content):
    binding = json.loads(content)
    monkeypatch.setattr(
        reconstructor,
        "_EXPECTED_BINDING_CORE",
        {
            "artifact_hashes": binding["artifact_hashes"],
            "thresholds": binding["thresholds"],
            "secondary": binding["secondary"],
            "gmm_audit": binding["gmm_audit"],
        },
    )


def test_production_binding_core_matches_authenticated_public_reports(reconstructor):
    primary_reports = bound_models._read_public(ROOT)
    bound_models._validate_public(primary_reports)
    secondary_reports = bound_secondary._read_reports(ROOT)
    probe = secondary_reports["seeds"]["completion"]["probes"]["result"]
    operating_points = probe["operating_points"]
    primary_artifacts = {
        "length-only.json": primary_reports["baseline"]["models"]["length-only"][
            "artifact_sha256"
        ],
        "logistic-l1.json": primary_reports["baseline"]["models"]["Logistic-L1"][
            "artifact_sha256"
        ],
        **primary_reports["transformer"]["artifact_hashes"],
        "gmm.json": primary_reports["gmm"]["artifact_hashes"]["gmm.json"],
    }
    assert probe["primary_artifact_sha256"] == primary_artifacts

    tabular_points = bound_secondary._tabular_points(secondary_reports["tabular"])
    seed_points = bound_secondary._seed_points(
        secondary_reports["seeds"],
        SimpleNamespace(stage1_threshold=operating_points["stage1_threshold"]),
    )
    expected = {
        "gmm_audit": {
            "alert_count": primary_reports["gmm"]["audit_alert_count"],
            "window_count": primary_reports["gmm"]["audit_window_count"],
        },
        "artifact_hashes": primary_artifacts,
        "thresholds": {
            "length_only": operating_points["length_threshold"],
            "logistic_l1": operating_points["stage1_threshold"],
            "transformer": operating_points["transformer_threshold"],
            "half_width": operating_points["half_width"],
            "monitor_boundary": primary_reports["gmm"]["threshold"],
        },
        "secondary": {
            "accepted_report_sha256": {
                role: digest
                for role, (_, digest) in bound_secondary.PUBLIC_REPORTS.items()
            },
            "device_type": "mps",
            "stage1_threshold": operating_points["stage1_threshold"],
            "vocabulary_sha256": primary_artifacts["vocabulary.json"],
            "tabular": [
                {
                    "name": name,
                    "artifact_sha256": digest,
                    "threshold": threshold,
                }
                for name, digest, threshold in tabular_points
            ],
            "seeds": [
                {
                    "seed": seed,
                    "weights_sha256": digest,
                    "transformer_threshold": threshold,
                    "half_width": half_width,
                    "reuses_primary": seed == 42,
                }
                for seed, digest, threshold, half_width in seed_points
            ],
        },
    }

    assert reconstructor._EXPECTED_BINDING_CORE == expected


def test_reconstructs_complete_evidence_without_source_or_model_reads(
    reconstructor, monkeypatch
):
    prepared = parse(evaluation_producer)
    session, *_ = synthetic_session(evaluation_producer, monkeypatch)
    produced = evaluation_producer.produce_internal_evidence(prepared, session)

    def forbidden(*args, **kwargs):
        pytest.fail("saved reconstruction attempted source or model access")

    for name in (
        "load_bound_models",
        "load_bound_secondary",
        "score_bound_secondary",
        "parse_internal_partition",
    ):
        monkeypatch.setattr(reconstructor, name, forbidden, raising=False)
    expect_synthetic_binding(
        reconstructor, monkeypatch, produced.private_outputs["bindings.json"]
    )

    result = reconstructor.reconstruct_internal_evidence(
        produced.private_outputs["predictions.jsonl"],
        produced.private_outputs["manifests.json"],
        produced.private_outputs["bindings.json"],
        produced.private_outputs["routing.json"],
    )

    assert result.row_count == len(produced.rows)
    assert result.domain_count == prepared.domain_count
    assert result.class_counts == {"0": 2, "1": 2}
    assert result.inference_counts == produced.inference_counts
    assert result.secondary_inference_counts == produced.secondary_inference_counts
    assert asdict(result.primary) == asdict(produced.primary)
    assert {
        str(prevalence): evaluation_producer._manifest_summary(outcome)
        for prevalence, outcome in result.manifests.items()
    } == produced.public_summary["manifests"]
    assert result.secondary == source_runner._secondary(produced)


@pytest.mark.parametrize(
    "mutation",
    [
        "primary_artifact",
        "primary_cutoff",
        "accepted_report",
        "secondary_artifact",
        "secondary_cutoff",
    ],
)
def test_rejects_repaired_binding_identity_or_cutoff_mutation(
    reconstructor, monkeypatch, mutation
):
    prepared = parse(evaluation_producer)
    session, *_ = synthetic_session(evaluation_producer, monkeypatch)
    produced = evaluation_producer.produce_internal_evidence(prepared, session)
    binding = json.loads(produced.private_outputs["bindings.json"])
    expect_synthetic_binding(
        reconstructor, monkeypatch, produced.private_outputs["bindings.json"]
    )
    if mutation == "primary_artifact":
        binding["artifact_hashes"]["synthetic.json"] = "f" * 64
    elif mutation == "primary_cutoff":
        binding["thresholds"]["length_only"] = 0.6
    elif mutation == "accepted_report":
        binding["secondary"]["accepted_report_sha256"]["tabular"] = "f" * 64
    elif mutation == "secondary_artifact":
        binding["secondary"]["tabular"][0]["artifact_sha256"] = "f" * 64
    else:
        binding["secondary"]["tabular"][0]["threshold"] = 0.21

    with pytest.raises(reconstructor.SavedEvidenceError):
        reconstructor.reconstruct_internal_evidence(
            produced.private_outputs["predictions.jsonl"],
            produced.private_outputs["manifests.json"],
            evaluation_producer._json_bytes(binding),
            produced.private_outputs["routing.json"],
        )
