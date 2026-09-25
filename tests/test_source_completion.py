"""Independent completion checks over producer-created temporary fixtures only."""

# Pytest resolves these imported fixture names through test function parameters.
# ruff: noqa: F811

import json
import math
import os
import shutil
from collections import Counter
from dataclasses import replace
from hashlib import sha256

import pytest
from test_source_runner import inputs, runner  # noqa: F401

from automated_phishing_detection import evaluation_producer, execution_receipt
from automated_phishing_detection.internal_scientific_checkpoints import (
    SCIENTIFIC_CHECKPOINT_NAMES,
)

PRIVATE_NAMES = {
    "predictions.jsonl",
    "manifests.json",
    "bindings.json",
    "secondary.json",
    "routing.json",
}
CHECKPOINT_NAMES = {
    "group_test.jsonl",
    "source-overlap.json",
    "source-reconstruction.json",
}


@pytest.fixture
def verifier():
    from automated_phishing_detection import source_completion

    return source_completion


@pytest.fixture
def published(runner, inputs, monkeypatch):
    from automated_phishing_detection import saved_evidence

    binding, paths, _, events = inputs
    runner._run_bound_internal(binding, paths)
    private_binding = _load(paths.attempt / "evidence/bindings.json")
    monkeypatch.setattr(
        saved_evidence,
        "_EXPECTED_BINDING_CORE",
        {
            "artifact_hashes": private_binding["artifact_hashes"],
            "thresholds": private_binding["thresholds"],
            "secondary": private_binding["secondary"],
            "gmm_audit": private_binding["gmm_audit"],
        },
    )
    return binding, paths, events


def _load(path):
    return json.loads(path.read_bytes())


def _write(path, value):
    path.write_bytes(execution_receipt._json_bytes(value, "fixture"))


def _relink(paths):
    """Repair digest links, without repairing schema or scientific identities."""
    reservation_hash = sha256(
        (paths.attempt / "reservation.json").read_bytes()
    ).hexdigest()
    claim = _load(paths.attempt / "finalize.claim")
    claim["reservation_sha256"] = reservation_hash
    _write(paths.attempt / "finalize.claim", claim)
    public = _load(paths.public_summary)
    public["execution"]["reservation_sha256"] = reservation_hash
    hashes = {
        name: sha256((paths.attempt / "evidence" / name).read_bytes()).hexdigest()
        for name in PRIVATE_NAMES
    }
    public["private_sha256"] = hashes
    _write(paths.public_summary, public)
    outcome = _load(paths.attempt / "outcome.json")
    outcome.update(
        reservation_sha256=reservation_hash,
        private_sha256=hashes,
        public_summary_sha256=sha256(paths.public_summary.read_bytes()).hexdigest(),
    )
    _write(paths.attempt / "outcome.json", outcome)


def test_valid_completion_returns_public_summary_with_single_safe_reads(
    verifier, runner, published, monkeypatch
):
    binding, paths, events = published
    expected = _load(paths.public_summary)
    original = runner._read_file_once
    reads = Counter()
    initial_rechecks = events.count("recheck")

    def observed(path, **kwargs):
        reads[path] += 1
        assert path not in (paths.source_csv, paths.suffix_rules)
        assert path not in vars(paths.artifacts).values()
        assert path not in vars(paths.secondary_artifacts).values()
        return original(path, **kwargs)

    monkeypatch.setattr(runner, "_read_file_once", observed)
    reconstruct = verifier.reconstruct_internal_evidence_and_population
    reconstruction_calls = []

    def observed_reconstruction(*args):
        reconstruction_calls.append(args)
        return reconstruct(*args)

    monkeypatch.setattr(
        verifier,
        "reconstruct_internal_evidence_and_population",
        observed_reconstruction,
    )
    assert (
        verifier.verify_internal_completion(binding, paths, producer_exit_code=0)
        == expected
    )
    assert len(reconstruction_calls) == 1
    assert set(reads) == {
        binding.root / "data/sources.json",
        binding.root / "reports/phiusiil-preparation-summary.json",
        paths.attempt / "reservation.json",
        paths.attempt / "finalize.claim",
        paths.attempt / "outcome.json",
        paths.public_summary,
        *(paths.attempt / "evidence" / name for name in PRIVATE_NAMES),
        *(paths.attempt / "checkpoints" / name for name in CHECKPOINT_NAMES),
        *(
            paths.attempt / "scientific-checkpoints" / name
            for name in SCIENTIFIC_CHECKPOINT_NAMES
        ),
    }
    assert set(reads.values()) == {1}
    assert events.count("recheck") == initial_rechecks + 2


def _relink_checkpoints(paths):
    directory = paths.attempt / "checkpoints"
    receipt = _load(directory / "source-reconstruction.json")
    receipt["checkpoint_sha256"] = {
        name: sha256((directory / name).read_bytes()).hexdigest()
        for name in CHECKPOINT_NAMES - {"source-reconstruction.json"}
    }
    receipt["reconstruction"]["private_sha256"]["source-overlap.json"] = receipt[
        "checkpoint_sha256"
    ]["source-overlap.json"]
    (directory / "source-reconstruction.json").write_bytes(
        evaluation_producer._json_bytes(receipt)
    )
    public = _load(paths.public_summary)
    public["checkpoint_sha256"] = {
        name: sha256((directory / name).read_bytes()).hexdigest()
        for name in CHECKPOINT_NAMES
    }
    public["source_reconstruction"] = receipt["reconstruction"]
    _write(paths.public_summary, public)
    _relink(paths)


@pytest.mark.parametrize("name", sorted(CHECKPOINT_NAMES))
def test_changed_checkpoint_bytes_rejected(verifier, published, name):
    binding, paths, *_ = published
    path = paths.attempt / "checkpoints" / name
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(verifier.CompletionVerificationError):
        verifier.verify_internal_completion(binding, paths, producer_exit_code=0)


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_row",
        "duplicate_ordinal",
        "wrong_record",
        "extra_domain",
        "retained_domain",
        "counts",
        "execution",
        "partition",
    ],
)
def test_relinked_source_checkpoint_tampering_rejected(verifier, published, mutation):
    binding, paths, *_ = published
    directory = paths.attempt / "checkpoints"
    path = directory / "source-overlap.json"
    overlap = _load(path)
    if mutation == "missing_row":
        overlap["rows"].pop()
    elif mutation == "duplicate_ordinal":
        overlap["rows"][1]["source_ordinal"] = 1
    elif mutation == "wrong_record":
        overlap["rows"][0]["record_id"] = overlap["rows"][1]["record_id"]
    elif mutation == "extra_domain":
        overlap["domains"].append("invented.com")
    elif mutation == "retained_domain":
        overlap["rows"][0]["registrable_domain"] = overlap["rows"][1][
            "registrable_domain"
        ]
    elif mutation in {"counts", "execution"}:
        receipt_path = directory / "source-reconstruction.json"
        receipt = _load(receipt_path)
        if mutation == "counts":
            receipt["reconstruction"]["counts"]["input_rows"] += 1
        else:
            receipt["execution"]["source_interface"] = "partition_only"
        receipt_path.write_bytes(evaluation_producer._json_bytes(receipt))
    else:
        partition = directory / "group_test.jsonl"
        rows = [json.loads(line) for line in partition.read_bytes().splitlines()]
        rows[0]["raw_url"] += "changed"
        partition.write_bytes(
            b"".join(evaluation_producer._json_bytes(row) for row in rows)
        )
    path.write_bytes(evaluation_producer._json_bytes(overlap))
    _relink_checkpoints(paths)
    with pytest.raises(verifier.CompletionVerificationError):
        verifier.verify_internal_completion(binding, paths, producer_exit_code=0)


@pytest.mark.parametrize("kind", ["symlink", "hardlink", "directory"])
def test_checkpoint_aliases_rejected(verifier, published, tmp_path, kind):
    binding, paths, *_ = published
    path = paths.attempt / "checkpoints"
    if kind != "directory":
        path /= "source-overlap.json"
    detached = tmp_path / "detached-checkpoint"
    path.rename(detached)
    if kind == "hardlink":
        os.link(detached, path)
    else:
        path.symlink_to(detached, target_is_directory=kind == "directory")
    with pytest.raises(verifier.CompletionVerificationError):
        verifier.verify_internal_completion(binding, paths, producer_exit_code=0)


@pytest.mark.parametrize("source_ordinal", [31, 32])
def test_relinked_canonical_group_with_conflicting_domains_rejected(
    verifier, published, source_ordinal
):
    binding, paths, *_ = published
    path = paths.attempt / "checkpoints/source-overlap.json"
    overlap = _load(path)
    duplicate = overlap["rows"][source_ordinal - 1]
    assert (
        sum(
            row["canonical_url_sha256"] == duplicate["canonical_url_sha256"]
            for row in overlap["rows"]
        )
        == 2
    )
    duplicate["registrable_domain"] = next(
        domain
        for domain in overlap["domains"]
        if domain != duplicate["registrable_domain"]
    )
    path.write_bytes(evaluation_producer._json_bytes(overlap))
    _relink_checkpoints(paths)
    with pytest.raises(verifier.CompletionVerificationError):
        verifier.verify_internal_completion(binding, paths, producer_exit_code=0)


@pytest.mark.parametrize("exit_code", [True, False, None, "0", 0.0, 1, -9])
def test_exit_status_must_be_observed_exact_zero_before_any_access(
    verifier, runner, inputs, monkeypatch, exit_code
):
    binding, paths, *_ = inputs

    def forbidden(*args, **kwargs):
        pytest.fail("invalid producer exit reached binding or file access")

    monkeypatch.setattr(runner, "_read_file_once", forbidden)
    monkeypatch.setattr(runner, "recheck_binding", forbidden)
    with pytest.raises(verifier.CompletionVerificationError, match="producer_exit"):
        verifier.verify_internal_completion(
            binding, paths, producer_exit_code=exit_code
        )


def test_binding_failure_precedes_output_reads(
    verifier, runner, published, monkeypatch
):
    binding, paths, *_ = published

    def invalid(*args):
        raise ValueError("private detail must not appear")

    monkeypatch.setattr(runner, "recheck_binding", invalid)
    monkeypatch.setattr(
        runner, "_read_file_once", lambda *args: pytest.fail("read output")
    )
    with pytest.raises(verifier.CompletionVerificationError) as caught:
        verifier.verify_internal_completion(binding, paths, producer_exit_code=0)
    assert "private detail" not in str(caught.value)


def test_changed_public_source_chain_rejected(verifier, published):
    binding, paths, *_ = published
    source = binding.root / "data/sources.json"
    source.write_bytes(source.read_bytes() + b" ")
    with pytest.raises(verifier.CompletionVerificationError):
        verifier.verify_internal_completion(binding, paths, producer_exit_code=0)


@pytest.mark.parametrize("field", ["revision", "contract_sha256", "runtime_json"])
def test_identity_is_rederived_from_supplied_binding(verifier, published, field):
    binding, paths, *_ = published
    changed = {
        "revision": "e" * 40,
        "contract_sha256": "e" * 64,
        "runtime_json": '{"fixture":false}',
    }
    binding = replace(binding, **{field: changed[field]})
    with pytest.raises(verifier.CompletionVerificationError):
        verifier.verify_internal_completion(binding, paths, producer_exit_code=0)


@pytest.mark.parametrize(
    ("record", "field", "value"),
    [
        ("reservation.json", "schema_version", True),
        ("reservation.json", "status", "complete"),
        ("reservation.json", "directory", "/invented/wrong-attempt"),
        ("reservation.json", "extra", "rejected"),
        ("finalize.claim", "schema_version", True),
        ("finalize.claim", "operation", "failure"),
        ("finalize.claim", "extra", "rejected"),
        ("outcome.json", "schema_version", True),
        ("outcome.json", "status", "failed"),
        ("outcome.json", "extra", "rejected"),
    ],
)
def test_exact_receipt_schemas_even_with_repaired_hashes(
    verifier, published, record, field, value
):
    binding, paths, *_ = published
    path = paths.attempt / record
    content = _load(path)
    content[field] = value
    _write(path, content)
    _relink(paths)
    with pytest.raises(verifier.CompletionVerificationError):
        verifier.verify_internal_completion(binding, paths, producer_exit_code=0)


def test_identity_tampering_cannot_be_hidden_by_repaired_links(verifier, published):
    binding, paths, *_ = published
    reservation = _load(paths.attempt / "reservation.json")
    reservation["identity"]["partition_sha256"] = "a" * 64
    _write(paths.attempt / "reservation.json", reservation)
    public = _load(paths.public_summary)
    public["execution"]["partition_sha256"] = "a" * 64
    _write(paths.public_summary, public)
    _relink(paths)
    with pytest.raises(verifier.CompletionVerificationError):
        verifier.verify_internal_completion(binding, paths, producer_exit_code=0)


@pytest.mark.parametrize(
    "record", ["reservation.json", "finalize.claim", "outcome.json"]
)
def test_receipt_bytes_must_be_canonical(verifier, published, record):
    binding, paths, *_ = published
    target = paths.attempt / record
    target.write_bytes(json.dumps(_load(target), indent=2).encode())
    # The reservation's canonicality must be checked separately from its links.
    if record == "reservation.json":
        _relink(paths)
    with pytest.raises(verifier.CompletionVerificationError):
        verifier.verify_internal_completion(binding, paths, producer_exit_code=0)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", True),
        ("status", "internal_evidence_composed"),
        ("source_binding", "caller_supplied_pins_only"),
        ("protected_evaluation_authorized", 0),
        ("protected_evaluation_authorized", True),
        ("row_count", True),
        ("row_count", 5),
        ("domain_count", 5),
        ("class_counts", {"0": 3, "1": 2}),
        ("extra", {"raw_url": "https://invented.invalid/"}),
    ],
)
def test_public_schema_and_counts_are_checked_after_repaired_hash(
    verifier, published, field, value
):
    binding, paths, *_ = published
    public = _load(paths.public_summary)
    public[field] = value
    _write(paths.public_summary, public)
    _relink(paths)
    with pytest.raises(verifier.CompletionVerificationError):
        verifier.verify_internal_completion(binding, paths, producer_exit_code=0)


@pytest.mark.parametrize(
    "tamper",
    [
        "metric_key",
        "metric_count",
        "metric_extra",
        "holm_cells",
        "holm_bool",
        "mcnemar",
    ],
)
def test_secondary_exact_shapes_and_frozen_family(verifier, published, tamper):
    binding, paths, *_ = published
    public = _load(paths.public_summary)
    secondary = public["secondary"]
    if tamper == "metric_key":
        secondary["metrics"]["extra"] = secondary["metrics"]["length_only"]
    elif tamper == "metric_count":
        secondary["metrics"]["length_only"]["counts"]["true_positives"] = True
    elif tamper == "metric_extra":
        secondary["metrics"]["length_only"]["precision"]["extra"] = "private"
    elif tamper == "holm_cells":
        secondary["holm"]["cells"] = secondary["holm"]["cells"][:3]
    elif tamper == "holm_bool":
        secondary["holm"]["family_size"] = True
    else:
        secondary["mcnemar"]["unexpected_contrast"] = None
    _write(paths.attempt / "evidence/secondary.json", secondary)
    _write(paths.public_summary, public)
    _relink(paths)
    with pytest.raises(verifier.CompletionVerificationError):
        verifier.verify_internal_completion(binding, paths, producer_exit_code=0)


def test_private_secondary_must_equal_public_secondary(verifier, published):
    binding, paths, *_ = published
    path = paths.attempt / "evidence/secondary.json"
    secondary = _load(path)
    secondary["metrics"]["length_only"]["precision"]["value"] = 0.1234
    _write(path, secondary)
    _relink(paths)
    with pytest.raises(verifier.CompletionVerificationError):
        verifier.verify_internal_completion(binding, paths, producer_exit_code=0)


@pytest.mark.parametrize(
    "field", ["partition_sha256", "source_csv_sha256", "suffix_rules_sha256"]
)
def test_private_bindings_must_match_authenticated_source_identity(
    verifier, published, field
):
    binding, paths, *_ = published
    path = paths.attempt / "evidence/bindings.json"
    data = _load(path)
    data[field] = "f" * 64
    _write(path, data)
    _relink(paths)
    with pytest.raises(verifier.CompletionVerificationError):
        verifier.verify_internal_completion(binding, paths, producer_exit_code=0)


@pytest.mark.parametrize("name", sorted(PRIVATE_NAMES))
def test_altered_private_bytes_rejected(verifier, published, name):
    binding, paths, *_ = published
    path = paths.attempt / "evidence" / name
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(verifier.CompletionVerificationError):
        verifier.verify_internal_completion(binding, paths, producer_exit_code=0)


@pytest.mark.parametrize(
    "mutation",
    [
        "feature",
        "primary_decision",
        "secondary_score",
        "seed_42_score",
        "monitor_probability",
        "negative_log_likelihood",
        "length_scoring_audit_json",
        "stage1_scoring_audit_json",
    ],
)
def test_repaired_prediction_science_mutation_is_reconstructed_and_rejected(
    verifier, published, mutation
):
    binding, paths, *_ = published
    path = paths.attempt / "evidence/predictions.jsonl"
    rows = [json.loads(line) for line in path.read_bytes().splitlines()]
    if mutation == "feature":
        rows[0]["features"][0] += 1.0
    elif mutation == "primary_decision":
        rows[0]["stage1_decision"] = 1 - rows[0]["stage1_decision"]
    elif mutation == "secondary_score":
        rows[0]["secondary_tabular"][0]["probability"] = 0.24
    elif mutation == "seed_42_score":
        rows[0]["secondary_seeds"][0]["transformer_probability"] = 0.21
    elif mutation.endswith("audit_json"):
        rows[0][mutation] = '{"repaired":true}'
    else:
        rows[0][mutation] += 0.01
    path.write_bytes(b"".join(evaluation_producer._json_bytes(row) for row in rows))
    _relink(paths)
    with pytest.raises(verifier.CompletionVerificationError):
        verifier.verify_internal_completion(binding, paths, producer_exit_code=0)


@pytest.mark.parametrize("mutation", ["routing_mask", "window", "alert_fraction"])
def test_repaired_routing_mutation_is_reconstructed_and_rejected(
    verifier, published, mutation
):
    binding, paths, *_ = published
    path = paths.attempt / "evidence/routing.json"
    routing = _load(path)
    if mutation == "routing_mask":
        routing["rows"][0]["drift_override"] = True
    elif mutation == "window":
        routing["windows"].append(
            {"start_position": 1, "end_position": 4, "score": 2.0, "alert": False}
        )
    else:
        routing["window_alert_fraction"] = 0.0
    path.write_bytes(evaluation_producer._json_bytes(routing))
    _relink(paths)
    with pytest.raises(verifier.CompletionVerificationError):
        verifier.verify_internal_completion(binding, paths, producer_exit_code=0)


@pytest.mark.parametrize(
    "mutation", ["audit", "length-only.json", "logistic-l1.json", "gmm.json"]
)
def test_repaired_retained_artifact_or_audit_mutation_rejected(
    verifier, published, mutation
):
    binding, paths, *_ = published
    path = paths.attempt / "evidence/bindings.json"
    retained = _load(path)
    if mutation == "audit":
        retained["gmm_audit"]["alert_count"] = 0
    else:
        retained["replay_artifacts"][mutation] = "Y2hhbmdlZA=="
    path.write_bytes(evaluation_producer._json_bytes(retained))
    _relink(paths)
    with pytest.raises(verifier.CompletionVerificationError):
        verifier.verify_internal_completion(binding, paths, producer_exit_code=0)


@pytest.mark.parametrize("location", ["attempt", "evidence"])
def test_extraneous_files_rejected(verifier, published, location):
    binding, paths, *_ = published
    directory = paths.attempt if location == "attempt" else paths.attempt / "evidence"
    (directory / "unexpected.txt").write_bytes(b"invented")
    with pytest.raises(verifier.CompletionVerificationError):
        verifier.verify_internal_completion(binding, paths, producer_exit_code=0)


@pytest.mark.parametrize(
    "name",
    [
        "reservation.json",
        "finalize.claim",
        "outcome.json",
        "evidence/secondary.json",
        "public",
    ],
)
def test_incomplete_publication_rejected(verifier, published, name):
    binding, paths, *_ = published
    target = paths.public_summary if name == "public" else paths.attempt / name
    target.unlink()
    with pytest.raises(verifier.CompletionVerificationError):
        verifier.verify_internal_completion(binding, paths, producer_exit_code=0)


@pytest.mark.parametrize(
    "kind", ["symlink", "hardlink", "evidence_symlink", "attempt_symlink"]
)
def test_aliases_rejected(verifier, published, kind, tmp_path):
    binding, paths, *_ = published
    detached = tmp_path / "detached"
    if kind in {"symlink", "hardlink"}:
        path = paths.attempt / "evidence/secondary.json"
        path.rename(detached)
        if kind == "symlink":
            path.symlink_to(detached)
        else:
            os.link(detached, path)
    else:
        path = (
            paths.attempt / "evidence" if kind == "evidence_symlink" else paths.attempt
        )
        path.rename(detached)
        path.symlink_to(detached, target_is_directory=True)
    with pytest.raises(verifier.CompletionVerificationError):
        verifier.verify_internal_completion(binding, paths, producer_exit_code=0)


def test_mutation_after_single_read_is_detected_at_final_identity_check(
    verifier, runner, published, monkeypatch
):
    binding, paths, *_ = published
    original = runner._read_file_once
    target = paths.attempt / "evidence/secondary.json"

    def mutate_after_read(path, **kwargs):
        content = original(path, **kwargs)
        if path == target:
            path.write_bytes(content + b" ")
        return content

    monkeypatch.setattr(runner, "_read_file_once", mutate_after_read)
    with pytest.raises(verifier.CompletionVerificationError):
        verifier.verify_internal_completion(binding, paths, producer_exit_code=0)


@pytest.mark.parametrize(
    "tamper",
    [
        "h2_pass",
        "missing_gate",
        "duplicate_gate",
        "audit",
        "h1_decision",
        "h3_evidence",
    ],
)
def test_frozen_primary_gates_cannot_be_rewritten_even_with_repaired_links(
    verifier, published, tamper
):
    binding, paths, *_ = published
    public = _load(paths.public_summary)
    hypotheses = public["primary"]["hypotheses"]
    if tamper == "h2_pass":
        hypotheses["H2"] = {"decision": "pass", "complete": True, "gates": []}
    elif tamper == "missing_gate":
        hypotheses["H1"]["gates"].pop()
    elif tamper == "duplicate_gate":
        hypotheses["H2"]["gates"][0] = hypotheses["H2"]["gates"][1]
    elif tamper == "audit":
        gate = hypotheses["H2"]["gates"][1]
        gate.update(numerator=0, estimate=0.0, status="pass")
        hypotheses["H2"]["decision"] = "undecided"
    elif tamper == "h1_decision":
        hypotheses["H1"]["decision"] = "supported"
    else:
        gate = hypotheses["H3"]["gates"][-1]
        gate.update(
            status="pass", estimate=0.0, numerator=0, denominator=50000, reason=None
        )
    _write(paths.public_summary, public)
    _relink(paths)
    with pytest.raises(verifier.CompletionVerificationError):
        verifier.verify_internal_completion(binding, paths, producer_exit_code=0)


@pytest.mark.parametrize("where", ["metric", "curve", "mcnemar", "holm", "projection"])
def test_secondary_roles_cannot_be_promoted_to_primary_evidence(
    verifier, published, where
):
    binding, paths, *_ = published
    public = _load(paths.public_summary)
    secondary = public["secondary"]
    metric = secondary["metrics"]["length_only"]
    if where == "metric":
        metric["analysis_role"] = "confirmatory_primary"
    elif where == "curve":
        metric["recall_at_fpr"]["analysis_role"] = "deployment_threshold"
    elif where == "mcnemar":
        secondary["mcnemar"]["internal_logistic_minus_length"]["analysis_role"] = (
            "independent_significance"
        )
    elif where == "holm":
        secondary["holm"]["analysis_role"] = "confirmatory_primary"
    else:
        metric["prevalence_projections"][0]["assumption"] = (
            "measured_deployment_prevalence"
        )
    _write(paths.attempt / "evidence/secondary.json", secondary)
    _write(paths.public_summary, public)
    _relink(paths)
    with pytest.raises(verifier.CompletionVerificationError):
        verifier.verify_internal_completion(binding, paths, producer_exit_code=0)


def test_negative_nested_counts_are_rejected(verifier, published):
    binding, paths, *_ = published
    public = _load(paths.public_summary)
    secondary = public["secondary"]
    secondary["mcnemar"]["internal_logistic_minus_length"]["domain_count"] = -1
    _write(paths.attempt / "evidence/secondary.json", secondary)
    _write(paths.public_summary, public)
    _relink(paths)
    with pytest.raises(verifier.CompletionVerificationError):
        verifier.verify_internal_completion(binding, paths, producer_exit_code=0)


@pytest.mark.parametrize("field", ["measured_count", "warmup_count"])
def test_prepared_manifest_has_exact_frozen_counts(verifier, published, field):
    binding, paths, *_ = published
    public = _load(paths.public_summary)
    manifest = {
        "status": "prepared",
        "sha256": "e" * 64,
        "measured_count": 10000,
        "warmup_count": 1000,
        "prevalence_basis_points": 10,
    }
    manifest[field] = 1
    public["manifests"]["10"] = manifest
    _write(paths.public_summary, public)
    _relink(paths)
    with pytest.raises(verifier.CompletionVerificationError):
        verifier.verify_internal_completion(binding, paths, producer_exit_code=0)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("length_only", -0.1),
        ("logistic_l1", 1.1),
        ("transformer", -0.1),
        ("half_width", -0.1),
        ("length_only", math.nextafter(math.nextafter(1.0, math.inf), math.inf)),
    ],
)
def test_private_threshold_bounds_are_frozen(verifier, published, field, value):
    binding, paths, *_ = published
    path = paths.attempt / "evidence/bindings.json"
    data = _load(path)
    data["thresholds"][field] = value
    _write(path, data)
    _relink(paths)
    with pytest.raises(verifier.CompletionVerificationError):
        verifier.verify_internal_completion(binding, paths, producer_exit_code=0)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("length_only", math.nextafter(1.0, math.inf)),
        ("logistic_l1", math.nextafter(1.0, math.inf)),
        ("transformer", math.nextafter(1.0, math.inf)),
        ("half_width", 0.75),
    ],
)
def test_any_private_threshold_change_is_rejected_even_when_numerically_valid(
    verifier, published, field, value
):
    binding, paths, *_ = published
    path = paths.attempt / "evidence/bindings.json"
    data = _load(path)
    data["thresholds"][field] = value
    _write(path, data)
    _relink(paths)
    with pytest.raises(verifier.CompletionVerificationError):
        verifier.verify_internal_completion(binding, paths, producer_exit_code=0)


def test_external_holm_slots_remain_unavailable(verifier, published):
    binding, paths, *_ = published
    public = _load(paths.public_summary)
    secondary = public["secondary"]
    cell = secondary["holm"]["cells"][2]
    cell["raw_pvalue"] = {"value": 0.01, "reason": None}
    cell["adjusted_pvalue"] = {"value": 0.04, "reason": None}
    _write(paths.attempt / "evidence/secondary.json", secondary)
    _write(paths.public_summary, public)
    _relink(paths)
    with pytest.raises(verifier.CompletionVerificationError):
        verifier.verify_internal_completion(binding, paths, producer_exit_code=0)


def test_snapshot_cannot_read_a_transient_cloned_tree(
    verifier, runner, published, monkeypatch, tmp_path
):
    binding, paths, *_ = published
    clone = tmp_path.parent / f"{tmp_path.name}-clone"
    saved = tmp_path.parent / f"{tmp_path.name}-saved"
    original_snapshot = verifier._output_snapshot
    original_recheck = runner.recheck_binding
    swapped = False

    def restore():
        nonlocal swapped
        if swapped:
            tmp_path.rename(clone)
            saved.rename(tmp_path)
            swapped = False

    def swap_after_snapshot(*args):
        nonlocal swapped
        snapshot = original_snapshot(*args)
        shutil.copytree(tmp_path, clone)
        cloned_paths = replace(
            paths, attempt=clone / "attempt", public_summary=clone / "summary.json"
        )
        target = cloned_paths.attempt / "evidence/predictions.jsonl"
        target.write_bytes(target.read_bytes() + b" ")
        _relink(cloned_paths)
        tmp_path.rename(saved)
        clone.rename(tmp_path)
        swapped = True
        return snapshot

    def restore_before_recheck(*args):
        restore()
        return original_recheck(*args)

    monkeypatch.setattr(verifier, "_output_snapshot", swap_after_snapshot)
    monkeypatch.setattr(runner, "recheck_binding", restore_before_recheck)
    try:
        with pytest.raises(verifier.CompletionVerificationError):
            verifier.verify_internal_completion(binding, paths, producer_exit_code=0)
    finally:
        restore()
