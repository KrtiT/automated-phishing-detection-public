"""Invented inputs exercise durable new-fit evidence and source alignment."""

import json
from hashlib import sha256

import pytest
from test_secondary_development import _fixture

from automated_phishing_detection import execution_receipt, secondary_development


@pytest.fixture
def member():
    from automated_phishing_detection import audited_permutation_member

    return audited_permutation_member


@pytest.fixture
def supplied(tmp_path, member):
    fixture = _fixture(secondary_development, train_count=64, validation_count=12)
    arguments = fixture["arguments"]
    attempt = execution_receipt.reserve_attempt(
        tmp_path / "seed_42", identity={"role": "synthetic_audited_control"}
    )
    return {
        "seed": 42,
        "attempt": attempt,
        "public_path": tmp_path / "seed_42.json",
        "pins": arguments["pins"],
        "preparation_summary": arguments["preparation_summary"],
        "suffix_rules": arguments["suffix_rules"],
        "train_content": arguments["train_content"],
        "validation_content": fixture["validation_content"],
    }


@pytest.mark.parametrize("seed", range(42, 47))
def test_source_to_consumed_labels_to_model_link_is_durable(member, supplied, seed):
    supplied["seed"] = seed
    result = member.run_reserved_member(**supplied)
    root = supplied["attempt"].directory
    summary = json.loads(result.read_bytes())
    linkage = json.loads((root / "evidence/fit-linkage.json").read_bytes())
    for name in ("fit-call-prepared.json", "fit-call-returned.json", "fit-state.json"):
        assert (
            sha256((root / name).read_bytes()).hexdigest()
            == linkage["receipt_sha256"][name]
        )
    assert (
        sha256((root / "evidence/model.json").read_bytes()).hexdigest()
        == linkage["model_sha256"]
    )
    assert linkage["partition_sha256"]["train"] == supplied["pins"].train_sha256
    prepared = json.loads((root / "fit-call-prepared.json").read_bytes())
    train = [json.loads(line) for line in supplied["train_content"].splitlines()]
    assert prepared["ordered_record_ids"] == [row["record_id"] for row in train]
    assert summary["analysis_stage"] == "post_review_new_fit_diagnostic"
    assert summary["historical_fit_verified"] is False
    assert summary["result"]["scoring_audit"]["portable_exact_parity"] is True
    assert summary["result"]["source_identity_verified"] is True
    assert "record_id" not in result.read_text()
    assert ".example" not in result.read_text()


@pytest.mark.parametrize(
    "destination", ["existing", "inside_attempt", "alias", "missing_parent"]
)
def test_unsafe_public_destination_stops_before_fit(
    member, supplied, monkeypatch, tmp_path, destination
):
    if destination == "existing":
        supplied["public_path"].write_bytes(b"preserved")
    elif destination == "inside_attempt":
        supplied["public_path"] = supplied["attempt"].directory / "summary.json"
    elif destination == "alias":
        alias = tmp_path / "alias"
        alias.symlink_to(tmp_path, target_is_directory=True)
        supplied["public_path"] = alias / "summary.json"
    else:
        supplied["public_path"] = tmp_path / "absent" / "summary.json"
    monkeypatch.setattr(
        member.development_runner,
        "_tabular",
        lambda *args, **kwargs: pytest.fail("fit before destination preflight"),
    )
    with pytest.raises(member.DiagnosticError, match="output_preflight"):
        member.run_reserved_member(**supplied)
    root = supplied["attempt"].directory
    assert not (root / "fit-call-prepared.json").exists()
    assert json.loads((root / "outcome.json").read_bytes())["status"] == "failed"
    if destination == "existing":
        assert supplied["public_path"].read_bytes() == b"preserved"


@pytest.mark.parametrize("stage", ["checkpoint", "scoring"])
def test_late_failure_preserves_fit_receipts(member, supplied, monkeypatch, stage):
    original_retain = member._retain

    def retain(attempt, name, content):
        if name == "fit-state.json":
            raise OSError("private path")
        original_retain(attempt, name, content)

    def scoring_failure(*args, **kwargs):
        raise ValueError("private score")

    if stage == "checkpoint":
        monkeypatch.setattr(member, "_retain", retain)
    else:
        monkeypatch.setattr(
            member.secondary_tabular, "_logistic_scores", scoring_failure
        )
    with pytest.raises(member.DiagnosticError, match="fit"):
        member.run_reserved_member(**supplied)
    root = supplied["attempt"].directory
    assert (root / "fit-call-prepared.json").exists()
    assert (root / "fit-call-returned.json").exists()
    assert (root / "fit-state.json").exists() == (stage == "scoring")
    assert json.loads((root / "outcome.json").read_bytes())["status"] == "failed"
    assert not supplied["public_path"].exists()


def test_publication_rejection_before_claim_retains_failure(
    member, supplied, monkeypatch
):
    def reject(*args, **kwargs):
        raise OSError("private output")

    monkeypatch.setattr(member.execution_receipt, "publish_completion", reject)
    with pytest.raises(member.DiagnosticError, match="publication"):
        member.run_reserved_member(**supplied)
    outcome = supplied["attempt"].directory / "outcome.json"
    assert json.loads(outcome.read_bytes())["stage"] == "publication"
    assert not supplied["public_path"].exists()


def test_fit_failure_preserves_prepared_evidence_and_blocks_retry(
    member, supplied, monkeypatch
):
    def refuse(*args, **kwargs):
        assert (supplied["attempt"].directory / "fit-call-prepared.json").exists()
        raise ValueError("private.example")

    monkeypatch.setattr(member.secondary_tabular.LogisticRegression, "fit", refuse)
    with pytest.raises(member.DiagnosticError, match="fit") as error:
        member.run_reserved_member(**supplied)
    root = supplied["attempt"].directory
    assert (root / "fit-call-prepared.json").exists()
    assert not (root / "fit-call-returned.json").exists()
    assert json.loads((root / "outcome.json").read_bytes())["status"] == "failed"
    assert not supplied["public_path"].exists()
    assert "private.example" not in str(error.value)
    with pytest.raises(member.DiagnosticError):
        member.run_reserved_member(**supplied)


def test_wrong_source_bytes_fail_before_fit(member, supplied, monkeypatch):
    supplied["train_content"] += b"\n"
    monkeypatch.setattr(
        member.secondary_tabular,
        "fit_label_permutation",
        lambda *args, **kwargs: pytest.fail("fit on wrong source"),
    )
    with pytest.raises(member.DiagnosticError, match="input_validation"):
        member.run_reserved_member(**supplied)


def test_receipt_failure_prevents_fit_and_success(member, supplied, monkeypatch):
    def refuse(*args):
        raise OSError("private path")

    monkeypatch.setattr(member, "_retain", refuse)
    monkeypatch.setattr(
        member.secondary_tabular.LogisticRegression,
        "fit",
        lambda *args: pytest.fail("fit after lost receipt"),
    )
    with pytest.raises(member.DiagnosticError):
        member.run_reserved_member(**supplied)
    assert not supplied["public_path"].exists()


def test_completed_member_cannot_fit_again(member, supplied, monkeypatch):
    member.run_reserved_member(**supplied)
    monkeypatch.setattr(
        member.secondary_tabular.LogisticRegression,
        "fit",
        lambda *args: pytest.fail("repeat fit"),
    )
    with pytest.raises(member.DiagnosticError):
        member.run_reserved_member(**supplied)
