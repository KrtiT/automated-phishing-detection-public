"""No-refit audits use invented source rows and immutable synthetic receipts."""

import importlib
import json
from dataclasses import asdict
from hashlib import sha256
from types import SimpleNamespace

import numpy as np
import pytest
from test_development_completion import _model
from test_secondary_development import _fixture

from automated_phishing_detection import (
    baselines,
    development_execution,
    execution_receipt,
    secondary_development,
    secondary_metrics,
    secondary_tabular,
)

MEMBERS = development_execution.STEPS[:-1]


def _bytes(value):
    return execution_receipt._json_bytes(value, "fixture")


def _load(path):
    return json.loads(path.read_bytes())


def _write(path, value):
    path.write_bytes(_bytes(value))


@pytest.fixture(scope="module")
def material():
    data = _fixture(secondary_development, validation_count=32)
    reference = secondary_development.build_training_reference(**data["arguments"])
    drift = secondary_development.evaluate_validation(
        reference, data["validation_content"]
    )
    return data, drift


@pytest.fixture
def retained(material, tmp_path, monkeypatch):
    data, drift = material
    root = tmp_path / "checkout"
    root.mkdir()
    base = SimpleNamespace(
        root=root,
        revision="e" * 40,
        contract_sha256=development_execution.BASE_PROFILE_SHA256,
        runtime_json='{"fixture":true}',
    )
    binding = development_execution.DevelopmentExecutionBinding(
        base,
        "67146228d636c16f02484998741c7a1545da68b209e693620efab22b2676cd43",
        development_execution.METHODS_SHA256,
        data["arguments"]["pins"],
        data["arguments"]["preparation_summary"],
    )
    rechecks = []
    monkeypatch.setattr(
        development_execution,
        "recheck_development_binding",
        lambda value: rechecks.append(value),
    )
    identity = {
        "kind": "secondary_development",
        "revision": "a" * 40,
        "execution_contract_sha256": binding.base.contract_sha256,
        "development_profile_sha256": binding.profile_sha256,
        "methods_contract_sha256": binding.methods_sha256,
        "runtime_sha256": sha256(base.runtime_json.encode()).hexdigest(),
        "pins": asdict(binding.pins),
        "ordered_steps": list(development_execution.STEPS),
    }
    original = tmp_path / "original-attempt"
    attempt = execution_receipt.reserve_attempt(original, identity=identity)
    rows = [
        {"record_id": row["record_id"], "label": row["is_phishing"], "probability": 0.5}
        for row in data["validation"]
    ]
    threshold = baselines.select_validation_threshold(
        [row["probability"] for row in rows], [row["label"] for row in rows]
    )
    members = []
    for member in MEMBERS:
        child = execution_receipt.reserve_attempt(
            original / member,
            identity={
                "root_reservation_sha256": attempt.reservation_sha256,
                "member": member,
            },
        )
        if member == "drift":
            outputs, result = drift.private_outputs, drift.public_summary
        else:
            audit = {
                "batch_size": 1,
                "warning_records": [],
                "portable_exact_parity": True,
                "threshold_role": "secondary_descriptive_operating_point",
                "platform_identity": baselines._platform_identity(),
                "max_absolute_decision_difference": 0.0,
                "max_absolute_probability_difference": 0.0,
            }
            outputs = {
                "model.json": _model(member, len(data["train"])),
                "validation-predictions.jsonl": b"".join(
                    secondary_tabular._json_bytes(row) for row in rows
                ),
                "threshold.json": secondary_tabular._json_bytes(threshold),
                "scoring-audit.json": secondary_tabular._json_bytes(audit),
            }
            result = {
                "model_kind": "formatting" if member == "formatting" else "permutation",
                "seed": 42 if member == "formatting" else int(member[-2:]),
                "row_count": len(rows),
                "class_counts": {"0": 16, "1": 16},
                "validation_threshold": threshold,
                "scoring_audit": audit,
                "analysis_role": "descriptive_secondary_not_primary",
                "score_metrics": {"average_precision": 0.5, "roc_auc": 0.5},
            }
        summary = {
            "schema_version": 1,
            "status": "development_member_completed",
            "member": member,
            "root_reservation_sha256": attempt.reservation_sha256,
            "reservation_sha256": child.reservation_sha256,
            "private_sha256": {
                name: sha256(content).hexdigest() for name, content in outputs.items()
            },
            "result": result,
        }
        marker = original / f"{member}.json"
        execution_receipt.publish_completion(
            child, private_outputs=outputs, public_summary=summary, public_path=marker
        )
        members.append(
            {
                "member": member,
                "acceptance": "producer_completed_preliminary_not_independently_accepted",
                "public_summary_sha256": sha256(marker.read_bytes()).hexdigest(),
                "summary": summary,
            }
        )
    failed = execution_receipt.reserve_attempt(
        original / "random_forest",
        identity={
            "root_reservation_sha256": attempt.reservation_sha256,
            "member": "random_forest",
        },
    )
    for item in (failed, attempt):
        execution_receipt.record_failure(
            item, stage="random_forest", error_type="SecondaryTabularError"
        )
    accounting = {
        "schema_version": 1,
        "record_kind": "secondary_development_attempt_accounting",
        "analysis_stage": "development_validation_only",
        "status": "failed_partial_evidence_retained",
        "aggregate_accepted": False,
        "execution": identity,
        "execution_observation": {
            "parent_exit_code": 2,
            "worker_exit_code": None,
            "successful_public_marker_present": False,
            "retries": 0,
            "resumes": 0,
        },
        "failure": {
            "member": "random_forest",
            "error_type": "SecondaryTabularError",
            "specific_check": "not_recorded",
            "fitted_state_retained": False,
            "root_outcome": _load(original / "outcome.json"),
            "member_outcome": _load(original / "random_forest/outcome.json"),
        },
        "completed_children": members,
        "unattempted_members": [],
        "receipt_and_summary_sha256": {},
    }
    result = SimpleNamespace(
        binding=binding,
        original=original,
        accounting=accounting,
        data=data,
        rechecks=rechecks,
    )
    _relink(result)
    return result


def _relink(fixture):
    root, report = fixture.original, fixture.accounting
    for child in report["completed_children"]:
        member = child["member"]
        marker = root / f"{member}.json"
        summary = _load(marker)
        summary["private_sha256"] = {
            path.name: sha256(path.read_bytes()).hexdigest()
            for path in (root / member / "evidence").iterdir()
        }
        if member == "drift":
            summary["result"]["private_sha256"] = summary["private_sha256"]
        _write(marker, summary)
        outcome_path = root / member / "outcome.json"
        outcome = _load(outcome_path)
        outcome.update(
            private_sha256=summary["private_sha256"],
            public_summary_sha256=sha256(marker.read_bytes()).hexdigest(),
        )
        _write(outcome_path, outcome)
        child.update(
            summary=summary,
            public_summary_sha256=sha256(marker.read_bytes()).hexdigest(),
        )
    paths = [
        "reservation.json",
        "outcome.json",
        *[f"{member}.json" for member in MEMBERS],
        "random_forest/reservation.json",
        "random_forest/outcome.json",
    ]
    report["receipt_and_summary_sha256"] = {
        name: sha256((root / name).read_bytes()).hexdigest() for name in paths
    }


def _module():
    name = "automated_phishing_detection.development_audit"
    assert importlib.util.find_spec(name) is not None, (
        "retained audit implementation is missing"
    )
    return importlib.import_module(name)


def _audit(fixture, **overrides):
    arguments = {
        "accounting_bytes": _bytes(fixture.accounting),
        "original_attempt": fixture.original,
        "train_bytes": fixture.data["arguments"]["train_content"],
        "validation_bytes": fixture.data["validation_content"],
        "suffix_bytes": fixture.data["arguments"]["suffix_rules"],
        **overrides,
    }
    return _module().audit_retained(fixture.binding, **arguments)


def test_audits_all_retained_members_without_fitting_or_rewriting(
    retained, monkeypatch
):
    def forbidden(*args, **kwargs):
        raise AssertionError("audit attempted a fit")

    for method in (
        "fit_formatting",
        "fit_label_permutation",
        "fit_random_forest",
        "_fit",
        "_fit_logistic",
        "_fit_forest",
    ):
        monkeypatch.setattr(secondary_tabular, method, forbidden)
    monkeypatch.setattr(secondary_tabular.LogisticRegression, "fit", forbidden)
    monkeypatch.setattr(secondary_tabular.RandomForestClassifier, "fit", forbidden)
    before = {
        str(path.relative_to(retained.original)): sha256(path.read_bytes()).hexdigest()
        for path in retained.original.rglob("*")
        if path.is_file()
    }
    result = _audit(retained)
    assert result["status"] == "retained_development_members_audited"
    assert result["fits"] == 0
    assert result["original_aggregate_accepted"] is False
    assert result["protected_evaluation_authorized"] is False
    assert result["original_worker_exit_code"] is None
    assert result["original_execution"]["revision"] == "a" * 40
    assert [member["member"] for member in result["members"]] == list(MEMBERS)
    assert len(retained.rechecks) == 2
    for member in result["members"][1:]:
        assert member["checks"]["saved_model_exact_scores"] is True
        assert member["checks"]["authenticated_validation_order_and_labels"] is True
        assert member["score_range"] == {
            "minimum": 0.5,
            "maximum": 0.5,
            "mean": 0.5,
            "population_standard_deviation": 0.0,
        }
    after = {
        str(path.relative_to(retained.original)): sha256(path.read_bytes()).hexdigest()
        for path in retained.original.rglob("*")
        if path.is_file()
    }
    assert before == after
    public = json.dumps(result)
    assert "raw_url" not in public
    assert "record_id" not in public
    assert str(retained.original) not in public


def test_reconstructs_every_fresh_permutation_and_discloses_missing_fit_digest(
    retained,
):
    result = _audit(retained)
    labels = np.asarray(
        [row["is_phishing"] for row in retained.data["train"]], dtype=np.int64
    )
    diagnostics = result["permutation_diagnostics"]
    assert [item["seed"] for item in diagnostics] == list(range(42, 47))
    for item in diagnostics:
        expected = np.random.Generator(np.random.PCG64(item["seed"])).permutation(
            labels
        )
        assert (
            item["permuted_labels_sha256"]
            == sha256(
                secondary_tabular._json_bytes({"labels": expected.tolist()})
            ).hexdigest()
        )
        assert item["label_agreement_count"] == int(
            np.count_nonzero(expected == labels)
        )
        assert item["class_counts"] == {"0": 128, "1": 128}
        assert item["fit_label_digest_status"] == "not_retained_by_original_producer"


@pytest.mark.parametrize("role", ["train_bytes", "validation_bytes", "suffix_bytes"])
def test_rejects_changed_authenticated_source_bytes(retained, role):
    with pytest.raises(_module().DevelopmentAuditError):
        _audit(retained, **{role: b"changed\n"})


def test_rejects_coherently_rehashed_wrong_validation_labels(retained):
    for member in MEMBERS[1:]:
        path = retained.original / member / "evidence/validation-predictions.jsonl"
        rows = [json.loads(line) for line in path.read_bytes().splitlines()]
        rows[0]["label"], rows[-1]["label"] = rows[-1]["label"], rows[0]["label"]
        path.write_bytes(b"".join(secondary_tabular._json_bytes(row) for row in rows))
    _relink(retained)
    with pytest.raises(
        _module().DevelopmentAuditError, match="validation_label_mismatch"
    ):
        _audit(retained)


def test_rejects_coherently_rehashed_scores_not_produced_by_saved_model(retained):
    member = "permutation_43"
    path = retained.original / member / "evidence"
    rows = [
        json.loads(line)
        for line in (path / "validation-predictions.jsonl").read_bytes().splitlines()
    ]
    for row in rows:
        row["probability"] = 0.6
    (path / "validation-predictions.jsonl").write_bytes(
        b"".join(secondary_tabular._json_bytes(row) for row in rows)
    )
    labels = [row["label"] for row in rows]
    scores = [row["probability"] for row in rows]
    threshold = baselines.select_validation_threshold(scores, labels)
    (path / "threshold.json").write_bytes(secondary_tabular._json_bytes(threshold))
    marker = retained.original / f"{member}.json"
    summary = _load(marker)
    summary["result"]["validation_threshold"] = threshold
    summary["result"]["score_metrics"] = {
        "average_precision": float(
            secondary_metrics.average_precision_score(labels, scores)
        ),
        "roc_auc": float(secondary_metrics.roc_auc_score(labels, scores)),
    }
    _write(marker, summary)
    _relink(retained)
    with pytest.raises(
        _module().DevelopmentAuditError, match="saved_model_score_mismatch"
    ):
        _audit(retained)


def test_rejects_coherently_rehashed_seed_swap(retained):
    destination = retained.original / "permutation_42/evidence/model.json"
    destination.write_bytes(
        (retained.original / "permutation_43/evidence/model.json").read_bytes()
    )
    _relink(retained)
    with pytest.raises(_module().DevelopmentAuditError):
        _audit(retained)


def test_rejects_coherently_rehashed_alternate_validation_membership(retained):
    def shift(identity):
        prefix, ordinal = identity.rsplit(":", 1)
        return f"{prefix}:{int(ordinal, 16) + 10000:016x}"

    for member in MEMBERS[1:]:
        path = retained.original / member / "evidence/validation-predictions.jsonl"
        rows = [json.loads(line) for line in path.read_bytes().splitlines()]
        for row in rows:
            row["record_id"] = shift(row["record_id"])
        path.write_bytes(b"".join(secondary_tabular._json_bytes(row) for row in rows))
    path = retained.original / "drift/evidence/validation-audit.json"
    audit = _load(path)
    for stream in audit["streams"].values():
        stream["record_ids"] = [shift(identity) for identity in stream["record_ids"]]
    path.write_bytes(secondary_tabular._json_bytes(audit))
    _relink(retained)
    with pytest.raises(
        _module().DevelopmentAuditError, match="validation_membership_mismatch"
    ):
        _audit(retained)


def test_rejects_file_symlink_and_unexpected_rf_state(retained, tmp_path):
    module = _module()
    original = retained.original / "formatting/evidence/model.json"
    copy = tmp_path / "model-copy.json"
    copy.write_bytes(original.read_bytes())
    original.unlink()
    original.symlink_to(copy)
    with pytest.raises(module.DevelopmentAuditError):
        _audit(retained)


def test_rejects_rf_fitted_state_added_after_the_failed_attempt(retained):
    (retained.original / "random_forest/model.json").write_bytes(b"{}")
    with pytest.raises(_module().DevelopmentAuditError):
        _audit(retained)


def test_rejects_directory_swap_during_scoring(retained, monkeypatch):
    original = secondary_tabular.SecondaryModel.score_urls
    swapped = False

    def score(model, urls):
        nonlocal swapped
        if not swapped:
            swapped = True
            path = retained.original / "formatting/evidence"
            path.rename(path.with_name("original-evidence"))
            path.mkdir()
            for source in path.with_name("original-evidence").iterdir():
                (path / source.name).write_bytes(source.read_bytes())
        return original(model, urls)

    monkeypatch.setattr(secondary_tabular.SecondaryModel, "score_urls", score)
    with pytest.raises(_module().DevelopmentAuditError):
        _audit(retained)


@pytest.mark.parametrize(
    "field,value",
    [
        ("aggregate_accepted", True),
        ("status", "completed"),
        ("unattempted_members", ["random_forest"]),
    ],
)
def test_rejects_rewritten_original_attempt_status(retained, field, value):
    retained.accounting[field] = value
    with pytest.raises(_module().DevelopmentAuditError):
        _audit(retained)


def test_snapshot_can_verify_retained_receipts_without_source_reads(retained):
    module = _module()
    with module.audited_snapshot(
        retained.binding,
        accounting_bytes=_bytes(retained.accounting),
        original_attempt=retained.original,
    ) as snapshot:
        report, contents = snapshot
        assert report == retained.accounting
        assert retained.original / "drift/evidence/training-reference.json" in contents
    assert len(retained.rechecks) == 2


def test_parent_validation_rechecks_saved_arithmetic_without_source_replay(
    retained, monkeypatch
):
    result = _audit(retained)

    def forbidden(*args, **kwargs):
        raise AssertionError("parent tried source replay")

    monkeypatch.setattr(secondary_development, "_partition", forbidden)
    monkeypatch.setattr(secondary_tabular.SecondaryModel, "score_urls", forbidden)
    module = _module()
    assert hasattr(module, "validate_audit_summary"), (
        "parent audit validator is missing"
    )
    observed = module.validate_audit_summary(
        result,
        binding=retained.binding,
        accounting_bytes=_bytes(retained.accounting),
        original_attempt=retained.original,
    )
    assert observed == result


@pytest.mark.parametrize(
    "change",
    [
        "extra_field",
        "scope",
        "member_summary",
        "member_order",
        "score_range",
        "seed",
        "counts",
        "original_label_digest",
        "permuted_label_digest",
        "agreement",
        "historical_digest_claim",
        "fit_count",
    ],
)
def test_parent_rejects_changed_public_audit_claims(retained, change):
    result = _audit(retained)
    if change == "extra_field":
        result["private_url"] = "https://must-not-appear.example/"
    elif change == "scope":
        result["members"][0]["checks"]["independent_drift_score_recomputation"] = True
    elif change == "member_summary":
        result["members"][1]["summary"]["result"]["seed"] = 46
    elif change == "member_order":
        result["members"][1], result["members"][2] = (
            result["members"][2],
            result["members"][1],
        )
    elif change == "score_range":
        result["members"][1]["score_range"]["minimum"] = 0.4
    elif change == "seed":
        result["permutation_diagnostics"][1]["seed"] = 42
    elif change == "counts":
        result["permutation_diagnostics"][0]["class_counts"] = {"0": 256, "1": 0}
    elif change == "original_label_digest":
        result["permutation_diagnostics"][1]["original_labels_sha256"] = "e" * 64
    elif change == "permuted_label_digest":
        result["permutation_diagnostics"][0]["permuted_labels_sha256"] = "invalid"
    elif change == "agreement":
        result["permutation_diagnostics"][0]["label_agreement_count"] = 257
    elif change == "historical_digest_claim":
        result["permutation_diagnostics"][0]["fit_label_digest_status"] = "verified"
    elif change == "fit_count":
        result["fits"] = 1
    with pytest.raises(_module().DevelopmentAuditError):
        _module().validate_audit_summary(
            result,
            binding=retained.binding,
            accounting_bytes=_bytes(retained.accounting),
            original_attempt=retained.original,
        )
