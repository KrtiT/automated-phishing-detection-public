"""Audit retained development members without fitting or rewriting the old run.

The caller authenticates this procedure and the accounting bytes before reading
research inputs. This audit binds those supplied bytes to original receipts and
checks saved predictions against unchanged development rows and saved models.
It cannot recover the original producer's missing fit-label digest or worker exit.
"""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType

import numpy as np

from . import (
    baselines,
    development_execution,
    execution_receipt,
    secondary_development,
    secondary_tabular,
    source_runner,
)
from . import (
    development_completion as completion,
)
from .development_execution import DevelopmentExecutionBinding

_MEMBERS = (
    "drift",
    "formatting",
    "permutation_42",
    "permutation_43",
    "permutation_44",
    "permutation_45",
    "permutation_46",
)
_STEPS = (*_MEMBERS, "random_forest")
_RECEIPTS = {"reservation.json", "finalize.claim", "outcome.json"}
_HASHED_RECORDS = {
    "reservation.json",
    "outcome.json",
    "random_forest/reservation.json",
    "random_forest/outcome.json",
    *(f"{member}.json" for member in _MEMBERS),
}
_DRIFT_CHECKS = {
    "retained_drift_arithmetic": True,
    "authenticated_training_and_validation_membership": True,
    "independent_drift_score_recomputation": False,
}
_TABULAR_CHECKS = {
    "authenticated_validation_order_and_labels": True,
    "saved_model_exact_scores": True,
    "metrics_and_cp_arithmetic": True,
}
_DRIFT_SCOPE = "retained_score_arithmetic_and_authenticated_membership_not_independent_drift_score_recomputation"


class DevelopmentAuditError(ValueError):
    """A safe check identifier without private paths or exception details."""


def _require(condition, symbol):
    if not condition:
        raise DevelopmentAuditError(symbol)


def _accounting(binding, content):
    _require(type(content) is bytes, "accounting_requires_bytes")
    report = source_runner._json(content)
    _require(type(report) is dict, "invalid_accounting_record")
    expected = {
        "schema_version": 1,
        "record_kind": "secondary_development_attempt_accounting",
        "analysis_stage": "development_validation_only",
        "status": "failed_partial_evidence_retained",
        "aggregate_accepted": False,
        "unattempted_members": [],
    }
    _require(
        completion._same({key: report.get(key) for key in expected}, expected),
        "original_attempt_status_mismatch",
    )
    identity = report["execution"]
    _require(
        completion._keys(
            identity,
            {
                "kind",
                "revision",
                "execution_contract_sha256",
                "development_profile_sha256",
                "methods_contract_sha256",
                "runtime_sha256",
                "pins",
                "ordered_steps",
            },
        ),
        "invalid_original_execution",
    )
    revision = identity["revision"]
    _require(
        type(revision) is str
        and len(revision) == 40
        and all(character in "0123456789abcdef" for character in revision),
        "invalid_original_revision",
    )
    expected_identity = {
        "kind": "secondary_development",
        "revision": revision,
        "execution_contract_sha256": binding.base.contract_sha256,
        "development_profile_sha256": binding.profile_sha256,
        "methods_contract_sha256": binding.methods_sha256,
        "runtime_sha256": sha256(binding.base.runtime_json.encode()).hexdigest(),
        "pins": asdict(binding.pins),
        "ordered_steps": list(_STEPS),
    }
    _require(
        completion._same(identity, expected_identity),
        "original_execution_binding_mismatch",
    )
    observation = report["execution_observation"]
    expected_observation = {
        "parent_exit_code": 2,
        "worker_exit_code": None,
        "successful_public_marker_present": False,
        "retries": 0,
        "resumes": 0,
    }
    _require(
        type(observation) is dict
        and completion._same(
            {key: observation.get(key) for key in expected_observation},
            expected_observation,
        ),
        "original_process_observation_mismatch",
    )
    failed = report["failure"]
    expected_failure = {
        "member": "random_forest",
        "error_type": "SecondaryTabularError",
        "specific_check": "not_recorded",
        "fitted_state_retained": False,
    }
    _require(
        type(failed) is dict
        and completion._same(
            {key: failed.get(key) for key in expected_failure}, expected_failure
        ),
        "original_failure_mismatch",
    )
    members = report["completed_children"]
    _require(
        type(members) is list
        and len(members) == len(_MEMBERS)
        and all(
            completion._keys(
                member,
                {"member", "acceptance", "public_summary_sha256", "summary"},
            )
            for member in members
        )
        and [member["member"] for member in members] == list(_MEMBERS)
        and all(
            member["acceptance"]
            == "producer_completed_preliminary_not_independently_accepted"
            for member in members
        ),
        "original_completed_members_mismatch",
    )
    hashes = report["receipt_and_summary_sha256"]
    _require(
        completion._keys(hashes, _HASHED_RECORDS)
        and all(
            type(value) is str and execution_receipt._SHA256.fullmatch(value)
            for value in hashes.values()
        ),
        "invalid_original_receipt_hashes",
    )
    return report


def _failed_receipt(contents, path, identity, expected_outcome):
    reservation = completion._canonical(
        {
            "schema_version": 1,
            "status": "reserved",
            "directory": str(path),
            "identity": identity,
        }
    )
    _require(
        contents[path / "reservation.json"] == reservation,
        "original_failed_reservation_mismatch",
    )
    digest = sha256(reservation).hexdigest()
    _require(
        contents[path / "finalize.claim"]
        == completion._canonical(
            {
                "schema_version": 1,
                "reservation_sha256": digest,
                "operation": "failure",
            }
        ),
        "original_failure_claim_mismatch",
    )
    outcome = {
        "schema_version": 1,
        "status": "failed",
        "reservation_sha256": digest,
        "stage": "random_forest",
        "error_type": "SecondaryTabularError",
    }
    _require(
        contents[path / "outcome.json"] == completion._canonical(outcome)
        and completion._same(expected_outcome, outcome),
        "original_failure_outcome_mismatch",
    )
    return digest


def _receipts(report, path, contents):
    for name, expected in report["receipt_and_summary_sha256"].items():
        _require(
            sha256(contents[path / name]).hexdigest() == expected,
            "original_receipt_hash_mismatch",
        )
    root_hash = _failed_receipt(
        contents, path, report["execution"], report["failure"]["root_outcome"]
    )
    _failed_receipt(
        contents,
        path / "random_forest",
        {"root_reservation_sha256": root_hash, "member": "random_forest"},
        report["failure"]["member_outcome"],
    )
    for child in report["completed_children"]:
        member = child["member"]
        marker = path / f"{member}.json"
        digest, summary, hashes = completion._receipt(
            contents,
            path / member,
            marker,
            {"root_reservation_sha256": root_hash, "member": member},
            completion._DRIFT_FILES if member == "drift" else completion._TABULAR_FILES,
        )
        _require(
            completion._keys(
                summary,
                {
                    "schema_version",
                    "status",
                    "member",
                    "root_reservation_sha256",
                    "reservation_sha256",
                    "private_sha256",
                    "result",
                },
            )
            and completion._same(
                {key: value for key, value in summary.items() if key != "result"},
                {
                    "schema_version": 1,
                    "status": "development_member_completed",
                    "member": member,
                    "root_reservation_sha256": root_hash,
                    "reservation_sha256": digest,
                    "private_sha256": hashes,
                },
            )
            and completion._same(summary, child["summary"])
            and sha256(contents[marker]).hexdigest() == child["public_summary_sha256"],
            "original_member_receipt_mismatch",
        )


@contextmanager
def audited_snapshot(
    binding: DevelopmentExecutionBinding,
    *,
    accounting_bytes: bytes,
    original_attempt: Path,
):
    """Authenticate old receipts and pin their exact files for the caller's audit.

    This checks publication identity, not the scientific contents. It opens no
    source partition or accepted model. The caller separately pins accounting
    bytes in its prospective execution profile before entering this context.
    """
    try:
        _require(type(binding) is DevelopmentExecutionBinding, "invalid_audit_binding")
        development_execution.recheck_development_binding(binding)
        report = _accounting(binding, accounting_bytes)
        path = execution_receipt._absolute_path(original_attempt)
        _require(
            not path.is_relative_to(binding.base.root),
            "original_attempt_inside_checkout",
        )
        directories = [
            (path, _RECEIPTS | set(_STEPS) | {f"{member}.json" for member in _MEMBERS}),
            (path / "random_forest", _RECEIPTS),
        ]
        for member in _MEMBERS:
            directories.extend(
                (
                    (path / member, _RECEIPTS | {"evidence"}),
                    (
                        path / member / "evidence",
                        completion._DRIFT_FILES
                        if member == "drift"
                        else completion._TABULAR_FILES,
                    ),
                )
            )
        with ExitStack() as stack:
            pinned = [
                (stack.enter_context(execution_receipt._directory(name)), expected)
                for name, expected in directories
            ]
            snapshot = []
            for directory, expected in pinned:
                completion._directory_contents(directory, expected)
                names = (
                    expected
                    - {"evidence"}
                    - (set(_STEPS) if directory.path == path else set())
                )
                for name in sorted(names):
                    state = execution_receipt._entry(directory, name)
                    _require(state is not None, "missing_retained_file")
                    snapshot.append((directory, name, source_runner._file_state(state)))
            contents = {
                directory.path / name: source_runner._read_file_once(
                    directory.path / name, expected_state=state
                )
                for directory, name, state in snapshot
            }
            _receipts(report, path, contents)
            yield report, MappingProxyType(contents)
            development_execution.recheck_development_binding(binding)
            for directory, expected in pinned:
                completion._directory_contents(directory, expected)
            for directory, name, state in snapshot:
                current = execution_receipt._entry(directory, name)
                _require(
                    current is not None and source_runner._file_state(current) == state,
                    "retained_file_changed_during_audit",
                )
    except DevelopmentAuditError:
        raise
    except Exception:
        raise DevelopmentAuditError("retained_audit_failed") from None


def _partitions(binding, train_bytes, validation_bytes, suffix_bytes):
    prepared, rules = secondary_development._preparation(
        binding.preparation_bytes, suffix_bytes, binding.pins
    )
    partitions, rows = {}, {}
    for split, content in (("train", train_bytes), ("validation", validation_bytes)):
        partitions[split] = secondary_development._partition(
            content, split, prepared["splits"][split], binding.pins, rules
        )
        rows[split] = tuple(source_runner._json(line) for line in content.splitlines())
    _require(
        not set(partitions["train"].record_ids).intersection(
            partitions["validation"].record_ids
        )
        and not set(partitions["train"].domains).intersection(
            partitions["validation"].domains
        ),
        "authenticated_training_validation_overlap",
    )
    return prepared, partitions, rows


def _drift_membership(contents, path, partitions):
    reference = completion._private_json(contents[path / "training-reference.json"])
    _require(
        reference["training_record_ids"] == list(partitions["train"].record_ids)
        and reference["training_domains"] == list(partitions["train"].domains),
        "training_membership_mismatch",
    )
    audit = completion._private_json(contents[path / "validation-audit.json"])
    validation = partitions["validation"]
    for stream in audit["streams"].values():
        positions = stream["input_row_positions"]
        _require(
            stream["record_ids"]
            == [validation.record_ids[index] for index in positions]
            and stream["domains"] == [validation.domains[index] for index in positions],
            "validation_membership_mismatch",
        )


def _tabular_scores(contents, path, identities, labels, validation):
    _require(
        identities == [row["record_id"] for row in validation],
        "validation_membership_mismatch",
    )
    _require(
        labels == [row["is_phishing"] for row in validation],
        "validation_label_mismatch",
    )
    scores = tuple(
        source_runner._json(line)["probability"]
        for line in contents[path / "validation-predictions.jsonl"].splitlines()
    )
    model = secondary_tabular.load_secondary_model_bytes(contents[path / "model.json"])
    rescored = model.score_urls(tuple(row["raw_url"] for row in validation))
    _require(rescored == scores, "saved_model_score_mismatch")
    return _score_range(scores)


def _score_range(scores):
    array = np.asarray(scores, dtype=np.float64)
    return {
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
        "mean": float(np.mean(array)),
        "population_standard_deviation": float(np.std(array, ddof=0)),
    }


def _permutations(training):
    labels = np.asarray([row["is_phishing"] for row in training], dtype=np.int64)
    original_hash = sha256(
        secondary_tabular._json_bytes({"labels": labels.tolist()})
    ).hexdigest()
    counts = {str(label): int(np.count_nonzero(labels == label)) for label in (0, 1)}
    diagnostics = []
    for seed in range(42, 47):
        permuted = np.random.Generator(np.random.PCG64(seed)).permutation(labels)
        actual = {
            str(label): int(np.count_nonzero(permuted == label)) for label in (0, 1)
        }
        _require(actual == counts, "permutation_class_count_mismatch")
        diagnostics.append(
            {
                "seed": seed,
                "bit_generator": "PCG64",
                "training_rows": len(training),
                "class_counts": counts,
                "original_labels_sha256": original_hash,
                "permuted_labels_sha256": sha256(
                    secondary_tabular._json_bytes({"labels": permuted.tolist()})
                ).hexdigest(),
                "label_agreement_count": int(np.count_nonzero(labels == permuted)),
                "fit_label_digest_status": "not_retained_by_original_producer",
            }
        )
    return diagnostics


def audit_retained(
    binding: DevelopmentExecutionBinding,
    *,
    accounting_bytes: bytes,
    original_attempt: Path,
    train_bytes: bytes,
    validation_bytes: bytes,
    suffix_bytes: bytes,
) -> dict:
    """Audit all seven completed members; no fit, selection, or old-file writes."""
    with audited_snapshot(
        binding, accounting_bytes=accounting_bytes, original_attempt=original_attempt
    ) as (report, contents):
        path = execution_receipt._absolute_path(original_attempt)
        with secondary_tabular._numerical_runtime():
            prepared, partitions, rows = _partitions(
                binding, train_bytes, validation_bytes, suffix_bytes
            )
            members = []
            for child in report["completed_children"]:
                member, summary = child["member"], child["summary"]
                evidence = path / member / "evidence"
                entry = {
                    "member": member,
                    "public_summary_sha256": child["public_summary_sha256"],
                    "summary": summary,
                }
                if member == "drift":
                    completion._drift(
                        contents,
                        evidence,
                        summary["result"],
                        summary["private_sha256"],
                        binding,
                        prepared,
                    )
                    _drift_membership(contents, evidence, partitions)
                    entry["checks"] = dict(_DRIFT_CHECKS)
                else:
                    identities, labels = completion._tabular(
                        contents, evidence, member, summary["result"], binding, prepared
                    )
                    entry["score_range"] = _tabular_scores(
                        contents, evidence, identities, labels, rows["validation"]
                    )
                    entry["checks"] = dict(_TABULAR_CHECKS)
                members.append(entry)
            result = _summary(
                binding, accounting_bytes, report, members, _permutations(rows["train"])
            )
        return result


def _summary(binding, accounting_bytes, report, members, diagnostics):
    return {
        "schema_version": 1,
        "status": "retained_development_members_audited",
        "analysis_stage": "development_validation_only",
        "protected_evaluation_authorized": False,
        "fits": 0,
        "original_aggregate_accepted": False,
        "original_accounting_sha256": sha256(accounting_bytes).hexdigest(),
        "original_execution": report["execution"],
        "original_reservation_sha256": report["receipt_and_summary_sha256"][
            "reservation.json"
        ],
        "original_worker_exit_code": None,
        "input_hashes": asdict(binding.pins),
        "members": members,
        "permutation_diagnostics": diagnostics,
        "drift_scope": _DRIFT_SCOPE,
    }


def _diagnostics(value, prepared):
    train = prepared["splits"]["train"]
    count = train["row_count"]
    expected_keys = {
        "seed",
        "bit_generator",
        "training_rows",
        "class_counts",
        "original_labels_sha256",
        "permuted_labels_sha256",
        "label_agreement_count",
        "fit_label_digest_status",
    }
    _require(type(value) is list and len(value) == 5, "invalid_permutation_diagnostics")
    original_digest = None
    for seed, diagnostic in zip(range(42, 47), value, strict=True):
        _require(
            completion._keys(diagnostic, expected_keys),
            "invalid_permutation_diagnostics",
        )
        expected = {
            "seed": seed,
            "bit_generator": "PCG64",
            "training_rows": count,
            "class_counts": train["class_counts"],
            "fit_label_digest_status": "not_retained_by_original_producer",
        }
        _require(
            completion._same({key: diagnostic[key] for key in expected}, expected),
            "permutation_diagnostic_identity_mismatch",
        )
        for key in ("original_labels_sha256", "permuted_labels_sha256"):
            _require(
                type(diagnostic[key]) is str
                and execution_receipt._SHA256.fullmatch(diagnostic[key]) is not None,
                "invalid_permutation_label_digest",
            )
        current_digest = diagnostic["original_labels_sha256"]
        _require(
            original_digest is None or original_digest == current_digest,
            "original_label_digest_mismatch",
        )
        original_digest = current_digest
        agreement = diagnostic["label_agreement_count"]
        minimum_agreement = max(0, 2 * max(train["class_counts"].values()) - count)
        _require(
            type(agreement) is int
            and minimum_agreement <= agreement <= count
            and (count - agreement) % 2 == 0,
            "invalid_permutation_agreement_count",
        )


def validate_audit_summary(
    summary: dict,
    *,
    binding: DevelopmentExecutionBinding,
    accounting_bytes: bytes,
    original_attempt: Path,
) -> dict:
    """Check the worker's public result against saved receipts and arithmetic.

    The supervisor must also observe a zero audit-worker exit and authenticate
    that worker's publication. This deliberately does not repeat source reads,
    URL scoring or permutation reconstruction; those claims remain bound to the
    authenticated audit producer, not independently replayed by this verifier.
    """
    with audited_snapshot(
        binding, accounting_bytes=accounting_bytes, original_attempt=original_attempt
    ) as (report, contents):
        with secondary_tabular._numerical_runtime():
            prepared = baselines._validate_preparation_summary(
                source_runner._json(binding.preparation_bytes)
            )
            _require(type(summary) is dict, "invalid_audit_summary")
            diagnostics = summary["permutation_diagnostics"]
            _diagnostics(diagnostics, prepared)
            path = execution_receipt._absolute_path(original_attempt)
            members, drift_ids, previous = [], None, None
            for child in report["completed_children"]:
                member, original = child["member"], child["summary"]
                evidence = path / member / "evidence"
                entry = {
                    "member": member,
                    "public_summary_sha256": child["public_summary_sha256"],
                    "summary": original,
                }
                if member == "drift":
                    drift_ids = completion._drift(
                        contents,
                        evidence,
                        original["result"],
                        original["private_sha256"],
                        binding,
                        prepared,
                    )
                    entry["checks"] = dict(_DRIFT_CHECKS)
                else:
                    observed = completion._tabular(
                        contents,
                        evidence,
                        member,
                        original["result"],
                        binding,
                        prepared,
                    )
                    _require(
                        observed[0] == drift_ids
                        and (previous is None or observed == previous),
                        "retained_validation_alignment_mismatch",
                    )
                    previous = observed
                    entry["checks"] = dict(_TABULAR_CHECKS)
                    entry["score_range"] = _score_range(
                        [
                            source_runner._json(line)["probability"]
                            for line in contents[
                                evidence / "validation-predictions.jsonl"
                            ].splitlines()
                        ]
                    )
                members.append(entry)
            expected = _summary(binding, accounting_bytes, report, members, diagnostics)
            _require(completion._same(summary, expected), "audit_summary_mismatch")
        return summary
