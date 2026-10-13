"""One five-seed diagnostic schedule using existing receipt and fit machinery.

This in-memory worker does not grant access, read datasets, or lift a whole-study
hold. Its authorized parent supplies the live admission recheck and frozen bytes.
No resume, replacement seed, selection on results, or standalone CLI is provided.
"""

import re
from dataclasses import asdict
from hashlib import sha256

from . import audited_permutation_evidence as evidence
from . import audited_permutation_member as member
from . import execution_receipt, secondary_development, secondary_tabular

SEEDS = tuple(range(42, 47))


def _retain(attempt, name, record):
    with execution_receipt._attempt_directory(attempt) as directory:
        for marker in ("finalize.claim", "outcome.json"):
            execution_receipt._require_absent(directory, marker)
        execution_receipt._install_record(
            directory, name, secondary_tabular._json_bytes(record)
        )


def run_reserved_family(
    *,
    attempt,
    public_path,
    pins,
    preparation_summary,
    suffix_rules,
    train_content,
    validation_content,
    protocol_sha256,
    accepted_comparison,
    accepted_comparison_sha256,
    require_admission,
):
    started = False
    active_seed = None
    results = []
    stage = "reservation"
    try:
        if not callable(require_admission):
            raise member.DiagnosticError("missing_live_admission_recheck")
        if (
            type(protocol_sha256) is not str
            or re.fullmatch(r"[0-9a-f]{64}", protocol_sha256) is None
        ):
            raise member.DiagnosticError("invalid_diagnostic_protocol_identity")
        if (
            type(accepted_comparison) is not bytes
            or sha256(accepted_comparison).hexdigest() != accepted_comparison_sha256
        ):
            raise member.DiagnosticError("accepted_comparison_hash_mismatch")
        _retain(
            attempt,
            "family-started.json",
            {
                "scheduled_seeds": list(SEEDS),
                "protocol_sha256": protocol_sha256,
                "accepted_comparison_sha256": accepted_comparison_sha256,
                "input_pins": asdict(pins),
            },
        )
        started = True
        stage = "admission"
        require_admission()
        stage = "output_preflight"
        member._output_preflight(attempt, public_path)
        with execution_receipt._attempt_directory(attempt) as directory:
            for seed in SEEDS:
                execution_receipt._require_absent(directory, f"seed-{seed}")
                execution_receipt._require_absent(
                    directory, f"seed-{seed}-summary.json"
                )
        stage = "input_validation"
        member._validated_rows(
            train_content, validation_content, preparation_summary, suffix_rules, pins
        )
        for seed in SEEDS:
            stage = "admission"
            require_admission()
            active_seed = seed
            stage = "member"
            child = execution_receipt.reserve_attempt(
                attempt.directory / f"seed-{seed}",
                identity={
                    "kind": "post_review_permutation_diagnostic",
                    "seed": seed,
                    "parent_reservation_sha256": attempt.reservation_sha256,
                    "protocol_sha256": protocol_sha256,
                    "input_pins": asdict(pins),
                },
            )
            arguments = {
                "seed": seed,
                "attempt": child,
                "public_path": attempt.directory / f"seed-{seed}-summary.json",
                "pins": pins,
                "preparation_summary": preparation_summary,
                "suffix_rules": suffix_rules,
                "train_content": train_content,
                "validation_content": validation_content,
            }
            result = member.run_reserved_member(
                **arguments, require_admission=require_admission
            )
            stage = "evidence_verification"
            verified = evidence.verify_completed_member(**arguments)
            summary = secondary_development._json(evidence._read_file_once(result))
            results.append(
                {"seed": seed, "result": summary["result"], "verification": verified}
            )
            active_seed = None
        stage = "admission"
        require_admission()
        stage = "publication"
        private = {
            "accepted-comparison.json": accepted_comparison,
            "members.json": secondary_tabular._json_bytes({"members": results}),
        }
        summary = {
            "schema_version": 1,
            "status": "audited_permutation_family_completed",
            "analysis_stage": "post_review_new_fit_diagnostic",
            "historical_fit_verified": False,
            "protected_evaluation_authorized": False,
            "scheduled_seeds": list(SEEDS),
            "protocol_sha256": protocol_sha256,
            "input_pins": asdict(pins),
            "accepted_comparison_sha256": accepted_comparison_sha256,
            "comparison_role": "retained_unshuffled_reference_not_refitted",
            "reservation_sha256": attempt.reservation_sha256,
            "members": results,
            "private_sha256": {
                name: sha256(content).hexdigest() for name, content in private.items()
            },
        }
        return execution_receipt.publish_completion(
            attempt,
            private_outputs=private,
            public_summary=summary,
            public_path=public_path,
        )
    except BaseException as error:
        if started:
            completed = [result["seed"] for result in results]
            try:
                with execution_receipt._attempt_directory(attempt) as directory:
                    claimed = (
                        execution_receipt._entry(directory, "finalize.claim")
                        is not None
                    )
                if not claimed:
                    _retain(
                        attempt,
                        "family-stopped.json",
                        {
                            "stage": stage,
                            "completed_seeds": completed,
                            "failed_seed": active_seed,
                            "unstarted_seeds": [
                                seed
                                for seed in SEEDS
                                if seed not in completed and seed != active_seed
                            ],
                        },
                    )
                member._record_unclaimed_failure(attempt, stage, error)
            except BaseException as failure:
                raise member.DiagnosticError(
                    "diagnostic_family_failure_record_incomplete"
                ) from failure
        raise member.DiagnosticError(f"diagnostic_family_{stage}_failed") from error
