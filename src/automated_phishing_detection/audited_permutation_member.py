"""One newly audited diagnostic member; caller owns scientific freeze/admission.

No data reader or standalone CLI is provided. The authorized caller supplies
hash-bound development bytes and a reserved attempt from the new diagnostic
protocol. Historical development execution and historical artifacts are unchanged.
"""

import json
from dataclasses import asdict
from hashlib import sha256

from . import (
    development_runner,
    execution_receipt,
    secondary_development,
    secondary_tabular,
)

_RECORDS = frozenset(
    {
        "member-started.json",
        "fit-call-prepared.json",
        "fit-call-returned.json",
        "fit-state.json",
    }
)


class DiagnosticError(ValueError):
    pass


def _retain(attempt, name, content):
    if name not in _RECORDS or type(content) is not bytes:
        raise DiagnosticError("invalid_receipt")
    with execution_receipt._attempt_directory(attempt) as directory:
        for marker in ("finalize.claim", "outcome.json"):
            execution_receipt._require_absent(directory, marker)
        execution_receipt._install_record(directory, name, content)


def _validated_rows(
    train_content, validation_content, preparation_summary, suffix_rules, pins
):
    prepared, rules = secondary_development._preparation(
        preparation_summary, suffix_rules, pins
    )
    partitions = [
        secondary_development._partition(
            content, split, prepared["splits"][split], pins, rules
        )
        for split, content in (
            ("train", train_content),
            ("validation", validation_content),
        )
    ]
    train, validation = partitions
    if not set(train.domains).isdisjoint(validation.domains) or not set(
        train.record_ids
    ).isdisjoint(validation.record_ids):
        raise DiagnosticError("development_identity_overlap")
    return tuple(
        secondary_development._json(line) for line in train_content.splitlines()
    ), tuple(
        secondary_development._json(line) for line in validation_content.splitlines()
    )


def _output_preflight(attempt, public_path):
    destination = execution_receipt._absolute_path(public_path)
    with execution_receipt._attempt_directory(attempt) as directory:
        if destination.is_relative_to(directory.path):
            raise DiagnosticError("public_summary_inside_attempt")
        with execution_receipt._directory(destination.parent) as parent:
            execution_receipt._require_absent(parent, destination.name)


def _record_unclaimed_failure(attempt, stage, error):
    with execution_receipt._attempt_directory(attempt) as directory:
        claimed = any(
            execution_receipt._entry(directory, name) is not None
            for name in ("finalize.claim", "outcome.json")
        )
    if not claimed:
        execution_receipt.record_failure(
            attempt, stage=stage, error_type=development_runner._failure_symbol(error)
        )


def run_reserved_member(
    *,
    seed,
    attempt,
    public_path,
    pins,
    preparation_summary,
    suffix_rules,
    train_content,
    validation_content,
    require_admission=None,
):
    started = False
    stage = "reservation"
    try:
        if type(seed) is not int or seed not in range(42, 47):
            raise DiagnosticError("invalid_permutation_seed")
        _retain(
            attempt,
            "member-started.json",
            secondary_tabular._json_bytes({"seed": seed, "input_pins": asdict(pins)}),
        )
        started = True
        stage = "output_preflight"
        _output_preflight(attempt, public_path)
        stage = "input_validation"
        train, validation = _validated_rows(
            train_content, validation_content, preparation_summary, suffix_rules, pins
        )
        retained = {}

        def record_fit(content):
            if require_admission is not None:
                require_admission()
            event = json.loads(content)["event"]
            expected = "fit_call_prepared" if not retained else "fit_call_returned"
            if event != expected:
                raise DiagnosticError("unexpected_fit_event")
            name = event.replace("_", "-") + ".json"
            _retain(attempt, name, content)
            retained[name] = content

        def record_state(content):
            if require_admission is not None:
                require_admission()
            if set(retained) != {"fit-call-prepared.json", "fit-call-returned.json"}:
                raise DiagnosticError("missing_fit_receipts")
            _retain(attempt, "fit-state.json", content)
            retained["fit-state.json"] = content

        stage = "fit"
        private, result = development_runner._tabular(
            f"permutation_{seed}",
            train,
            validation,
            fit_input_receipt=record_fit,
            checkpoint=record_state,
        )
        stage = "linkage_verification"
        if set(retained) != _RECORDS - {"member-started.json"}:
            raise DiagnosticError("missing_fit_receipts")
        if (
            json.loads(retained["fit-call-returned.json"])["prepared_receipt_sha256"]
            != sha256(retained["fit-call-prepared.json"]).hexdigest()
        ):
            raise DiagnosticError("fit_receipt_link_mismatch")
        secondary_tabular.validate_fit_checkpoint_bytes(
            retained["fit-state.json"], model_bytes=private["model.json"]
        )
        linkage = {
            "analysis_stage": "post_review_new_fit_diagnostic",
            "historical_fit_verified": False,
            "seed": seed,
            "partition_sha256": {
                "train": pins.train_sha256,
                "validation": pins.validation_sha256,
            },
            "preparation_summary_sha256": pins.preparation_summary_sha256,
            "suffix_rules_sha256": pins.suffix_rules_sha256,
            "receipt_sha256": {
                name: sha256(content).hexdigest() for name, content in retained.items()
            },
            "model_sha256": sha256(private["model.json"]).hexdigest(),
        }
        private["fit-linkage.json"] = secondary_tabular._json_bytes(linkage)
        result["source_identity_verified"] = True
        summary = {
            "schema_version": 1,
            "status": "audited_permutation_member_completed",
            "analysis_stage": "post_review_new_fit_diagnostic",
            "historical_fit_verified": False,
            "protected_evaluation_authorized": False,
            "reservation_sha256": attempt.reservation_sha256,
            "result": result,
            "private_sha256": development_runner._hashes(private),
            "receipt_sha256": linkage["receipt_sha256"],
            "input_pins": asdict(pins),
        }
        stage = "publication"
        if require_admission is not None:
            require_admission()
        return execution_receipt.publish_completion(
            attempt,
            private_outputs=private,
            public_summary=summary,
            public_path=public_path,
        )
    except BaseException as error:
        if started:
            try:
                _record_unclaimed_failure(attempt, stage, error)
            except BaseException as failure:
                raise DiagnosticError(
                    "diagnostic_failure_record_incomplete"
                ) from failure
        raise DiagnosticError(f"diagnostic_{stage}_failed") from error
