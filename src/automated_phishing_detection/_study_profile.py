"""Declare the existing whole-study candidate without granting execution access."""

REQUIRED = (
    "scripts/run_study.py",
    "scripts/run_prepared_internal_evaluation.py",
    "scripts/run_prepared_external_evaluation.py",
    *(
        f"src/automated_phishing_detection/{name}.py"
        for name in (
            "_study_cli_protocol",
            "_study_profile",
            "study_runner",
            "_study_run_body",
            "_study_run_schema",
            "_study_root_records",
            "study_preparation_retention",
            "_internal_handoff_validation",
            "_external_completion_files",
            "_external_completion_records",
        )
    ),
)


def _source_pair():
    from ._external_completion_files import PAYLOAD_NAMES
    from ._external_completion_records import _LOGICAL_NAMES
    from ._internal_handoff_validation import SNAPSHOT_NAMES

    return {
        "order": ("internal", "external"),
        "source_interface": "retained_study_preparation_v1",
        "internal": {
            "script": "scripts/run_prepared_internal_evaluation.py",
            "fixed_flags": ("--worker",),
            "snapshot_names": sorted(SNAPSHOT_NAMES),
        },
        "external": {
            "script": "scripts/run_prepared_external_evaluation.py",
            "fixed_flags": (),
            "private_output_names": sorted(PAYLOAD_NAMES),
            "snapshot_names": sorted(_LOGICAL_NAMES),
        },
    }


def _branch(checkpoints, outputs):
    return {
        "checkpoint_names": checkpoints,
        "private_output_names": outputs,
        "snapshot_names": (
            "attempt/reservation.json",
            *(f"attempt/{name}" for name in checkpoints),
            "attempt/finalize.claim",
            "attempt/outcome.json",
            *(f"attempt/evidence/{name}" for name in outputs),
            "public-summary.json",
        ),
    }


def _retention():
    from ._study_root_records import EXTRA_NAMES, HOLD_ORDER, ORDER

    return {
        "whole_study_hold": _branch(HOLD_ORDER, HOLD_ORDER),
        "study_evidence_published": _branch(ORDER, ORDER + EXTRA_NAMES),
    }


def projection():
    from ._study_cli_protocol import ARGUMENTS, SCRIPT
    from ._study_run_schema import PROTOCOL, STAGES
    from .study_preparation_retention import PREPARATION_ORDER

    return {
        "protocol": PROTOCOL,
        "command": {"script": SCRIPT, "arguments": ARGUMENTS},
        "stage_names": STAGES,
        "preparation": {
            "protocol": "study-preparation-v1",
            "reservation_name": "reservation.json",
            "payload_names": PREPARATION_ORDER,
        },
        "source_pair": _source_pair(),
        "retention": _retention(),
    }
