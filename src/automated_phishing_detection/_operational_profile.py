"""Bound public candidate metadata; unresolved choices keep execution closed."""

import json
from dataclasses import asdict, dataclass, field
from hashlib import sha256

from . import _operational_cell_protocol as protocol
from . import _study_profile as study
from . import execution_preflight as preflight
from ._checkpoint_codec import canonical_bytes
from ._external_records_validation import pins
from ._external_source_profile import _execution
from .operational_schedule import planned_cells, validate_cell

_REQUIRED = (
    "data/operational-workloads-v1.json",
    "data/http-replay-contract-v1.json",
    "data/shift-execution-contract-v1.json",
    "data/evaluation-manifest-contract-v1.json",
    "src/automated_phishing_detection/_operational_profile.py",
    "src/automated_phishing_detection/_operational_cell_protocol.py",
    "src/automated_phishing_detection/operational_schedule.py",
    protocol.SERVICE_SCRIPT,
    protocol.CLIENT_SCRIPT,
    "pyproject.toml",
    "uv.lock",
    *study.REQUIRED,
)


class OperationalProfileError(ValueError):
    """Public operational metadata could not be authenticated."""


@dataclass(frozen=True)
class CandidateOperationalProfile:
    """Immutable candidate bytes; construction never establishes authority."""

    canonical_bytes: bytes = field(repr=False)

    def projection(self) -> dict:
        return json.loads(self.canonical_bytes)

    @property
    def profile_sha256(self) -> str:
        return sha256(self.canonical_bytes).hexdigest()

    @property
    def protected_evaluation_ready(self) -> bool:
        return False


def _require(condition):
    if not condition:
        raise OperationalProfileError("invalid_operational_profile")


def _schedule():
    cells = planned_cells()
    _require(tuple(cell.ordinal for cell in cells) == tuple(range(1, 126)))
    return {
        "cells": [asdict(validate_cell(cell)) for cell in cells],
        "reference_ordinal": 1,
        "primary_http_ordinals": list(range(21, 26)),
    }


def _retention():
    return {
        "protocol": protocol.PROTOCOL,
        "working_names": protocol.WORKING_NAMES,
        "private_output_names": protocol.PRIVATE_NAMES,
        "snapshot_names": protocol.SNAPSHOT_NAMES,
    }


def _commands():
    return {
        "service": {
            "script": protocol.SERVICE_SCRIPT,
            "arguments": protocol.SERVICE_ARGUMENTS,
        },
        "client": {
            "script": protocol.CLIENT_SCRIPT,
            "arguments": protocol.COMMON_ARGUMENTS,
        },
    }


def _unadopted():
    return {
        "protective_deadlines_seconds": dict.fromkeys(
            ("startup", "shutdown", "terminate", "kill")
        ),
        "session_exclusivity": "pending_review",
        "pre_prediction_policy": {
            "scope": "whole_study",
            "dataset_check": "both_datasets_once_before_predictions",
            "missing_promised_task": "hold_entire_study",
            "status": "selected_for_review_not_adopted",
        },
    }


def _projection(binding):
    bound = pins(binding)
    _require(set(_REQUIRED) <= set(bound))
    return {
        "schema_version": 1,
        "profile_id": "operational-candidate-v1",
        "status": "incomplete_closed_candidate",
        "protected_evaluation_ready": False,
        "protected_evaluation_authorized": False,
        "execution": _execution(binding, bound),
        "bound_file_sha256": bound,
        "schedule": _schedule(),
        "commands": _commands(),
        "retention": _retention(),
        "whole_study": study.projection(),
        "manifest_sha256": {
            "http": "original_compact_sorted_utf8_json_without_newline",
            "shift": "exact_retained_test_jsonl_bytes_in_retained_order",
        },
        **_unadopted(),
    }


def resolve_operational_profile(
    binding: preflight.ExecutionBinding,
) -> CandidateOperationalProfile:
    """Authenticate only public metadata, before and after candidate construction."""
    try:
        _require(type(binding) is preflight.ExecutionBinding)
        preflight.recheck_binding(binding)
        result = CandidateOperationalProfile(canonical_bytes(_projection(binding)))
        preflight.recheck_binding(binding)
        return result
    except (ValueError, TypeError, KeyError, OSError, RecursionError):
        raise OperationalProfileError("invalid_operational_profile") from None
