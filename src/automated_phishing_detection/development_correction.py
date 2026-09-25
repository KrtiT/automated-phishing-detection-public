"""Bind the prospective RF correction and the unchanged stopped-attempt record."""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from . import execution_preflight
from .development_execution import (
    DevelopmentExecutionBinding,
    bind_development_execution,
    recheck_development_binding,
)
from .source_runner import _json

PROFILE_PATH = "data/development-correction-contract-v2.json"
PROFILE_SHA256 = "1d806d536dc77b5a085264950e64b2bde3db4ab02afa33c6adb827d8b7a07f73"
BASE_PROFILE_SHA256 = "67146228d636c16f02484998741c7a1545da68b209e693620efab22b2676cd43"
ACCOUNTING_PATH = "reports/secondary-development-v1-attempt-1.json"
ACCOUNTING_SHA256 = "c372aa5d6563a0c75d297d0178154588bfbcd93451a3f14fa06a064a84cbd02e"
HISTORY_PINS = {
    "data/development-correction-contract-v1.json": "61739fa0638ae822bf54639cf3a485c0b7b8a3a236a80221d824df7f79a090a9",
    "reports/secondary-development-correction-v1-attempt-1.json": "aa9aac8b02d611ed362069abd5b8103d863f8d01acd39d0f99292830b725ddee",
}


class CorrectionError(ValueError):
    """A symbolic correction binding or execution failure."""


@dataclass(frozen=True)
class CorrectionBinding:
    development: DevelopmentExecutionBinding
    profile_sha256: str
    accounting_bytes: bytes


def _require(condition, symbol):
    if not condition:
        raise CorrectionError(symbol)


def validate_profile(value):
    """The entire reviewed policy is fixed, including its explanatory limitations."""
    try:
        content = (
            json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False) + "\n"
        ).encode("ascii")
    except (ValueError, TypeError, UnicodeError):
        raise CorrectionError("invalid_profile_policy") from None
    _require(
        type(value) is dict and sha256(content).hexdigest() == PROFILE_SHA256,
        "invalid_profile_policy",
    )


def bind_correction(
    root: Path, *, expected_revision: str, expected_profile_sha256: str
) -> CorrectionBinding:
    """No supplied research input, original attempt or output path is accepted here."""
    _require(expected_profile_sha256 == PROFILE_SHA256, "profile_hash_mismatch")
    development = bind_development_execution(
        root,
        expected_revision=expected_revision,
        expected_profile_sha256=BASE_PROFILE_SHA256,
    )
    base = development.base
    profile = execution_preflight._read_regular(base.root, PROFILE_PATH)
    _require(
        sha256(profile).hexdigest() == expected_profile_sha256, "profile_hash_mismatch"
    )
    validate_profile(_json(profile))
    expected = {
        PROFILE_PATH: expected_profile_sha256,
        ACCOUNTING_PATH: ACCOUNTING_SHA256,
        **HISTORY_PINS,
    }
    execution_preflight._historical_v2_committed_files(
        base.root, base.revision, expected
    )
    accounting = execution_preflight._read_regular(base.root, ACCOUNTING_PATH)
    _require(
        sha256(accounting).hexdigest() == ACCOUNTING_SHA256, "accounting_hash_mismatch"
    )
    recheck_development_binding(development)
    for name, digest in expected.items():
        _require(
            sha256(execution_preflight._read_regular(base.root, name)).hexdigest()
            == digest,
            "correction_binding_changed",
        )
    return CorrectionBinding(development, expected_profile_sha256, accounting)


def recheck_correction(binding: CorrectionBinding) -> None:
    _require(type(binding) is CorrectionBinding, "invalid_correction_binding")
    current = bind_correction(
        binding.development.base.root,
        expected_revision=binding.development.base.revision,
        expected_profile_sha256=binding.profile_sha256,
    )
    _require(current == binding, "correction_binding_changed")
