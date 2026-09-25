"""Bind the prospective zero-fit probe correction to immutable public history."""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from . import development_execution, execution_preflight, seed_probe_execution
from .seed_probe_execution import (
    SeedProbeExecutionBinding,
    bind_seed_probe_execution,
    recheck_seed_probe_binding,
)
from .source_runner import _json

PROFILE_PATH = "data/seed-probe-correction-contract-v1.json"
PROFILE_SHA256 = "46a659ddc809f998a12abaae0f5c6e353c2f965844487faf87c0637b9d7697a4"
BASE_PROFILE_SHA256 = seed_probe_execution.PROFILE_SHA256
ACCOUNTING_PATH = "reports/secondary-seed-probe-v2-attempt-1.json"
ACCOUNTING_SHA256 = "cf1fc0e6e41839464def2955b4475492b4839e637d4e563d4c94324057e74cb4"
METHODS_SHA256 = seed_probe_execution.METHODS_SHA256
HISTORY_PINS = {
    seed_probe_execution.PROFILE_PATH: BASE_PROFILE_SHA256,
    seed_probe_execution.BASE_EXECUTION_PROFILE_PATH: (
        seed_probe_execution.BASE_EXECUTION_PROFILE_SHA256
    ),
    seed_probe_execution.STOPPED_ATTEMPT_PATH: (
        seed_probe_execution.STOPPED_ATTEMPT_SHA256
    ),
    seed_probe_execution.METHODS_PATH: METHODS_SHA256,
    development_execution.METHODS_PATH: development_execution.METHODS_SHA256,
    seed_probe_execution.ACCEPTED_PATH: seed_probe_execution.ACCEPTED_SHA256,
}


class SeedProbeCorrectionError(ValueError):
    """A safe symbolic failure of the prospective correction binding."""


@dataclass(frozen=True)
class SeedProbeCorrectionBinding:
    seed_probe: SeedProbeExecutionBinding
    profile_sha256: str
    accounting_bytes: bytes

    @property
    def base(self):
        return self.seed_probe.base

    @property
    def protected_evaluation_ready(self):
        return False


def _require(condition, symbol):
    if not condition:
        raise SeedProbeCorrectionError(symbol)


def validate_profile(value) -> None:
    """Freeze the complete correction policy, including all limitations."""
    try:
        content = (
            json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False) + "\n"
        ).encode("ascii")
    except (ValueError, TypeError, UnicodeError):
        raise SeedProbeCorrectionError("invalid_profile_policy") from None
    _require(
        type(value) is dict and sha256(content).hexdigest() == PROFILE_SHA256,
        "invalid_profile_policy",
    )


def bind_seed_probe_correction(
    root: Path, *, expected_revision: str, expected_profile_sha256: str
) -> SeedProbeCorrectionBinding:
    """Authenticate public metadata without accepting any research-data path."""
    _require(expected_profile_sha256 == PROFILE_SHA256, "profile_hash_mismatch")
    seed_probe = bind_seed_probe_execution(
        root,
        expected_revision=expected_revision,
        expected_profile_sha256=BASE_PROFILE_SHA256,
    )
    base = seed_probe.base
    expected = {
        PROFILE_PATH: PROFILE_SHA256,
        ACCOUNTING_PATH: ACCOUNTING_SHA256,
        **HISTORY_PINS,
    }
    execution_preflight._historical_v2_committed_files(
        base.root, base.revision, expected
    )
    profile = execution_preflight._read_regular(base.root, PROFILE_PATH)
    _require(sha256(profile).hexdigest() == PROFILE_SHA256, "profile_hash_mismatch")
    try:
        validate_profile(_json(profile))
    except SeedProbeCorrectionError:
        raise
    except (ValueError, TypeError, UnicodeError):
        raise SeedProbeCorrectionError("invalid_profile_policy") from None
    accounting = execution_preflight._read_regular(base.root, ACCOUNTING_PATH)
    _require(
        sha256(accounting).hexdigest() == ACCOUNTING_SHA256,
        "accounting_hash_mismatch",
    )
    recheck_seed_probe_binding(seed_probe)
    for relative, digest in expected.items():
        current = execution_preflight._read_regular(base.root, relative)
        _require(
            sha256(current).hexdigest() == digest,
            "seed_probe_correction_binding_changed",
        )
    return SeedProbeCorrectionBinding(seed_probe, PROFILE_SHA256, accounting)


def recheck_seed_probe_correction(binding: SeedProbeCorrectionBinding) -> None:
    """Rebind every public input and reject any correction identity change."""
    _require(
        type(binding) is SeedProbeCorrectionBinding,
        "invalid_seed_probe_correction_binding",
    )
    current = bind_seed_probe_correction(
        binding.base.root,
        expected_revision=binding.base.revision,
        expected_profile_sha256=binding.profile_sha256,
    )
    _require(current == binding, "seed_probe_correction_binding_changed")
