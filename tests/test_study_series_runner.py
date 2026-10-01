"""The private root composes held history and rejects a nonadoptable policy."""

import asyncio
import json
from dataclasses import replace
from importlib import import_module
from importlib.util import find_spec

import pytest
from study_series_execution_fixtures import bind, public_case

from automated_phishing_detection._checkpoint_codec import canonical_bytes

__all__ = ["public_case"]


def api():
    name = "automated_phishing_detection.study_series_runner"
    assert find_spec(name), "missing private series root orchestration"
    return import_module(name)


def test_candidate_policy_rejects_before_history_or_output_access(
    public_case, monkeypatch
):
    module, public = api(), bind(public_case)
    candidate = json.loads(public.policy_bytes) | {
        "status": "development_candidate_header_only"
    }
    public = replace(public, policy_bytes=canonical_bytes(candidate))

    def forbidden(*args):
        pytest.fail("candidate accessed protected history or outputs")

    monkeypatch.setattr(module, "hold_outputs", forbidden)
    with pytest.raises(ValueError, match="invalid_series_adoption_header"):
        asyncio.run(
            module._run_series_bound(
                public,
                hold_history=forbidden,
                lifecycle_check=lambda: None,
            )
        )


def test_private_root_exposes_no_standalone_execution_entry():
    module = api()
    assert not hasattr(module, "run_study_series")
    assert not hasattr(module, "run_series")
