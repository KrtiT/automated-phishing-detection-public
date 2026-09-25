"""External carriers do not coerce controls into internal binary records."""

import importlib
import importlib.util
from dataclasses import FrozenInstanceError, fields
from types import SimpleNamespace

import pytest


def test_external_carrier_preserves_nullable_label_and_separate_primary_scores():
    name = "automated_phishing_detection.external_evidence_types"
    assert importlib.util.find_spec(name), "missing external evidence carriers"
    module = importlib.import_module(name)
    record, primary = SimpleNamespace(is_phishing=None, role="tranco"), object()
    row = module.ScoredExternalRow(
        record, primary, (), (), (0.0,) * 26, 0.2, 0, False, False
    )
    assert row.record is record and row.record.is_phishing is None
    assert row.primary is primary
    assert tuple(field.name for field in fields(row)) == (
        "record",
        "primary",
        "secondary_tabular",
        "secondary_seeds",
        "standardized_monitor_features",
        "policy_probability",
        "policy_decision",
        "drift_override",
        "logical_stage2_mask",
    )
    with pytest.raises(FrozenInstanceError):
        row.policy_decision = 1
