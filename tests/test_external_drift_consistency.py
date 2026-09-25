from dataclasses import replace

import pytest
from external_composition_fixtures import composition_inputs, producer_module


@pytest.mark.parametrize(
    "field",
    [
        "scaler_mean",
        "scaler_scale",
        "portable_state_sha256",
        "mmd_calibration",
        "psi_calibration",
        "audit_domains",
    ],
)
def test_replay_state_must_match_exact_retained_snapshot_before_scoring(
    monkeypatch, field
):
    module = producer_module()
    prepared, session, primary_calls, _ = composition_inputs(monkeypatch)
    reference = session.drift.reference
    if field in ("scaler_mean", "scaler_scale"):
        value = tuple(number + 1.0 for number in getattr(reference, field))
    elif field == "portable_state_sha256":
        value = "f" * 64
    elif field == "audit_domains":
        value = ("changed.example", *reference.audit_domains[1:])
    else:
        calibration = getattr(reference, field)
        value = replace(calibration, threshold=calibration.threshold + 1.0)
    reference = replace(reference, **{field: value})
    session = replace(session, drift=replace(session.drift, reference=reference))
    with pytest.raises(module.ExternalProducerError):
        module.produce_external_evidence(prepared, session)
    assert session.evaluation.primary.scorer.urls == []
    assert primary_calls == [[], [], []]
