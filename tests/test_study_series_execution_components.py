"""Real component comparison within public binding, with invented preflight only."""

from dataclasses import replace
from pathlib import Path

import pytest
from study_series_adoption_fixtures import make_case, refresh
from study_series_components_fixtures import arguments, case
from study_series_execution_fixtures import api, bind

from automated_phishing_detection import execution_preflight as preflight


def setup(tmp_path, monkeypatch):
    selected = case()
    expected = arguments(selected)
    envelope = make_case()
    envelope.envelope["profile"] = selected.profile
    refresh(envelope)
    selected.envelope = envelope.envelope
    selected.envelope_path = tmp_path / "invented-envelope.json"
    module = api()
    monkeypatch.setattr(
        preflight, "bind_execution", lambda *args, **kwargs: selected.base
    )
    monkeypatch.setattr(preflight, "recheck_binding", lambda *args: None)
    monkeypatch.setattr(
        module.admission_io,
        "committed_policy",
        lambda *args: module.policy.policy_bytes(),
    )
    monkeypatch.setattr(
        module.external,
        "resolve_external_source_profile",
        lambda *args: expected["external"],
    )
    monkeypatch.setattr(
        module.operational,
        "resolve_operational_profile",
        lambda *args: expected["operational"],
    )
    _reads(selected, monkeypatch)
    return selected, expected


def _reads(selected, monkeypatch):
    original_read = preflight._read_regular

    def public_only(root, name):
        if root == selected.base.root:
            assert name == "reports/rq2-gmm-development-v1-summary.json"
            return selected.audit
        assert root == Path("/")
        assert root / name == selected.envelope_path
        return original_read(root, name)

    monkeypatch.setattr(preflight, "_read_regular", public_only)


def test_real_comparison_preserves_both_original_component_bytes(tmp_path, monkeypatch):
    selected, unused = setup(tmp_path, monkeypatch)
    result = bind(selected)
    assert result.components.original_external_bytes == selected.before[0]
    assert result.components.original_operational_bytes == selected.before[1]
    assert result.components.current_external_bytes == selected.after[0]
    assert result.components.current_operational_bytes == selected.after[1]
    assert not result.authorizes_execution
    api().recheck_series_public_execution(result)


@pytest.mark.parametrize(
    "target", ("runtime", "scope", "audit", "external", "operational")
)
def test_real_component_join_cannot_be_skipped(tmp_path, monkeypatch, target):
    selected, expected = setup(tmp_path, monkeypatch)
    if target == "runtime":
        selected.base = replace(selected.base, runtime_json='{"changed":true}')
    elif target == "scope":
        selected.base = replace(
            selected.base, source_hashes=selected.base.source_hashes[:-1]
        )
    elif target == "audit":
        selected.audit = b"changed"
    else:
        expected[target] = replace(expected[target], canonical_bytes=b"changed")
    with pytest.raises(api().SeriesPublicBindingError):
        bind(selected)
