from dataclasses import FrozenInstanceError, replace

import pytest
from study_series_execution_fixtures import api, bind, public_case, seal

from automated_phishing_detection import execution_preflight as preflight

__all__ = ["public_case"]


def test_shared_public_binding_remains_non_authorizing(public_case):
    result = bind(public_case)
    assert result.authorizes_execution is False
    assert result.base is public_case.base
    assert result.components is public_case.components
    assert result.external is public_case.external
    assert result.operational is public_case.operational
    assert public_case.events == ["preflight", "recheck"]
    result.deadlines["startup"] = 1
    assert result.deadlines["startup"] == 300
    with pytest.raises(FrozenInstanceError):
        result.authorizes_execution = True


@pytest.mark.parametrize(
    "field",
    ("expected_profile_sha256", "expected_envelope_sha256", "expected_revision"),
)
def test_independent_wrong_pin_rejects_before_preflight(public_case, field):
    with pytest.raises(api().SeriesPublicBindingError):
        bind(public_case, **{field: "0" * (40 if field.endswith("revision") else 64)})
    assert public_case.events == []


def test_development_directive_never_reaches_preflight(public_case):
    public_case.envelope["operator_directive"]["scope"] = "development_only"
    with pytest.raises(api().SeriesPublicBindingError):
        bind(public_case)
    assert public_case.events == []


def test_wrong_returned_base_rejects(public_case, monkeypatch):
    monkeypatch.setattr(
        preflight,
        "bind_execution",
        lambda *args, **kwargs: replace(public_case.base, revision="c" * 40),
    )
    with pytest.raises(api().SeriesPublicBindingError):
        bind(public_case)


def test_recheck_reauthenticates_original_pin_and_record(public_case):
    result = bind(public_case)
    api().recheck_series_public_execution(result)
    with pytest.raises(api().SeriesPublicBindingError):
        api().recheck_series_public_execution(replace(result, policy_bytes=b"changed"))
    public_case.envelope_path.write_bytes(b"changed")
    with pytest.raises(api().SeriesPublicBindingError):
        api().recheck_series_public_execution(result)


def test_root_must_be_exact_without_alias_normalization(public_case):
    with pytest.raises(api().SeriesPublicBindingError):
        api().bind_series_public_execution(
            str(public_case.base.root) + "/../repository", **seal(public_case)
        )
    assert public_case.events == []
