from dataclasses import replace

import pytest
from study_series_execution_fixtures import api, bind, digest, public_case, seal

from automated_phishing_detection import execution_preflight as preflight
from automated_phishing_detection._checkpoint_codec import canonical_bytes

__all__ = ["public_case"]


@pytest.mark.parametrize("content", (b"not-json", b"[]", b"null", b'{"profile":null}'))
def test_pinned_but_malformed_envelope_rejects_before_preflight(public_case, content):
    arguments = seal(public_case)
    public_case.envelope_path.write_bytes(content)
    arguments["expected_envelope_sha256"] = digest(content)
    with pytest.raises(api().SeriesPublicBindingError):
        api().bind_series_public_execution(public_case.base.root, **arguments)
    assert public_case.events == []


@pytest.mark.parametrize("operation", ("space", "duplicate", "null_profile"))
def test_noncanonical_or_wrong_nested_profile_cannot_reach_preflight(
    public_case, operation
):
    arguments = seal(public_case)
    content = canonical_bytes(public_case.envelope)
    content = {
        "space": content + b" ",
        "duplicate": b'{"revoked":false,' + content[1:],
        "null_profile": canonical_bytes(public_case.envelope | {"profile": None}),
    }[operation]
    public_case.envelope_path.write_bytes(content)
    arguments["expected_envelope_sha256"] = digest(content)
    with pytest.raises(api().SeriesPublicBindingError):
        api().bind_series_public_execution(public_case.base.root, **arguments)
    assert public_case.events == []


@pytest.mark.parametrize("stage", ("preflight", "components", "recheck", "policy"))
def test_ordinary_failure_is_symbolic_and_interrupt_is_preserved(
    public_case, monkeypatch, stage
):
    owner, name = {
        "preflight": (preflight, "bind_execution"),
        "components": (api(), "_components"),
        "recheck": (preflight, "recheck_binding"),
        "policy": (api().admission_io, "committed_policy"),
    }[stage]
    for error in (RuntimeError("invented private diagnostic"), KeyboardInterrupt()):

        def fail(*args, **kwargs):
            raise error

        monkeypatch.setattr(owner, name, fail)
        expected = (
            api().SeriesPublicBindingError
            if isinstance(error, Exception)
            else KeyboardInterrupt
        )
        with pytest.raises(expected) as caught:
            bind(public_case)
        if isinstance(error, Exception):
            assert str(caught.value) == "invalid_series_public_binding"
            assert caught.value.__suppress_context__
        else:
            assert caught.value is error


@pytest.mark.parametrize(
    "field,value",
    (
        ("revision", "c" * 40),
        ("contract_sha256", "c" * 64),
        ("root", "/invented/alias"),
    ),
)
def test_wrong_base_identity_never_resolves_components(
    public_case, monkeypatch, field, value
):
    monkeypatch.setattr(
        preflight,
        "bind_execution",
        lambda *args, **kwargs: replace(public_case.base, **{field: value}),
    )
    monkeypatch.setattr(
        api(), "_components", lambda *args: pytest.fail("wrong base reached components")
    )
    with pytest.raises(api().SeriesPublicBindingError):
        bind(public_case)


@pytest.mark.parametrize("target", ("envelope", "policy"))
def test_drift_during_final_recheck_never_returns_binding(
    public_case, monkeypatch, target
):
    def drift(base):
        if target == "envelope":
            public_case.envelope_path.write_bytes(b"changed")
        else:
            monkeypatch.setattr(
                api().admission_io, "committed_policy", lambda *args: b"changed"
            )

    monkeypatch.setattr(preflight, "recheck_binding", drift)
    with pytest.raises(api().SeriesPublicBindingError):
        bind(public_case)


@pytest.mark.parametrize("value", (None, {}, object()))
def test_recheck_rejects_fabricated_shape(value):
    with pytest.raises(api().SeriesPublicBindingError):
        api().recheck_series_public_execution(value)
