"""Malformed public claims and failed authentication never open a candidate."""

import pytest
from operational_profile_fixtures import REQUIRED, api, profile_case, resolve

from automated_phishing_detection import execution_preflight as preflight

__all__ = ["profile_case"]


@pytest.mark.parametrize("name", REQUIRED)
def test_missing_required_public_identity_is_rejected(profile_case, name):
    hashes = profile_case.hashes.copy()
    del hashes[name]
    with pytest.raises(api().OperationalProfileError):
        resolve(profile_case, source_hashes=tuple(sorted(hashes.items())))
    assert len(profile_case.checks) == 1


@pytest.mark.parametrize("value", [None, {}, [], True, "binding"])
def test_binding_requires_exact_type_before_authentication(profile_case, value):
    with pytest.raises(api().OperationalProfileError):
        api().resolve_operational_profile(value)
    assert profile_case.checks == []


@pytest.mark.parametrize(
    "field,value",
    [
        ("revision", "F" * 40),
        ("revision", True),
        ("contract_sha256", "short"),
        ("contract_sha256", None),
        ("runtime_json", {}),
        ("runtime_json", "[]"),
        ("runtime_json", '{"duplicate":1,"duplicate":1}'),
        ("runtime_json", '{ "noncanonical":true}'),
        ("runtime_json", '{"invalid":NaN}'),
    ],
)
def test_invalid_execution_projection_cannot_enter_candidate(
    profile_case, field, value
):
    with pytest.raises(api().OperationalProfileError):
        resolve(profile_case, **{field: value})


@pytest.mark.parametrize("value", [True, None, "F" * 64, "short", 1])
def test_any_malformed_bound_hash_is_rejected(profile_case, value):
    hashes = profile_case.hashes | {"other_bound_public_file": value}
    with pytest.raises(api().OperationalProfileError):
        resolve(profile_case, source_hashes=tuple(sorted(hashes.items())))


@pytest.mark.parametrize(
    "entries", [[], {}, (("name",),), ((True, "a" * 64),), (["name", "a" * 64],)]
)
def test_source_identity_entries_require_exact_types(profile_case, entries):
    with pytest.raises(api().OperationalProfileError):
        resolve(profile_case, source_hashes=entries)


@pytest.mark.parametrize("different", [False, True])
def test_duplicate_source_names_never_collapse(profile_case, different):
    name, digest = profile_case.binding.source_hashes[0]
    duplicate = (name, "e" * 64 if different else digest)
    with pytest.raises(api().OperationalProfileError):
        resolve(
            profile_case, source_hashes=(*profile_case.binding.source_hashes, duplicate)
        )


@pytest.mark.parametrize("failed", [1, 2])
@pytest.mark.parametrize("error", [OSError("private"), ValueError("private")])
def test_before_and_after_authentication_failures_are_symbolic(
    profile_case, monkeypatch, failed, error
):
    checks = []

    def reject(binding):
        checks.append(binding)
        if len(checks) == failed:
            raise error

    monkeypatch.setattr(preflight, "recheck_binding", reject)
    with pytest.raises(api().OperationalProfileError) as caught:
        resolve(profile_case)
    assert str(caught.value) == "invalid_operational_profile"
    assert len(checks) == failed


@pytest.mark.parametrize("original", [KeyboardInterrupt("first"), SystemExit(17)])
def test_non_exception_authentication_interruption_is_not_rewritten(
    profile_case, monkeypatch, original
):
    def interrupt(binding):
        raise original

    monkeypatch.setattr(preflight, "recheck_binding", interrupt)
    with pytest.raises(BaseException) as caught:
        resolve(profile_case)
    assert caught.value is original


@pytest.mark.parametrize(
    "field,value",
    [("revision", "c" * 40), ("contract_sha256", "c" * 64), ("runtime_json", "{}")],
)
def test_authenticated_execution_change_always_changes_profile_digest(
    profile_case, field, value
):
    original = resolve(profile_case)
    assert (
        resolve(profile_case, **{field: value}).profile_sha256
        != original.profile_sha256
    )
