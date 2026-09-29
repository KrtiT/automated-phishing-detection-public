"""Historical holder cleanup preserves rejection and interruption associations."""

from dataclasses import replace

import pytest
from retained_study_hold_fixtures import (
    api,
    execution_case,
    manager,
    prepared,
    prior_hold,
)

__all__ = ["execution_case", "prepared", "prior_hold"]


def test_original_body_exception_is_preserved_without_cleanup_failure(prior_hold):
    original = ValueError("invented body failure")
    with pytest.raises(ValueError) as caught:
        with manager(prior_hold):
            raise original
    assert caught.value is original


def test_cleanup_rejection_retains_original_failure_association(prior_hold):
    original = ValueError("invented body failure")
    original.preparation_progress = b"invented progress"
    with pytest.raises(ValueError) as caught:
        with manager(prior_hold):
            prior_hold.authorization.paths.public_summary.write_bytes(b"changed")
            raise original
    assert caught.value is not original
    assert caught.value.preparation_progress == b"invented progress"


def test_interruption_survives_cleanup_rejection(prior_hold):
    original = KeyboardInterrupt("invented interruption")
    with pytest.raises(KeyboardInterrupt) as caught:
        with manager(prior_hold):
            prior_hold.authorization.paths.public_summary.write_bytes(b"changed")
            raise original
    assert caught.value is original


@pytest.mark.parametrize("field", ("runtime_sha256", "execution_contract_sha256"))
def test_current_identity_mismatch_rejects(prior_hold, monkeypatch, field):
    original = api().bound_preparation_context

    def changed(binding):
        identity, source, buffers, profile = original(binding)
        return identity | {field: "0" * 64}, source, buffers, profile

    monkeypatch.setattr(api(), "bound_preparation_context", changed)
    with pytest.raises(ValueError), manager(prior_hold):
        pytest.fail("different runtime or contract was accepted")


def test_current_binding_revision_mismatch_rejects(prior_hold):
    prior_hold.binding = replace(prior_hold.binding, revision="0" * 40)
    with pytest.raises(ValueError), manager(prior_hold):
        pytest.fail("inconsistent current binding revision was accepted")


@pytest.mark.parametrize("directory", ("attempt", "evidence"))
def test_nonprivate_historical_directory_rejects(prior_hold, directory):
    root = prior_hold.authorization.paths.attempt
    selected = root if directory == "attempt" else root / "evidence"
    selected.chmod(0o755)
    with pytest.raises(ValueError), manager(prior_hold):
        pytest.fail("nonprivate historical directory was accepted")


def test_source_completion_file_cannot_turn_hold_into_success(prior_hold):
    root = prior_hold.authorization.paths.attempt
    (root / "source-results.json").write_bytes(b"{}")
    with pytest.raises(ValueError), manager(prior_hold):
        pytest.fail("success-shaped root was accepted as a hold")
