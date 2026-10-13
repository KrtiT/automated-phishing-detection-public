"""The five-seed diagnostic is bounded, ordered, and cannot silently resume."""

import json
from hashlib import sha256

import pytest
from test_audited_permutation_member import member as member
from test_audited_permutation_member import supplied as supplied

from automated_phishing_detection import execution_receipt


@pytest.fixture
def family_arguments(supplied, tmp_path):
    return {
        **{key: value for key, value in supplied.items() if key != "seed"},
        "attempt": execution_receipt.reserve_attempt(
            tmp_path / "family", identity={"role": "synthetic_family"}
        ),
        "public_path": tmp_path / "family-summary.json",
        "protocol_sha256": "a" * 64,
        "accepted_comparison": b"synthetic frozen unshuffled comparison",
        "accepted_comparison_sha256": sha256(
            b"synthetic frozen unshuffled comparison"
        ).hexdigest(),
        "require_admission": lambda: None,
    }


def _run(arguments):
    from automated_phishing_detection import audited_permutation_family

    return audited_permutation_family.run_reserved_family(**arguments)


def test_family_completes_every_seed_once_with_independent_readback(family_arguments):
    calls = []
    family_arguments["require_admission"] = lambda: calls.append(True)
    public = _run(family_arguments)
    summary = json.loads(public.read_bytes())
    assert summary["scheduled_seeds"] == list(range(42, 47))
    assert [item["seed"] for item in summary["members"]] == list(range(42, 47))
    assert all(
        item["verification"]["status"] == "evidence_consistent"
        for item in summary["members"]
    )
    assert len(calls) >= 7
    assert summary["historical_fit_verified"] is False
    root = family_arguments["attempt"].directory
    assert (
        root / "evidence/accepted-comparison.json"
    ).read_bytes() == family_arguments["accepted_comparison"]
    with pytest.raises(ValueError):
        _run(family_arguments)


def test_admission_is_checked_before_any_fit(family_arguments, member, monkeypatch):
    def deny():
        raise ValueError("whole study held")

    family_arguments["require_admission"] = deny
    monkeypatch.setattr(
        member.development_runner,
        "_tabular",
        lambda *args, **kwargs: pytest.fail("fit while held"),
    )
    with pytest.raises(ValueError):
        _run(family_arguments)
    assert not family_arguments["public_path"].exists()
    assert not (family_arguments["attempt"].directory / "seed-42").exists()


def test_wrong_comparison_bytes_stop_before_fit(family_arguments, member, monkeypatch):
    family_arguments["accepted_comparison"] += b"modified"
    monkeypatch.setattr(
        member.development_runner,
        "_tabular",
        lambda *args, **kwargs: pytest.fail("fit with wrong comparator"),
    )
    with pytest.raises(ValueError):
        _run(family_arguments)


def test_revocation_during_member_preparation_prevents_fit(
    family_arguments, member, monkeypatch
):
    original = member._validated_rows
    calls = []
    revoked = []

    def validate(*args):
        result = original(*args)
        calls.append(True)
        if len(calls) == 2:
            revoked.append(True)
        return result

    def check():
        if revoked:
            raise ValueError("revoked while preparing member")

    fits = []
    original_fit = member.secondary_tabular.LogisticRegression.fit

    def fit(*args, **kwargs):
        fits.append(True)
        return original_fit(*args, **kwargs)

    monkeypatch.setattr(member, "_validated_rows", validate)
    monkeypatch.setattr(member.secondary_tabular.LogisticRegression, "fit", fit)
    family_arguments["require_admission"] = check
    with pytest.raises(ValueError):
        _run(family_arguments)
    assert fits == []


def test_mid_family_failure_keeps_completed_failed_and_unstarted_members(
    family_arguments, member, monkeypatch
):
    original = member.run_reserved_member
    seen = []

    def run(**kwargs):
        seen.append(kwargs["seed"])
        if kwargs["seed"] == 44:
            raise ValueError("private scientific failure")
        return original(**kwargs)

    monkeypatch.setattr(member, "run_reserved_member", run)
    with pytest.raises(ValueError):
        _run(family_arguments)
    assert seen == [42, 43, 44]
    root = family_arguments["attempt"].directory
    ledger = json.loads((root / "family-stopped.json").read_bytes())
    assert ledger["completed_seeds"] == [42, 43]
    assert ledger["failed_seed"] == 44
    assert ledger["unstarted_seeds"] == [45, 46]
    assert (root / "seed-42/evidence/model.json").exists()
    assert (root / "seed-43/evidence/model.json").exists()
    assert not family_arguments["public_path"].exists()
    with pytest.raises(ValueError):
        _run(family_arguments)
    assert seen == [42, 43, 44]
