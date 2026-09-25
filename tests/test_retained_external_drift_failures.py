"""Mutated snapshots cannot substitute another accepted chain or leak values."""

import json
from hashlib import sha256
from types import ModuleType, SimpleNamespace

import pytest
from retained_external_drift_fixtures import chain as chain
from retained_external_drift_fixtures import restorer as restorer

from automated_phishing_detection import bound_drift, bound_secondary, phiusiil


@pytest.mark.parametrize(
    "change",
    ["absent", "list", "reverse", "duplicate", "extra", "pair_list", "content", "name"],
)
def test_public_input_inventory_is_exact_and_ordered(
    restorer: ModuleType, chain: SimpleNamespace, change: str
) -> None:
    changes = {
        "absent": (),
        "list": list(chain.public),
        "reverse": chain.public[::-1],
        "duplicate": (chain.public[0], chain.public[0], chain.public[2]),
        "extra": (*chain.public, chain.public[0]),
        "pair_list": (list(chain.public[0]), *chain.public[1:]),
        "content": ((chain.public[0][0], "private-content"), *chain.public[1:]),
        "name": (("private-path", chain.public[0][1]), *chain.public[1:]),
    }
    with pytest.raises(bound_drift.BoundDriftError):
        restorer.restore_external_drift(*chain.arguments[:2], changes[change])


@pytest.mark.parametrize("position", [0, 1])
@pytest.mark.parametrize("invalid", [None, "private-value", bytearray(b"private")])
def test_private_snapshots_must_be_exact_bytes(
    restorer: ModuleType, chain: SimpleNamespace, position: int, invalid: object
) -> None:
    arguments = list(chain.arguments)
    arguments[position] = invalid
    with pytest.raises(bound_drift.BoundDriftError):
        restorer.restore_external_drift(*arguments)


@pytest.mark.parametrize("position", [0, 1])
def test_tampered_private_snapshots_fail_against_report_hashes(
    restorer: ModuleType, chain: SimpleNamespace, position: int
) -> None:
    arguments = list(chain.arguments)
    arguments[position] += b" "
    with pytest.raises(bound_drift.BoundDriftError):
        restorer.restore_external_drift(*arguments)


@pytest.mark.parametrize("position", [1, 2])
def test_public_chain_tampering_precedes_retained_snapshot_loading(
    restorer: ModuleType,
    chain: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    position: int,
) -> None:
    from automated_phishing_detection import retained_drift

    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("unbound public inputs reached retained snapshot loader")

    monkeypatch.setattr(retained_drift, "load_retained_drift_reference", forbidden)
    public = tuple(
        (name, content + b" " if index == position else content)
        for index, (name, content) in enumerate(chain.public)
    )
    with pytest.raises(bound_drift.BoundDriftError):
        restorer.restore_external_drift(*chain.arguments[:2], public)


def test_unbound_source_bytes_are_rejected_before_source_parsing(
    restorer: ModuleType, chain: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("unbound source bytes reached the source parser")

    monkeypatch.setattr(phiusiil, "_load_source_spec", forbidden)
    public = (*chain.public[:2], (chain.public[2][0], b"private malformed source"))
    with pytest.raises(bound_drift.BoundDriftError):
        restorer.restore_external_drift(*chain.arguments[:2], public)


def test_fixed_report_attestation_cannot_be_replaced_by_matching_private_hashes(
    restorer: ModuleType, chain: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(
        bound_secondary.PUBLIC_REPORTS, "tabular", (chain.public[0][0], "f" * 64)
    )
    with pytest.raises(bound_drift.BoundDriftError, match="accepted"):
        restorer.restore_external_drift(*chain.arguments)


@pytest.mark.parametrize("field", ["status", "completion_summary_sha256"])
def test_accepted_report_still_passes_existing_chain_semantic_checks(
    restorer: ModuleType,
    chain: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
) -> None:
    report = json.loads(chain.public[0][1])
    report[field] = "private-invalid-report-state"
    content = json.dumps(report).encode()
    public = ((chain.public[0][0], content), *chain.public[1:])
    monkeypatch.setitem(
        bound_secondary.PUBLIC_REPORTS,
        "tabular",
        (public[0][0], sha256(content).hexdigest()),
    )
    with pytest.raises(bound_drift.BoundDriftError) as caught:
        restorer.restore_external_drift(*chain.arguments[:2], public)
    assert "private" not in str(caught.value)
    assert caught.value.__suppress_context__
