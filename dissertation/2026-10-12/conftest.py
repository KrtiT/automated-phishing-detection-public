"""Relocate public aggregate fixtures without editing frozen source snapshots."""

from hashlib import sha256
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parent


@pytest.fixture(autouse=True)
def public_aggregate_locations(request, monkeypatch):
    target = PACKAGE / "research-snapshot/test_corrected_ledger.py"
    if Path(request.module.__file__).resolve() != target:
        return
    from corrected_ledger import PINS

    original = PACKAGE.parent / "2026-10-04/aggregate-data"
    current = PACKAGE / "aggregate-data"
    for directory, name in (
        (original, "primary-gates.csv"),
        (current, "paired-and-dependence-audit.json"),
    ):
        assert sha256((directory / name).read_bytes()).hexdigest() == PINS[name]
    monkeypatch.setattr(request.module, "AGGREGATES", original)
    monkeypatch.setattr(request.module, "REVISION", current)
