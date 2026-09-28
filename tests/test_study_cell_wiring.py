"""Adopted cell passes live admissions into the existing owned pair observer."""

import asyncio
from types import SimpleNamespace

from operational_cell_runner_fixtures import orchestration, setup
from operational_input_fixtures import candidates, manifests

__all__ = ["candidates", "manifests"]


def observer(case):
    async def observe(attempt, **options):
        assert callable(options["study_admissions"])
        assert options["service_command"][1].endswith("scripts/run_study_child.py")
        assert options["client_command"][1].endswith("scripts/run_study_child.py")
        return case.observation

    return observe


def test_private_study_cell_supplies_admission_factory(
    tmp_path, manifests, monkeypatch
):
    case = setup(tmp_path, manifests, monkeypatch)
    orchestration(case, monkeypatch)
    authorization = SimpleNamespace(
        base=case.binding,
        envelope_path=tmp_path / "approval.json",
        envelope_sha256="a" * 64,
    )
    ledger = SimpleNamespace(
        authorization=authorization, source_results_sha256="b" * 64
    )

    monkeypatch.setattr(case.module, "_observe_pair_with_writer", observer(case))
    state = case.module.CellProgress()
    result = asyncio.run(
        case.module._run(
            state,
            case.binding,
            case.profile,
            case.accepted,
            case.cell,
            case.paths,
            case.artifacts,
            case.deadlines,
            study_admissions=ledger,
        )
    )
    assert result.observation is case.observation and result.snapshot is case.snapshot
