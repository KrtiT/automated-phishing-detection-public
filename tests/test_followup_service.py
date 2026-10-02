import asyncio
import importlib
import importlib.util
import io
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from http_run_codec_fixtures import complete_run


def extension():
    name = "automated_phishing_detection.followup_service"
    assert importlib.util.find_spec(name), "bounded service comparison not implemented"
    return importlib.import_module(name)


def test_schedule_has_exactly_eighty_arms_and_alternates_within_each_pair():
    schedule = extension().schedule()
    assert len(schedule) == 80
    assert len({tuple(arm.values()) for arm in schedule}) == 80
    for offset in range(0, 80, 2):
        first, second = schedule[offset : offset + 2]
        assert first["workload"] == second["workload"]
        assert first["concurrency"] == second["concurrency"]
        assert first["pair"] == second["pair"]
        expected = ["shared", "worker"] if first["pair"] % 2 else ["worker", "shared"]
        assert [first["client"], second["client"]] == expected
    assert {arm["pair"] for arm in schedule} == set(range(1, 11))


def test_synthetic_requests_are_fixed_unique_and_have_no_class_prevalence():
    module = extension()
    rows = module.synthetic_requests()
    assert rows == module.synthetic_requests()
    assert len(rows) == len({row.record_id for row in rows}) == 10000
    assert all(".example/" in row.raw_url for row in rows)
    assert not any(hasattr(row, "is_phishing") for row in rows)


def test_success_percentiles_and_failure_percentiles_have_explicit_denominators():
    module = extension()
    run = complete_run()
    summary = module.describe_run(run)
    assert summary["request_count"] == 10000
    assert summary["request_errors"] == 6
    assert summary["success_latency"]["count"] == 9994
    assert summary["failure_latency"]["count"] == 6
    assert summary["all_latency"]["count"] == 10000
    assert "prevalence_basis_points" not in summary
    assert "workload" not in summary


def test_response_check_keeps_scheduling_and_prediction_agreement_separate():
    module = extension()
    baseline = complete_run()
    rows = list(baseline.measured)
    rows[10] = replace(
        rows[10],
        response=rows[10].response.model_copy(
            update={"admission_sequence": rows[11].response.admission_sequence}
        ),
    )
    rows[11] = replace(
        rows[11],
        response=rows[11].response.model_copy(
            update={
                "admission_sequence": baseline.measured[10].response.admission_sequence
            }
        ),
    )
    candidate = replace(baseline, measured=tuple(rows))
    result = module.response_agreement(baseline, candidate)
    assert result["both_successful"] == 9994
    assert result["exact_except_request_id"] == 9992
    assert result["prediction_agreement"] == 9994
    assert result["noncomparable_errors"] == 6


def test_response_check_rejects_changed_record_order():
    module = extension()
    baseline = complete_run()
    changed = replace(baseline, measured=tuple(reversed(baseline.measured)))
    with pytest.raises(ValueError, match="order"):
        module.response_agreement(baseline, changed)


def pairs():
    return [
        {"pair": index, "shared_p95_ms": 300.0, "worker_p95_ms": 150.0}
        for index in range(1, 11)
    ]


def test_operational_requirement_uses_all_ten_pairs_and_fixed_bootstrap():
    result = extension().primary_summary(
        pairs(), [150.0] * 100000, errors=0, exact_agreement=100000
    )
    assert result["ratios"] == [0.5] * 10
    assert result["median_ratio"] == 0.5
    assert result["ratio_interval_97_5"] == [0.5, 0.5]
    assert result["bootstrap_replicates"] == 10000
    assert result["bootstrap_seed"] == 20261002
    assert result["S_requirement_met"] is True


@pytest.mark.parametrize(
    "change", ["missing_pair", "duplicate_pair", "wrong_denominator"]
)
def test_no_incomplete_or_repeated_subset_can_support_requirement(change):
    selected = pairs()
    latencies = [150.0] * 100000
    if change == "missing_pair":
        selected.pop()
    elif change == "duplicate_pair":
        selected[-1] = selected[0]
    else:
        latencies.pop()
    with pytest.raises(ValueError):
        extension().primary_summary(
            selected, latencies, errors=0, exact_agreement=100000
        )


@pytest.mark.parametrize(
    "errors,agreement,expected",
    [(99, 99901, False), (100, 99900, False), (0, 99999, False)],
)
def test_errors_and_any_response_mismatch_are_not_hidden(errors, agreement, expected):
    result = extension().primary_summary(
        pairs(), [150.0] * (100000 - errors), errors=errors, exact_agreement=agreement
    )
    assert result["S_requirement_met"] is expected
    assert result["requirements"]["errors_below_one_per_thousand"] is (errors < 100)


def test_empty_success_arm_is_an_adverse_result_not_an_exception():
    selected = pairs()
    selected[0]["worker_p95_ms"] = None
    result = extension().primary_summary(selected, [], errors=100000, exact_agreement=0)
    assert result["S_requirement_met"] is False
    assert result["ratio_interval_97_5"] is None
    assert result["undefined_pairs"] == [1]


def runner():
    path = Path(__file__).resolve().parents[1] / "scripts/run_followup_service.py"
    assert path.exists(), "bounded service runner not implemented"
    spec = importlib.util.spec_from_file_location("followup_service_runner", path)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


@pytest.mark.parametrize(
    "thermal,expected",
    [
        (
            "Note: No thermal warning level has been recorded\nNote: No performance warning level has been recorded\nNote: No CPU power status has been recorded\n",
            None,
        ),
        ("Thermal_Level = 1", "thermal_or_performance_not_normal"),
        ("CPU_Speed_Limit = 80", "thermal_or_performance_not_normal"),
        ("", "thermal_or_performance_not_normal"),
    ],
)
def test_thermal_guard_does_not_ignore_warning_or_unknown_state(thermal, expected):
    assert runner().thermal_violation(thermal) == expected


def test_no_model_control_never_loads_research_artifact(monkeypatch):
    module = extension()

    def forbidden(*args):
        pytest.fail("no-model control tried to load a model")

    monkeypatch.setattr(module.fixed_cascade, "load_logistic_l1_artifact", forbidden)
    with module.scorer_session("no_model", Path("unused")) as scorer:
        response = scorer.scan("https://example.test")
        assert response.stage1_probability == 0.25
        assert response.decision == 0
        assert not response.transformer_evaluated
        assert scorer.counts.completed_requests == 1
        with pytest.raises(ValueError, match="drift"):
            scorer.scan("https://example.test", drift_override=True)
        assert scorer.counts.failed_requests == 1


def test_structural_service_uses_unchanged_singleton_scoring(monkeypatch):
    module = extension()
    model = SimpleNamespace(validation_threshold_record={"threshold": 0.5})
    calls = []

    def score(selected, urls):
        assert selected is model
        calls.append(urls)
        return (0.5,), {"audit": "synthetic"}

    monkeypatch.setattr(module.fixed_cascade, "score_logistic_l1_authoritative", score)
    scorer = module.StructuralScorer(model)
    result = scorer.scan("https://raw.example/path")
    assert calls == [["https://raw.example/path"]]
    assert result.decision == result.fixed_decision == 1
    assert scorer.counts.transformer_forward_attempts == 0


def test_owned_process_is_cleaned_if_launch_receipt_cannot_be_written(
    tmp_path, monkeypatch
):
    module = runner()
    calls = []
    process = SimpleNamespace(pid=987, stdin=io.BytesIO(), returncode=0)

    def wait(*args):
        calls.append("wait")
        return 0

    process.wait = wait
    monkeypatch.setattr(module.subprocess, "Popen", lambda *args, **kwargs: process)

    def write(path, value):
        if path.name == "launch.json":
            raise OSError("synthetic storage failure")

    monkeypatch.setattr(module, "write_json", write)

    async def exercise():
        async with module.owned_server(["unused"], tmp_path):
            pytest.fail("startup should not complete")

    with pytest.raises(OSError, match="storage"):
        asyncio.run(exercise())
    assert calls == ["wait"]
    assert process.stdin.closed


def test_environmental_violation_cancels_measurement_and_stops_schedule(monkeypatch):
    module = runner()
    stopped = []

    async def exercise():
        started = asyncio.Event()

        async def measurement(*args):
            started.set()
            try:
                await asyncio.sleep(30)
            finally:
                stopped.append(True)

        async def monitor(*args):
            await started.wait()
            raise module.ConditionsError("synthetic interruption")

        monkeypatch.setattr(module, "measure_schedule", measurement)
        monkeypatch.setattr(module, "monitor_conditions", monitor)
        await module.guarded_schedule(None, Path("unused"), None, (), "a" * 64)

    with pytest.raises(module.ConditionsError, match="interruption"):
        asyncio.run(exercise())
    assert stopped == [True]


@pytest.mark.parametrize("client", ["shared", "worker"])
def test_small_separate_process_control_shuts_down_cleanly(tmp_path, client):
    module = runner()
    script_directory = str(Path(module.__file__).parent)
    program = f"""
import sys
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0, {script_directory!r})
import run_followup_service as runner
runner.verify_execution = lambda *args: None
runner.serve(SimpleNamespace(context_root=Path('.'), manifest=None, workload='no_model', socket_fd=int(sys.argv[-1])))
"""

    async def exercise():
        async with module.owned_server(
            [sys.executable, "-c", program], tmp_path
        ) as base_url:
            replay = (
                module.http_replay.replay_run
                if client == "shared"
                else module.worker_connection_replay.replay_worker_connections
            )
            result = await replay(
                base_url,
                extension().synthetic_requests()[:128],
                manifest_sha256="a" * 64,
                prevalence_basis_points=100,
                concurrency=64,
                run_index=1,
                warmup_count=64,
            )
            run = result if client == "shared" else result.run
            assert len(run.measured) == 128
            assert all(
                row.error is None and row.response.probability == 0.25
                for row in run.measured
            )
            assert run.after_measured.completed_requests == 192

    asyncio.run(exercise())
    assert json.loads((tmp_path / "exit.json").read_text())["exit_code"] == 0
    assert not json.loads((tmp_path / "exit.json").read_text())["forced"]
