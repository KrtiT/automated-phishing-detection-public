"""The original and retained adapters share the real scientific producer."""

import pytest
from test_external_source_runner import configured_case, session_owner

from automated_phishing_detection import external_source_runner as runner


@pytest.mark.parametrize("count", [0, 5, 256])
def test_acquired_inputs_use_real_producer_without_source_reads(
    tmp_path, monkeypatch, count
):
    assert callable(getattr(runner.body, "produce_inputs", None))
    case = configured_case(runner, tmp_path, monkeypatch, count)
    state = runner.body.ExternalRun(case.binding, case.paths, case.handoff)
    runner.body.preflight(state)

    def forbidden(*args, **kwargs):
        pytest.fail("acquired input seam reopened an original source")

    monkeypatch.setattr(runner.body, "_read_file_once", forbidden)
    with session_owner(
        case,
        case.binding,
        case.paths.artifacts,
        case.paths.secondary_artifacts,
        case.paths.drift_artifacts,
    ) as session:
        runner.body.produce_inputs(
            state, session, case.decoded, case.prepared, case.suffix
        )
    assert len(state.outputs) == 36
    assert len(case.session.evaluation.primary.scorer.urls) == count
    assert all(
        state.outputs[name] == content
        for name, content in case.produced.private_outputs.items()
    )
