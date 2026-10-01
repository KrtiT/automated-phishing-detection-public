import importlib
import importlib.util
import json
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest
from study_history_cell_fixtures import candidates, case, history, manifests

__all__ = ["candidates", "case", "history", "manifests"]


def api():
    name = "automated_phishing_detection.study_series_cell"
    assert importlib.util.find_spec(name), "missing series cell science adapter"
    return importlib.import_module(name)


def arguments(history):
    prior = history.arguments
    return dict(
        expected_snapshot_sha256=prior["expected_snapshot_sha256"],
        metadata_bytes=b"invalid metadata not to be parsed",
        profile_bytes=b"invalid profile not to be parsed",
        internal_snapshot=prior["internal_snapshot"],
        external_snapshot=prior["external_snapshot"],
        descriptor_bytes=prior["descriptor_bytes"],
        binding_bytes=prior["binding_bytes"],
        manifest_bytes=history.working.inputs.manifest_bytes,
        expected_metadata_sha256="1" * 64,
        expected_profile_sha256="2" * 64,
        expected_descriptor_sha256=prior["expected_descriptor_sha256"],
        expected_binding_sha256=prior["expected_binding_sha256"],
        expected_cell_reservation_sha256=prior["expected_cell_reservation_sha256"],
        expected_attempt_directory=prior["expected_attempt_directory"],
    )


def test_result_is_frozen_non_authorizing_and_decodes_fresh_views(history):
    result = api().SeriesCellScience(
        tuple(history.values.items()),
        SimpleNamespace(computational=history.working.inputs),
        history.working.summary_bytes,
        history.working.reservation_sha256,
    )
    assert result.authorizes_execution is False
    assert result.run == history.working.run
    assert result.summary == history.working.summary
    result.summary.clear()
    result.run.after_measured.admitted_requests = 0
    assert result.summary == history.working.summary
    assert result.run == history.working.run
    with pytest.raises(FrozenInstanceError):
        result.summary_bytes = b"{}"
    with pytest.raises(FrozenInstanceError):
        result.authorizes_execution = True


@pytest.mark.parametrize("member", ("attempt/run.json", "public-summary.json"))
def test_stale_payload_pin_rejects_before_context_or_science(history, member):
    module = api()
    payloads = dict(history.values)
    payloads[member] += b" "
    with pytest.raises(
        module.SeriesCellScienceError, match="^invalid_series_cell_science$"
    ):
        module.verify_series_cell_science(tuple(payloads.items()), **arguments(history))


def test_scientific_output_bytes_do_not_leak_in_error(history):
    module = api()
    payloads = dict(history.values)
    payloads["attempt/run.json"] = b"sensitive invented invalid body"
    with pytest.raises(module.SeriesCellScienceError) as caught:
        module.verify_series_cell_science(tuple(payloads.items()), **arguments(history))
    assert str(caught.value) == "invalid_series_cell_science"
    assert json.dumps(str(caught.value)).find("sensitive") == -1
