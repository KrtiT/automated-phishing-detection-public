"""Real shift transport with invented models and an unexecuted scheduling prefix."""

import asyncio
import sys
import threading
from dataclasses import replace

import study_authorized_process_science_fixtures as science
from study_authorized_process_cell_fixtures import _accepted, _retain_source_results
from study_authorized_process_root_fixtures import setup

from automated_phishing_detection import (
    bound_runtime,
    fixed_cascade,
    gmm_monitor,
    operational_cell_runner,
    source_runner,
)
from automated_phishing_detection._study_authorized_cell import run_adopted_cell
from automated_phishing_detection._study_run_paths import cell_paths
from automated_phishing_detection.operational_input_transport import (
    retain_operational_root_inputs,
)
from automated_phishing_detection.operational_schedule import cell_for_ordinal
from automated_phishing_detection.selective_inference import (
    InferenceCounts,
    RequestScores,
)
from automated_phishing_detection.study_operational_records import retain_accepted_cell


class ShiftFixtureScorer:
    def __init__(self, cascade):
        self.cascade = cascade
        self.owner = threading.get_ident()
        self.counts = InferenceCounts(0, 0, 0, 0)

    def __enter__(self):
        return self

    def __exit__(self, *arguments):
        return None

    def _require_owner(self):
        assert threading.get_ident() == self.owner

    def score_all(self, raw_url):
        return self._score(raw_url, False, True)

    def scan(self, raw_url, *, drift_override=False):
        return self._score(raw_url, drift_override, False)

    def _score(self, raw_url, override, all_models):
        self._require_owner()
        probabilities, audit = fixed_cascade.score_logistic_l1_authoritative(
            self.cascade.stage1_model, (raw_url,)
        )
        fixed = fixed_cascade.score_fixed_cascade(
            probabilities,
            (0.15,),
            stage1_threshold=self.cascade.stage1_threshold,
            transformer_threshold=self.cascade.transformer_threshold,
            half_width=self.cascade.half_width,
        )
        band = fixed.transformer_invoked[0]
        invoked = band or override or all_models
        self._completed(invoked)
        decision = int(0.15 >= self.cascade.transformer_threshold)
        return RequestScores(
            probabilities[0],
            0.15 if invoked else None,
            fixed.decisions[0],
            decision if override else fixed.decisions[0],
            band,
            override,
            band or override,
            invoked,
            audit,
        )

    def _completed(self, invoked):
        self.counts = replace(
            self.counts,
            completed_requests=self.counts.completed_requests + 1,
            transformer_forward_attempts=self.counts.transformer_forward_attempts
            + int(invoked),
            successful_transformer_scores=self.counts.successful_transformer_scores
            + int(invoked),
        )


def setup_shift(inputs, tmp_path, monkeypatch):
    monkeypatch.setattr(science, "FixturePrimaryScorer", ShiftFixtureScorer)
    case = setup(inputs, tmp_path, monkeypatch)
    bootstrap = case.root.parent / "bootstrap/sitecustomize.py"
    bootstrap.write_text(
        "import study_authorized_process_science_fixtures as science\n"
        "from study_authorized_process_shift_fixtures import "
        "ShiftFixtureScorer, install_shift_owner\n"
        "science.FixturePrimaryScorer = ShiftFixtureScorer\n"
        "from study_authorized_process_child_fixtures import bootstrap\n"
        "bootstrap()\ninstall_shift_owner()\n"
    )
    return case


def install_shift_owner():
    if "--role" not in sys.argv:
        return
    if sys.argv[sys.argv.index("--role") + 1] != "service":
        return
    with source_runner.open_bound_evaluation_session() as session:
        models = session.primary.models
    bound_runtime.load_bound_models = lambda *arguments: models
    bound_runtime.SelectiveCascade = ShiftFixtureScorer
    gmm_monitor.score_feature_matrix = science.score_feature_matrix


def _simulate_unexecuted_prefix(ledger):
    assert ledger.completed_cells == 0 and len(ledger.entries) == 2
    ledger.completed_cells = 120
    ledger.entries.extend(
        {"fixture_only_unexecuted_prefix": ordinal, "role": role}
        for ordinal in range(1, 121)
        for role in ("service", "client")
    )


def run_shift_cell(case, sources, monkeypatch):
    auth, ledger = case.authorization, case.ledger
    accepted = _accepted(case, sources)
    content = _retain_source_results(case, accepted)
    auth.paths.cells_directory.mkdir(mode=0o700)
    auth.paths.accepted_inputs_directory.parent.chmod(0o700)
    monkeypatch.setattr(operational_cell_runner, "recheck_binding", lambda _: None)
    with retain_operational_root_inputs(
        auth.paths.accepted_inputs_directory, accepted_inputs=accepted.metadata_bytes
    ):
        ledger.inputs_retained(accepted, content)
        _simulate_unexecuted_prefix(ledger)
        return _execute_shift(auth, ledger, accepted)


def _execute_shift(auth, ledger, accepted):
    cell = cell_for_ordinal(121)
    ledger.cell_started(cell)
    completion = asyncio.run(
        run_adopted_cell(
            auth,
            accepted,
            cell,
            paths=cell_paths(auth.paths, 121),
            admissions=ledger,
        )
    )
    ledger.cell_accepted(
        cell,
        retain_accepted_cell(completion, accepted=accepted),
        pair_intent_bytes=dict(completion.snapshot.payloads)[
            "attempt/process-pair-intent.json"
        ],
    )
    return completion
