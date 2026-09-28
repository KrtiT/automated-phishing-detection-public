"""Real adopted root metadata and retained preparation from invented originals."""

from types import SimpleNamespace

from prepared_internal_fixtures import _restore
from study_authorized_process_inputs_fixtures import external_inputs, internal_inputs
from study_authorized_process_metadata_fixtures import (
    authorization,
    bind_metadata,
    write_child_bootstrap,
)
from study_authorized_process_science_fixtures import install_science

from automated_phishing_detection import _study_preparation_files as files
from automated_phishing_detection import execution_receipt, study_preparation_runner
from automated_phishing_detection._adopted_study_ledger import AdmissionLedger
from automated_phishing_detection._adopted_study_records import (
    study_identity,
    study_intent,
)
from automated_phishing_detection._study_run_records import prediction_barrier


def setup(inputs, tmp_path, monkeypatch, mode="success"):
    base, original, unused, events = inputs
    case = SimpleNamespace(
        base=internal_inputs(base, original),
        original=original,
        envelope_path=tmp_path / "invented-envelope.json",
    )
    archive = tmp_path / "invented.zip"
    case.archive_pins = external_inputs(archive)
    case.preparation_paths = study_preparation_runner.StudyPreparationPaths(
        original.source_csv, original.suffix_rules, archive, original.attempt
    )
    install_science(case, monkeypatch)
    bind_metadata(case, monkeypatch)
    case.authorization = authorization(case)
    _prepare(case, monkeypatch)
    _root(case)
    write_child_bootstrap(case, monkeypatch, mode)
    _diagnostics(monkeypatch)
    for path in (original.source_csv, original.suffix_rules, archive):
        path.unlink()
    return case


def _prepare(case, monkeypatch):
    for name in ("attempt", "internal", "external"):
        selected = getattr(case.authorization.paths, name)
        (selected if name == "attempt" else selected.attempt).parent.mkdir(
            parents=True, exist_ok=True
        )
    monkeypatch.setattr(
        study_preparation_runner, "recheck_binding", lambda binding: None
    )
    snapshot = study_preparation_runner._run_bound_preparation(
        case.base, case.preparation_paths
    )
    case.preparation = _restore(snapshot, case.base)[0]


def _diagnostics(monkeypatch):
    from automated_phishing_detection import owned_worker

    original = owned_worker._stream_hash

    def stream_hash(stream):
        stream.seek(0)
        print(stream.read().decode())
        return original(stream)

    monkeypatch.setattr(owned_worker, "_stream_hash", stream_hash)


def _root(case):
    auth = case.authorization
    attempt = execution_receipt.reserve_attempt(
        auth.paths.attempt, identity=study_identity(auth)
    )
    intent = study_intent(auth, attempt)
    execution = study_identity(auth) | {
        "reservation_sha256": attempt.reservation_sha256
    }
    from automated_phishing_detection._adopted_study_records import scientific_execution

    barrier, held = prediction_barrier(
        case.preparation, execution=scientific_execution(execution)
    )
    assert held is False
    with execution_receipt._directory(attempt.directory) as directory:
        states = {"reservation.json": files.capture(directory, "reservation.json")}
        for name, content in (
            ("study-intent.json", intent),
            ("prediction-barrier.json", barrier),
        ):
            states[name] = files.append(directory, states, name, content)
    case.ledger = AdmissionLedger(auth, attempt, intent)
    case.ledger.barrier_retained(case.preparation, barrier)
