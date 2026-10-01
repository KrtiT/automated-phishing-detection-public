"""Synthetic failed-root histories with complete opaque source and cell receipts."""

import importlib
import importlib.util
import json
from types import SimpleNamespace

from adopted_study_profile_evidence_fixtures import _bindings, _frames
from adopted_study_profile_fixtures import _authorization, _contents, _root, make_case
from stopped_study_authorization_fixtures import (
    _finalization,
    _truncate,
    refresh_accounting,
)
from study_history_snapshot_source_fixtures import (
    digest,
    external_source,
    hashes,
    internal_source,
)

from automated_phishing_detection import _adopted_study_records as records
from automated_phishing_detection._checkpoint_codec import canonical_bytes


def api():
    name = "automated_phishing_detection.study_history_snapshots"
    assert importlib.util.find_spec(name), "missing opaque accepted-history verifier"
    return importlib.import_module(name)


def _profile_case(prepared, manifests):
    original = make_case(prepared, manifests)
    source_profile = {
        "execution": json.loads(original.original.contents["source-results.json"])[
            "accepted_inputs"
        ]["execution"]
    }
    source_profile["execution"]["execution_contract_sha256"] = (
        original.base.contract_sha256
    )
    original.profile["components"]["external"] = digest(canonical_bytes(source_profile))
    return original, source_profile


def make_history(prepared, manifests, *, prefix=1):
    original, source_profile = _profile_case(prepared, manifests)
    authorization = _authorization(original)
    reserved, execution = _root(original, authorization)
    contents = _contents(
        original, authorization, execution, original.profile["components"]["external"]
    )
    accounting = json.loads(contents["study-accounting.json"])
    scientific = json.loads(records.decoded(accounting["scientific_accounting_bytes"]))
    _truncate(accounting, scientific, prefix, 0)
    case = SimpleNamespace(
        authorization=authorization,
        execution=execution,
        contents=contents,
        accounting=accounting,
        scientific=scientific,
        source_profile=source_profile,
        profile=original.profile,
        payloads={"attempt/reservation.json": reserved},
    )
    case.payloads.update(_finalization(execution["reservation_sha256"]))
    sources(case)
    return case


def sources(case):
    source = json.loads(case.contents["source-results.json"])
    metadata = source["accepted_inputs"]
    case.internal = internal_source(
        metadata["internal"], case.profile["paths"]["internal-attempt"]
    )
    case.external = external_source(
        metadata["external"],
        metadata["internal"],
        case.internal,
        case.profile["paths"]["external-attempt"],
        case.source_profile,
    )
    source["accepted_inputs_sha256"] = digest(canonical_bytes(metadata))
    case.source = source
    refresh_sources(case)


def refresh_sources(case):
    content = canonical_bytes(case.source)
    case.contents["source-results.json"] = content
    ledger = case.accounting["authorization_ledger"]
    ledger.update(
        source_results_sha256=digest(content),
        accepted_inputs_sha256=case.source["accepted_inputs_sha256"],
        handoff_sha256=case.source["accepted_inputs"]["external"]["execution"][
            "internal_handoff_sha256"
        ],
    )
    _frames(ledger, case.execution, _bindings(ledger, case.execution))
    refresh_accounting(case)


def complete_history(prepared, manifests, *, prefix=1):
    from study_history_snapshot_cell_fixtures import cells

    case = make_history(prepared, manifests, prefix=prefix)
    cells(case)
    refresh_accounting(case)
    return case


def arguments(case):
    return dict(
        expected_profile_sha256=case.authorization.profile_sha256,
        expected_envelope_sha256=case.authorization.envelope_sha256,
        expected_root_snapshot_sha256=case.pins,
        internal_payloads=tuple(case.internal.items()),
        external_payloads=tuple(case.external.items()),
        cell_snapshots=case.cells,
        expected_internal_sha256=hashes(case.internal),
        expected_external_sha256=hashes(case.external),
        expected_cells_sha256=tuple(
            (ordinal, hashes(dict(payloads))) for ordinal, _, payloads in case.cells
        ),
    )


def verify(case, **changes):
    return api().verify_study_history_snapshots(
        case.snapshot, **(arguments(case) | changes)
    )
