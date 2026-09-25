"""Entirely invented, in-memory preparation snapshots; no process authority."""

import importlib
import json
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import pytest
from phishvn_source_fixtures import bundle, members, record
from test_source_overlap import source as source_fixture

from automated_phishing_detection import source_checkpoints, study_preparation_inputs
from automated_phishing_detection._checkpoint_codec import canonical_bytes
from automated_phishing_detection._phishvn_archive import PhishVNSourcePins
from automated_phishing_detection._study_preparation_records import (
    PreparedStudySnapshot,
)
from automated_phishing_detection.execution_receipt import Attempt
from automated_phishing_detection.source_overlap import SourceOverlapPins
from automated_phishing_detection.study_feasibility import (
    assess_preparation_feasibility,
)
from automated_phishing_detection.study_preparation_retention import PREPARATION_ORDER


@pytest.fixture
def api():
    return importlib.import_module(
        "automated_phishing_detection.retained_study_preparation"
    )


def source_claims(source):
    split = source.report["splits"]["group_test"]
    return {
        "expected_sha256": source.report["output_hashes"]["group_test.jsonl"],
        "source_csv_sha256": source.pins["source_csv_sha256"],
        "suffix_rules_sha256": source.pins["suffix_rules_sha256"],
        "expected_row_count": split["row_count"],
        "expected_domain_count": split["domain_count"],
        "expected_class_counts": split["class_counts"],
    }


def publisher_rows():
    return [
        record("training", "train"),
        record("gold", url="HTTPS://gold.example:443/path"),
        record("certified", source="tinnhiem_web", label="benign"),
        record("secondary", tier="silver"),
        record("control", source="tranco", tier="silver", label="benign"),
        record("overlap", url="https://invalid-label.example/path"),
        record("invalid", url="bare.example"),
    ]


def identity(source, archive):
    return {
        "kind": "study_preparation_only",
        "protocol": "study-preparation-v1",
        "revision": "c" * 40,
        "execution_contract_sha256": "d" * 64,
        "runtime_sha256": "e" * 64,
        "source_profile_sha256": "f" * 64,
        **source.pins,
        "partition_sha256": source.report["output_hashes"]["group_test.jsonl"],
        **archive.pins,
    }


def completion(outputs, execution, reservation):
    return canonical_bytes(
        {
            "schema_version": 1,
            "protocol": "study-preparation-v1",
            "status": "preparation_only",
            "protected_evaluation_authorized": False,
            "scoring_authorized": False,
            "execution": execution,
            "reservation_sha256": reservation,
            "input_sha256": {
                name: sha256(content).hexdigest() for name, content in outputs.items()
            },
        }
    )


def internal_inputs(source):
    return study_preparation_inputs.prepare_internal_inputs(
        source.buffers["csv_bytes"],
        source.buffers["suffix_rules_bytes"],
        source.buffers["source_spec_bytes"],
        source.buffers["preparation_summary_bytes"],
        pins=SourceOverlapPins(**source.pins),
        source=source_claims(source),
    )


def external_inputs(source, domains, rows):
    archive = bundle(members(publisher_rows() if rows is None else rows))
    publisher = study_preparation_inputs.decode_external_inputs(
        archive.content,
        source.buffers["suffix_rules_bytes"],
        archive_pins=PhishVNSourcePins(**archive.pins),
        suffix_rules_sha256=source.pins["suffix_rules_sha256"],
    )
    external = study_preparation_inputs.prepare_external_inputs(
        publisher,
        source.buffers["suffix_rules_bytes"],
        overlap_domains=domains,
    )
    return archive, publisher, external


def make_case(rows=None):
    source = source_fixture.__wrapped__()
    reconstructed, internal = internal_inputs(source)
    archive, publisher, external = external_inputs(
        source, reconstructed.overlap_domains, rows
    )
    case = SimpleNamespace(
        source=source,
        reconstructed=reconstructed,
        internal=internal,
        external=external,
        publisher=publisher,
        archive=archive,
    )
    case.identity, case.reservation = identity(source, archive), "a" * 64
    outputs = retained_outputs(case)
    outputs["preparation-complete.json"] = completion(
        outputs, case.identity, case.reservation
    )
    case.snapshot = PreparedStudySnapshot(
        case.reservation, tuple((name, outputs[name]) for name in PREPARATION_ORDER)
    )
    case.completion = sha256(outputs["preparation-complete.json"]).hexdigest()
    return case


def retained_outputs(case):
    attempt = Attempt(Path("/invented/preparation"), case.reservation)
    return {
        "suffix-rules.dat": case.source.buffers["suffix_rules_bytes"],
        **source_checkpoints._contents(attempt, case.identity, case.reconstructed),
        **case.publisher.private_outputs,
        "publisher-summary.json": canonical_bytes(case.publisher.public_summary),
        **case.external.private_outputs,
        "preparation-summary.json": canonical_bytes(case.external.public_summary),
        "feasibility.json": assess_preparation_feasibility(
            case.internal.records, case.external
        ),
    }


@pytest.fixture
def retained_case():
    return make_case()


def restore(api, case, **changes):
    arguments = {
        "expected_identity": case.identity,
        "expected_reservation_sha256": case.reservation,
        "expected_completion_sha256": case.completion,
        "source_spec_bytes": case.source.buffers["source_spec_bytes"],
        "preparation_summary_bytes": case.source.buffers["preparation_summary_bytes"],
    } | changes
    return api.restore_study_preparation(case.snapshot, **arguments)


def changed(case, replacements, *, repin=False):
    outputs = dict(case.snapshot.payloads) | replacements
    receipt = json.loads(outputs["preparation-complete.json"])
    receipt["input_sha256"] = {
        name: sha256(outputs[name]).hexdigest() for name in PREPARATION_ORDER[:-1]
    }
    outputs["preparation-complete.json"] = canonical_bytes(receipt)
    result = SimpleNamespace(**vars(case))
    result.snapshot = replace(
        case.snapshot,
        payloads=tuple((name, outputs[name]) for name in PREPARATION_ORDER),
    )
    if repin:
        result.completion = sha256(outputs["preparation-complete.json"]).hexdigest()
    return result
