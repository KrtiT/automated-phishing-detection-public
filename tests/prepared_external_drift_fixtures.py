"""Invented retained drift declarations aligned to invented preparation metadata."""

import json
from dataclasses import asdict, replace
from hashlib import sha256

from test_retained_drift import SOURCE_SHA256, _bytes, _rebind, _snapshots


def _replace_source(value, source):
    if type(value) is str:
        return value.replace(SOURCE_SHA256, source)
    if type(value) is list:
        return [_replace_source(item, source) for item in value]
    if type(value) is dict:
        return {key: _replace_source(item, source) for key, item in value.items()}
    return value


def aligned_drift(source, report):
    summary = json.loads(report)
    splits = summary["splits"]
    data = _snapshots(
        training_count=splits["train"]["row_count"],
        validation_count=splits["validation"]["row_count"],
        constant=True,
    )
    pins = replace(
        data["pins"],
        source_csv_sha256=json.loads(source)["phiusiil"]["csv_sha256"],
        train_sha256=summary["output_hashes"]["train.jsonl"],
        validation_sha256=summary["output_hashes"]["validation.jsonl"],
        preparation_summary_sha256=sha256(report).hexdigest(),
        suffix_rules_sha256=json.loads(source)["public_suffix_list"]["sha256"],
    )
    reference = _replace_source(
        json.loads(data["training_reference"]), pins.source_csv_sha256
    )
    audit = _replace_source(
        json.loads(data["validation_audit"]), pins.source_csv_sha256
    )
    public = data["expected_drift_summary"]
    for document in (reference, audit, public):
        document["input_hashes"] = asdict(pins)
    reference["validation_declaration_sha256"] = sha256(
        _bytes(splits["validation"])
    ).hexdigest()
    return _rebind(reference, audit, public, report, pins), source
