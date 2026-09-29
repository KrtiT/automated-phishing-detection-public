"""Closed derived completion metadata and independently pinned publisher ancestry."""

from . import _retained_preparation_records as records
from ._study_urlnorm_policy import REPRESENTATION
from ._study_urlnorm_scope import FIELDS, PIN_FIELDS


def validate(value):
    records.require(
        type(value) is dict and set(value) == {"representation", *PIN_FIELDS}
    )
    records.require(value["representation"] == REPRESENTATION)
    for name in PIN_FIELDS:
        records.digest(value[name])
    return dict(value)


def derivation(continuation):
    records.require(type(continuation) is dict and set(continuation) == set(FIELDS))
    return validate(
        {name: value for name, value in continuation.items() if name != "prior_profile"}
    )


def publisher_versions(outputs, version, lineage=None):
    for name in ("publisher-source.json", "publisher-summary.json"):
        publisher = records.load(outputs[name])
        records.require(type(publisher) is dict)
        records.require(type(publisher.get("schema_version")) is int)
        records.require(publisher["schema_version"] == version)
        if lineage is not None:
            records.require(publisher.get("representation") == REPRESENTATION)
            records.require(
                publisher.get("parent_source_sha256")
                == lineage["publisher_source_sha256"]
            )
            records.require(
                publisher.get("parent_summary_sha256")
                == lineage["publisher_summary_sha256"]
            )


def project(envelope, expected, outputs):
    records.require(type(envelope) is dict)
    version = envelope.get("schema_version")
    records.require(type(version) is int and version in (1, 2))
    fields = set(expected) | ({"derivation"} if version == 2 else set())
    records.require(set(envelope) == fields)
    lineage = validate(envelope["derivation"]) if version == 2 else None
    publisher_versions(outputs, version, lineage)
    return (
        expected
        if version == 1
        else expected
        | {
            "schema_version": 2,
            "protocol": "study-preparation-derived-v1",
            "derivation": lineage,
        }
    )
