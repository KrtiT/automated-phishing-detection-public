"""Synthetic aggregate, privacy and header-alignment checks."""

import json

import pytest
from retained_metadata_preflight import summarize


def test_aggregate_retains_all_strata_without_exporting_records():
    headers = ["split", "source", "tier", "label", "domain", "url"]
    rows = [
        [
            "train",
            "registry",
            "gold",
            "benign",
            "private.test",
            "https://private.test/secret",
        ],
        [
            "train",
            "registry",
            "gold",
            "benign",
            "private.test",
            "https://private.test/other",
        ],
        ["test", "registry", "gold", "benign", "", "unknown"],
    ]
    document = {
        "headers": {"table": headers},
        "rows": [{"source_member": "table", "cells": cells} for cells in rows],
    }
    result = summarize(document)
    assert sum(row["rows"] for row in result) == 3
    assert result[1]["distinct_publisher_domain_strings"] == 1
    assert result[0]["empty_publisher_domain_rows"] == 1
    assert "private.test" not in json.dumps(result)
    assert "secret" not in json.dumps(result)


def test_header_misalignment_fails():
    with pytest.raises(ValueError, match="header and cell"):
        summarize(
            {
                "headers": {"table": ["split"]},
                "rows": [{"source_member": "table", "cells": []}],
            }
        )
