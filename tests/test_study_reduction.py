"""The root composes existing kernels without reclassifying incomplete evidence."""

import importlib
import importlib.util

import pytest
from test_study_operational_records import api as records_api


def api():
    name = "automated_phishing_detection.study_reduction"
    assert importlib.util.find_spec(name), "missing whole-study reduction"
    return importlib.import_module(name)


def test_incomplete_accounting_cannot_reduce_or_claim_success():
    with pytest.raises(api().StudyReductionError):
        api().reduce_accepted_study(None, records_api().freeze_cell_accounting(()))
