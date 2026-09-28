import json
from dataclasses import FrozenInstanceError
from hashlib import sha256

import pytest
from study_admission_fixtures import frame, module


@pytest.mark.parametrize("role", ("internal", "external", "service", "client"))
def test_fixed_roles_round_trip_exact_canonical_bytes(role):
    admission = frame(role=role)
    content = admission.canonical_bytes
    assert module().decode_admission_frame(content) == admission
    assert admission.sha256 == sha256(content).hexdigest()
    assert len(content) <= 4096
    with pytest.raises(FrozenInstanceError):
        admission.role = "replacement"


@pytest.mark.parametrize(
    "changes",
    [
        {"role": "arbitrary"},
        {"parent_pid": True},
        {"parent_pid": 0},
        {"profile_sha256": "A" * 64},
        {"command_sha256": "short"},
        {"predecessor_sha256": "1" * 64},
        {"accepted_inputs_sha256": "1" * 64},
        {"cell_binding_sha256": "1" * 64},
    ],
)
def test_invalid_frame_fields_fail_closed(changes):
    with pytest.raises(module().StudyAdmissionError):
        frame(**changes)


@pytest.mark.parametrize("role", ("external", "service", "client"))
def test_role_requires_predecessor(role):
    with pytest.raises(module().StudyAdmissionError):
        frame(role=role, predecessor_sha256=None)


@pytest.mark.parametrize("field", ("accepted_inputs_sha256", "cell_binding_sha256"))
def test_cell_role_requires_accepted_inputs_and_cell(field):
    with pytest.raises(module().StudyAdmissionError):
        frame(role="service", **{field: None})


@pytest.mark.parametrize("mutation", ("extra", "missing", "space", "duplicate"))
def test_noncanonical_or_open_schema_is_rejected(mutation):
    content = frame().canonical_bytes
    value = json.loads(content)
    if mutation == "extra":
        value["ready"] = True
    elif mutation == "missing":
        del value["role"]
    elif mutation == "duplicate":
        content = content[:-1] + b',"role":"internal"}'
    if mutation in ("extra", "missing", "space"):
        content = json.dumps(value).encode()
    with pytest.raises(module().StudyAdmissionError):
        module().decode_admission_frame(content)
