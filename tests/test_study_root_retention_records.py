import json
from hashlib import sha256

import pytest
from study_root_retention_fixtures import api, append_all, complete, manager, root_case

from automated_phishing_detection._checkpoint_codec import canonical_bytes


@pytest.mark.parametrize("prefix", [0, 1, 2])
def test_terminal_failure_accounting_can_follow_any_actual_prefix(tmp_path, prefix):
    module, case = api(), root_case(tmp_path)
    original = RuntimeError("invented stop")
    with pytest.raises(RuntimeError) as caught:
        with manager(module, case) as writer:
            for name in tuple(case.contents)[:prefix]:
                writer.append(name, case.contents[name])
            writer.append(
                "study-accounting.json", case.contents["study-accounting.json"]
            )
            raise original
    assert caught.value is original
    assert (case.attempt.directory / "study-accounting.json").exists()
    assert not writer.publishing


@pytest.mark.parametrize("success", [False, True])
@pytest.mark.parametrize(
    "field",
    [
        "schema_version",
        "protocol",
        "status",
        "execution",
        "accounting_sha256",
        "private_sha256",
        "projection",
        "extra",
        "missing",
    ],
)
def test_public_envelope_mutation_never_starts_publication(tmp_path, success, field):
    module, case = api(), root_case(tmp_path, success=success)
    if field == "missing":
        case.public.pop("status")
    elif field == "extra":
        case.public["protected_evaluation_authorized"] = True
    elif field == "projection":
        case.public["study" if success else "feasibility"] = {"forged": True}
    else:
        case.public[field] = True if field == "schema_version" else "forged"
    with pytest.raises(module.StudyRootRetentionError):
        with manager(module, case) as writer:
            append_all(writer, case)
            complete(writer, case)
    assert not writer.publishing
    assert not (case.attempt.directory / "finalize.claim").exists()


@pytest.mark.parametrize(
    "field,value",
    [
        ("predictions_started", True),
        ("predictions_started", 0),
        ("status", "invented"),
        ("execution", {}),
        ("feasibility_sha256", "0" * 64),
        ("feasibility", {"shortages": []}),
        ("feasibility", {"shortages": {}}),
    ],
)
def test_barrier_link_rejects_consistent_outer_rehash(tmp_path, field, value):
    module, case = api(), root_case(tmp_path)
    barrier = json.loads(case.contents["prediction-barrier.json"])
    barrier[field] = value
    content = canonical_bytes(barrier)
    case.contents["prediction-barrier.json"] = content
    case.public["private_sha256"]["prediction-barrier.json"] = sha256(
        content
    ).hexdigest()
    with pytest.raises(module.StudyRootRetentionError):
        with manager(module, case) as writer:
            append_all(writer, case)
            complete(writer, case)
    assert not writer.publishing


@pytest.mark.parametrize(
    "content",
    [
        b"{}",
        b'{"duplicate":1,"duplicate":1}\n',
        b'{"value":NaN}\n',
        b"[]\n",
        bytearray(b"{}\n"),
        "{}\n",
        b'{"value":1e999}\n',
    ],
)
def test_checkpoint_is_exact_canonical_finite_object_bytes(tmp_path, content):
    module, case = api(), root_case(tmp_path)
    with pytest.raises(module.StudyRootRetentionError):
        with manager(module, case) as writer:
            writer.append("study-intent.json", content)
    assert not (case.attempt.directory / "study-intent.json").exists()


@pytest.mark.parametrize(
    "names",
    [
        ("prediction-barrier.json",),
        ("source-results.json",),
        ("study-intent.json", "study-intent.json"),
        ("../other.json",),
        ("study-intent.json", "study-accounting.json", "prediction-barrier.json"),
    ],
)
def test_fixed_names_order_and_single_creation(tmp_path, names):
    module, case = api(), root_case(tmp_path)
    with pytest.raises(module.StudyRootRetentionError):
        with manager(module, case) as writer:
            for name in names:
                writer.append(name, canonical_bytes({"invented": True}))


@pytest.mark.parametrize(
    "extra", [[], {"invented.json": b"{}\n"}, {"operational-summary.json": b"{}\n"}]
)
def test_extra_output_inventory_is_closed(tmp_path, extra):
    module, case = api(), root_case(tmp_path, success=True)
    with pytest.raises(module.StudyRootRetentionError):
        with manager(module, case) as writer:
            append_all(writer, case)
            writer.complete(extra_outputs=extra, public_summary=case.public)
    assert not writer.publishing
