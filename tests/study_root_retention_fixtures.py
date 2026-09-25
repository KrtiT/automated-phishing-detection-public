import importlib
from hashlib import sha256
from types import SimpleNamespace

from automated_phishing_detection import execution_receipt as receipt
from automated_phishing_detection._checkpoint_codec import canonical_bytes


def root_case(tmp_path, *, success=False):
    parent = tmp_path.resolve()
    identity = {"kind": "whole_study", "protocol": "study-root-v1"}
    attempt = receipt.reserve_attempt(parent / "root", identity=identity)
    feasibility = {"shortages": [] if success else [{"requirement": "invented"}]}
    execution = identity | {"reservation_sha256": attempt.reservation_sha256}
    barrier = {
        "execution": execution,
        "status": "necessary_capacity_present" if success else "whole_study_hold",
        "predictions_started": False,
        "feasibility": feasibility,
        "feasibility_sha256": sha256(canonical_bytes(feasibility)).hexdigest(),
    }
    contents = {
        "study-intent.json": canonical_bytes({"invented": True}),
        "prediction-barrier.json": canonical_bytes(barrier),
    }
    if success:
        contents["source-results.json"] = canonical_bytes({"invented": True})
    contents["study-accounting.json"] = canonical_bytes({"invented": True})
    extra = reduction_bytes() if success else {}
    public = public_record(execution, contents, extra, feasibility)
    return SimpleNamespace(
        attempt=attempt,
        identity=identity,
        public_path=parent / "public.json",
        contents=contents,
        extra=extra,
        public=public,
    )


def reduction_bytes():
    return {
        "operational-summary.json": canonical_bytes({"groups": []}),
        "study-evidence.json": canonical_bytes({"primary": {}}),
    }


def public_record(execution, contents, extra, feasibility):
    result = {
        "schema_version": 1,
        "protocol": "study-root-v1",
        "status": "study_evidence_published" if extra else "whole_study_hold",
        "execution": execution,
        "accounting_sha256": sha256(contents["study-accounting.json"]).hexdigest(),
        "private_sha256": {
            name: sha256(content).hexdigest()
            for name, content in (contents | extra).items()
        },
    }
    return result | (
        {"operational": {"groups": []}, "study": {"primary": {}}}
        if extra
        else {"feasibility": feasibility}
    )


def api():
    return importlib.import_module("automated_phishing_detection.study_root_retention")


def manager(module, case):
    return module.hold_study_root(
        case.attempt, case.public_path, expected_identity=case.identity
    )


def append_all(writer, case):
    for name, content in case.contents.items():
        writer.append(name, content)


def complete(writer, case):
    return writer.complete(extra_outputs=case.extra, public_summary=case.public)
