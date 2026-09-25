import importlib
from contextlib import contextmanager
from hashlib import sha256

CONTENTS = {
    "accepted-inputs.json": b'{"accepted":"invented"}\n',
    "descriptor.json": b'{"descriptor":"invented"}\n',
    "binding.json": b'{"binding":"invented"}\n',
    "manifest": b"invented full manifest\n",
}
RESERVATION = "1" * 64


def module():
    return importlib.import_module(
        "automated_phishing_detection.operational_input_transport"
    )


def expectations():
    return {
        "expected_binding_sha256": sha256(CONTENTS["binding.json"]).hexdigest(),
        "expected_cell_reservation_sha256": RESERVATION,
    }


@contextmanager
def retained(tmp_path):
    transport = module()
    root, cell = tmp_path.resolve() / "accepted-inputs", tmp_path.resolve() / "cell-001"
    with transport.retain_operational_root_inputs(
        root, accepted_inputs=CONTENTS["accepted-inputs.json"]
    ):
        with transport.retain_operational_cell_inputs(
            cell,
            descriptor=CONTENTS["descriptor.json"],
            binding=CONTENTS["binding.json"],
            manifest=CONTENTS["manifest"],
        ):
            yield root, cell


def fake_restore(monkeypatch):
    calls, restored = [], object()

    def restore(contents, expected):
        calls.append((contents, expected))
        return restored

    monkeypatch.setattr(module(), "_restore", restore)
    return calls, restored


def written(tmp_path):
    with retained(tmp_path) as paths:
        return paths
