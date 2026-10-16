"""Pins for synthetic test checkouts; never authorize historical execution.

Two compatibility reports are privacy projections in this public repository.
Authenticate their public identities before using those bytes in the existing
mock base binding. Keep the original research contracts and production binders
unchanged. These reports are not inputs to the downstream methods chains.
"""

import json
from hashlib import sha256


def synthetic_public_pins(root, historical):
    projections = json.loads(
        (root / "privacy/2026-10-04/publication.json").read_text()
    )["report_projections"]
    pins = dict(historical)
    for name, expected in historical.items():
        actual = sha256((root / name).read_bytes()).hexdigest()
        if actual == expected:
            continue
        projection = projections[name]
        assert projection["original_sha256"] == expected
        assert projection["public_sha256"] == actual
        assert projection["execution_authority"] is False
        assert name in {
            "reports/inference-compatibility-v1.json",
            "reports/inference-compatibility-v1-preflight-correction.json",
        }
        pins[name] = actual
    return pins
