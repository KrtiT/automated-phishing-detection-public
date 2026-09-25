"""Real child transports/replayers with explicitly synthetic scoring, not model proof."""

import asyncio
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from operational_owner_fixtures import models
from test_live_monitor import gmm
from test_selective_service import SyntheticScorer

from automated_phishing_detection import _operational_child_cli as cli
from automated_phishing_detection import _operational_child_context as context
from automated_phishing_detection import bound_runtime, gmm_monitor, saved_evidence
from automated_phishing_detection._checkpoint_codec import canonical_bytes
from automated_phishing_detection._operational_profile import (
    CandidateOperationalProfile,
)
from automated_phishing_detection.execution_preflight import ExecutionBinding
from automated_phishing_detection.operational_cell_client import _run_bound_client
from automated_phishing_detection.operational_cell_service import _run_bound_service


def install_synthetic_owner():
    bound = models()
    primary = saved_evidence._EXPECTED_BINDING_CORE
    points = primary["thresholds"]
    bound.artifact_hashes = tuple(sorted(primary["artifact_hashes"].items()))
    bound.length_only.validation_threshold_record["threshold"] = points["length_only"]
    bound.cascade.stage1_threshold = points["logistic_l1"]
    bound.cascade.transformer_threshold = points["transformer"]
    bound.cascade.half_width = points["half_width"]
    bound.cascade.stage1_model = SimpleNamespace(
        score_urls=lambda urls: (0.2,) * len(urls)
    )
    bound.monitor_boundary = points["monitor_boundary"]
    bound.gmm = gmm.__wrapped__()
    bound_runtime.recheck_binding = lambda binding: None
    bound_runtime.load_bound_models = lambda *arguments: bound
    bound_runtime.SelectiveCascade = synthetic_scorer
    gmm_monitor.score_feature_matrix = lambda *arguments: np.asarray([-100.0])


def synthetic_scorer(model):
    scorer = SyntheticScorer()
    scorer._require_owner = lambda: None
    return scorer


def main(role, binding_values, profile_bytes, mode):
    os.umask(0o077)
    binding = ExecutionBinding(Path(binding_values[0]), *binding_values[1:])
    profile = CandidateOperationalProfile(profile_bytes)
    arguments = cli.parser(role).parse_args()
    keywords = cli._keywords(arguments, role)
    for name in (
        "expected_revision",
        "expected_contract_sha256",
        "expected_operational_profile_sha256",
    ):
        keywords.pop(name)
    context.recheck_binding = lambda binding: None
    install_synthetic_owner()
    runner = _run_bound_service if role == "service" else _run_bound_client
    asyncio.run(runner(binding, profile, **keywords))
    if mode == "client_nonzero" and role == "client":
        raise SystemExit(17)
    if mode == "corrupt_trace" and role == "client":
        destination = Path(os.environ["APD_ATTEMPT_DIRECTORY"]) / "run.json"
        value = json.loads(destination.read_bytes())
        value["trace"]["rows"][0]["monitor_nll"] = -99.0
        destination.write_bytes(canonical_bytes(value))


def write_children(case, *, mode="success"):
    root = Path(__file__).resolve().parents[1]
    scripts = case.binding.root / "scripts"
    scripts.mkdir(parents=True)
    binding = case.binding
    values = (
        str(binding.root),
        binding.revision,
        binding.contract_sha256,
        binding.source_hashes,
        binding.runtime_json,
    )
    for role in ("service", "client"):
        content = (
            "import sys\n"
            f"sys.path[:0] = {[str(root / 'src'), str(root / 'tests')]!r}\n"
            "from operational_cell_runner_children import main\n"
            f"main({role!r}, {values!r}, {case.profile.canonical_bytes!r}, {mode!r})\n"
        )
        (scripts / f"run_operational_{role}.py").write_text(content)
