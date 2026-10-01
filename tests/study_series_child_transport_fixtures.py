"""Private temporary copies of complete invented series child inputs."""

from importlib import import_module
from importlib.util import find_spec

from automated_phishing_detection import operational_input_transport as original


def api():
    name = "automated_phishing_detection._study_series_child_transport"
    assert find_spec(name), "missing held series child transport"
    return import_module(name)


def written(tmp_path, case):
    root, cell = tmp_path.resolve() / "inputs", tmp_path.resolve() / "cell"
    with original.retain_operational_root_inputs(root, accepted_inputs=case.metadata):
        with original.retain_operational_cell_inputs(
            cell,
            descriptor=case.descriptor,
            binding=case.binding,
            manifest=case.manifest,
        ):
            pass
    return root, cell


def hold(paths, case, **changes):
    expected = dict(
        profile_bytes=case.profile,
        frame=case.frame,
        expected_cell_reservation_sha256="3" * 64,
    )
    return api().hold_series_child_inputs(*paths, **(expected | changes))
