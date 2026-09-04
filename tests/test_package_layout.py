from importlib.metadata import entry_points, version
from importlib.util import find_spec


def test_installable_package_is_importable():
    assert find_spec("automated_phishing_detection") is not None


def test_phiusiil_preparation_module_is_packaged():
    assert find_spec("automated_phishing_detection.phiusiil") is not None


def test_rq1_baseline_module_is_packaged():
    assert find_spec("automated_phishing_detection.baselines") is not None


def test_package_version_and_single_console_command_are_frozen():
    assert version("automated-phishing-detection") == "0.1.0.dev0"
    project_commands = [
        entry_point
        for entry_point in entry_points(group="console_scripts")
        if entry_point.dist.name == "automated-phishing-detection"
    ]

    assert [(command.name, command.value) for command in project_commands] == [
        ("phishing-research", "automated_phishing_detection.cli:main")
    ]
