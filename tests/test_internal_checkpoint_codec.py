"""Neutral checkpoint encoding preserves the established secondary wire format."""

from test_evaluation_producer import _bound_secondary_fixture

from automated_phishing_detection import evaluation_producer
from automated_phishing_detection._checkpoint_codec import (
    canonical_bytes,
    project_member_bindings,
)


def test_neutral_json_encoding_keeps_ascii_escaping_and_terminal_newline():
    assert canonical_bytes({"value": "é"}) == b'{"value":"\\u00e9"}\n'


def test_binding_projection_retains_only_established_member_fields():
    binding = evaluation_producer._secondary_binding(_bound_secondary_fixture(), 0.5)
    expected = project_member_bindings(binding)
    binding["seeds"][0]["private"] = "not-a-checkpoint-field"
    assert project_member_bindings(binding) == expected
