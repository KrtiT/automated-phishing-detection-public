"""Run retained layout checks with the original aggregate-evidence binding."""

import importlib.util
import sys
from types import SimpleNamespace

import expand


def main():
    previous = expand.HERE.parent / "consistency-review-20261004"
    sys.path.insert(0, str(previous))
    specification = importlib.util.spec_from_file_location("previous_layout", previous / "layout_check.py")
    wrapper = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(wrapper)
    wrapper.revise = SimpleNamespace(
        ROOT=expand.ROOT,
        BASE=expand.CONTEXT / "deliverables/Tallam_Dissertation_Layout_Reviewed_2026-10-02",
        SOURCE=expand.SOURCE,
        MANUSCRIPT=expand.MANUSCRIPT,
        OUTPUT=expand.OUTPUT,
    )
    wrapper.main()


if __name__ == "__main__":
    main()
