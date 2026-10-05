"""Reuse the full table/figure checks against the newly rendered edition."""

import shutil
import sys
import unittest

import revise

sys.path.insert(0, str(revise.HERE.parent))
import check_layout_revision_20261002 as check
import test_layout_revision_20261002 as tests


def main():
    root = revise.ROOT / "layout-audit"
    root.mkdir(exist_ok=True)
    shutil.copy2(revise.BASE / "layout-audit/layout-ledger.json", root / "layout-ledger.json")
    check.layout.ROOT = root
    check.layout.BASE = revise.BASE
    check.layout.SOURCE = revise.SOURCE
    check.layout.MANUSCRIPT = revise.MANUSCRIPT
    check.layout.OUTPUT = revise.OUTPUT
    check.main()
    tests.ROOT = revise.ROOT
    tests.MANUSCRIPT = revise.MANUSCRIPT
    names = [name for name in unittest.defaultTestLoader.getTestCaseNames(tests.LayoutTests)
             if name != "test_all_table_text_and_manuscript_preserved"]
    suite = unittest.TestSuite(tests.LayoutTests(name) for name in names)
    with (root / "layout-tests.txt").open("w") as stream:
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise ValueError("Layout regression test failure")
    print(f"{result.testsRun} layout tests passed; exact authorized prose changes checked by audit.py.")


if __name__ == "__main__":
    main()
