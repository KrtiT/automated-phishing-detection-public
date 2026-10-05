import importlib
import importlib.util
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


class CompletePackageTests(unittest.TestCase):
    def checker(self):
        self.assertIsNotNone(importlib.util.find_spec("check_complete_package_20261002"))
        return importlib.import_module("check_complete_package_20261002")

    def test_pdf_normalization_preserves_meaningful_characters(self):
        checker = self.checker()
        self.assertEqual(checker.normalize("ﬁnal\n soft\u00adhyphen"), "finalsofthyphen")
        self.assertNotEqual(checker.normalize("0.203249"), checker.normalize("0.203294"))

    def test_hash_inventory_rejects_changed_bytes(self):
        checker = self.checker()
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "evidence.txt"
            source.write_text("preserved")
            inventory = {source.name: checker.digest(source)}
            self.assertEqual(checker.verify_hashes(root, inventory), 1)
            source.write_text("changed")
            with self.assertRaisesRegex(ValueError, "Hash mismatch"):
                checker.verify_hashes(root, inventory)

    def test_hash_inventory_rejects_path_escape(self):
        checker = self.checker()
        with TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "Unsafe"):
                checker.verify_hashes(Path(temporary), {"../outside": "0" * 64})

    def test_table_check_rejects_dropped_rows(self):
        checker = self.checker()
        checker.require_table([["A"], ["B"]], [["A"], ["B"]], "sample")
        with self.assertRaisesRegex(ValueError, "Table mismatch"):
            checker.require_table([["A"]], [["A"], ["B"]], "sample")

    def test_packager_rejects_sensitive_member_and_duplicate(self):
        self.assertIsNotNone(importlib.util.find_spec("package_complete_20261002"))
        package = importlib.import_module("package_complete_20261002")
        with TemporaryDirectory() as temporary:
            source = Path(temporary) / "data.txt"
            source.write_text("safe")
            selected = {}
            package.add_member(selected, source, "evidence/safe.txt")
            with self.assertRaises(ValueError):
                package.add_member(selected, source, "evidence/safe.txt")
            for target in ["../escape.txt", "private-authorization.json", "predictions.jsonl", "model.json"]:
                with self.assertRaises(ValueError):
                    package.add_member({}, source, target)

    def test_archive_detects_changed_content(self):
        self.assertIsNotNone(importlib.util.find_spec("package_complete_20261002"))
        package = importlib.import_module("package_complete_20261002")
        from zipfile import ZipFile

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "entry.txt"
            source.write_text("original")
            archive = root / "test.zip"
            with ZipFile(archive, "w") as stream:
                stream.write(source, "package/entry.txt")
            expected = {"package/entry.txt": package.digest(source)}
            package.verify_archive(archive, expected)
            source.write_text("changed")
            with self.assertRaisesRegex(ValueError, "Archive content"):
                package.verify_archive(archive, {"package/entry.txt": package.digest(source)})


if __name__ == "__main__":
    unittest.main()
