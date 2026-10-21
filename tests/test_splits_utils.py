"""diffpeak.data.splits 的划分项解析测试。"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diffpeak.data.splits import (  # noqa: E402
    load_split_entries,
    mask_path_for_image,
    resolve_under_root,
    sample_id_from_image_path,
)


class LoadSplitEntriesTest(unittest.TestCase):
    def test_accepts_sequence_and_preserves_order(self) -> None:
        entries = ["train/b_sat.jpg", "train/a_sat.jpg"]
        loaded = load_split_entries(entries)

        self.assertEqual(loaded, entries)
        self.assertIsNot(loaded, entries)

    def test_reads_json_array(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.json"
            path.write_text(
                json.dumps(["train/a_sat.jpg", "train/b_sat.jpg"]),
                encoding="utf-8",
            )

            loaded = load_split_entries(path)

        self.assertEqual(loaded, ["train/a_sat.jpg", "train/b_sat.jpg"])

    def test_invalid_json_content_raises(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            for contents in ("{}", '"single"', "[]", "[1, 2]", '["ok", 3]'):
                with self.subTest(contents=contents):
                    path.write_text(contents, encoding="utf-8")
                    with self.assertRaises(ValueError):
                        load_split_entries(path)

    def test_missing_file_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            load_split_entries("does-not-exist-split.json")

    def test_invalid_type_raises(self) -> None:
        with self.assertRaises(TypeError):
            load_split_entries(123)  # type: ignore[arg-type]


class PathResolutionTest(unittest.TestCase):
    def test_resolve_under_root_accepts_relative_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            resolved = resolve_under_root(root, "train/a_sat.jpg")

            self.assertEqual(resolved, (root / "train/a_sat.jpg").resolve())

    def test_resolve_under_root_rejects_escape_and_absolute(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(ValueError):
                resolve_under_root(root, "../outside.jpg")
            with self.assertRaises(ValueError):
                resolve_under_root(root, "/etc/passwd")
            with self.assertRaises(ValueError):
                resolve_under_root(root, "train/../../outside.jpg")

    def test_mask_path_derivation(self) -> None:
        self.assertEqual(
            mask_path_for_image("train/100034_sat.jpg"),
            Path("train/100034_mask.png"),
        )
        self.assertEqual(
            mask_path_for_image("train/abc_SAT.jpg"),
            Path("train/abc_mask.png"),
        )
        with self.assertRaises(ValueError):
            mask_path_for_image("train/100034.jpg")

    def test_sample_id_strips_sat_suffix(self) -> None:
        self.assertEqual(
            sample_id_from_image_path("train/100034_sat.jpg"), "100034"
        )
        self.assertEqual(
            sample_id_from_image_path("test/abc_SAT.jpg"), "abc"
        )


if __name__ == "__main__":
    unittest.main()
