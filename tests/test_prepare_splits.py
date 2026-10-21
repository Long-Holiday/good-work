"""scripts/prepare_splits.py 的数据划分测试。

测试在临时目录中构造最小数据树，不读取或修改真实 DeepGlobe 数据。
核心断言：train/val/unlabeled 三个清单两两不相交，且每次划分可复现。
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "prepare_splits.py"

_spec = importlib.util.spec_from_file_location("prepare_splits_under_test", SCRIPT)
if _spec is None or _spec.loader is None:  # pragma: no cover
    raise ImportError(f"无法加载 {SCRIPT}")
prepare_splits = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(prepare_splits)


class PrepareSplitsTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.base = Path(self._tmp.name)
        self.root = self.base / "data"
        for split in ("train", "valid", "test"):
            (self.root / split).mkdir(parents=True, exist_ok=True)

    def _add_labeled(self, names: list[str]) -> None:
        for name in names:
            (self.root / "train" / f"{name}_sat.jpg").write_bytes(b"image")
            (self.root / "train" / f"{name}_mask.png").write_bytes(b"mask")

    def _add_unlabeled(self, split: str, names: list[str]) -> None:
        for name in names:
            (self.root / split / f"{name}_sat.jpg").write_bytes(b"image")

    def _run(self, output: Path, *extra: str) -> None:
        argv = [
            "prepare_splits.py",
            "--root",
            str(self.root),
            "--output",
            str(output),
            *extra,
        ]
        old_argv = sys.argv
        try:
            sys.argv = argv
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
                io.StringIO()
            ):
                prepare_splits.main()
        finally:
            sys.argv = old_argv

    def _load(self, output: Path, name: str):
        return json.loads((output / name).read_text(encoding="utf-8"))

    def test_split_manifests_are_disjoint_and_complete(self) -> None:
        labeled = [f"train_{index:02d}" for index in range(10)]
        self._add_labeled(labeled)
        self._add_unlabeled("valid", ["valid_a", "valid_b"])
        self._add_unlabeled("test", ["test_a"])

        output = self.base / "splits"
        self._run(output, "--val-fraction", "0.3", "--seed", "42")

        train = self._load(output, "train.json")
        val = self._load(output, "val.json")
        unlabeled = self._load(output, "unlabeled.json")
        protocol = self._load(output, "protocol.json")

        train_set, val_set, unlabeled_set = set(train), set(val), set(unlabeled)
        self.assertFalse(train_set & val_set)
        self.assertFalse(train_set & unlabeled_set)
        self.assertFalse(val_set & unlabeled_set)

        expected_labeled = {f"train/{name}_sat.jpg" for name in labeled}
        self.assertEqual(train_set | val_set, expected_labeled)
        self.assertEqual(
            unlabeled_set,
            {
                "valid/valid_a_sat.jpg",
                "valid/valid_b_sat.jpg",
                "test/test_a_sat.jpg",
            },
        )

        self.assertEqual(len(train), 7)
        self.assertEqual(len(val), 3)
        self.assertEqual(protocol["train_count"], len(train))
        self.assertEqual(protocol["val_count"], len(val))
        self.assertEqual(protocol["seed"], 42)
        self.assertEqual(protocol["val_fraction"], 0.3)
        self.assertEqual(protocol["source"], "labeled train only")

        for manifest in (train, val, unlabeled):
            self.assertEqual(manifest, sorted(manifest))
            self.assertTrue(all(not item.endswith("_mask.png") for item in manifest))

    def test_same_seed_reproduces_split(self) -> None:
        self._add_labeled([f"train_{index:02d}" for index in range(8)])

        first = self.base / "splits_first"
        second = self.base / "splits_second"
        self._run(first, "--val-fraction", "0.25", "--seed", "7")
        self._run(second, "--val-fraction", "0.25", "--seed", "7")

        self.assertEqual(
            self._load(first, "train.json"), self._load(second, "train.json")
        )
        self.assertEqual(
            self._load(first, "val.json"), self._load(second, "val.json")
        )

    def test_refuses_to_overwrite_existing_manifest(self) -> None:
        self._add_labeled([f"train_{index:02d}" for index in range(4)])
        output = self.base / "splits"
        self._run(output, "--seed", "1")
        before = (output / "train.json").read_bytes()

        with self.assertRaises(SystemExit):
            self._run(output, "--seed", "2")

        self.assertEqual((output / "train.json").read_bytes(), before)

    def test_rejects_image_without_mask(self) -> None:
        (self.root / "train" / "broken_sat.jpg").write_bytes(b"image")
        self._add_labeled(["ok"])

        with self.assertRaises(SystemExit):
            self._run(self.base / "splits")

    def test_rejects_invalid_val_fraction(self) -> None:
        self._add_labeled(["a", "b"])
        for fraction in ("0", "1", "-0.1", "1.5"):
            with self.subTest(fraction=fraction):
                with self.assertRaises(SystemExit):
                    self._run(
                        self.base / f"splits_{fraction.replace('.', '_')}",
                        "--val-fraction",
                        fraction,
                    )


if __name__ == "__main__":
    unittest.main()
