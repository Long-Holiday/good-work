"""run_probe_suite / probe_followup 的候选筛选与覆盖护栏测试。

只构造临时 JSON 证据，不启动子进程、不导入 torch。
"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def _load_script(module_name: str, filename: str):
    spec = importlib.util.spec_from_file_location(
        module_name, ROOT / "scripts" / filename
    )
    if spec is None or spec.loader is None:  # pragma: no cover
        raise ImportError(filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


suite = _load_script("run_probe_suite_under_test", "run_probe_suite.py")
followup = _load_script("probe_followup_under_test", "probe_followup.py")


class CandidateFixture(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.results = self.root / "results_sam2"
        self.results.mkdir()

    def candidates_for(self, module) -> Path:
        path = self.results / module.__name__
        path.mkdir()
        return path

    def make(
        self,
        root: Path,
        name: str,
        *,
        iou=0.3,
        updates=10,
        finite=True,
        status="ok",
        train_hash="train-A",
        val_hash="val-A",
        weights=True,
        validation=True,
        splits=True,
    ) -> Path:
        directory = root / name
        directory.mkdir()
        (directory / "results.json").write_text(
            json.dumps(
                {
                    "status": status,
                    "parameters": {"all_finite": finite},
                    "measurement": {"optimizer": {"successful_updates": updates}},
                }
            ),
            encoding="utf-8",
        )
        if validation:
            (directory / "validation.json").write_text(
                json.dumps({"status": "ok", "metrics": {"iou": iou}}),
                encoding="utf-8",
            )
        if splits:
            (directory / "splits.json").write_text(
                json.dumps(
                    {
                        "train": {"seed": 2026, "subset_entries_sha256": train_hash},
                        "val": {"seed": 2026, "subset_entries_sha256": val_hash},
                    }
                ),
                encoding="utf-8",
            )
        if weights:
            (directory / "weights.pt").write_bytes(b"weights")
        return directory


class SelectionTest(CandidateFixture):
    def test_selects_best_valid_candidate(self):
        for module in (suite, followup):
            with self.subTest(module=module.__name__):
                root = self.candidates_for(module)
                self.make(root, "X_lr_3e-05", iou=0.40, updates=12)
                self.make(root, "X_lr_0.0001", iou=0.50, updates=12)
                self.make(root, "X_lr_0.0003", iou=0.30, updates=0)
                selection = module._select_candidate(root.glob("X_lr_*"))
                self.assertEqual(selection["selected"]["name"], "X_lr_0.0001")
                failed = {
                    row["name"] for row in selection["candidates"] if not row["valid"]
                }
                self.assertEqual(failed, {"X_lr_0.0003"})
                self.assertFalse(selection["tie"])

    def test_missing_or_nonfinite_iou_is_not_selected(self):
        for module in (suite, followup):
            with self.subTest(module=module.__name__):
                root = self.candidates_for(module)
                self.make(root, "X_lr_1e-04", iou=None, updates=8)
                self.make(root, "X_lr_2e-04", iou=float("nan"), updates=8)
                self.make(root, "X_lr_3e-04", iou=0.2, updates=8)
                selection = module._select_candidate(root.glob("X_lr_*"))
                self.assertEqual(selection["selected"]["name"], "X_lr_3e-04")

    def test_all_failed_candidates_raise_instead_of_selecting_files(self):
        for module in (suite, followup):
            with self.subTest(module=module.__name__):
                root = self.candidates_for(module)
                self.make(root, "X_lr_3e-05", updates=0)
                self.make(root, "X_lr_0.0001", updates=0, finite=False)
                with self.assertRaises(RuntimeError) as raised:
                    module._select_candidate(root.glob("X_lr_*"))
                message = str(raised.exception)
                self.assertIn("所有候选均未通过", message)
                self.assertIn("no_successful_updates", message)
                self.assertIn("parameters_not_finite", message)

    def test_zero_update_validation_and_weights_are_not_success(self):
        # C 选择曾只检查文件存在；0 次实际更新的“成功”文件必须被排除。
        for module in (suite, followup):
            with self.subTest(module=module.__name__):
                root = self.candidates_for(module)
                self.make(root, "C_lr_2e-05", updates=0, iou=0.9)
                with self.assertRaises(RuntimeError):
                    module._select_candidate(root.glob("C_lr_*"))

    def test_missing_validation_or_weights_is_invalid(self):
        for module in (suite, followup):
            with self.subTest(module=module.__name__):
                root = self.candidates_for(module)
                self.make(root, "X_lr_1e-04", validation=False)
                self.make(root, "X_lr_2e-04", weights=False)
                rows = {
                    row["name"]: row
                    for row in (module._candidate(path) for path in root.glob("X_lr_*"))
                }
                self.assertIn(
                    "validation_iou_missing_or_nonfinite",
                    rows["X_lr_1e-04"]["reasons"],
                )
                self.assertIn("weights_missing", rows["X_lr_2e-04"]["reasons"])

    def test_mixed_train_or_val_subsets_are_rejected(self):
        for module in (suite, followup):
            with self.subTest(module=module.__name__):
                root = self.candidates_for(module)
                self.make(root, "X_lr_1e-04", train_hash="train-A")
                self.make(root, "X_lr_2e-04", train_hash="train-B")
                with self.assertRaises(RuntimeError) as raised:
                    module._select_candidate(root.glob("X_lr_*"))
                self.assertIn("子集不一致", str(raised.exception))

    def test_exact_tie_uses_lowest_lr_and_is_recorded(self):
        for module in (suite, followup):
            with self.subTest(module=module.__name__):
                root = self.candidates_for(module)
                self.make(root, "X_lr_0.0003", iou=0.5)
                self.make(root, "X_lr_3e-05", iou=0.5)
                self.make(root, "X_lr_0.0001", iou=0.1)
                selection = module._select_candidate(root.glob("X_lr_*"))
                self.assertEqual(selection["selected"]["name"], "X_lr_3e-05")
                self.assertTrue(selection["tie"])
                self.assertEqual(
                    set(selection["tied"]), {"X_lr_3e-05", "X_lr_0.0003"}
                )
                event = module._selection_event("A", selection)
                self.assertTrue(event["tie"])
                self.assertEqual(event["selected"], "X_lr_3e-05")

    def test_d_system_gate_requires_actual_updates(self):
        for module in (suite, followup):
            with self.subTest(module=module.__name__):
                root = self.candidates_for(module)
                self.make(
                    root, "D_system_off", updates=0,
                    validation=False, weights=False,
                )
                self.make(
                    root, "D_system_on", updates=4,
                    validation=False, weights=False,
                )
                off = module._candidate(
                    root / "D_system_off",
                    require_validation=False,
                    require_weights=False,
                )
                on = module._candidate(
                    root / "D_system_on",
                    require_validation=False,
                    require_weights=False,
                )
                self.assertFalse(off["valid"])
                self.assertIn("no_successful_updates", off["reasons"])
                self.assertTrue(on["valid"])


class OverwriteGuardTest(CandidateFixture):
    def test_link_refuses_file_directory_and_dangling_symlink(self):
        for module in (suite, followup):
            with self.subTest(module=module.__name__):
                link = self.root / f"{module.__name__}_selected"
                link.write_text("keep", encoding="utf-8")
                with self.assertRaises(FileExistsError):
                    module._link_selected(link, self.results)
                self.assertEqual(link.read_text(encoding="utf-8"), "keep")
                link.unlink()

                link.mkdir()
                with self.assertRaises(FileExistsError):
                    module._link_selected(link, self.results)
                link.rmdir()

                link.symlink_to(self.root / "missing", target_is_directory=True)
                with self.assertRaises(FileExistsError):
                    module._link_selected(link, self.results)
                self.assertTrue(link.is_symlink())
                link.unlink()

                module._link_selected(link, self.results / "target")
                self.assertTrue(link.is_symlink())

    def test_followup_main_refuses_suite_before_creating_parent_link(self):
        (self.root / "configs").mkdir()
        (self.root / "configs/probe_sam2_constant_lr.yaml").write_text("{}", encoding="utf-8")
        (self.root / "suite_sam2_followup.jsonl").write_text("keep\n", encoding="utf-8")
        with patch.object(
            sys, "argv", ["probe_followup.py", "--root", str(self.root)]
        ):
            with self.assertRaises(FileExistsError):
                followup.main()
        self.assertFalse((self.root / "results_sam2/B_selected").exists())
        self.assertEqual(
            (self.root / "suite_sam2_followup.jsonl").read_text(encoding="utf-8"),
            "keep\n",
        )

    def test_suite_main_refuses_suite_before_writing_constant_config(self):
        (self.root / "suite_sam2_learning.jsonl").write_text("keep\n", encoding="utf-8")
        with patch.object(
            sys, "argv",
            ["run_probe_suite.py", "--root", str(self.root), "--phase", "learning"],
        ):
            with self.assertRaises(FileExistsError):
                suite.main()
        self.assertFalse(
            (self.root / "configs/probe_sam2_constant_lr.yaml").exists()
        )
        self.assertEqual(
            (self.root / "suite_sam2_learning.jsonl").read_text(encoding="utf-8"),
            "keep\n",
        )

    def test_followup_refuses_existing_parent_link_before_any_run(self):
        (self.root / "configs").mkdir()
        (self.root / "configs/probe_sam2_constant_lr.yaml").write_text("{}", encoding="utf-8")
        (self.results / "B_selected").symlink_to("missing", target_is_directory=True)
        with patch.object(
            sys, "argv", ["probe_followup.py", "--root", str(self.root)]
        ):
            with self.assertRaises(FileExistsError):
                followup.main()
        self.assertFalse((self.root / "suite_sam2_followup.jsonl").exists())

    def test_suite_refuses_existing_a_selected_before_learning_run(self):
        (self.results / "A_selected").symlink_to("missing", target_is_directory=True)
        with patch.object(
            sys, "argv",
            ["run_probe_suite.py", "--root", str(self.root), "--phase", "learning"],
        ):
            with self.assertRaises(FileExistsError):
                suite.main()
        self.assertFalse((self.root / "suite_sam2_learning.jsonl").exists())
        self.assertFalse(
            (self.root / "configs/probe_sam2_constant_lr.yaml").exists()
        )

    def test_followup_requires_constant_lr_config(self):
        with patch.object(
            sys, "argv", ["probe_followup.py", "--root", str(self.root)]
        ):
            with self.assertRaises(FileNotFoundError) as raised:
                followup.main()
        self.assertIn("probe_sam2_constant_lr.yaml", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
