"""DeepGlobeDataset 的单通道山脊场、有效区域与 padding 测试。

数据集只接受带 mask 的监督影像；官方 valid/test 影像没有标注，应通过
``unlabeled.json`` 做推理，而不能被当作训练样本。目标不再包含离散点或
双通道高斯热图。
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diffpeak.data.deepglobe import DeepGlobeDataset  # noqa: E402
from diffpeak.data.ridge import (  # noqa: E402
    BACKGROUND_VALUE,
    FIELD_TARGET_VERSION,
    PADDING_VALUE,
    REGION_FAR_BACKGROUND,
)


class DeepGlobeDatasetTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name) / "data"
        (self.root / "train").mkdir(parents=True)

    def _write_pair(
        self,
        stem: str,
        road: bool,
        *,
        size=(64, 64),
        row_slice=None,
    ) -> None:
        rng = np.random.default_rng(0)
        height, width = size
        image = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
        mask = np.zeros((height, width, 3), dtype=np.uint8)
        if road:
            if row_slice is None:
                row_slice = (height // 2, height // 2 + 1)
            mask[row_slice[0] : row_slice[1], 10 : width - 10] = 255
        Image.fromarray(image, mode="RGB").save(
            self.root / "train" / f"{stem}_sat.jpg"
        )
        Image.fromarray(mask, mode="RGB").save(
            self.root / "train" / f"{stem}_mask.png"
        )

    def _split_file(self, entries) -> Path:
        path = Path(self._tmp.name) / "split.json"
        path.write_text(json.dumps(entries), encoding="utf-8")
        return path

    def _dataset(
        self,
        entries,
        *,
        image_size: int = 64,
        heatmap_size: int = 16,
    ) -> DeepGlobeDataset:
        return DeepGlobeDataset(
            self.root,
            self._split_file(entries),
            image_size=image_size,
            heatmap_size=heatmap_size,
            seed=7,
            ridge_config={"center_radius": 1.0, "boundary_band": 2.0},
        )

    def test_sample_shapes_ranges_and_field_keys(self) -> None:
        self._write_pair("road", road=True)
        dataset = self._dataset(["train/road_sat.jpg"])

        sample = dataset[0]

        self.assertEqual(tuple(sample["image"].shape), (3, 64, 64))
        self.assertEqual(sample["image"].dtype, torch.float32)
        self.assertGreaterEqual(float(sample["image"].min()), 0.0)
        self.assertLessEqual(float(sample["image"].max()), 1.0)
        self.assertEqual(tuple(sample["mask"].shape), (1, 64, 64))
        self.assertEqual(tuple(sample["skeleton"].shape), (1, 64, 64))
        self.assertEqual(tuple(sample["heatmaps"].shape), (1, 16, 16))
        self.assertEqual(sample["heatmaps"].dtype, torch.float32)
        self.assertEqual(tuple(sample["field_valid_mask"].shape), (1, 16, 16))
        self.assertEqual(sample["field_valid_mask"].dtype, torch.float32)
        self.assertEqual(tuple(sample["field_regions"].shape), (4, 16, 16))
        self.assertEqual(sample["field_regions"].dtype, torch.float32)
        self.assertEqual(sample["id"], "road")
        self.assertEqual(sample["original_size"], (64, 64))
        self.assertEqual(sample["resized_size"], (64, 64))

        # 有符号场的数值域与有效区域约定。
        field = sample["heatmaps"][0]
        self.assertGreaterEqual(float(field.min()), BACKGROUND_VALUE)
        self.assertLessEqual(float(field.max()), 1.0)
        valid = sample["field_valid_mask"][0] > 0
        self.assertTrue(bool(valid.all()))
        self.assertTrue(bool((field[valid] >= BACKGROUND_VALUE).all()))
        # 四个区域互斥且每个有效像素恰好属于一个区域。
        regions = sample["field_regions"]
        self.assertTrue(bool(((regions > 0).sum(dim=0) == 1).all()))
        self.assertTrue(bool((regions[:, ~valid].sum() == 0)))

        # 极细（1 像素）道路在下采样后仍然保留为山脊。
        self.assertGreater(float(field.max()), 0.9)
        self.assertTrue(bool((field > 0).any()))

        # 骨架位于道路掩码内。
        self.assertFalse(bool((sample["skeleton"] * (1 - sample["mask"])).any()))

    def test_empty_road_is_background_not_zero(self) -> None:
        self._write_pair("empty", road=False)
        dataset = self._dataset(["train/empty_sat.jpg"])

        sample = dataset[0]

        self.assertEqual(float(sample["mask"].sum()), 0.0)
        self.assertEqual(float(sample["skeleton"].sum()), 0.0)
        field = sample["heatmaps"][0]
        valid = sample["field_valid_mask"][0] > 0
        self.assertTrue(bool((field[valid] == BACKGROUND_VALUE).all()))
        regions = sample["field_regions"]
        self.assertTrue(bool((regions[REGION_FAR_BACKGROUND][valid] == 1).all()))
        self.assertTrue(bool(((regions > 0).sum(dim=0) == 1).all()))
        # padding 仍然使用独立的 0 约定。
        self.assertTrue(
            bool((field[~valid] == PADDING_VALUE).all())
        )

    def test_non_square_padding_uses_zero_and_valid_mask(self) -> None:
        self._write_pair("wide", road=True, size=(32, 64))
        dataset = self._dataset(["train/wide_sat.jpg"])

        sample = dataset[0]

        self.assertEqual(sample["original_size"], (32, 64))
        self.assertEqual(sample["resized_size"], (32, 64))
        valid = sample["field_valid_mask"][0] > 0
        self.assertTrue(bool(valid[:8].all()))
        self.assertFalse(bool(valid[8:].any()))
        field = sample["heatmaps"][0]
        self.assertTrue(bool((field[~valid] == PADDING_VALUE).all()))
        self.assertTrue(
            bool((sample["field_regions"][:, ~valid].sum() == 0))
        )
        # 有效区域里的背景必须是 -1，而不是 0。
        self.assertTrue(bool((field[valid] == BACKGROUND_VALUE).any()))

    def test_field_report_is_json_serialisable(self) -> None:
        self._write_pair("road", road=True)
        dataset = self._dataset(["train/road_sat.jpg"])

        report = dataset.field_report(0)
        encoded = json.dumps(report, ensure_ascii=False)

        self.assertIn(FIELD_TARGET_VERSION, encoded)
        self.assertEqual(report["field_target_version"], FIELD_TARGET_VERSION)
        self.assertEqual(
            sum(report["region_counts"].values()), report["valid_pixels"]
        )
        self.assertEqual(report["unassigned_valid_pixels"], 0)
        # 细道路保留诊断来自源分辨率几何投影，而不是最终热图掩码。
        self.assertGreater(report["source_road_pixels"], 0)
        self.assertEqual(report["projected_road_recall"], 1.0)
        self.assertEqual(report["projected_skeleton_recall"], 1.0)
        self.assertIn("road_coverage_retention_vs_nearest", report)

    def test_missing_mask_is_rejected(self) -> None:
        """没有 GT 的影像不能进入监督数据集。"""
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        Image.fromarray(image, mode="RGB").save(
            self.root / "train" / "unlabeled_sat.jpg"
        )

        with self.assertRaises(FileNotFoundError):
            self._dataset(["train/unlabeled_sat.jpg"])

    def test_path_escape_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self._dataset(["../outside_sat.jpg"])

    def test_negative_and_out_of_range_index(self) -> None:
        self._write_pair("road", road=True)
        dataset = self._dataset(["train/road_sat.jpg"])

        negative = dataset[-1]

        self.assertEqual(negative["id"], "road")
        with self.assertRaises(IndexError):
            dataset[1]

    def test_legacy_sampler_arguments_are_not_accepted(self) -> None:
        self._write_pair("road", road=True)
        with self.assertRaises(TypeError):
            DeepGlobeDataset(
                self.root,
                self._split_file(["train/road_sat.jpg"]),
                image_size=64,
                heatmap_size=16,
                sampling_config={"positive": {}},
            )

    def test_same_seed_reproduces_first_access(self) -> None:
        self._write_pair("road", road=True)
        entries = ["train/road_sat.jpg"]

        first = self._dataset(entries)[0]
        second = self._dataset(entries)[0]

        self.assertTrue(torch.equal(first["heatmaps"], second["heatmaps"]))
        self.assertTrue(
            torch.equal(first["field_valid_mask"], second["field_valid_mask"])
        )
        self.assertTrue(torch.equal(first["field_regions"], second["field_regions"]))
        self.assertTrue(torch.equal(first["image"], second["image"]))

    def test_fixed_target_when_augmentation_disabled(self) -> None:
        """augment=False 时同一 epoch 内重复读取不应改变目标。"""
        self._write_pair("road", road=True)
        dataset = self._dataset(["train/road_sat.jpg"])

        first = dataset[0]
        second = dataset[0]

        self.assertTrue(torch.equal(first["heatmaps"], second["heatmaps"]))
        self.assertTrue(torch.equal(first["skeleton"], second["skeleton"]))


if __name__ == "__main__":
    unittest.main()
