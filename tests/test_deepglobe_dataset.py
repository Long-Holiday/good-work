"""DeepGlobeDataset 的无标注/无道路边界测试。

数据集只接受带 mask 的监督影像；官方 valid/test 影像没有标注，应通过
``unlabeled.json`` 做推理，而不能被当作训练样本。该测试同时检查无道路
影像的双通道目标热图保持全零。
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


class DeepGlobeDatasetTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name) / "data"
        (self.root / "train").mkdir(parents=True)

    def _write_pair(self, stem: str, road: bool) -> None:
        rng = np.random.default_rng(0)
        image = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
        mask = np.zeros((64, 64, 3), dtype=np.uint8)
        if road:
            mask[32, 10:54] = 255
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

    def _dataset(self, entries) -> DeepGlobeDataset:
        return DeepGlobeDataset(
            self.root,
            self._split_file(entries),
            image_size=64,
            heatmap_size=16,
            seed=7,
            heatmap_sigma=0.5,
            target_min_distance=1.0,
            sampling_config={
                "positive": {
                    "min_spacing": 8.0,
                    "max_spacing": 16.0,
                    "junction_probability": 0.0,
                    "endpoint_probability": 0.0,
                },
                "negative": {
                    "boundary_count": (0, 0),
                    "background_count": (0, 0),
                },
            },
        )

    def test_sample_shapes_and_ranges(self) -> None:
        self._write_pair("road", road=True)
        dataset = self._dataset(["train/road_sat.jpg"])

        sample = dataset[0]

        self.assertEqual(tuple(sample["image"].shape), (3, 64, 64))
        self.assertEqual(sample["image"].dtype, torch.float32)
        self.assertGreaterEqual(float(sample["image"].min()), 0.0)
        self.assertLessEqual(float(sample["image"].max()), 1.0)
        self.assertEqual(tuple(sample["mask"].shape), (1, 64, 64))
        self.assertEqual(tuple(sample["skeleton"].shape), (1, 64, 64))
        self.assertEqual(tuple(sample["heatmaps"].shape), (2, 16, 16))
        self.assertGreater(float(sample["heatmaps"][0].sum()), 0.0)
        self.assertEqual(float(sample["heatmaps"][1].sum()), 0.0)
        self.assertEqual(sample["id"], "road")
        self.assertEqual(sample["original_size"], (64, 64))

    def test_no_road_sample_has_zero_heatmaps(self) -> None:
        self._write_pair("empty", road=False)
        dataset = self._dataset(["train/empty_sat.jpg"])

        sample = dataset[0]

        self.assertEqual(float(sample["mask"].sum()), 0.0)
        self.assertEqual(float(sample["skeleton"].sum()), 0.0)
        self.assertEqual(float(sample["heatmaps"].sum()), 0.0)

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

    def test_same_seed_reproduces_first_access(self) -> None:
        self._write_pair("road", road=True)
        entries = ["train/road_sat.jpg"]

        first = self._dataset(entries)[0]
        second = self._dataset(entries)[0]

        self.assertTrue(torch.equal(first["heatmaps"], second["heatmaps"]))
        self.assertTrue(torch.equal(first["image"], second["image"]))

    def test_fixed_layout_when_augmentation_disabled(self) -> None:
        """augment=False 时同一 epoch 内重复读取不应改变提示布局。"""
        self._write_pair("road", road=True)
        dataset = self._dataset(["train/road_sat.jpg"])

        first = dataset[0]
        second = dataset[0]

        self.assertTrue(torch.equal(first["heatmaps"], second["heatmaps"]))


if __name__ == "__main__":
    unittest.main()
