"""diffpeak.rewards 的奖励与空图边界测试。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diffpeak.rewards import (  # noqa: E402
    RewardConfig,
    compute_reward,
    image_confusion_metrics,
)


def _empty(size: int = 16) -> np.ndarray:
    return np.zeros((size, size), dtype=bool)


def _line(size: int = 16) -> np.ndarray:
    mask = np.zeros((size, size), dtype=bool)
    mask[size // 2, 2 : size - 2] = True
    return mask


class ImageConfusionMetricsTest(unittest.TestCase):
    def test_both_empty_is_perfect(self) -> None:
        metrics = image_confusion_metrics(_empty(), _empty())

        self.assertEqual(metrics["tp"], 0.0)
        self.assertEqual(metrics["fp"], 0.0)
        self.assertEqual(metrics["fn"], 0.0)
        self.assertEqual(metrics["tn"], 256.0)
        self.assertEqual(metrics["iou"], 1.0)
        self.assertEqual(metrics["f1"], 1.0)

    def test_one_side_empty_scores_zero(self) -> None:
        prediction_empty = image_confusion_metrics(_empty(), _line())
        target_empty = image_confusion_metrics(_line(), _empty())

        for metrics in (prediction_empty, target_empty):
            self.assertEqual(metrics["iou"], 0.0)
            self.assertEqual(metrics["f1"], 0.0)
            self.assertGreater(metrics["fn"] + metrics["fp"], 0.0)

    def test_identical_masks_score_one(self) -> None:
        mask = _line()
        metrics = image_confusion_metrics(mask, mask.copy())

        self.assertEqual(metrics["iou"], 1.0)
        self.assertEqual(metrics["f1"], 1.0)

    def test_non_binary_float_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            image_confusion_metrics(np.array([[0.0, 0.5]]), np.zeros((1, 2)))
        with self.assertRaises(ValueError):
            image_confusion_metrics(np.array([[np.nan, 0.0]]), np.zeros((1, 2)))
        with self.assertRaises(ValueError):
            image_confusion_metrics(np.zeros((2, 2)), np.zeros((3, 3)))
        with self.assertRaises(ValueError):
            image_confusion_metrics(np.zeros((2, 2, 1)), np.zeros((2, 2, 1)))


class ComputeRewardTest(unittest.TestCase):
    def test_both_empty_without_points_gets_full_reward(self) -> None:
        result = compute_reward(_empty(), _empty(), point_count=0, config=RewardConfig())

        self.assertAlmostEqual(result["reward"], 1.0, places=6)
        self.assertEqual(result["iou"], 1.0)
        self.assertEqual(result["f1"], 1.0)
        self.assertEqual(result["cldice"], 1.0)
        self.assertEqual(result["path"], 0.0)
        self.assertFalse(result["path_valid"])
        self.assertEqual(result["point_penalty"], 0.0)

    def test_both_empty_still_pays_point_cost(self) -> None:
        result = compute_reward(_empty(), _empty(), point_count=32, config=RewardConfig())

        self.assertAlmostEqual(result["reward"], 0.95, places=6)
        self.assertAlmostEqual(result["point_penalty"], 0.05, places=6)
        self.assertEqual(result["path_requested"], 0)

    def test_only_one_side_empty_has_zero_quality(self) -> None:
        for prediction, target in ((_empty(), _line()), (_line(), _empty())):
            with self.subTest(prediction_empty=not prediction.any()):
                result = compute_reward(
                    prediction, target, point_count=32, config=RewardConfig()
                )
                self.assertEqual(result["iou"], 0.0)
                self.assertEqual(result["f1"], 0.0)
                self.assertEqual(result["cldice"], 0.0)
                self.assertAlmostEqual(result["reward"], -0.05, places=6)

    def test_identical_nonempty_masks_get_full_quality(self) -> None:
        mask = _line()
        result = compute_reward(
            mask, mask.copy(), point_count=0, config=RewardConfig()
        )

        self.assertAlmostEqual(result["iou"], 1.0, places=6)
        self.assertAlmostEqual(result["f1"], 1.0, places=6)
        self.assertAlmostEqual(result["cldice"], 1.0, places=6)
        self.assertAlmostEqual(result["reward"], 1.0, places=6)

    def test_missing_path_renormalizes_remaining_weights(self) -> None:
        config = RewardConfig(
            iou_weight=0.6,
            cldice_weight=0.4,
            path_weight=0.2,
        )
        result = compute_reward(_empty(), _empty(), point_count=0, config=config)

        self.assertFalse(result["path_valid"])
        self.assertAlmostEqual(result["reward"], 1.0, places=6)

    def test_point_cost_uses_reference_not_cap(self) -> None:
        base = compute_reward(_empty(), _empty(), point_count=0, config=RewardConfig())
        more = compute_reward(_empty(), _empty(), point_count=64, config=RewardConfig())

        self.assertAlmostEqual(more["reward"], base["reward"] - 0.1, places=6)
        self.assertAlmostEqual(more["point_penalty"], 0.1, places=6)

    def test_invalid_arguments_raise(self) -> None:
        with self.assertRaises(TypeError):
            compute_reward(_empty(), _empty(), point_count=True, config=RewardConfig())
        with self.assertRaises(TypeError):
            compute_reward(_empty(), _empty(), point_count=1.5, config=RewardConfig())  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            compute_reward(_empty(), _empty(), point_count=-1, config=RewardConfig())
        with self.assertRaises(TypeError):
            compute_reward(_empty(), _empty(), point_count=0, config={})  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            compute_reward(np.zeros((2, 2)), np.zeros((3, 3)), point_count=0, config=RewardConfig())


class PathRewardTest(unittest.TestCase):
    def test_identical_short_network_scores_one(self) -> None:
        """相同短路网：重复源端点复用同一匹配，路径项必须为 1。"""
        mask = np.zeros((10, 10), dtype=bool)
        mask[5, 2:7] = True
        config = RewardConfig(
            iou_weight=0.0,
            cldice_weight=0.0,
            path_weight=1.0,
            point_cost=0.0,
            path_samples=32,
            path_match_radius=8,
            seed=0,
        )

        result = compute_reward(mask, mask.copy(), point_count=0, config=config)

        self.assertTrue(result["path_valid"])
        self.assertGreater(result["path_requested"], 0)
        self.assertEqual(result["path_matched"], result["path_requested"])
        self.assertAlmostEqual(result["path"], 1.0, places=6)
        self.assertAlmostEqual(result["reward"], 1.0, places=6)

    def test_path_is_valid_when_requested_even_if_no_endpoint_matches(self) -> None:
        """全断路时路径项按 0 分计入，而不是移除路径权重后重新归一化。"""
        target = np.zeros((24, 40), dtype=bool)
        prediction = np.zeros((24, 40), dtype=bool)
        target[10:13, 2:30] = True
        prediction[12:15, 2:30] = True

        config = RewardConfig(
            iou_weight=0.3,
            cldice_weight=0.2,
            path_weight=0.5,
            point_cost=0.0,
            path_samples=32,
            path_match_radius=0,
            seed=0,
        )

        result = compute_reward(prediction, target, point_count=0, config=config)

        self.assertGreater(result["iou"], 0.0)
        self.assertGreater(result["path_requested"], 0)
        self.assertEqual(result["path_matched"], 0)
        self.assertTrue(result["path_valid"])
        self.assertEqual(result["path"], 0.0)
        # 路径项有效但得分为 0：保留原权重（0.3/0.2），不重新归一化。
        self.assertAlmostEqual(
            result["reward"], 0.3 * result["iou"], places=6
        )


class RewardConfigTest(unittest.TestCase):
    def test_defaults_match_design_document(self) -> None:
        config = RewardConfig()

        self.assertEqual(config.iou_weight, 0.6)
        self.assertEqual(config.cldice_weight, 0.4)
        self.assertEqual(config.path_weight, 0.0)
        self.assertEqual(config.point_cost, 0.05)
        self.assertEqual(config.point_reference, 32)

    def test_invalid_values_raise(self) -> None:
        invalid = (
            {"iou_weight": -1.0},
            {"cldice_weight": -1.0},
            {"path_weight": -1.0},
            {"point_cost": -1.0},
            {"point_reference": 0},
            {"path_samples": 0},
            {"path_match_radius": -1},
        )
        for kwargs in invalid:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    RewardConfig(**kwargs)


if __name__ == "__main__":
    unittest.main()
