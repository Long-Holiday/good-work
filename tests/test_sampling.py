"""目标热图渲染与点采样测试。

覆盖：
* 高斯响应逐点取 max（不是 sum）；
* 无道路样本不产生任何正/负目标点；
* 目标点的最小间距筛选与固定随机种子可复现。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diffpeak.data.sampling import (  # noqa: E402
    RoadTargetSampler,
    render_gaussian_heatmap,
    select_spread_points,
)


class RenderGaussianHeatmapTest(unittest.TestCase):
    def test_empty_points_give_zero_heatmap(self) -> None:
        heatmap = render_gaussian_heatmap(np.empty((0, 2)), size=16, sigma=1.0)

        self.assertEqual(heatmap.shape, (16, 16))
        self.assertEqual(heatmap.dtype, np.float32)
        self.assertFalse(heatmap.any())

    def test_single_point_peaks_at_its_pixel(self) -> None:
        heatmap = render_gaussian_heatmap(np.array([[5.0, 7.0]]), size=16, sigma=1.0)

        self.assertAlmostEqual(float(heatmap[5, 7]), 1.0, places=6)
        self.assertEqual(np.unravel_index(int(np.argmax(heatmap)), heatmap.shape), (5, 7))
        self.assertAlmostEqual(
            float(heatmap[5, 6]), float(heatmap[5, 8]), places=6
        )
        self.assertAlmostEqual(
            float(heatmap[4, 7]), float(heatmap[6, 7]), places=6
        )
        self.assertLessEqual(float(heatmap.max()), 1.0)

    def test_overlapping_points_use_max_not_sum(self) -> None:
        points = np.array([[4.0, 4.0], [4.0, 5.0]])
        heatmap = render_gaussian_heatmap(points, size=9, sigma=1.0)

        # 两个高斯在该像素上的和会大于 1；取 max 时必须恰好保持 1。
        self.assertAlmostEqual(float(heatmap[4, 4]), 1.0, places=6)
        self.assertAlmostEqual(float(heatmap[4, 5]), 1.0, places=6)
        self.assertLessEqual(float(heatmap.max()), 1.0)

    def test_nonpositive_sigma_rounds_and_clips(self) -> None:
        heatmap = render_gaussian_heatmap(np.array([[2.6, 3.4]]), size=8, sigma=0.0)
        self.assertEqual(float(heatmap[3, 3]), 1.0)
        self.assertEqual(float(heatmap.sum()), 1.0)

        clipped = render_gaussian_heatmap(np.array([[-5.0, 100.0]]), size=8, sigma=0.0)
        self.assertEqual(float(clipped[0, 7]), 1.0)

    def test_nonfinite_points_are_ignored(self) -> None:
        heatmap = render_gaussian_heatmap(
            np.array([[np.nan, 1.0], [np.inf, 2.0]]), size=8, sigma=1.0
        )
        self.assertFalse(heatmap.any())

    def test_point_far_outside_heatmap_is_ignored(self) -> None:
        heatmap = render_gaussian_heatmap(np.array([[0.0, 1000.0]]), size=8, sigma=1.0)
        self.assertFalse(heatmap.any())

    def test_invalid_size_raises(self) -> None:
        with self.assertRaises(ValueError):
            render_gaussian_heatmap(np.empty((0, 2)), size=0, sigma=1.0)


class NoRoadTargetTest(unittest.TestCase):
    def test_no_road_produces_no_positive_or_negative_points(self) -> None:
        sampler = RoadTargetSampler()
        mask = np.zeros((32, 32), dtype=bool)
        skeleton = np.zeros_like(mask)

        targets = sampler.sample(
            skeleton=skeleton,
            road_mask=mask,
            rng=np.random.default_rng(0),
        )

        self.assertEqual(targets.positive_yx.shape, (0, 2))
        self.assertEqual(targets.negative_yx.shape, (0, 2))
        self.assertEqual(targets.positive_priorities.shape, (0,))

    def test_empty_road_ignores_nonempty_skeleton(self) -> None:
        """无道路时即使骨架异常非空，也不应凭空生成目标点。"""
        sampler = RoadTargetSampler()
        mask = np.zeros((16, 16), dtype=bool)
        skeleton = np.zeros_like(mask)
        skeleton[8, 2:12] = True

        targets = sampler.sample(
            skeleton=skeleton,
            road_mask=mask,
            rng=np.random.default_rng(1),
        )

        self.assertEqual(len(targets.positive_yx), 0)
        self.assertEqual(len(targets.negative_yx), 0)

    def test_shape_mismatch_raises(self) -> None:
        sampler = RoadTargetSampler()
        with self.assertRaises(ValueError):
            sampler.sample(
                skeleton=np.zeros((8, 8), dtype=bool),
                road_mask=np.zeros((9, 8), dtype=bool),
                rng=np.random.default_rng(0),
            )


class SelectSpreadPointsTest(unittest.TestCase):
    def test_enforces_min_distance(self) -> None:
        points = np.array([[0.0, 0.0], [0.0, 1.0], [0.0, 10.0]])

        kept = select_spread_points(
            points, min_distance=5.0, rng=np.random.default_rng(0)
        )

        self.assertEqual(kept.shape, (2, 2))
        for first in range(len(kept)):
            for second in range(first + 1, len(kept)):
                self.assertGreaterEqual(
                    float(np.linalg.norm(kept[first] - kept[second])), 5.0
                )

    def test_same_seed_is_reproducible(self) -> None:
        points = np.array(
            [[0.0, 0.0], [0.0, 2.0], [0.0, 4.0], [0.0, 6.0], [0.0, 8.0]]
        )

        first = select_spread_points(
            points, min_distance=3.0, rng=np.random.default_rng(123)
        )
        second = select_spread_points(
            points, min_distance=3.0, rng=np.random.default_rng(123)
        )

        self.assertTrue(np.array_equal(first, second))

    def test_structural_priority_is_kept_first(self) -> None:
        points = np.array([[0.0, 0.0], [0.0, 1.0]])
        priorities = np.array([1, 0])

        kept = select_spread_points(
            points,
            min_distance=5.0,
            rng=np.random.default_rng(0),
            priorities=priorities,
        )

        self.assertTrue(np.array_equal(kept, np.array([[0.0, 1.0]])))

    def test_zero_distance_returns_all_points(self) -> None:
        points = np.array([[0.0, 0.0], [0.0, 1.0]])
        kept = select_spread_points(
            points, min_distance=0.0, rng=np.random.default_rng(0)
        )
        self.assertTrue(np.array_equal(kept, points))


if __name__ == "__main__":
    unittest.main()
