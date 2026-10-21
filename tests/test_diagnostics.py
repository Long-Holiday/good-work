"""diffpeak.diagnostics 的提示质量与组内诊断边界测试。"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diffpeak.diagnostics import (  # noqa: E402
    PROMPT_METRIC_KEYS,
    REWARD_GROUP_METRIC_KEYS,
    prompt_quality,
    reward_group_diagnostics,
)


def _empty_points():
    return (
        torch.empty(0, 2, dtype=torch.float32),
        torch.empty(0, dtype=torch.long),
    )


def _zeros_heatmap(size: int = 8) -> torch.Tensor:
    return torch.zeros(2, size, size, dtype=torch.float32)


class PromptQualityEmptyTest(unittest.TestCase):
    def test_no_points_still_reports_heatmap_metrics(self) -> None:
        target = np.zeros((4, 4), dtype=bool)
        result = prompt_quality(
            _empty_points(),
            target,
            _zeros_heatmap(4),
            image_size=8,
            threshold=0.5,
        )

        self.assertEqual(tuple(result), PROMPT_METRIC_KEYS)
        self.assertTrue(math.isnan(result["positive_on_road_rate"]))
        self.assertTrue(math.isnan(result["negative_on_background_rate"]))
        self.assertTrue(math.isnan(result["point_outside_rate"]))
        self.assertTrue(math.isnan(result["skeleton_coverage"]))
        self.assertEqual(result["no_positive_rate"], 1.0)
        self.assertEqual(result["heatmap_above_threshold_rate"], 0.0)
        self.assertEqual(result["heatmap_mean"], 0.0)
        self.assertEqual(result["heatmap_max"], 0.0)

    def test_negative_only_is_not_counted_as_no_negative(self) -> None:
        target = np.zeros((4, 4), dtype=bool)
        points = (
            torch.tensor([[1.0, 1.0]]),
            torch.tensor([0], dtype=torch.long),
        )
        result = prompt_quality(
            points, target, _zeros_heatmap(4), image_size=8, threshold=0.5
        )

        self.assertEqual(result["no_positive_rate"], 1.0)
        self.assertTrue(math.isnan(result["positive_on_road_rate"]))
        self.assertEqual(result["negative_on_background_rate"], 1.0)
        self.assertEqual(result["point_outside_rate"], 0.0)


class PromptQualityPointTest(unittest.TestCase):
    def test_positive_negative_and_padding_rates(self) -> None:
        target = np.zeros((4, 4), dtype=bool)
        target[0, :] = True  # 第一行是道路
        points = (
            torch.tensor(
                [
                    [0.0, 0.0],  # 正点，道路
                    [1.0, 0.0],  # 正点，道路
                    [4.0, 0.0],  # 正点，落在右侧 padding
                    [0.0, 1.0],  # 负点，有效背景
                    [5.0, 5.0],  # 负点，落在 padding
                ],
                dtype=torch.float32,
            ),
            torch.tensor([1, 1, 1, 0, 0], dtype=torch.long),
        )

        result = prompt_quality(
            points,
            target,
            _zeros_heatmap(4),
            image_size=8,
            threshold=0.5,
        )

        self.assertAlmostEqual(result["positive_on_road_rate"], 2.0 / 3.0)
        self.assertAlmostEqual(result["negative_on_background_rate"], 1.0 / 2.0)
        self.assertEqual(result["no_positive_rate"], 0.0)
        self.assertAlmostEqual(result["point_outside_rate"], 2.0 / 5.0)

    def test_half_pixel_coordinates_use_nearest_pixel(self) -> None:
        target = np.zeros((4, 4), dtype=bool)
        target[1, 1] = True
        points = (
            torch.tensor([[1.4, 0.6]], dtype=torch.float32),
            torch.tensor([1], dtype=torch.long),
        )

        result = prompt_quality(
            points, target, _zeros_heatmap(4), image_size=8, threshold=0.5
        )

        self.assertEqual(result["positive_on_road_rate"], 1.0)
        self.assertEqual(result["point_outside_rate"], 0.0)


class PromptQualityHeatmapTest(unittest.TestCase):
    def test_padding_is_excluded_by_geometry_pixel_centers(self) -> None:
        # target 有效区域 6x6、画布 8、热图 4：热图中心为
        # 0.5/2.5/4.5/6.5，只有前 3 行/列落在 [−0.5, 5.5) 内，
        # 因此有效热图区域是 3x3，而不是 round(6*4/8)^2=2x2。
        target = np.zeros((6, 6), dtype=bool)
        heatmap = torch.zeros(2, 4, 4, dtype=torch.float32)
        heatmap[0, 2, 2] = 0.6  # 有效
        heatmap[1, 2, 0] = 0.4  # 有效
        heatmap[0, 3, 3] = 0.9  # padding，必须排除

        result = prompt_quality(
            _empty_points(),
            target,
            heatmap,
            image_size=8,
            threshold=(0.5, 0.3),
        )

        self.assertAlmostEqual(result["heatmap_above_threshold_rate"], 2.0 / 9.0)
        self.assertAlmostEqual(result["heatmap_mean"], 1.0 / 18.0)
        self.assertAlmostEqual(result["heatmap_max"], 0.6)

    def test_per_channel_thresholds_are_merged_per_pixel(self) -> None:
        target = np.zeros((6, 6), dtype=bool)
        heatmap = torch.zeros(2, 4, 4, dtype=torch.float32)
        heatmap[0, 0, 0] = 0.6  # 超过两个阈值
        heatmap[1, 1, 1] = 0.4  # 仅超过负通道阈值 0.3

        scalar = prompt_quality(
            _empty_points(), target, heatmap, image_size=8, threshold=0.5
        )
        per_channel = prompt_quality(
            _empty_points(),
            target,
            heatmap,
            image_size=8,
            threshold=(0.5, 0.3),
        )
        as_tensor = prompt_quality(
            _empty_points(),
            target,
            heatmap,
            image_size=8,
            threshold=torch.tensor([0.5, 0.3]),
        )

        self.assertAlmostEqual(scalar["heatmap_above_threshold_rate"], 1.0 / 9.0)
        self.assertAlmostEqual(
            per_channel["heatmap_above_threshold_rate"], 2.0 / 9.0
        )
        self.assertAlmostEqual(
            as_tensor["heatmap_above_threshold_rate"], 2.0 / 9.0
        )

    def test_padding_only_heatmap_stays_zero(self) -> None:
        target = np.zeros((4, 4), dtype=bool)
        heatmap = torch.zeros(2, 4, 4, dtype=torch.float32)
        heatmap[:, 2:, 2:] = 1.0  # 对应右侧/底部 padding

        result = prompt_quality(
            _empty_points(), target, heatmap, image_size=8, threshold=0.5
        )

        self.assertEqual(result["heatmap_above_threshold_rate"], 0.0)
        self.assertEqual(result["heatmap_mean"], 0.0)
        self.assertEqual(result["heatmap_max"], 0.0)


class PromptQualitySkeletonCoverageTest(unittest.TestCase):
    def test_local_coverage_uses_radius(self) -> None:
        target = np.zeros((8, 8), dtype=bool)
        target[3, 1:7] = True
        points = (
            torch.tensor([[1.0, 3.0]], dtype=torch.float32),
            torch.tensor([1], dtype=torch.long),
        )

        near = prompt_quality(
            points, target, _zeros_heatmap(8), image_size=8,
            threshold=0.5, radius=1.0,
        )
        default = prompt_quality(
            points, target, _zeros_heatmap(8), image_size=8,
            threshold=0.5,
        )

        # 6 个骨架像素中，距离 (1, 3) <= 1 的是 x=1、2。
        self.assertAlmostEqual(near["skeleton_coverage"], 2.0 / 6.0)
        # 默认半径 8 覆盖整条 6 像素线。
        self.assertEqual(default["skeleton_coverage"], 1.0)

    def test_empty_prompt_is_zero_but_empty_gt_is_nan(self) -> None:
        target = np.zeros((8, 8), dtype=bool)
        target[3, 1:7] = True
        negative_only = (
            torch.tensor([[1.0, 3.0]]),
            torch.tensor([0], dtype=torch.long),
        )
        no_positive = prompt_quality(
            negative_only, target, _zeros_heatmap(8), image_size=8, threshold=0.5
        )

        empty_target = prompt_quality(
            (
                torch.tensor([[1.0, 3.0]]),
                torch.tensor([1], dtype=torch.long),
            ),
            np.zeros((8, 8), dtype=bool),
            _zeros_heatmap(8),
            image_size=8,
            threshold=0.5,
        )

        # GT 有骨架但没有正提示：0，而不是被宏平均忽略的 NaN。
        self.assertEqual(no_positive["skeleton_coverage"], 0.0)
        # GT 本身没有骨架：无法定义覆盖，NaN。
        self.assertTrue(math.isnan(empty_target["skeleton_coverage"]))

    def test_outside_positive_points_do_not_contribute_coverage(self) -> None:
        target = np.zeros((8, 8), dtype=bool)
        target[3, 1:7] = True
        outside_positive = (
            torch.tensor([[20.0, 20.0]]),  # image_size=8，位于 padding 外
            torch.tensor([1], dtype=torch.long),
        )

        result = prompt_quality(
            outside_positive,
            target,
            _zeros_heatmap(8),
            image_size=8,
            threshold=0.5,
        )

        self.assertEqual(result["point_outside_rate"], 1.0)
        self.assertEqual(result["skeleton_coverage"], 0.0)


class PromptQualityValidationTest(unittest.TestCase):
    def test_invalid_inputs_raise(self) -> None:
        target = np.zeros((4, 4), dtype=bool)
        heatmap = _zeros_heatmap(4)
        good_points = (
            torch.tensor([[0.0, 0.0]]),
            torch.tensor([1], dtype=torch.long),
        )

        with self.assertRaises(ValueError):
            prompt_quality(
                (torch.tensor([[0.0, 0.0]]), torch.tensor([1, 0])),
                target,
                heatmap,
                image_size=8,
                threshold=0.5,
            )
        with self.assertRaises(ValueError):
            prompt_quality(
                (torch.tensor([[0.0, 0.5, 1.0]]), torch.tensor([1])),
                target,
                heatmap,
                image_size=8,
                threshold=0.5,
            )
        with self.assertRaises(ValueError):
            prompt_quality(
                good_points, target, torch.zeros(1, 4, 4), image_size=8,
                threshold=0.5,
            )
        with self.assertRaises(ValueError):
            prompt_quality(
                good_points,
                np.full((4, 4), 0.5, dtype=np.float32),
                heatmap,
                image_size=8,
                threshold=0.5,
            )
        with self.assertRaises(ValueError):
            prompt_quality(
                good_points, np.zeros((9, 4), dtype=bool), heatmap,
                image_size=8, threshold=0.5,
            )
        with self.assertRaises(ValueError):
            prompt_quality(
                good_points, target, heatmap, image_size=8,
                threshold=(0.1, 0.2, 0.3),
            )
        with self.assertRaises(ValueError):
            prompt_quality(
                good_points, target, heatmap, image_size=8,
                threshold=0.5, radius=-1.0,
            )
        with self.assertRaises(ValueError):
            prompt_quality(
                good_points,
                target,
                torch.full((2, 4, 4), float("nan")),
                image_size=8,
                threshold=0.5,
            )


class RewardGroupDiagnosticsTest(unittest.TestCase):
    def test_same_image_groups_are_not_mixed(self) -> None:
        rows = [
            # 轨迹优先：g0b0, g0b1, g1b0, g1b1
            {"quality_reward": 0.4, "point_penalty": 0.0, "point_count": 10},
            {"quality_reward": 0.3, "point_penalty": 0.0, "point_count": 30},
            {"quality_reward": 0.6, "point_penalty": 0.0, "point_count": 20},
            {"quality_reward": 0.7, "point_penalty": 0.0, "point_count": 10},
        ]
        advantages = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float64)

        result = reward_group_diagnostics(rows, advantages)

        self.assertEqual(tuple(result), REWARD_GROUP_METRIC_KEYS)
        self.assertAlmostEqual(result["quality_reward_mean"], 0.5)
        self.assertAlmostEqual(result["point_penalty_mean"], 0.0)
        self.assertAlmostEqual(result["reward_mean"], 0.5)
        self.assertAlmostEqual(result["point_count_mean"], 17.5)
        self.assertAlmostEqual(result["quality_reward_group_std"], 0.15)
        self.assertAlmostEqual(result["point_penalty_group_std"], 0.0)
        self.assertAlmostEqual(result["reward_group_std"], 0.15)
        self.assertAlmostEqual(result["point_count_group_std"], 7.5)
        self.assertAlmostEqual(result["advantage_abs_mean"], 0.5)
        self.assertAlmostEqual(result["advantage_group_std"], 0.5)
        # 逐图分别为 +1 与 −1；若错误地跨图汇总会得到约 −0.30。
        self.assertAlmostEqual(result["advantage_point_count_corr"], 0.0)
        self.assertAlmostEqual(result["reward_point_count_corr"], 0.0)
        self.assertAlmostEqual(result["quality_reward_point_count_corr"], 0.0)
        self.assertEqual(result["constant_reward_group_rate"], 0.0)
        self.assertEqual(result["undefined_advantage_corr_rate"], 0.0)
        self.assertEqual(result["no_point_rate"], 0.0)
        self.assertEqual(result["n_images"], 2.0)
        self.assertEqual(result["n_trajectories"], 2.0)

    def test_constant_groups_give_nan_correlations(self) -> None:
        rows = [
            # g0b0, g0b1
            {"quality_reward": 1.0, "point_penalty": 0.0, "point_count": 5},
            {"quality_reward": 0.7, "point_penalty": 0.1, "point_count": 0},
            # g1b0, g1b1
            {"quality_reward": 1.0, "point_penalty": 0.0, "point_count": 5},
            {"quality_reward": 0.5, "point_penalty": 0.2, "point_count": 10},
            # g2b0, g2b1
            {"quality_reward": 1.0, "point_penalty": 0.0, "point_count": 5},
            {"quality_reward": 0.3, "point_penalty": 0.3, "point_count": 20},
        ]
        advantages = torch.tensor(
            [[0.5, 1.0], [0.0, 0.0], [0.5, -1.0]], dtype=torch.float64
        )

        result = reward_group_diagnostics(rows, advantages)

        self.assertAlmostEqual(result["point_count_group_std"], 4.0824829046)
        self.assertAlmostEqual(
            result["quality_reward_group_std"], 0.0816496581
        )
        self.assertAlmostEqual(result["point_penalty_group_std"], 0.0408248290)
        self.assertAlmostEqual(result["no_point_rate"], 1.0 / 6.0)
        self.assertAlmostEqual(result["reward_mean"], 0.65)
        self.assertAlmostEqual(result["constant_reward_group_rate"], 0.5)
        self.assertAlmostEqual(result["undefined_advantage_corr_rate"], 0.5)
        # b0 的优势与点数都常量 -> NaN；b1 为完全负相关 -> −1。
        self.assertAlmostEqual(result["advantage_point_count_corr"], -1.0)
        self.assertAlmostEqual(result["reward_point_count_corr"], -1.0)
        self.assertAlmostEqual(result["quality_reward_point_count_corr"], -1.0)

    def test_all_constant_groups_report_nan_correlation(self) -> None:
        rows = [
            {"quality_reward": 1.0, "point_penalty": 0.0, "point_count": 5},
            {"quality_reward": 1.0, "point_penalty": 0.0, "point_count": 5},
        ]
        advantages = torch.zeros(2, 1, dtype=torch.float64)

        result = reward_group_diagnostics(rows, advantages)

        self.assertTrue(math.isnan(result["advantage_point_count_corr"]))
        self.assertTrue(math.isnan(result["reward_point_count_corr"]))
        self.assertTrue(math.isnan(result["quality_reward_point_count_corr"]))
        self.assertEqual(result["constant_reward_group_rate"], 1.0)
        self.assertEqual(result["undefined_advantage_corr_rate"], 1.0)

    def test_empty_input_returns_nan_metrics(self) -> None:
        result = reward_group_diagnostics([], torch.empty(0, 0))

        self.assertEqual(tuple(result), REWARD_GROUP_METRIC_KEYS)
        self.assertEqual(result["n_images"], 0.0)
        self.assertEqual(result["n_trajectories"], 0.0)
        self.assertTrue(math.isnan(result["reward_mean"]))

    def test_shape_and_field_validation(self) -> None:
        row = {"quality_reward": 1.0, "point_penalty": 0.0, "point_count": 5}

        with self.assertRaises(ValueError):
            reward_group_diagnostics([row], torch.zeros(2, 2))
        with self.assertRaises(ValueError):
            reward_group_diagnostics([row], torch.zeros(2))
        with self.assertRaises(KeyError):
            reward_group_diagnostics(
                [{"quality_reward": 1.0, "point_count": 5}],
                torch.zeros(1, 1),
            )
        with self.assertRaises(ValueError):
            reward_group_diagnostics(
                [{"quality_reward": float("nan"), "point_penalty": 0.0,
                  "point_count": 5}],
                torch.zeros(1, 1),
            )
        with self.assertRaises(ValueError):
            reward_group_diagnostics(
                [row], torch.tensor([[float("inf")]], dtype=torch.float64)
            )


if __name__ == "__main__":
    unittest.main()
