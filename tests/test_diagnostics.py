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
    FIELD_METRIC_KEYS,
    ORACLE_ROW_KEYS,
    PERTURBATION_MODES,
    PROMPT_METRIC_KEYS,
    REWARD_GROUP_METRIC_KEYS,
    build_binary_prompt_field,
    build_oracle_prompt_fields,
    build_ridge_prompt_field,
    field_quality,
    oracle_prompt_diagnostics,
    perturb_prompt_field,
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


class FieldQualityTest(unittest.TestCase):
    def test_constant_background_field_reports_collapse(self) -> None:
        field = np.full((1, 8, 8), -1.0, dtype=np.float32)
        target = np.zeros((8, 8), dtype=bool)
        target[3, 2:6] = True

        result = field_quality(field, target, image_size=8)

        self.assertEqual(tuple(result), FIELD_METRIC_KEYS)
        self.assertEqual(result["valid_pixels"], 64)
        self.assertEqual(result["valid_field_mean"], -1.0)
        self.assertEqual(result["foreground_rate"], 0.0)
        self.assertEqual(result["foreground_positive_rate"], 0.0)
        self.assertEqual(result["foreground_collapse_rate"], 1.0)
        self.assertEqual(result["background_negative_rate"], 1.0)
        self.assertEqual(result["collapsed"], 1.0)
        # 4/64 道路像素与背景一致，剩余背景也一致。
        self.assertAlmostEqual(result["sign_agreement"], 60.0 / 64.0)

    def test_ridge_field_is_not_collapsed_and_tracks_foreground(self) -> None:
        target = np.zeros((16, 16), dtype=bool)
        target[7:9, 2:14] = True
        ridge = build_ridge_prompt_field(target)
        field = ridge[None, ...]

        result = field_quality(field, target, image_size=16)

        self.assertEqual(result["collapsed"], 0.0)
        self.assertGreater(result["valid_field_std"], 0.0)
        self.assertEqual(result["background_field_mean"], -1.0)
        self.assertGreater(result["foreground_field_mean"], -1.0)
        self.assertGreater(result["foreground_positive_rate"], 0.9)
        self.assertEqual(result["background_negative_rate"], 1.0)
        self.assertGreater(result["sign_agreement"], 0.9)

    def test_valid_crop_excludes_right_bottom_padding(self) -> None:
        field = np.full((16, 16), -1.0, dtype=np.float32)
        field[:, 8:] = 99.0
        field[8:, :] = 99.0

        result = field_quality(
            field, target=np.zeros((8, 8), dtype=bool), image_size=16
        )

        # field 与画布同分辨率时，前 8x8 像素正好是有效区域。
        self.assertEqual(result["valid_pixels"], 64)
        self.assertEqual(result["field_max"], 99.0)
        self.assertLessEqual(result["valid_field_max"], -1.0)
        self.assertEqual(result["valid_background_rate"], 1.0)

    def test_target_and_field_resolutions_differ(self) -> None:
        # engine 约定：field 1x16x16，target 64x64 有效 crop，image_size=64。
        field = np.zeros((1, 16, 16), dtype=np.float32)
        target = np.zeros((64, 64), dtype=bool)
        target[:32] = True

        result = field_quality(field, target, image_size=64)

        self.assertEqual(result["valid_pixels"], 256)
        self.assertAlmostEqual(result["target_foreground_rate"], 0.5)
        self.assertEqual(result["foreground_positive_rate"], 0.0)

    def test_without_target_foreground_metrics_are_nan(self) -> None:
        result = field_quality(
            np.zeros((1, 4, 4), dtype=np.float32), image_size=4
        )
        self.assertEqual(result["valid_pixels"], 16)
        self.assertTrue(math.isnan(result["target_foreground_rate"]))
        self.assertTrue(math.isnan(result["foreground_field_mean"]))
        self.assertTrue(math.isnan(result["sign_agreement"]))
        # 常量零场没有任何前景结构，视为塌缩。
        self.assertEqual(result["collapsed"], 1.0)

    def test_validation_errors(self) -> None:
        with self.assertRaises(ValueError):
            field_quality(np.zeros((1, 2, 4, 4)), image_size=4)
        with self.assertRaises(ValueError):
            field_quality(np.full((1, 4, 4), np.nan), image_size=4)
        with self.assertRaises(ValueError):
            field_quality(np.zeros((1, 4, 4)), np.zeros((8, 8)), image_size=None)


class OracleFieldTest(unittest.TestCase):
    def test_binary_and_empty_fields(self) -> None:
        mask = np.zeros((6, 6), dtype=bool)
        mask[2:4, 1:5] = True
        binary = build_binary_prompt_field(mask)
        self.assertTrue(np.all(binary[mask] == 1.0))
        self.assertTrue(np.all(binary[~mask] == -1.0))

        empty = build_binary_prompt_field(np.zeros((4, 4), dtype=bool))
        self.assertTrue(np.all(empty == -1.0))

    def test_ridge_semantics(self) -> None:
        mask = np.zeros((16, 16), dtype=bool)
        mask[7:9, 2:14] = True
        ridge = build_ridge_prompt_field(mask)

        self.assertTrue(np.all(ridge[~mask] == -1.0))
        self.assertTrue(np.all(ridge[mask] > -1.0))
        self.assertAlmostEqual(float(ridge.max()), 1.0, places=5)
        self.assertTrue(np.isfinite(ridge).all())

        # 全道路与空道路都必须给出确定结果。
        full = build_ridge_prompt_field(np.ones((8, 8), dtype=bool))
        self.assertTrue(np.isfinite(full).all())
        empty = build_ridge_prompt_field(np.zeros((8, 8), dtype=bool))
        self.assertTrue(np.all(empty == -1.0))

    def test_build_oracle_prompt_fields_kinds(self) -> None:
        mask = np.zeros((8, 8), dtype=bool)
        mask[3:5, 1:7] = True
        fields = build_oracle_prompt_fields(mask)
        self.assertEqual(set(fields), {"ridge", "binary"})
        with self.assertRaises(ValueError):
            build_oracle_prompt_fields(mask, kinds=("unknown",))


class PerturbationTest(unittest.TestCase):
    def setUp(self) -> None:
        mask = np.zeros((16, 16), dtype=bool)
        mask[7:9, 2:14] = True
        self.field = build_ridge_prompt_field(mask)

    def test_none_zero_and_deterministic_shuffle(self) -> None:
        none = perturb_prompt_field(self.field, "none")
        self.assertTrue(np.array_equal(none, self.field))

        zero = perturb_prompt_field(self.field, "zero")
        self.assertFalse(zero.any())

        first = perturb_prompt_field(self.field, "shuffled", seed=3)
        second = perturb_prompt_field(self.field, "shuffled", seed=3)
        self.assertTrue(np.array_equal(first, second))
        self.assertEqual(sorted(first.reshape(-1)), sorted(self.field.reshape(-1)))

    def test_local_perturbation_is_bounded_and_seeded(self) -> None:
        first = perturb_prompt_field(self.field, "local", seed=1, strength=0.5)
        second = perturb_prompt_field(self.field, "local", seed=1, strength=0.5)
        self.assertTrue(np.array_equal(first, second))
        self.assertGreaterEqual(float(first.min()), -1.0)
        self.assertLessEqual(float(first.max()), 1.0)
        self.assertGreater(float(np.abs(first - self.field).sum()), 0.0)
        # 局部扰动应只改变一小块区域。
        changed = np.count_nonzero(first != self.field)
        self.assertLess(changed, self.field.size)

    def test_invalid_mode_and_valid_mask(self) -> None:
        with self.assertRaises(ValueError):
            perturb_prompt_field(self.field, "unknown")
        with self.assertRaises(ValueError):
            perturb_prompt_field(
                self.field, "shuffled", valid_mask=np.ones((4, 4), dtype=bool)
            )
        valid = np.zeros_like(self.field, dtype=bool)
        valid[:, :4] = True
        shuffled = perturb_prompt_field(
            self.field, "shuffled", seed=0, valid_mask=valid
        )
        # 未被选中的区域保持不变。
        self.assertTrue(np.array_equal(shuffled[:, 4:], self.field[:, 4:]))


class OracleInterfaceDiagnosticsTest(unittest.TestCase):
    @staticmethod
    def _decode(prompt: torch.Tensor) -> torch.Tensor:
        return prompt * 6.0 - 0.5

    def setUp(self) -> None:
        self.mask = np.zeros((16, 16), dtype=bool)
        self.mask[7:9, 2:14] = True
        self.fields = build_oracle_prompt_fields(self.mask)

    def test_rows_cover_kinds_scales_and_perturbations(self) -> None:
        result = oracle_prompt_diagnostics(
            self.fields,
            self._decode,
            self.mask,
            scales=(0.5, 1.0),
            perturbations=PERTURBATION_MODES,
        )

        rows = result["rows"]
        self.assertEqual(
            len(rows), len(self.fields) * 2 * len(PERTURBATION_MODES)
        )
        for row in rows:
            self.assertEqual(tuple(row), ORACLE_ROW_KEYS)
        summary = result["summary"]
        self.assertEqual(summary["n_rows"], len(rows))
        self.assertIn(summary["best_kind"], {"ridge", "binary"})
        self.assertIn(summary["best_perturbation"], set(PERTURBATION_MODES))
        self.assertIn("ridge", summary["mean_iou_by_kind"])
        self.assertIn("zero", summary["mean_iou_by_perturbation"])

    def test_zero_prompt_and_local_perturbation_change_output(self) -> None:
        result = oracle_prompt_diagnostics(
            {"binary": self.fields["binary"]},
            self._decode,
            self.mask,
            scales=(1.0,),
            perturbations=("none", "zero", "shuffled", "local"),
        )
        by_mode = {row["perturbation"]: row for row in result["rows"]}
        self.assertAlmostEqual(by_mode["none"]["iou"], 1.0)
        self.assertLess(by_mode["zero"]["iou"], 0.5)
        self.assertEqual(by_mode["none"]["mean_abs_delta"], 0.0)
        self.assertGreater(by_mode["zero"]["mean_abs_delta"], 0.0)
        self.assertGreater(by_mode["shuffled"]["mean_abs_delta"], 0.0)
        self.assertGreater(by_mode["local"]["mean_abs_delta"], 0.0)

    def test_decode_output_can_be_resized_to_target(self) -> None:
        def tiny_decode(prompt: torch.Tensor) -> torch.Tensor:
            return torch.ones(1, 1, 8, 8, dtype=prompt.dtype)

        result = oracle_prompt_diagnostics(
            {"binary": self.fields["binary"]},
            tiny_decode,
            np.ones((16, 16), dtype=bool),
            scales=(1.0,),
            perturbations=("none",),
        )
        self.assertAlmostEqual(result["rows"][0]["iou"], 1.0)

    def test_scale_is_applied_after_clamp(self) -> None:
        seen: dict[str, float] = {}

        def capture(prompt: torch.Tensor) -> torch.Tensor:
            seen["max"] = float(prompt.max())
            seen["min"] = float(prompt.min())
            return prompt * 6.0 - 0.5

        oracle_prompt_diagnostics(
            {"binary": self.fields["binary"]},
            capture,
            self.mask,
            scales=(2.0,),
            perturbations=("none",),
        )
        # U = scale * clamp(H)：H=1, scale=2 时必须保留 2，而不是被夹回 1。
        self.assertAlmostEqual(seen["max"], 2.0)
        self.assertAlmostEqual(seen["min"], -2.0)

    def test_validation_errors(self) -> None:
        with self.assertRaises(ValueError):
            oracle_prompt_diagnostics({}, self._decode, self.mask)
        with self.assertRaises(ValueError):
            oracle_prompt_diagnostics(
                self.fields, self._decode, self.mask, scales=(0.0,)
            )
        with self.assertRaises(ValueError):
            oracle_prompt_diagnostics(
                self.fields, self._decode, self.mask, perturbations=("bad",)
            )
        with self.assertRaises(ValueError):
            oracle_prompt_diagnostics(
                self.fields,
                self._decode,
                self.mask,
                terminal_mapping="identity",
            )


if __name__ == "__main__":
    unittest.main()
