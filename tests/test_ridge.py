"""单通道有符号山脊场的几何与区域测试。

覆盖直路、弯路、交叉口、近邻道路、极细道路、空图、边界接触、满道路、
非等比影像的 padding，以及下采样与坐标映射的一致性。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diffpeak.data.geometry import build_geometry  # noqa: E402
from diffpeak.data.ridge import (  # noqa: E402
    BACKGROUND_VALUE,
    PADDING_VALUE,
    REGION_CENTER,
    REGION_FAR_BACKGROUND,
    REGION_NAMES,
    REGION_OUTSIDE_BAND,
    REGION_PADDING,
    REGION_REST_ROAD,
    RIDGE_VALUE,
    RidgeFieldConfig,
    build_ridge_field,
    build_source_ridge_field,
    coerce_ridge_config,
    field_statistics,
    regions_from_one_hot,
    regions_to_one_hot,
    source_retention_statistics,
)


class SourceRidgeFieldTest(unittest.TestCase):
    def test_straight_road_profile_decreases_from_center(self) -> None:
        road = np.zeros((17, 17), dtype=bool)
        road[6:11, 2:15] = True

        field, skeleton = build_source_ridge_field(road)

        self.assertTrue(skeleton.any())
        self.assertFalse(np.any(skeleton & ~road))
        # 竖直剖面：从中心线向两侧单调下降，背景固定为 -1。
        profile = field[5:12, 8]
        self.assertTrue(np.all(np.diff(profile[1:4]) >= -1e-6))
        self.assertTrue(np.all(np.diff(profile[4:7]) <= 1e-6))
        self.assertEqual(float(field.min()), BACKGROUND_VALUE)
        self.assertEqual(float(field.max()), RIDGE_VALUE)
        self.assertTrue(np.all(field[~road] == BACKGROUND_VALUE))
        # 中心线显式 +1。
        self.assertTrue(np.all(field[skeleton] == RIDGE_VALUE))
        # 紧贴背景的道路边界为 0，不是负值。
        boundary_rows = np.flatnonzero(road[:, 8])
        self.assertAlmostEqual(float(field[boundary_rows[0], 8]), 0.0, places=6)
        self.assertAlmostEqual(float(field[boundary_rows[-1], 8]), 0.0, places=6)

    def test_empty_road_is_all_background(self) -> None:
        road = np.zeros((20, 20), dtype=bool)

        field, skeleton = build_source_ridge_field(road)

        self.assertFalse(skeleton.any())
        self.assertTrue(np.all(field == BACKGROUND_VALUE))

    def test_full_road_and_edge_touching_road_are_defined(self) -> None:
        for road in (np.ones((16, 16), dtype=bool),):
            field, skeleton = build_source_ridge_field(road)
            self.assertTrue(np.all(np.isfinite(field)))
            self.assertGreaterEqual(float(field.min()), BACKGROUND_VALUE)
            self.assertLessEqual(float(field.max()), RIDGE_VALUE)
            self.assertTrue(skeleton.any())

        edge_road = np.zeros((16, 16), dtype=bool)
        edge_road[0:4, 2:14] = True
        field, skeleton = build_source_ridge_field(edge_road)
        self.assertTrue(skeleton.any())
        self.assertFalse(np.any(skeleton & ~edge_road))
        self.assertTrue(np.all(field[~edge_road] == BACKGROUND_VALUE))
        # 图像边缘按背景处理：最外圈道路像素到边界距离为 0。
        self.assertAlmostEqual(float(field[0, 8]), 0.0, places=6)

    def test_curved_road_and_intersection_keep_ridge(self) -> None:
        road = np.zeros((40, 40), dtype=bool)
        # L 形弯路。
        road[8:12, 6:30] = True
        road[8:30, 26:30] = True
        # 交叉口：横路与竖路相交。
        road[18:22, 2:38] = True

        field, skeleton = build_source_ridge_field(road)

        self.assertTrue(skeleton.any())
        self.assertFalse(np.any(skeleton & ~road))
        self.assertTrue(np.all(field[skeleton] == RIDGE_VALUE))
        self.assertTrue(np.all(field[~road] == BACKGROUND_VALUE))
        # 交叉口中心仍属于道路并有中心线覆盖。
        self.assertTrue(road[20, 20])
        self.assertTrue(field[20, 20] > 0.0)

    def test_neighboring_roads_remain_separate(self) -> None:
        road = np.zeros((32, 40), dtype=bool)
        road[6:9, 4:36] = True
        road[20:23, 4:36] = True

        field, _skeleton = build_source_ridge_field(road)

        self.assertTrue(field[7, 20] > 0.9)
        self.assertTrue(field[21, 20] > 0.9)
        # 两条道路之间的背景仍然是平底 -1，没有被距离场填成正响应。
        self.assertTrue(np.all(field[12:17, 4:36] == BACKGROUND_VALUE))


class HeatmapRidgeFieldTest(unittest.TestCase):
    def test_thin_one_pixel_road_survives_downsample(self) -> None:
        road = np.zeros((64, 64), dtype=bool)
        road[32, :] = True
        geometry = build_geometry(64, 64, image_size=64, heatmap_size=16)

        result = build_ridge_field(road, geometry)

        positive_rows = np.unique(np.nonzero(result.field > 0.0)[0])
        self.assertEqual(positive_rows.tolist(), [8])
        self.assertTrue(np.all(result.field[8] == RIDGE_VALUE))
        self.assertTrue(np.all(result.road_mask[8]))
        # 其余有效行是 -1 背景，没有被细路下采样污染。
        self.assertTrue(np.all(result.field[np.arange(16) != 8] == BACKGROUND_VALUE))
        stats = field_statistics(
            result.field, result.valid_mask, result.regions
        )
        self.assertGreater(stats["road_coverage_fraction_of_valid"], 0.0)
        self.assertAlmostEqual(
            stats["road_coverage_fraction_of_valid"], 1.0 / 16.0, places=6
        )

    def test_padding_is_zero_and_valid_background_is_minus_one(self) -> None:
        # 32x64 内容在 64x64 画布上只占上半部分，下半部分是 padding。
        road = np.zeros((32, 64), dtype=bool)
        road[8:12, :] = True
        geometry = build_geometry(32, 64, image_size=64, heatmap_size=16)

        result = build_ridge_field(road, geometry)

        valid_rows = result.valid_mask[:, 0]
        self.assertTrue(np.all(valid_rows[:8]))
        self.assertFalse(np.any(valid_rows[8:]))
        self.assertTrue(np.all(result.field[~result.valid_mask] == PADDING_VALUE))
        self.assertTrue(
            np.all(
                result.field[result.valid_mask & ~result.road_mask]
                == BACKGROUND_VALUE
            )
        )
        self.assertTrue(np.all(result.regions[~result.valid_mask] == REGION_PADDING))
        self.assertTrue(np.all(result.regions[result.valid_mask] >= REGION_CENTER))

    def test_regions_are_disjoint_and_cover_all_valid_pixels(self) -> None:
        road = np.zeros((48, 48), dtype=bool)
        road[20:25, 5:43] = True
        road[8:40, 22:26] = True
        geometry = build_geometry(48, 48, image_size=64, heatmap_size=16)

        result = build_ridge_field(road, geometry)

        region_ids = set(np.unique(result.regions[result.valid_mask]).tolist())
        self.assertEqual(
            region_ids, {REGION_CENTER, REGION_REST_ROAD, REGION_OUTSIDE_BAND, REGION_FAR_BACKGROUND}
        )
        stats = field_statistics(
            result.field, result.valid_mask, result.regions
        )
        self.assertEqual(stats["unassigned_valid_pixels"], 0)
        self.assertEqual(
            sum(stats["region_counts"].values()), int(result.valid_mask.sum())
        )
        self.assertEqual(set(stats["region_counts"]), set(REGION_NAMES.values()))

    def test_downsample_interval_boundaries_on_non_integer_scale(self) -> None:
        """非整数缩放时，每个热图单元覆盖的源区间与像素中心映射一致。"""
        source_height, source_width = 37, 53
        road = np.zeros((source_height, source_width), dtype=bool)
        road[10:13, 4:48] = True
        road[25:32, 20:23] = True
        geometry = build_geometry(
            source_height, source_width, image_size=64, heatmap_size=20
        )
        heatmap_size = geometry.heatmap_size
        image_size = geometry.image_size

        result = build_ridge_field(road, geometry)

        def axis_intervals(source_length: int, resized_length: int):
            scale = (resized_length / source_length) * (
                heatmap_size / image_size
            )
            span = 1.0 / scale
            intervals = []
            for cell in range(heatmap_size):
                low = cell * span - 0.5
                high = low + span
                indices = [
                    index
                    for index in range(source_length)
                    if low - 1e-9 <= index < high - 1e-9
                ]
                intervals.append(indices)
            return intervals

        y_intervals = axis_intervals(source_height, geometry.resized_height)
        x_intervals = axis_intervals(source_width, geometry.resized_width)
        for row, y_indices in enumerate(y_intervals):
            for column, x_indices in enumerate(x_intervals):
                if not result.valid_mask[row, column]:
                    self.assertFalse(bool(result.road_mask[row, column]))
                    continue
                if not y_indices or not x_indices:
                    continue
                expected = bool(road[np.ix_(y_indices, x_indices)].any())
                self.assertEqual(bool(result.road_mask[row, column]), expected)

    def test_region_one_hot_contract(self) -> None:
        road = np.zeros((48, 48), dtype=bool)
        road[20:25, 5:43] = True
        road[8:40, 22:26] = True
        geometry = build_geometry(48, 48, image_size=64, heatmap_size=16)
        result = build_ridge_field(road, geometry)

        one_hot = regions_to_one_hot(result.regions)

        self.assertEqual(one_hot.shape, (4, 16, 16))
        self.assertEqual(one_hot.dtype, np.float32)
        self.assertTrue(np.all(one_hot.sum(axis=0)[result.valid_mask] == 1))
        self.assertTrue(np.all(one_hot[:, ~result.valid_mask] == 0))
        self.assertTrue(
            np.array_equal(
                regions_from_one_hot(one_hot, result.valid_mask), result.regions
            )
        )

    def test_empty_road_has_only_far_background(self) -> None:
        geometry = build_geometry(32, 32, image_size=64, heatmap_size=16)

        result = build_ridge_field(np.zeros((32, 32), dtype=bool), geometry)

        self.assertTrue(np.all(result.field[result.valid_mask] == BACKGROUND_VALUE))
        self.assertTrue(
            np.all(result.regions[result.valid_mask] == REGION_FAR_BACKGROUND)
        )
        self.assertEqual(int(np.count_nonzero(result.regions == REGION_CENTER)), 0)
        self.assertEqual(
            int(np.count_nonzero(result.regions == REGION_OUTSIDE_BAND)), 0
        )
        self.assertTrue(np.all(result.field[~result.valid_mask] == PADDING_VALUE))

    def test_road_coverage_and_valid_mask_are_transform_equivariant(self) -> None:
        """翻转/旋转后道路覆盖与有效区域对齐；场差异只可能来自骨架平局。"""
        road = np.zeros((48, 48), dtype=bool)
        road[20:25, 5:43] = True
        road[8:40, 22:26] = True
        geometry = build_geometry(48, 48, image_size=64, heatmap_size=16)
        baseline = build_ridge_field(road, geometry)

        transforms = {
            "flip_x": (road[:, ::-1], lambda value: value[:, ::-1]),
            "flip_y": (road[::-1], lambda value: value[::-1]),
            "rot90": (np.rot90(road, 1), lambda value: np.rot90(value, 1)),
            "rot180": (np.rot90(road, 2), lambda value: np.rot90(value, 2)),
        }
        for name, (mask, transform) in transforms.items():
            with self.subTest(transform=name):
                result = build_ridge_field(mask, geometry)
                self.assertTrue(
                    np.array_equal(result.road_mask, transform(baseline.road_mask))
                )
                self.assertTrue(
                    np.array_equal(result.valid_mask, transform(baseline.valid_mask))
                )
                # 背景（-1）检测同样保持对齐；padding 由独立掩码标记，
                # 不能直接用数值 0 判断（有效道路边界也可能为 0）。
                self.assertTrue(
                    np.array_equal(
                        result.field < 0, transform(baseline.field < 0)
                    )
                )

    def test_skeleton_maps_into_positive_heatmap_cell(self) -> None:
        road = np.zeros((64, 64), dtype=bool)
        road[18:46, 28:32] = True
        geometry = build_geometry(64, 64, image_size=64, heatmap_size=16)

        result = build_ridge_field(road, geometry)
        skeleton_yx = np.argwhere(result.source_skeleton)
        heatmap_yx = geometry.original_yx_to_heatmap_yx(
            skeleton_yx.astype(np.float64)
        )
        heatmap_yx = np.rint(heatmap_yx).astype(int)
        inside = (
            (heatmap_yx[:, 0] >= 0)
            & (heatmap_yx[:, 0] < geometry.heatmap_size)
            & (heatmap_yx[:, 1] >= 0)
            & (heatmap_yx[:, 1] < geometry.heatmap_size)
        )
        heatmap_yx = heatmap_yx[inside]

        self.assertTrue(
            np.all(result.field[heatmap_yx[:, 0], heatmap_yx[:, 1]] > 0.0)
        )

    def test_non_square_geometry_uses_aspect_preserving_mapping(self) -> None:
        # 16x64 内容缩放到 64x64 时只有上半部分 16 行有效。
        road = np.zeros((16, 64), dtype=bool)
        road[6:10, :] = True
        geometry = build_geometry(16, 64, image_size=64, heatmap_size=16)

        result = build_ridge_field(road, geometry)

        self.assertTrue(np.all(result.valid_mask[:4]))
        self.assertFalse(np.any(result.valid_mask[4:]))
        self.assertTrue(np.all(result.field[4:] == PADDING_VALUE))

    def test_source_smaller_than_heatmap_is_supported(self) -> None:
        road = np.zeros((8, 8), dtype=bool)
        road[3:5, 1:7] = True
        geometry = build_geometry(8, 8, image_size=64, heatmap_size=16)

        result = build_ridge_field(road, geometry)

        self.assertEqual(result.field.shape, (16, 16))
        self.assertTrue(np.all(np.isfinite(result.field)))
        self.assertTrue(result.road_mask.any())
        stats = field_statistics(result.field, result.valid_mask, result.regions)
        self.assertGreater(stats["ridge_fraction_of_valid"], 0.0)

    def test_source_retention_reports_thin_road_gain(self) -> None:
        """单像素道路：覆盖最大值保留 16 个单元，中心采样一个也没有。"""
        road = np.zeros((64, 64), dtype=bool)
        road[32, :] = True
        geometry = build_geometry(64, 64, image_size=64, heatmap_size=16)

        result = build_ridge_field(road, geometry)
        stats = source_retention_statistics(result)

        self.assertEqual(stats["source_road_pixels"], 64)
        self.assertEqual(stats["source_skeleton_pixels"], 64)
        self.assertEqual(stats["projected_road_recall"], 1.0)
        self.assertEqual(stats["projected_skeleton_recall"], 1.0)
        self.assertEqual(stats["projected_skeleton_center_recall"], 1.0)
        self.assertEqual(stats["conservative_road_cells"], 16)
        self.assertEqual(stats["nearest_road_cells"], 0)
        self.assertIsNone(stats["road_coverage_retention_vs_nearest"])
        self.assertEqual(stats["conservative_road_extra_cells"], 16)

    def test_source_retention_handles_empty_road(self) -> None:
        geometry = build_geometry(32, 32, image_size=64, heatmap_size=16)
        result = build_ridge_field(np.zeros((32, 32), dtype=bool), geometry)

        stats = source_retention_statistics(result)

        self.assertEqual(stats["source_road_pixels"], 0)
        self.assertIsNone(stats["projected_road_recall"])
        self.assertIsNone(stats["projected_skeleton_recall"])
        self.assertEqual(stats["nearest_road_cells"], 0)
        self.assertEqual(stats["conservative_road_cells"], 0)

    def test_config_coercion_and_validation(self) -> None:
        self.assertEqual(coerce_ridge_config(None), RidgeFieldConfig())
        self.assertEqual(
            coerce_ridge_config({"center_radius": 2, "boundary_band": 1}).center_radius,
            2,
        )
        with self.assertRaises(ValueError):
            coerce_ridge_config({"unknown": 1})
        with self.assertRaises(ValueError):
            RidgeFieldConfig(center_radius=-1)
        with self.assertRaises(ValueError):
            RidgeFieldConfig(epsilon=0)
        # NaN / Inf 不会触发普通的 < 0 或 <= 0 比较，必须显式拒绝。
        for field_name in ("center_radius", "boundary_band", "epsilon"):
            for value in (float("nan"), float("inf"), float("-inf")):
                with self.subTest(field=field_name, value=value):
                    with self.assertRaises(ValueError):
                        RidgeFieldConfig(**{field_name: value})
        with self.assertRaises(ValueError):
            build_source_ridge_field(
                np.zeros((4, 4), dtype=bool), epsilon=float("nan")
            )


if __name__ == "__main__":
    unittest.main()
