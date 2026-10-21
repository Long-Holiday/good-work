"""PeakExtractor 的确定性峰值提取测试。

覆盖设计文档要求的关键检查：
* 零热图不产生任何提示点；
* 平顶峰的代表点选择完全确定；
* 正负通道坐标重合时的固定消歧规则；
* 目标高斯热图能被最终提取器回提；
* 右下填充区域的峰在 NMS 前被排除。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage
from skimage.morphology import h_maxima, local_maxima

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diffpeak.data.sampling import render_gaussian_heatmap  # noqa: E402
from diffpeak.prompts import (  # noqa: E402
    PeakExtractor,
    _ChannelConfig,
    _MappedPeak,
    _Peak,
    _detect_peaks,
    _nms_peaks,
    extract_numpy,
    target_recovery_report,
)

_DETERMINISTIC_CONFIG = {
    "sigma": 0.0,
    "h": 0.0,
    "threshold": 0.3,
    "min_distance": 4.0,
}


def _legacy_detect_peaks(heatmap: np.ndarray, config: _ChannelConfig):
    """优化前 ``_detect_peaks`` 的逐字实现，作为等价性基准。"""
    if config.sigma > 0:
        smooth = ndimage.gaussian_filter(
            heatmap, sigma=float(config.sigma), mode="nearest"
        )
    else:
        smooth = heatmap
    if not np.any(smooth > 0):
        return []
    if config.h > 0:
        peak_mask = h_maxima(smooth, float(config.h))
    else:
        peak_mask = local_maxima(smooth, connectivity=smooth.ndim)
    peak_mask = np.asarray(peak_mask, dtype=bool)
    if config.threshold > -np.inf:
        peak_mask &= smooth >= float(config.threshold)
    if not peak_mask.any():
        return []
    components, num_components = ndimage.label(
        peak_mask, structure=np.ones((3, 3), dtype=bool)
    )
    peaks = []
    for component_id in range(1, int(num_components) + 1):
        ys, xs = np.nonzero(components == component_id)
        if len(ys) == 0:
            continue
        values = smooth[ys, xs]
        max_value = float(values.max())
        tied = values >= max_value - 1e-7
        tied_y = ys[tied].astype(np.float64)
        tied_x = xs[tied].astype(np.float64)
        center_y = float(ys.mean())
        center_x = float(xs.mean())
        distance_sq = (tied_y - center_y) ** 2 + (tied_x - center_x) ** 2
        order = np.lexsort((tied_x, tied_y, distance_sq))
        chosen = int(order[0])
        peaks.append(
            _Peak(
                score=max_value,
                y_heat=float(tied_y[chosen]),
                x_heat=float(tied_x[chosen]),
            )
        )
    return peaks


def _legacy_nms_peaks(peaks, min_distance: float):
    """优化前 ``_nms_peaks`` 的逐字实现，作为等价性基准。"""
    ordered = sorted(peaks, key=lambda peak: (-peak.score, peak.y_heat, peak.x_heat))
    if min_distance <= 0:
        return ordered
    minimum_sq = float(min_distance) ** 2
    kept = []
    for peak in ordered:
        if all(
            (peak.y_heat - other.y_heat) ** 2
            + (peak.x_heat - other.x_heat) ** 2
            >= minimum_sq
            for other in kept
        ):
            kept.append(peak)
    return kept


class PeakOptimisationEquivalenceTest(unittest.TestCase):
    """逐字旧实现与新实现的精确等价性。"""

    CONFIG = _ChannelConfig(sigma=0.0, h=0.0, threshold=0.3, min_distance=4.0)

    def _assert_detect_equal(self, heatmap: np.ndarray, config=None) -> None:
        config = config or self.CONFIG
        expected = _legacy_detect_peaks(heatmap, config)
        actual = _detect_peaks(heatmap, config)
        self.assertEqual(
            [(peak.score, peak.y_heat, peak.x_heat) for peak in actual],
            [(peak.score, peak.y_heat, peak.x_heat) for peak in expected],
        )

    def _assert_nms_equal(self, peaks, min_distance: float) -> None:
        expected = _legacy_nms_peaks(peaks, min_distance)
        actual = _nms_peaks(peaks, min_distance)
        self.assertEqual(actual, expected)
        self.assertEqual(
            [(peak.score, peak.y_heat, peak.x_heat) for peak in actual],
            [(peak.score, peak.y_heat, peak.x_heat) for peak in expected],
        )

    def test_detect_matches_legacy_on_dense_random_heatmaps(self) -> None:
        rng = np.random.default_rng(20241021)
        for size in (1, 7, 16, 24):
            with self.subTest(size=size):
                self._assert_detect_equal(rng.random((size, size)))
        # 低阈值 + 大量零像素，覆盖只有部分区域成峰的情况。
        random_values = rng.random((20, 20))
        sparse = np.where(random_values > 0.85, random_values, 0.0)
        self._assert_detect_equal(sparse)

    def test_detect_matches_legacy_with_ties_and_plateaus(self) -> None:
        heatmap = np.zeros((12, 14))
        heatmap[1:5, 2:7] = 1.0  # 大平顶
        heatmap[7, 9:13] = 0.8  # 1x4 平顶
        heatmap[8:11, 4:6] = 0.8  # 2x2 平顶
        heatmap[2:10, 11] = 0.6  # 细长平顶
        for config in (
            self.CONFIG,
            _ChannelConfig(sigma=0.0, h=0.0, threshold=0.0, min_distance=4.0),
            _ChannelConfig(sigma=0.5, h=0.05, threshold=0.3, min_distance=4.0),
            _ChannelConfig(sigma=1.0, h=0.2, threshold=0.3, min_distance=4.0),
        ):
            with self.subTest(config=config):
                self._assert_detect_equal(heatmap, config)

    def test_detect_matches_legacy_for_single_pixel_components(self) -> None:
        heatmap = np.full((17, 19), 0.01)
        heatmap[::2, ::2] = 1.0
        self._assert_detect_equal(heatmap)

    def test_nms_matches_legacy_dense_and_boundaries(self) -> None:
        rng = np.random.default_rng(7)
        peaks = [
            _Peak(
                score=float(rng.random()),
                y_heat=float(rng.integers(-20, 21)),
                x_heat=float(rng.integers(-20, 21)),
            )
            for _ in range(120)
        ]
        for min_distance in (0.0, 1.0, 2.0, 3.5, 4.0, 5.0, 10.0, 1e-12):
            with self.subTest(min_distance=min_distance):
                self._assert_nms_equal(peaks, min_distance)

    def test_nms_matches_legacy_on_exact_integer_boundary(self) -> None:
        # (0,0) 与 (3,4) 的平方距离恰好等于 5^2，必须保留；
        # (3,0) 恰好等于 3^2 于 min_distance=3 时保留、2.9 时压制。
        peaks = [
            _Peak(score=1.0, y_heat=0.0, x_heat=0.0),
            _Peak(score=0.9, y_heat=3.0, x_heat=4.0),
            _Peak(score=0.8, y_heat=3.0, x_heat=0.0),
            _Peak(score=0.7, y_heat=0.0, x_heat=4.0),
        ]
        for min_distance in (3.0, 2.9, 4.0, 5.0, 5.0000001):
            with self.subTest(min_distance=min_distance):
                self._assert_nms_equal(peaks, min_distance)

    def test_nms_matches_legacy_with_tied_scores(self) -> None:
        peaks = [
            _Peak(score=0.5, y_heat=float(y), x_heat=float(x))
            for y in range(8)
            for x in range(8)
        ]
        for min_distance in (0.0, 1.0, 2.5):
            with self.subTest(min_distance=min_distance):
                self._assert_nms_equal(peaks, min_distance)

    def test_nms_matches_legacy_for_mapped_peaks(self) -> None:
        rng = np.random.default_rng(11)
        peaks = [
            _MappedPeak(
                score=float(rng.random()),
                y_heat=float(rng.integers(0, 32)),
                x_heat=float(rng.integers(0, 32)),
                x_image=float(rng.integers(0, 128)),
                y_image=float(rng.integers(0, 128)),
                label=int(rng.integers(0, 2)),
            )
            for _ in range(80)
        ]
        for min_distance in (0.0, 2.0, 4.0, 9.0):
            with self.subTest(min_distance=min_distance):
                actual = _nms_peaks(peaks, min_distance)
                expected = _legacy_nms_peaks(peaks, min_distance)
                self.assertEqual(actual, expected)
                self.assertTrue(
                    all(left is right for left, right in zip(actual, expected))
                )

    def test_nms_rejects_invalid_min_distance_like_config(self) -> None:
        peaks = [_Peak(score=1.0, y_heat=0.0, x_heat=0.0)]
        for value in (-1.0, float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                _nms_peaks(peaks, value)

    def test_nms_handles_extreme_coordinates_without_overflow(self) -> None:
        peaks = [
            _Peak(score=1.0, y_heat=1e150, x_heat=1e150),
            _Peak(score=0.9, y_heat=1e150, x_heat=1e150),
            _Peak(score=0.8, y_heat=0.0, x_heat=0.0),
        ]
        self._assert_nms_equal(peaks, 1e-12)

    def test_nms_falls_back_for_nonfinite_coordinates(self) -> None:
        peaks = [
            _Peak(score=1.0, y_heat=float("inf"), x_heat=float("inf")),
            _Peak(score=0.9, y_heat=float("inf"), x_heat=float("inf")),
            _Peak(score=0.8, y_heat=0.0, x_heat=0.0),
        ]
        self._assert_nms_equal(peaks, 4.0)

    def test_nms_matches_legacy_overflow_for_huge_finite_coordinates(self) -> None:
        # 坐标差的平方超过 float64 时旧实现会抛 OverflowError。这些峰相距
        # 极远、不会被网格比较到，因此新实现必须先整体退回旧路径，才能
        # 连异常行为都精确一致。
        for magnitude in (1e155, 1e200):
            with self.subTest(magnitude=magnitude):
                peaks = [
                    _Peak(score=1.0, y_heat=magnitude, x_heat=0.0),
                    _Peak(score=0.5, y_heat=0.0, x_heat=0.0),
                ]
                with self.assertRaises(OverflowError):
                    _legacy_nms_peaks(peaks, 4.0)
                with self.assertRaises(OverflowError):
                    _nms_peaks(peaks, 4.0)

    def test_nms_matches_legacy_for_huge_python_int_coordinates(self) -> None:
        # 超出 float 范围但可精确比较的 Python int 坐标：旧实现用整数
        # 运算不会溢出，新实现退回同一路径。
        peaks = [
            _Peak(score=1.0, y_heat=10**400, x_heat=0),
            _Peak(score=0.5, y_heat=0, x_heat=0),
        ]
        self._assert_nms_equal(peaks, 4.0)



class PeakExtractorTest(unittest.TestCase):
    def test_zero_heatmap_returns_no_points(self) -> None:
        extractor = PeakExtractor()
        heatmaps = torch.zeros(1, 2, 32, 32)

        points = extractor(heatmaps, image_size=32)

        self.assertEqual(len(points), 1)
        coords, labels = points[0]
        self.assertIsInstance(coords, torch.Tensor)
        self.assertIsInstance(labels, torch.Tensor)
        self.assertEqual(tuple(coords.shape), (0, 2))
        self.assertEqual(tuple(labels.shape), (0,))
        self.assertEqual(coords.dtype, torch.float32)
        self.assertEqual(labels.dtype, torch.int64)

    def test_zero_heatmap_numpy_interface(self) -> None:
        heatmaps = np.zeros((2, 16, 16), dtype=np.float32)

        points = extract_numpy(heatmaps, image_size=16)

        self.assertEqual(len(points), 1)
        coords, labels = points[0]
        self.assertEqual(coords.shape, (0, 2))
        self.assertEqual(labels.shape, (0,))
        self.assertEqual(coords.dtype, np.float32)
        self.assertEqual(labels.dtype, np.int64)

    def test_plateau_representative_is_deterministic(self) -> None:
        """1x4 平顶：中心并列时固定选较小的 x。"""
        heatmaps = np.zeros((1, 2, 16, 16), dtype=np.float32)
        heatmaps[0, 0, 10, 10:14] = 1.0
        extractor = PeakExtractor(_DETERMINISTIC_CONFIG)

        first = extractor(heatmaps, image_size=16)[0]
        second = extractor(heatmaps, image_size=16)[0]
        third = extractor(torch.from_numpy(heatmaps), image_size=16)[0]

        self.assertTrue(torch.equal(first[0], second[0]))
        self.assertTrue(torch.equal(first[0], third[0]))
        self.assertTrue(torch.equal(first[1], second[1]))
        self.assertEqual(tuple(first[0].shape), (1, 2))
        self.assertEqual(first[1].tolist(), [1])
        # 平顶列为 10..13，中心 11.5；两个最近点中按 (y, x) 取 x=11。
        self.assertEqual(first[0].tolist(), [[11.0, 10.0]])

    def test_same_coordinate_tie_removes_both(self) -> None:
        heatmaps = np.zeros((1, 2, 16, 16), dtype=np.float32)
        heatmaps[0, 0, 8, 8] = 0.8
        heatmaps[0, 1, 8, 8] = 0.8
        heatmaps[0, 0, 3, 3] = 1.0
        extractor = PeakExtractor(_DETERMINISTIC_CONFIG)

        coords, labels = extractor(heatmaps, image_size=16)[0]

        self.assertEqual(tuple(coords.shape), (1, 2))
        self.assertEqual(labels.tolist(), [1])
        self.assertEqual(coords.tolist(), [[3.0, 3.0]])

    def test_same_coordinate_tie_can_leave_no_prompts(self) -> None:
        heatmaps = np.zeros((1, 2, 16, 16), dtype=np.float32)
        heatmaps[0, :, 8, 8] = 0.8
        coords, labels = PeakExtractor(_DETERMINISTIC_CONFIG)(
            heatmaps, image_size=16
        )[0]
        self.assertEqual(tuple(coords.shape), (0, 2))
        self.assertEqual(labels.numel(), 0)

    def test_same_coordinate_higher_score_wins(self) -> None:
        heatmaps = np.zeros((1, 2, 16, 16), dtype=np.float32)
        heatmaps[0, 0, 8, 8] = 0.5
        heatmaps[0, 1, 8, 8] = 0.9
        extractor = PeakExtractor(_DETERMINISTIC_CONFIG)

        coords, labels = extractor(heatmaps, image_size=16)[0]

        self.assertEqual(labels.tolist(), [0])
        self.assertEqual(coords.tolist(), [[8.0, 8.0]])

    def test_same_coordinate_positive_higher_score_wins(self) -> None:
        heatmaps = np.zeros((1, 2, 16, 16), dtype=np.float32)
        heatmaps[0, 0, 8, 8] = 0.9
        heatmaps[0, 1, 8, 8] = 0.5
        extractor = PeakExtractor(_DETERMINISTIC_CONFIG)

        coords, labels = extractor(heatmaps, image_size=16)[0]

        self.assertEqual(labels.tolist(), [1])
        self.assertEqual(coords.tolist(), [[8.0, 8.0]])

    def test_peaks_in_padding_are_removed_before_nms(self) -> None:
        heatmaps = np.zeros((1, 2, 8, 8), dtype=np.float32)
        heatmaps[0, 0, 2, 2] = 1.0  # 有效区域
        heatmaps[0, 0, 6, 6] = 1.0  # 右下填充区域
        extractor = PeakExtractor(_DETERMINISTIC_CONFIG)

        coords, labels = extractor(
            heatmaps, image_size=16, valid_sizes=[(8, 8)]
        )[0]

        self.assertEqual(labels.tolist(), [1])
        self.assertEqual(len(coords), 1)
        self.assertAlmostEqual(float(coords[0, 0]), 4.5, places=5)
        self.assertAlmostEqual(float(coords[0, 1]), 4.5, places=5)

    def test_invalid_inputs_raise(self) -> None:
        extractor = PeakExtractor()
        with self.assertRaises(ValueError):
            extractor(torch.zeros(1, 3, 8, 8), image_size=16)
        with self.assertRaises(ValueError):
            extractor(torch.zeros(1, 2, 8, 10), image_size=16)
        with self.assertRaises(ValueError):
            extractor(torch.full((1, 2, 8, 8), float("nan")), image_size=16)
        with self.assertRaises(ValueError):
            PeakExtractor({"unknown": 1})
        with self.assertRaises(TypeError):
            PeakExtractor({"positive": 1})  # type: ignore[dict-item]

    def test_per_channel_override(self) -> None:
        extractor = PeakExtractor(
            {"threshold": 0.3, "channels": {"negative": {"threshold": 0.6}}}
        )

        self.assertEqual(extractor.positive_config["threshold"], 0.3)
        self.assertEqual(extractor.negative_config["threshold"], 0.6)

    def test_scalar_pair_and_mapping_overrides(self) -> None:
        paired = PeakExtractor({"threshold": (0.25, 0.55)})
        mapped = PeakExtractor(
            {"h": {"positive": 0.02, "negative": 0.08}}
        )

        self.assertEqual(paired.positive_config["threshold"], 0.25)
        self.assertEqual(paired.negative_config["threshold"], 0.55)
        self.assertEqual(mapped.positive_config["h"], 0.02)
        self.assertEqual(mapped.negative_config["h"], 0.08)


class TargetRecoveryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.heatmap_size = 32
        self.image_size = 32

    def _render(
        self,
        positive_yx: np.ndarray,
        negative_yx: np.ndarray,
        sigma: float = 1.0,
    ) -> torch.Tensor:
        heatmaps = np.zeros((1, 2, self.heatmap_size, self.heatmap_size), dtype=np.float32)
        heatmaps[0, 0] = render_gaussian_heatmap(
            positive_yx, self.heatmap_size, sigma
        )
        heatmaps[0, 1] = render_gaussian_heatmap(
            negative_yx, self.heatmap_size, sigma
        )
        return torch.from_numpy(heatmaps)

    def test_target_gaussians_are_recovered_by_final_extractor(self) -> None:
        positive_yx = np.array([[7.0, 13.0]])
        negative_yx = np.array([[20.0, 6.0]])
        heatmaps = self._render(positive_yx, negative_yx)

        report = target_recovery_report(
            heatmaps,
            expected_coords=np.array([[13.0, 7.0], [6.0, 20.0]]),
            expected_labels=np.array([1, 0]),
            image_size=self.image_size,
        )

        self.assertTrue(report["recovered"])
        # 默认 match_radius=None 时按热图量化误差自动取值。
        self.assertAlmostEqual(report["match_radius"], 2.0, places=6)
        self.assertEqual(report["expected"], 2)
        self.assertEqual(report["predicted"], 2)
        self.assertEqual(report["matched"], 2)
        self.assertEqual(report["missed"], 0)
        self.assertEqual(report["extra"], 0)
        self.assertEqual(report["positive"]["matched"], 1)
        self.assertEqual(report["negative"]["matched"], 1)
        self.assertEqual(report["empty_sample_count"], 0)

    def test_final_transition_noise_preserves_isolated_target_peaks(self) -> None:
        from diffpeak.models.diffusion import GaussianDiffusion

        class ZeroNoise(torch.nn.Module):
            def forward(self, state, timestep, conditions):
                return torch.zeros_like(state)

        heatmaps = self._render(np.array([[7.0, 13.0]]), np.array([[20.0, 6.0]]))
        diffusion = GaussianDiffusion(timesteps=8, sampling_steps=2, min_std=0.02)
        state = diffusion.alphas_cumprod[0].sqrt() * (2 * heatmaps - 1)
        mean, std = diffusion.transition(ZeroNoise(), state, 0, -1, [state] * 4)
        for seed in range(8):
            with self.subTest(seed=seed):
                noise = torch.randn(state.shape, generator=torch.Generator().manual_seed(seed))
                sampled = ((mean + std * noise + 1) / 2).clamp(0, 1)
                report = target_recovery_report(
                    sampled,
                    expected_coords=np.array([[13.0, 7.0], [6.0, 20.0]]),
                    expected_labels=np.array([1, 0]),
                    image_size=32,
                )
                self.assertTrue(report["recovered"])
                self.assertEqual(report["extra"], 0)

    def test_zero_target_heatmap_is_correctly_empty(self) -> None:
        heatmaps = torch.zeros(1, 2, self.heatmap_size, self.heatmap_size)

        report = target_recovery_report(
            heatmaps,
            expected_coords=np.empty((0, 2)),
            expected_labels=np.empty((0,)),
            image_size=self.image_size,
        )

        self.assertTrue(report["recovered"])
        self.assertEqual(report["predicted"], 0)
        self.assertEqual(report["empty_sample_count"], 1)
        self.assertEqual(report["empty_correct_count"], 1)

    def test_missing_expected_peak_is_reported(self) -> None:
        heatmaps = torch.zeros(1, 2, self.heatmap_size, self.heatmap_size)

        report = target_recovery_report(
            heatmaps,
            expected_coords=np.array([[13.0, 7.0]]),
            expected_labels=np.array([1]),
            image_size=self.image_size,
        )

        self.assertFalse(report["recovered"])
        self.assertEqual(report["missed"], 1)
        self.assertEqual(report["positive"]["recall"], 0.0)

    def test_heatmap_space_expected_points_are_converted(self) -> None:
        heatmaps = self._render(np.array([[4.0, 5.0]]), np.empty((0, 2)))

        report = target_recovery_report(
            heatmaps,
            expected_coords=np.array([[5.0, 4.0]]),
            expected_labels=np.array([1]),
            image_size=self.image_size,
            expected_space="heatmap",
            match_radius=2.0,
        )

        self.assertTrue(report["recovered"])
        self.assertEqual(report["matched"], 1)


if __name__ == "__main__":
    unittest.main()
