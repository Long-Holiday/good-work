"""diffpeak.losses 的单元测试。

重点覆盖设计文档要求的“空标注/空掩码”边界：
* 空标注不会让 Dice 或 soft-clDice 产生 NaN，也不会抬高损失；
* 空标注时分割损失退化为逐像素 BCE；
* 辅助头在空道路/空骨架目标上仍然是有限的。
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diffpeak.losses import (  # noqa: E402
    _align_binary_target,
    _soft_cldice_loss,
    _soft_skeleton,
    auxiliary_loss,
    segmentation_loss,
)


class SoftSkeletonTest(unittest.TestCase):
    """细线骨架与自比较 clDice 的回归测试。

    1 像素宽的十字形腐蚀核不能把整条细线腐蚀掉；目标与自身比较时，
    clDice 必须接近 1（损失接近 0）。
    """

    def _thin_line(self, vertical: bool = False) -> torch.Tensor:
        image = torch.zeros(1, 1, 32, 32)
        if vertical:
            image[0, 0, 5:27, 16] = 1.0
        else:
            image[0, 0, 16, 5:27] = 1.0
        return image

    def test_thin_horizontal_line_skeleton_is_nonempty(self) -> None:
        line = self._thin_line(vertical=False)

        skeleton = _soft_skeleton(line, iterations=3)

        self.assertGreater(float(skeleton.sum().item()), 0.0)
        self.assertTrue(torch.equal(skeleton, line))

    def test_thin_vertical_line_skeleton_is_nonempty(self) -> None:
        line = self._thin_line(vertical=True)

        skeleton = _soft_skeleton(line, iterations=3)

        self.assertGreater(float(skeleton.sum().item()), 0.0)
        self.assertTrue(torch.equal(skeleton, line))

    def test_thin_line_against_itself_has_near_zero_cldice_loss(self) -> None:
        for vertical in (False, True):
            with self.subTest(vertical=vertical):
                line = self._thin_line(vertical=vertical)
                loss = _soft_cldice_loss(line, line)

                self.assertTrue(torch.isfinite(loss))
                self.assertLess(float(loss.item()), 1e-5)

    def test_thin_line_disjoint_prediction_is_penalized(self) -> None:
        target = self._thin_line(vertical=False)
        prediction = self._thin_line(vertical=True)

        loss = _soft_cldice_loss(prediction, target)

        self.assertGreater(float(loss.item()), 0.5)


class SegmentationLossTest(unittest.TestCase):
    def test_empty_target_equals_pixel_bce(self) -> None:
        """整个 batch 都是空标注时，Dice 与 soft-clDice 必须为零。"""
        torch.manual_seed(0)
        logits = torch.randn(2, 1, 7, 9)
        targets = torch.zeros_like(logits)

        loss = segmentation_loss(logits, targets)
        expected = F.binary_cross_entropy_with_logits(
            logits, torch.zeros_like(logits)
        )

        self.assertTrue(torch.isfinite(loss))
        self.assertAlmostEqual(loss.item(), expected.item(), places=6)

    def test_empty_target_zero_logits_is_log_two(self) -> None:
        """空标注 + 全零 logits 时，损失恰好是 ln(2)。"""
        logits = torch.zeros(2, 1, 8, 8)
        targets = torch.zeros_like(logits)

        loss = segmentation_loss(logits, targets)

        self.assertAlmostEqual(loss.item(), math.log(2.0), places=6)

    def test_empty_target_strong_negative_logits_is_near_zero(self) -> None:
        """空标注且预测强烈为背景时，损失应接近 0 且梯度有限。"""
        logits = torch.full((1, 1, 6, 6), -30.0, requires_grad=True)
        targets = torch.zeros_like(logits)

        loss = segmentation_loss(logits, targets)
        loss.backward()

        self.assertLess(loss.item(), 1e-9)
        self.assertIsNotNone(logits.grad)
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_empty_sample_does_not_change_dice_or_cldice(self) -> None:
        """混合 batch 中，空标注样本不参与 Dice/clDice 的平均。"""
        logits_mixed = torch.zeros(2, 1, 8, 8)
        targets_mixed = torch.stack(
            [torch.zeros(1, 8, 8), torch.ones(1, 8, 8)]
        )

        logits_single = torch.zeros(1, 1, 8, 8)
        targets_single = torch.ones(1, 1, 8, 8)

        mixed = segmentation_loss(logits_mixed, targets_mixed)
        single = segmentation_loss(logits_single, targets_single)

        self.assertAlmostEqual(mixed.item(), single.item(), places=6)

    def test_accepts_bhw_and_b1hw_without_changing_value(self) -> None:
        torch.manual_seed(1)
        logits_3d = torch.randn(2, 5, 5)
        targets_3d = (torch.rand(2, 5, 5) > 0.5).float()

        loss_3d = segmentation_loss(logits_3d, targets_3d)
        loss_4d = segmentation_loss(logits_3d.unsqueeze(1), targets_3d.unsqueeze(1))

        self.assertAlmostEqual(loss_3d.item(), loss_4d.item(), places=6)

    def test_fp16_large_image_loss_is_finite_float32(self) -> None:
        """1024² 的 fp16 logits 不能因 Dice 求和溢出而产生 Inf/NaN。"""
        logits = torch.zeros(1, 1, 1024, 1024, dtype=torch.float16)
        targets = torch.ones_like(logits)

        # 1M 个元素的和远超 fp16 最大值 65504，必须在 float32 中累加。
        self.assertGreater(logits.numel(), 65504)

        loss = segmentation_loss(logits, targets)

        self.assertEqual(loss.dtype, torch.float32)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreaterEqual(float(loss.item()), 0.0)

    def test_invalid_inputs_raise(self) -> None:
        with self.assertRaises(ValueError):
            segmentation_loss(torch.zeros(1, 2, 4, 4), torch.zeros(1, 2, 4, 4))
        with self.assertRaises(ValueError):
            segmentation_loss(torch.zeros(1, 1, 4, 4), torch.zeros(1, 1, 5, 5))
        with self.assertRaises(ValueError):
            segmentation_loss(
                torch.zeros(1, 1, 4, 4), torch.zeros(1, 1, 4, 4), cldice_weight=-1.0
            )
        with self.assertRaises(TypeError):
            segmentation_loss(torch.zeros(1, 1, 4, 4, dtype=torch.long), torch.zeros(1, 1, 4, 4))


class AuxiliaryLossTest(unittest.TestCase):
    def test_empty_masks_and_skeletons_are_finite_and_near_zero(self) -> None:
        """空道路、空骨架目标不应产生 NaN，也不应被 Dice 抬高。"""
        road_logits = torch.full((2, 1, 4, 4), -30.0)
        skeleton_logits = torch.full((2, 1, 4, 4), -30.0)
        masks = torch.zeros(2, 16, 16)
        skeletons = torch.zeros(2, 16, 16)

        loss = auxiliary_loss((road_logits, skeleton_logits), masks, skeletons)

        self.assertTrue(torch.isfinite(loss))
        self.assertLess(loss.item(), 1e-8)
        self.assertEqual(loss.ndim, 0)

    def test_empty_target_loss_is_finite_with_zero_logits(self) -> None:
        road_logits = torch.zeros(1, 1, 4, 4)
        skeleton_logits = torch.zeros(1, 1, 4, 4)
        masks = torch.zeros(1, 8, 8)
        skeletons = torch.zeros(1, 8, 8)

        loss = auxiliary_loss(
            {"road": road_logits, "skeleton": skeleton_logits}, masks, skeletons
        )

        self.assertAlmostEqual(loss.item(), math.log(2.0), places=6)

    def test_skeleton_downsample_uses_max_pooling(self) -> None:
        """整数倍下采样时，骨架目标用 max pooling 保留单像素宽中心线。"""
        target = torch.zeros(1, 1, 4, 4)
        target[0, 0, 1, 1] = 1.0
        target[0, 0, 3, 3] = 1.0
        reference = torch.zeros(1, 1, 2, 2)

        aligned = _align_binary_target(
            target, reference, preserve_connectivity=True
        )

        expected = F.max_pool2d(target, kernel_size=2, stride=2)
        self.assertTrue(torch.equal(aligned, expected))
        self.assertEqual(float(aligned.sum().item()), 2.0)

    def test_non_integer_ratio_falls_back_to_nearest(self) -> None:
        target = torch.zeros(1, 1, 5, 5)
        target[0, 0, 2, 2] = 1.0
        reference = torch.zeros(1, 1, 2, 2)

        aligned = _align_binary_target(
            target, reference, preserve_connectivity=True
        )
        expected = F.interpolate(target, size=(2, 2), mode="nearest")

        self.assertTrue(torch.equal(aligned, expected))

    def test_single_tensor_aux_logits_are_supported(self) -> None:
        aux_logits = torch.zeros(1, 2, 4, 4)
        masks = torch.zeros(1, 8, 8)
        skeletons = torch.zeros(1, 8, 8)

        loss = auxiliary_loss(aux_logits, masks, skeletons)

        self.assertAlmostEqual(loss.item(), math.log(2.0), places=6)

    def test_single_tensor_aux_logits_require_two_channels(self) -> None:
        with self.assertRaises(ValueError):
            auxiliary_loss(
                torch.zeros(1, 3, 4, 4), torch.zeros(1, 8, 8), torch.zeros(1, 8, 8)
            )

    def test_fp16_large_image_aux_loss_is_finite_float32(self) -> None:
        """辅助头在 fp16 AMP 下也必须用 float32 累加区域 Dice。"""
        aux_logits = torch.zeros(1, 2, 1024, 1024, dtype=torch.float16)
        masks = torch.ones(1, 1024, 1024, dtype=torch.float16)
        skeletons = torch.ones(1, 1024, 1024, dtype=torch.float16)

        self.assertGreater(masks.numel(), 65504)

        loss = auxiliary_loss(aux_logits, masks, skeletons)

        self.assertEqual(loss.dtype, torch.float32)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreaterEqual(float(loss.item()), 0.0)

    def test_auxiliary_padding_has_no_loss_or_gradient(self) -> None:
        logits = torch.randn(1, 2, 4, 4, requires_grad=True)
        target = torch.zeros(1, 1, 16, 16)
        target[:, :, :8, 2:4] = 1
        loss = auxiliary_loss(logits, target, target, valid_sizes=[(8, 8)])
        perturbed = logits.detach().clone()
        perturbed[:, :, 2:, :] = 100
        perturbed[:, :, :, 2:] = -100
        other = auxiliary_loss(perturbed, target, target, valid_sizes=[(8, 8)])
        torch.testing.assert_close(loss, other)
        loss.backward()
        self.assertEqual(float(logits.grad[:, :, 2:, :].abs().sum()), 0)
        self.assertEqual(float(logits.grad[:, :, :, 2:].abs().sum()), 0)
        self.assertGreater(float(logits.grad[:, :, :2, :2].abs().sum()), 0)

    def test_auxiliary_loss_requires_two_outputs(self) -> None:
        with self.assertRaises(ValueError):
            auxiliary_loss((torch.zeros(1, 1, 4, 4),), torch.zeros(1, 8, 8), torch.zeros(1, 8, 8))
        with self.assertRaises(ValueError):
            auxiliary_loss({"road": torch.zeros(1, 1, 4, 4)}, torch.zeros(1, 8, 8), torch.zeros(1, 8, 8))


if __name__ == "__main__":
    unittest.main()
