"""GRPO 数学工具测试：组内优势、裁剪目标与高斯 KL 梯度。"""

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

from diffpeak.grpo import (  # noqa: E402
    gaussian_kl_divergence,
    group_relative_advantages,
    grpo_policy_loss,
)


class GroupRelativeAdvantagesTest(unittest.TestCase):
    def test_multiple_images_are_normalized_independently(self) -> None:
        rewards = torch.tensor([[1.0, 10.0], [3.0, 20.0]], dtype=torch.float64)

        advantages = group_relative_advantages(rewards, delta=1e-3)

        expected = torch.tensor([[-1.0, -1.0], [1.0, 1.0]], dtype=torch.float64)
        torch.testing.assert_close(advantages, expected)

    def test_numpy_input_and_negative_dim(self) -> None:
        rewards = np.array([[1.0, 10.0], [3.0, 20.0]])

        advantages = group_relative_advantages(rewards, dim=-2)

        expected = np.array([[-1.0, -1.0], [1.0, 1.0]])
        np.testing.assert_allclose(advantages.numpy(), expected)

    def test_output_is_detached(self) -> None:
        rewards = torch.tensor(
            [[0.0], [1.0]], dtype=torch.float64, requires_grad=True
        )

        advantages = group_relative_advantages(rewards)

        self.assertFalse(advantages.requires_grad)
        self.assertIsNone(advantages.grad_fn)
        self.assertEqual(advantages.dtype, torch.float64)

    def test_constant_group_has_zero_advantages(self) -> None:
        rewards = torch.full((4, 3), 7.5)

        advantages = group_relative_advantages(rewards, delta=1e-4)

        torch.testing.assert_close(advantages, torch.zeros_like(advantages))

    def test_low_variance_group_is_scaled_not_skipped(self) -> None:
        # std = 5e-6 < delta = 1e-4，分母取 delta，优势应为 +/-0.05 而非 0。
        rewards = torch.tensor([[0.0], [1e-5]], dtype=torch.float64)

        advantages = group_relative_advantages(rewards, delta=1e-4)

        torch.testing.assert_close(
            advantages,
            torch.tensor([[-0.05], [0.05]], dtype=torch.float64),
        )
        self.assertTrue(bool((advantages != 0).all()))

    def test_large_variance_uses_actual_std(self) -> None:
        rewards = torch.tensor([[0.0], [2.0]], dtype=torch.float64)

        advantages = group_relative_advantages(rewards, delta=1e-4)

        torch.testing.assert_close(
            advantages,
            torch.tensor([[-1.0], [1.0]], dtype=torch.float64),
        )

    def test_delta_must_be_strictly_positive_and_finite(self) -> None:
        rewards = torch.tensor([0.0, 1.0])
        for delta in (0.0, -1e-4, math.nan, math.inf, -math.inf):
            with self.subTest(delta=delta):
                with self.assertRaises(ValueError):
                    group_relative_advantages(rewards, delta=delta)
        with self.assertRaises(ValueError):
            group_relative_advantages(rewards, delta="bad")

    def test_non_finite_rewards_are_rejected(self) -> None:
        for bad in (math.nan, math.inf, -math.inf):
            rewards = torch.tensor([0.0, bad])
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    group_relative_advantages(rewards)

        with self.assertRaises(ValueError):
            group_relative_advantages(np.array([0.0, np.nan]))

    def test_invalid_groups_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            group_relative_advantages(torch.empty(0))
        with self.assertRaises(ValueError):
            group_relative_advantages(torch.tensor(1.0))
        with self.assertRaises(ValueError):
            group_relative_advantages(torch.tensor([1.0]))
        with self.assertRaises(ValueError):
            group_relative_advantages(torch.ones(1, 3))
        with self.assertRaises(ValueError):
            group_relative_advantages(torch.ones(3, 2), dim=2)


class PolicyLossTest(unittest.TestCase):
    def test_loss_uses_clipped_surrogate(self) -> None:
        ratio = torch.tensor([0.8, 1.0, 1.2], dtype=torch.float64)
        advantage = torch.tensor([1.0, -0.5, 2.0], dtype=torch.float64)

        loss = grpo_policy_loss(ratio, advantage, clip_epsilon=0.1)

        clipped = torch.tensor([0.9, 1.0, 1.1], dtype=torch.float64)
        expected = -torch.minimum(ratio * advantage, clipped * advantage).mean()
        torch.testing.assert_close(loss, expected)

    def test_gradient_inside_clip_region(self) -> None:
        ratio = torch.tensor([1.0, 1.0], dtype=torch.float64, requires_grad=True)
        advantage = torch.tensor([1.0, -1.0], dtype=torch.float64)

        grpo_policy_loss(ratio, advantage, clip_epsilon=0.2).backward()

        # dL/dr_i = -A_i / N。
        torch.testing.assert_close(
            ratio.grad,
            torch.tensor([-0.5, 0.5], dtype=torch.float64),
        )

    def test_zero_gradient_when_positive_advantage_is_clipped(self) -> None:
        ratio = torch.tensor([1.5], dtype=torch.float64, requires_grad=True)
        advantage = torch.tensor([1.0], dtype=torch.float64)

        grpo_policy_loss(ratio, advantage, clip_epsilon=0.2).backward()

        torch.testing.assert_close(ratio.grad, torch.zeros_like(ratio))

    def test_gradient_when_positive_advantage_is_below_clip(self) -> None:
        ratio = torch.tensor([0.5], dtype=torch.float64, requires_grad=True)
        advantage = torch.tensor([1.0], dtype=torch.float64)

        grpo_policy_loss(ratio, advantage, clip_epsilon=0.2).backward()

        torch.testing.assert_close(
            ratio.grad, torch.tensor([-1.0], dtype=torch.float64)
        )

    def test_clip_validation(self) -> None:
        ratio = torch.tensor([1.0])
        advantage = torch.tensor([1.0])
        for epsilon in (-0.1, math.nan, math.inf):
            with self.subTest(epsilon=epsilon):
                with self.assertRaises(ValueError):
                    grpo_policy_loss(ratio, advantage, epsilon)


class GaussianKLTest(unittest.TestCase):
    def test_kl_value_and_mean_gradient(self) -> None:
        torch.manual_seed(0)
        reference = torch.randn(2, 1, 2, 2, dtype=torch.float64)
        std = torch.full((2, 1, 2, 2), 0.5, dtype=torch.float64)
        mean = (reference + 0.1).requires_grad_(True)

        kl = gaussian_kl_divergence(mean, reference, std)
        kl.sum().backward()

        expected = 0.5 * (
            ((mean.detach() - reference) / std).pow(2)
            .reshape(2, -1)
            .sum(dim=1)
        )
        torch.testing.assert_close(kl, expected)
        torch.testing.assert_close(
            mean.grad, (mean.detach() - reference) / std.pow(2)
        )
        self.assertTrue(bool(torch.isfinite(mean.grad).all()))

    def test_kl_uses_floor_for_tiny_std(self) -> None:
        mean = torch.zeros(2, 1, 1, 1, dtype=torch.float64)
        reference = torch.ones(2, 1, 1, 1, dtype=torch.float64)
        std = torch.zeros(2, 1, 1, 1, dtype=torch.float64)

        kl = gaussian_kl_divergence(mean, reference, std)

        self.assertTrue(bool(torch.isfinite(kl).all()))
        self.assertTrue(bool((kl > 0).all()))


if __name__ == "__main__":
    unittest.main()
