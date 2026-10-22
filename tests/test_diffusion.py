"""GaussianDiffusion 的转移、对数概率与采样测试。

关键设计约束：
* 任意转移（包括末步)的高斯 std 都大于零（由 min_std 保证）；
* ``log_prob`` 对所有动作维度求 **sum**，而不是 mean；
* 采样记录中的 old_log_prob 必须有限，可直接用于 GRPO 比值。
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diffpeak.models.diffusion import GaussianDiffusion  # noqa: E402


class _ZeroDiT(nn.Module):
    """始终预测零噪声的占位去噪器。"""

    def forward(self, state: torch.Tensor, timestep, conditions) -> torch.Tensor:
        return torch.zeros_like(state)


class GaussianDiffusionTransitionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.diffusion = GaussianDiffusion(
            timesteps=50,
            sampling_steps=4,
            min_std=0.02,
            heatmap_size=8,
        )
        self.dit = _ZeroDiT()
        self.conditions = [torch.zeros(2, 4, 8, 8) for _ in range(4)]

    def test_invalid_policy_variance_and_schedule_are_rejected(self) -> None:
        for std in (0.0, -0.02, float("nan"), float("inf")):
            with self.subTest(min_std=std), self.assertRaises(ValueError):
                GaussianDiffusion(min_std=std)
        for eta in (-1.0, 1.1, float("nan"), float("inf")):
            with self.subTest(eta=eta), self.assertRaises(ValueError):
                GaussianDiffusion(eta=eta)
        for start, end in ((0.0, 0.02), (0.1, 0.01), (0.01, 1.0), (float("nan"), 0.02)):
            with self.subTest(beta=(start, end)), self.assertRaises(ValueError):
                GaussianDiffusion(beta_start=start, beta_end=end)

    def test_sampling_schedule_is_descending(self) -> None:
        schedule = self.diffusion.sampling_schedule.tolist()

        self.assertEqual(len(schedule), self.diffusion.sampling_steps + 1)
        self.assertEqual(schedule[0], self.diffusion.timesteps - 1)
        # 末尾的 -1 表示 alpha_bar=1 的干净端点，确保最后一步真正去噪到 x0。
        self.assertEqual(schedule[-1], -1)
        self.assertTrue(all(a > b for a, b in zip(schedule, schedule[1:])))

    def test_single_sampling_step_schedule(self) -> None:
        diffusion = GaussianDiffusion(timesteps=50, sampling_steps=1)

        self.assertEqual(diffusion.sampling_schedule.tolist(), [49, -1])

    def test_sampling_steps_equal_timesteps_covers_all_steps(self) -> None:
        diffusion = GaussianDiffusion(timesteps=10, sampling_steps=10)

        schedule = diffusion.sampling_schedule.tolist()

        self.assertEqual(schedule, list(range(9, -1, -1)) + [-1])
        self.assertEqual(len(schedule), diffusion.sampling_steps + 1)

    def test_final_step_std_is_nonzero(self) -> None:
        """末步 previous_timestep=-1 仍必须有非零高斯方差。"""
        state = torch.zeros(2, 2, 8, 8)

        mean, std = self.diffusion.transition(
            self.dit, state, timestep=0, previous_timestep=-1, conditions=self.conditions
        )

        self.assertEqual(tuple(mean.shape), tuple(state.shape))
        self.assertTrue(torch.isfinite(mean).all())
        self.assertTrue(torch.isfinite(std).all())
        self.assertTrue(torch.all(std >= self.diffusion.min_std))
        self.assertTrue(torch.all(std > 0.0))
        # std 允许以 (B,1,1,1) 广播形式返回；展开后每个动作维度都是下限。
        self.assertTrue(
            torch.equal(std.expand_as(state), torch.full_like(state, 0.02))
        )

    def test_every_transition_respects_min_std(self) -> None:
        state = torch.randn(2, 2, 8, 8)
        for timestep, previous in ((10, 5), (1, 0), (25, 12)):
            with self.subTest(timestep=timestep):
                _, std = self.diffusion.transition(
                    self.dit,
                    state,
                    timestep=timestep,
                    previous_timestep=previous,
                    conditions=self.conditions,
                )
                self.assertTrue(torch.all(std >= self.diffusion.min_std))

    def test_eta_adds_variance_above_min_std(self) -> None:
        diffusion = GaussianDiffusion(
            timesteps=50, sampling_steps=4, min_std=1e-4, eta=1.0
        )
        state = torch.zeros(1, 2, 8, 8)

        _, std = diffusion.transition(
            self.dit,
            state,
            timestep=25,
            previous_timestep=10,
            conditions=[torch.zeros(1, 4, 8, 8) for _ in range(4)],
        )

        self.assertTrue(torch.isfinite(std).all())
        self.assertTrue(torch.all(std >= diffusion.min_std))
        self.assertGreater(float(std.max().item()), float(diffusion.min_std))

    def test_invalid_configuration_raises(self) -> None:
        with self.assertRaises(ValueError):
            GaussianDiffusion(min_std=0.0)
        with self.assertRaises(ValueError):
            GaussianDiffusion(timesteps=10, sampling_steps=11)


class GaussianDiffusionLogProbTest(unittest.TestCase):
    def setUp(self) -> None:
        self.diffusion = GaussianDiffusion(timesteps=50, sampling_steps=4)

    def test_log_prob_sums_over_all_action_dimensions(self) -> None:
        torch.manual_seed(0)
        action = torch.randn(2, 2, 4, 5, dtype=torch.float64)
        mean = torch.randn(2, 2, 4, 5, dtype=torch.float64)
        std = torch.full((2, 2, 4, 5), 0.1, dtype=torch.float64)

        log_prob = self.diffusion.log_prob(action, mean, std)

        standardized = (action - mean) / std
        elementwise = (
            -0.5 * standardized * standardized
            - torch.log(std)
            - 0.5 * math.log(2.0 * math.pi)
        )
        expected = elementwise.reshape(elementwise.shape[0], -1).sum(dim=1)

        self.assertEqual(tuple(log_prob.shape), (2,))
        self.assertTrue(torch.allclose(log_prob, expected, atol=1e-10, rtol=1e-10))

    def test_log_prob_at_mean_is_sum_not_mean(self) -> None:
        mean = torch.randn(2, 2, 3, 3)
        std = torch.full_like(mean, 0.2)

        log_prob = self.diffusion.log_prob(mean, mean, std)

        numel = mean[0].numel()
        per_element = -math.log(0.2) - 0.5 * math.log(2.0 * math.pi)
        expected = per_element * numel

        self.assertTrue(
            torch.allclose(
                log_prob, torch.full_like(log_prob, expected), atol=1e-6, rtol=1e-6
            )
        )

    def test_log_prob_clamps_zero_std(self) -> None:
        mean = torch.zeros(1, 2, 2, 2)
        action = torch.ones_like(mean)

        with_zero = self.diffusion.log_prob(action, mean, torch.zeros_like(mean))
        with_floor = self.diffusion.log_prob(
            action, mean, torch.full_like(mean, 1e-12)
        )

        self.assertTrue(torch.isfinite(with_zero).all())
        self.assertTrue(torch.allclose(with_zero, with_floor, atol=1e-6, rtol=1e-6))


class GaussianDiffusionSamplingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.diffusion = GaussianDiffusion(
            timesteps=10,
            sampling_steps=3,
            min_std=0.02,
            heatmap_size=8,
        )
        self.dit = _ZeroDiT()
        self.conditions = [
            torch.zeros(2, 4, 8, 8),
            torch.zeros(2, 4, 4, 4),
            torch.zeros(2, 4, 2, 2),
            torch.zeros(2, 4, 1, 1),
        ]

    def test_sample_records_have_finite_sum_log_prob(self) -> None:
        generator = torch.Generator().manual_seed(7)

        heatmaps, records = self.diffusion.sample(
            self.dit, self.conditions, generator=generator, record=True
        )

        self.assertEqual(tuple(heatmaps.shape), (2, 2, 8, 8))
        self.assertTrue(torch.isfinite(heatmaps).all())
        self.assertTrue(torch.all(heatmaps >= 0.0))
        self.assertTrue(torch.all(heatmaps <= 1.0))
        self.assertEqual(len(records), self.diffusion.sampling_steps)

        previous_timestep = self.diffusion.timesteps - 1
        for record in records:
            self.assertEqual(tuple(record["state"].shape), (2, 2, 8, 8))
            self.assertEqual(tuple(record["action"].shape), (2, 2, 8, 8))
            self.assertEqual(tuple(record["old_log_prob"].shape), (2,))
            self.assertTrue(torch.isfinite(record["old_log_prob"]).all())
            self.assertLess(record["timestep"], previous_timestep + 1)
            previous_timestep = record["timestep"]

    def test_sampling_uses_background_prior_and_preserves_random_draws(self) -> None:
        for steps in (1, 3, 10):
            with self.subTest(sampling_steps=steps):
                diffusion = GaussianDiffusion(timesteps=10, sampling_steps=steps)
                seed = 17
                expected_generator = torch.Generator().manual_seed(seed)
                noise = torch.randn((2, 2, 8, 8), generator=expected_generator)
                expected_state = noise - diffusion.alphas_cumprod[-1].sqrt()
                heatmaps, records = diffusion.sample(
                    self.dit, self.conditions,
                    generator=torch.Generator().manual_seed(seed), record=True,
                )
                torch.testing.assert_close(records[0]["state"], expected_state)
                mean, std = diffusion.transition(
                    self.dit, expected_state, records[0]["timestep"],
                    records[0]["previous_timestep"], self.conditions,
                )
                expected_action = mean + std * torch.randn(
                    expected_state.shape, generator=expected_generator
                )
                torch.testing.assert_close(records[0]["action"], expected_action)
                plain = diffusion.sample(
                    self.dit, self.conditions,
                    generator=torch.Generator().manual_seed(seed), record=False,
                )
                torch.testing.assert_close(plain, heatmaps)

    def test_sample_without_record_returns_heatmap_only(self) -> None:
        heatmaps = self.diffusion.sample(self.dit, self.conditions)

        self.assertIsInstance(heatmaps, torch.Tensor)
        self.assertEqual(tuple(heatmaps.shape), (2, 2, 8, 8))

    def test_single_step_sampling_ends_at_clean_endpoint(self) -> None:
        diffusion = GaussianDiffusion(
            timesteps=8, sampling_steps=1, min_std=0.02, heatmap_size=8
        )

        heatmaps, records = diffusion.sample(
            self.dit, self.conditions, record=True
        )

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["timestep"], 7)
        self.assertEqual(records[0]["previous_timestep"], -1)
        self.assertEqual(tuple(records[0]["old_log_prob"].shape), (2,))
        self.assertTrue(torch.isfinite(records[0]["old_log_prob"]).all())
        self.assertTrue(torch.isfinite(heatmaps).all())

    def test_training_losses_are_finite_for_zero_target(self) -> None:
        target = torch.zeros(2, 2, 8, 8)

        losses = self.diffusion.training_losses(self.dit, self.conditions, target)

        self.assertEqual(set(losses), {"diff", "heat"})
        self.assertTrue(torch.isfinite(losses["diff"]))
        self.assertTrue(torch.isfinite(losses["heat"]))
        self.assertGreaterEqual(float(losses["diff"].item()), 0.0)
        self.assertGreaterEqual(float(losses["heat"].item()), 0.0)

    def test_training_losses_reject_wrong_channels(self) -> None:
        with self.assertRaises(ValueError):
            self.diffusion.training_losses(
                self.dit, self.conditions, torch.zeros(2, 1, 8, 8)
            )
        with self.assertRaises(ValueError):
            self.diffusion.training_losses(
                self.dit, self.conditions, torch.zeros(2, 2, 8)
            )


if __name__ == "__main__":
    unittest.main()
