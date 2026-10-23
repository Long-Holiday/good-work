"""GaussianDiffusion 的单通道有符号场转移、损失、对数概率与采样测试。

关键设计约束：
* 目标场直接为 ``[-1, 1]`` 的单通道有符号场，不做仿射目标变换；
* 任意转移（包括末步）的高斯 std 都大于零（由 min_std 保证）；
* 采样从纯噪声开始，不使用旧背景均值补偿；终端裁剪只作用于返回场，
  records 中保存的是裁剪前的原始高斯动作；
* ``log_prob`` 对所有动作维度求 **sum**，而不是 mean；
* 扩散 MSE 只在有效像素上平均；场重建损失按四个互斥区域平衡。
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


class _TrainableDiT(nn.Module):
    """输出依赖参数与第一尺度条件的占位去噪器，用于检查采样梯度。"""

    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, state: torch.Tensor, timestep, conditions) -> torch.Tensor:
        # 条件通道数多于单通道潜变量，取前 latent_channels 个通道。
        return self.scale * conditions[0][:, : state.shape[1]].to(state.dtype)


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

    def test_single_channel_protocol_and_timestep_validation(self) -> None:
        diffusion = GaussianDiffusion()
        self.assertEqual(diffusion.latent_channels, 1)
        with self.assertRaises(ValueError):
            GaussianDiffusion(latent_channels=2)
        with self.assertRaises(ValueError):
            GaussianDiffusion(timesteps=10, heat_loss_max_timestep=11)
        with self.assertRaises(ValueError):
            GaussianDiffusion(timesteps=10, heat_loss_max_timestep=-1)

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
        state = torch.zeros(2, 1, 8, 8)

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
        state = torch.randn(2, 1, 8, 8)
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
        state = torch.zeros(1, 1, 8, 8)

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
        action = torch.randn(2, 1, 4, 5, dtype=torch.float64)
        mean = torch.randn(2, 1, 4, 5, dtype=torch.float64)
        std = torch.full((2, 1, 4, 5), 0.1, dtype=torch.float64)

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
        mean = torch.randn(2, 1, 3, 3)
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
        mean = torch.zeros(1, 1, 2, 2)
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

        self.assertEqual(tuple(heatmaps.shape), (2, 1, 8, 8))
        self.assertTrue(torch.isfinite(heatmaps).all())
        self.assertTrue(torch.all(heatmaps >= -1.0))
        self.assertTrue(torch.all(heatmaps <= 1.0))
        self.assertEqual(len(records), self.diffusion.sampling_steps)

        previous_timestep = self.diffusion.timesteps - 1
        for record in records:
            self.assertEqual(tuple(record["state"].shape), (2, 1, 8, 8))
            self.assertEqual(tuple(record["action"].shape), (2, 1, 8, 8))
            self.assertEqual(tuple(record["old_log_prob"].shape), (2,))
            self.assertTrue(torch.isfinite(record["old_log_prob"]).all())
            self.assertLess(record["timestep"], previous_timestep + 1)
            previous_timestep = record["timestep"]

    def test_sampling_starts_from_pure_noise_without_background_prior(self) -> None:
        for steps in (1, 3, 10):
            with self.subTest(sampling_steps=steps):
                diffusion = GaussianDiffusion(timesteps=10, sampling_steps=steps)
                seed = 17
                expected_generator = torch.Generator().manual_seed(seed)
                expected_state = torch.randn((2, 1, 8, 8), generator=expected_generator)
                heatmaps, records = diffusion.sample(
                    self.dit, self.conditions,
                    generator=torch.Generator().manual_seed(seed), record=True,
                )
                # 新协议没有背景均值补偿：初始状态就是纯高斯噪声。
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

    def test_terminal_clamp_happens_after_recording_raw_action(self) -> None:
        diffusion = GaussianDiffusion(
            timesteps=8, sampling_steps=1, min_std=2.0, heatmap_size=8
        )

        heatmaps, records = diffusion.sample(
            self.dit,
            self.conditions,
            generator=torch.Generator().manual_seed(11),
            record=True,
        )

        action = records[0]["action"]
        # min_std=2.0 保证至少部分原始动作落到 [-1, 1] 之外。
        self.assertTrue(bool(torch.any(action.abs() > 1.0)))
        torch.testing.assert_close(heatmaps, action.clamp(-1.0, 1.0))
        self.assertTrue(torch.isfinite(records[0]["old_log_prob"]).all())

    def test_sample_without_record_returns_signed_field(self) -> None:
        heatmaps = self.diffusion.sample(self.dit, self.conditions)

        self.assertIsInstance(heatmaps, torch.Tensor)
        self.assertEqual(tuple(heatmaps.shape), (2, 1, 8, 8))
        self.assertTrue(torch.all(heatmaps >= -1.0))
        self.assertTrue(torch.all(heatmaps <= 1.0))

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

    def test_sample_differentiable_matches_sample_and_propagates_gradients(self) -> None:
        """同一 generator 下可微采样与 ``sample`` 数值一致，但保留计算图。"""
        dit = _TrainableDiT()
        conditions = [
            torch.randn_like(item).requires_grad_(True) for item in self.conditions
        ]
        seed = 23

        plain = self.diffusion.sample(
            dit,
            conditions,
            generator=torch.Generator().manual_seed(seed),
            record=False,
        )
        differentiable = self.diffusion.sample_differentiable(
            dit,
            conditions,
            generator=torch.Generator().manual_seed(seed),
            record=False,
        )

        self.assertFalse(plain.requires_grad)
        self.assertTrue(differentiable.requires_grad)
        torch.testing.assert_close(differentiable, plain)

        differentiable.square().sum().backward()
        self.assertIsNotNone(dit.scale.grad)
        self.assertTrue(bool(torch.count_nonzero(dit.scale.grad)))
        self.assertIsNotNone(conditions[0].grad)
        self.assertTrue(bool(torch.count_nonzero(conditions[0].grad)))

    def test_sample_differentiable_supports_recording(self) -> None:
        dit = _TrainableDiT()
        conditions = [
            torch.zeros(2, 4, 8, 8),
            torch.zeros(2, 4, 4, 4),
            torch.zeros(2, 4, 2, 2),
            torch.zeros(2, 4, 1, 1),
        ]

        heatmaps, records = self.diffusion.sample_differentiable(
            dit,
            conditions,
            generator=torch.Generator().manual_seed(5),
            record=True,
        )

        self.assertTrue(heatmaps.requires_grad)
        self.assertEqual(len(records), self.diffusion.sampling_steps)
        for record in records:
            self.assertFalse(record["state"].requires_grad)
            self.assertFalse(record["action"].requires_grad)
            self.assertTrue(torch.isfinite(record["old_log_prob"]).all())
        heatmaps.sum().backward()
        self.assertIsNotNone(dit.scale.grad)


class GaussianDiffusionLossTest(unittest.TestCase):
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

    def test_training_losses_are_finite_for_signed_target(self) -> None:
        target = torch.zeros(2, 1, 8, 8)

        losses = self.diffusion.training_losses(self.dit, self.conditions, target)

        self.assertEqual(set(losses), {"diff", "heat"})
        self.assertTrue(torch.isfinite(losses["diff"]))
        self.assertTrue(torch.isfinite(losses["heat"]))
        self.assertGreaterEqual(float(losses["diff"].item()), 0.0)
        self.assertGreaterEqual(float(losses["heat"].item()), 0.0)

    def test_training_losses_can_return_differentiable_signed_prediction(self) -> None:
        """``return_prediction=True`` 额外返回可反传的有符号 ``heat_pred``。"""
        dit = _TrainableDiT()
        target = torch.rand(2, 1, 8, 8) * 2.0 - 1.0
        conditions = [torch.randn_like(item) for item in self.conditions]

        losses = self.diffusion.training_losses(
            dit, conditions, target, return_prediction=True
        )

        self.assertEqual(set(losses), {"diff", "heat", "heat_pred"})
        prediction = losses["heat_pred"]
        self.assertEqual(tuple(prediction.shape), tuple(target.shape))
        self.assertTrue(torch.isfinite(prediction).all())
        self.assertTrue(torch.all(prediction >= -1.0))
        self.assertTrue(torch.all(prediction <= 1.0))
        self.assertTrue(prediction.requires_grad)

        (losses["diff"] + losses["heat"] + prediction.sum()).backward()
        self.assertIsNotNone(dit.scale.grad)
        self.assertTrue(bool(torch.count_nonzero(dit.scale.grad)))

    def test_training_losses_reject_wrong_channels(self) -> None:
        with self.assertRaises(ValueError):
            self.diffusion.training_losses(
                self.dit, self.conditions, torch.zeros(2, 2, 8, 8)
            )
        with self.assertRaises(ValueError):
            self.diffusion.training_losses(
                self.dit, self.conditions, torch.zeros(2, 1, 8)
            )

    def test_diff_loss_only_averages_valid_pixels(self) -> None:
        target = torch.zeros(2, 1, 8, 8)
        valid = torch.zeros(2, 1, 8, 8, dtype=torch.bool)
        valid[0, :, :4, :4] = True  # 第二个样本完全无效，必须被排除。

        torch.manual_seed(0)
        losses = self.diffusion.training_losses(
            self.dit, self.conditions, target, valid_mask=valid
        )

        torch.manual_seed(0)
        timesteps = torch.randint(0, self.diffusion.timesteps, (2,))
        noise = torch.randn_like(target)
        alpha_bar = self.diffusion._extract(
            self.diffusion.alphas_cumprod, timesteps, target.shape
        )
        sqrt_alpha = self.diffusion._extract(
            self.diffusion.sqrt_alphas_cumprod, timesteps, target.shape
        )
        sqrt_one_minus = self.diffusion._extract(
            self.diffusion.sqrt_one_minus_alphas_cumprod, timesteps, target.shape
        )
        z_t = sqrt_alpha * target + sqrt_one_minus * noise
        squared_error = (torch.zeros_like(z_t) - noise) ** 2
        valid_float = valid.to(squared_error.dtype)
        expected = (squared_error * valid_float).sum() / valid_float.sum()

        self.assertTrue(torch.allclose(losses["diff"], expected, atol=1e-7, rtol=1e-7))

    def test_field_loss_balances_four_regions_and_skips_missing(self) -> None:
        prediction = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
        target = torch.zeros_like(prediction)
        valid = torch.ones_like(prediction, dtype=torch.bool)
        regions = torch.zeros(1, 4, 2, 2)
        regions[0, 0, 0, 0] = 1.0
        regions[0, 1, 0, 1] = 1.0
        regions[0, 2, 1, 0] = 1.0
        regions[0, 3, 1, 1] = 1.0

        loss = self.diffusion._field_loss(prediction, target, valid, regions)

        # 每个区域均方误差为 1、4、9、16，四区域等权平均。
        self.assertAlmostEqual(float(loss), 7.5, places=6)

        missing = regions.clone()
        missing[0, 1] = 0.0
        partial = self.diffusion._field_loss(prediction, target, valid, missing)
        # 缺失区域被跳过，仅对其余三个区域取平均。
        self.assertAlmostEqual(float(partial), (1.0 + 9.0 + 16.0) / 3.0, places=6)

    def test_field_loss_excludes_invalid_samples(self) -> None:
        prediction = torch.tensor(
            [
                [[[10.0, 10.0], [10.0, 10.0]]],
                [[[1.0, 2.0], [3.0, 4.0]]],
            ]
        )
        target = torch.zeros_like(prediction)
        valid = torch.ones(2, 1, 2, 2, dtype=torch.bool)
        valid[0] = False
        regions = torch.zeros(2, 4, 2, 2)
        regions[1, 0, 0, 0] = 1.0
        regions[1, 1, 0, 1] = 1.0
        regions[1, 2, 1, 0] = 1.0
        regions[1, 3, 1, 1] = 1.0

        batched = self.diffusion._field_loss(prediction, target, valid, regions)
        single = self.diffusion._field_loss(
            prediction[1:2], target[1:2], valid[1:2], regions[1:2]
        )

        self.assertAlmostEqual(float(batched), float(single), places=6)

    def test_field_loss_fallback_balances_by_target_sign(self) -> None:
        target = torch.full((1, 1, 2, 2), -1.0)
        target[0, 0, 0, 0] = 1.0
        prediction = target.clone()
        prediction[0, 0, 0, 0] = 2.0  # 唯一正区域像素误差为 1。
        valid = torch.ones_like(target, dtype=torch.bool)

        fallback = self.diffusion._field_loss(prediction, target, valid)
        plain = ((prediction - target) ** 2).mean()

        # 无区域输入时按正/负两个区域等权：正区域均值 1，负区域均值 0。
        self.assertAlmostEqual(float(fallback), 0.5, places=6)
        self.assertAlmostEqual(float(plain), 0.25, places=6)

    def test_field_loss_is_zero_without_valid_regions(self) -> None:
        prediction = target = torch.zeros(2, 1, 4, 4)
        valid = torch.zeros(2, 1, 4, 4, dtype=torch.bool)
        regions = torch.zeros(2, 4, 4, 4)

        loss = self.diffusion._field_loss(prediction, target, valid, regions)

        self.assertEqual(float(loss), 0.0)
        self.assertTrue(torch.isfinite(loss))

    def test_training_losses_validate_masks_and_regions(self) -> None:
        target = torch.zeros(2, 1, 8, 8)
        with self.assertRaises(ValueError):
            self.diffusion.training_losses(
                self.dit, self.conditions, target, valid_mask=torch.ones(2, 1, 4, 4)
            )
        with self.assertRaises(TypeError):
            self.diffusion.training_losses(
                self.dit, self.conditions, target, valid_mask=[1, 1]
            )
        with self.assertRaises(ValueError):
            self.diffusion.training_losses(
                self.dit,
                self.conditions,
                target,
                field_regions=torch.zeros(2, 3, 8, 8),
            )
        with self.assertRaises(TypeError):
            self.diffusion.training_losses(
                self.dit, self.conditions, target, field_regions=[[0.0]]
            )

    def test_heat_loss_respects_timestep_limit(self) -> None:
        diffusion = GaussianDiffusion(
            timesteps=10,
            sampling_steps=3,
            heat_loss_max_timestep=0,
        )

        losses = diffusion.training_losses(
            self.dit,
            self.conditions,
            torch.rand(2, 1, 8, 8) * 2.0 - 1.0,
        )

        self.assertEqual(float(losses["heat"]), 0.0)


if __name__ == "__main__":
    unittest.main()
