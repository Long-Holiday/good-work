"""联合 ABC GRPO 的轻量单元/集成测试。

测试不加载 SAM2 或 DeepGlobe 数据，只用微型条件编码器、DiT、扩散替身和
可微密集热图解码器验证：

* Bernoulli 终端对数概率/KL 只对有效像素求和且使用 float64；
* 采样动作的生成器设备语义与可复现性；
* 联合组同时训练 condition_encoder、DiT 与 SAM mask decoder；
* 同一组相对优势同时驱动扩散转移与终端 Bernoulli 策略；
* 扩散项按 G*T*B、终端项按 G*B 归一，每轮只 step 一次；
* 常量组在无 KL/anchor 时完全跳过 backward/optimizer；
* 终端热图 detach、参考 mask decoder 冻结、参考条件只编码一次；
* 密集提示的奖励点计数为 0，不产生稀疏点惩罚。
"""

from __future__ import annotations

import copy
import math
import sys
import unittest
from pathlib import Path
from unittest import mock

import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diffpeak.engine import DiffPeakEngine  # noqa: E402
from diffpeak.joint_rl import (  # noqa: E402
    _sample_terminal_actions,
    bernoulli_kl_divergence,
    bernoulli_log_prob,
    joint_grpo_group,
)
from diffpeak.rewards import RewardConfig  # noqa: E402


# ---------------------------------------------------------------------------
# 微型模型
# ---------------------------------------------------------------------------
class _TinyDecoder(nn.Module):
    def __init__(self, weight: float = 2.0, bias: float = 0.0) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(float(weight)))
        self.bias = nn.Parameter(torch.tensor(float(bias)))


class _TinySam(nn.Module):
    """提供 encode 与可微 decode_heatmaps 的最小 SAM 替身。"""

    def __init__(self, decoder_weight: float = 2.0, decoder_bias: float = 0.0) -> None:
        super().__init__()
        # 冻结的图像编码器参数：联合训练不应修改它。
        self.encoder_scale = nn.Parameter(torch.tensor(1.0), requires_grad=False)
        self.decoder = _TinyDecoder(decoder_weight, decoder_bias)
        self.encode_calls = 0
        self.decode_calls: list[dict[str, object]] = []

    def encode(self, images: torch.Tensor, valid_sizes=None) -> torch.Tensor:
        self.encode_calls += 1
        return images.mean(dim=(1, 2, 3), keepdim=True)

    def decode_heatmaps(
        self,
        embeddings: torch.Tensor,
        heatmaps: torch.Tensor,
        output_size=(4, 4),
        valid_sizes=None,
        *,
        decoder: nn.Module | None = None,
    ) -> torch.Tensor:
        module = self.decoder if decoder is None else decoder
        self.decode_calls.append(
            {
                "decoder": module,
                "uses_reference": decoder is not None,
                "heatmaps_requires_grad": bool(heatmaps.requires_grad),
                "heatmap_sum": float(heatmaps.detach().double().sum().cpu()),
                "grad_enabled": torch.is_grad_enabled(),
                "valid_sizes": valid_sizes,
            }
        )
        signed = heatmaps[:, 0:1] - heatmaps[:, 1:2]
        logits = signed * module.weight + module.bias
        if tuple(logits.shape[-2:]) != tuple(output_size):
            logits = F.interpolate(
                logits, size=tuple(output_size), mode="nearest"
            )
        return logits


class _TinyConditionEncoder(nn.Module):
    def __init__(self, value: float = 0.5, height: int = 4, width: int = 4) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(float(value)))
        self.height = int(height)
        self.width = int(width)
        self.calls = 0

    def forward(self, images: torch.Tensor, embeddings: torch.Tensor):
        self.calls += 1
        batch_size = int(images.shape[0])
        base = torch.ones(
            batch_size,
            2,
            self.height,
            self.width,
            device=images.device,
            dtype=images.dtype,
        )
        condition = base * self.weight
        auxiliary = torch.zeros(
            batch_size,
            2,
            self.height,
            self.width,
            device=images.device,
            dtype=images.dtype,
        )
        return [condition, condition * 0.5, condition * 0.25, condition * 0.125], auxiliary


class _TinyDit(nn.Module):
    def __init__(self, value: float = 1.0) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(float(value)))


class _TinyDiffusion:
    """确定性 DDIM 替身：sample 与 replay 的 transition 完全一致。"""

    def __init__(self, steps: int = 2, constant: bool = False) -> None:
        self.steps = int(steps)
        self.constant = bool(constant)
        self.sample_calls = 0
        self.sample_seeds: list[int | None] = []
        self.transitions: list[dict[str, object]] = []

    def _heat(self, batch_size: int, height: int, width: int, device) -> torch.Tensor:
        # 非恒定模式交替 on/off，保证两条轨迹奖励不同。
        if self.constant:
            on = True
        else:
            on = self.sample_calls % 2 == 0
        positive = torch.full(
            (batch_size, 1, height, width),
            1.0 if on else 0.0,
            device=device,
            dtype=torch.float32,
        )
        negative = torch.zeros_like(positive)
        return torch.cat([positive, negative], dim=1)

    def sample(self, dit, conditions, generator=None, record: bool = False):
        self.sample_seeds.append(
            int(generator.initial_seed()) if generator is not None else None
        )
        batch_size, _, height, width = (
            int(conditions[0].shape[0]),
            int(conditions[0].shape[1]),
            int(conditions[0].shape[2]),
            int(conditions[0].shape[3]),
        )
        heatmaps = self._heat(batch_size, height, width, conditions[0].device)
        state = heatmaps * 2.0 - 1.0
        records: list[dict[str, object]] = []
        for timestep in range(self.steps, 0, -1):
            mean, std = self.transition(
                dit, state, timestep, timestep - 1, conditions
            )
            action = mean
            records.append(
                {
                    "state": state.detach().cpu(),
                    "action": action.detach().cpu(),
                    "timestep": timestep,
                    "previous_timestep": timestep - 1,
                    "old_log_prob": self.log_prob(action, mean, std).detach().cpu(),
                }
            )
            state = action
        self.sample_calls += 1
        if record:
            return heatmaps, records
        return heatmaps

    def transition(self, dit, state, timestep, previous_timestep, conditions):
        self.transitions.append(
            {
                "grad_enabled": torch.is_grad_enabled(),
                "conditions_requires_grad": bool(conditions[0].requires_grad),
                "conditions_grad_fn": conditions[0].grad_fn is not None,
            }
        )
        # 同时依赖 DiT 与条件编码器，便于检查两条梯度路径。
        mean = state * 0.5 + conditions[0] * dit.weight * 0.01
        std = torch.ones_like(state) * 0.5
        return mean, std

    @staticmethod
    def log_prob(action, mean, std):
        action64 = action.to(torch.float64)
        mean64 = mean.to(torch.float64)
        std64 = std.to(torch.float64).clamp_min(1e-12)
        elementwise = (
            -0.5 * ((action64 - mean64) / std64) ** 2
            - torch.log(std64)
            - 0.5 * math.log(2.0 * math.pi)
        )
        return elementwise.reshape(elementwise.shape[0], -1).sum(dim=1)


# ---------------------------------------------------------------------------
# 引擎/优化器替身
# ---------------------------------------------------------------------------
class _RecordingScaler:
    def __init__(self) -> None:
        self.losses: list[float] = []

    def is_enabled(self) -> bool:
        return False

    def scale(self, loss):
        self.losses.append(float(loss.detach()))
        return loss

    def unscale_(self, optimizer) -> None:
        return None

    def step(self, optimizer) -> None:
        optimizer.step()

    def update(self) -> None:
        return None

    def get_scale(self) -> float:
        return 1.0


class _RecordingScheduler:
    def __init__(self) -> None:
        self.steps = 0

    def step(self) -> None:
        self.steps += 1


def _freeze(module: nn.Module) -> nn.Module:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    return module


def _make_engine(
    *,
    constant: bool = False,
    kl_weight: float = 0.0,
    anchor_weight: float = 0.0,
    update_epochs: int = 1,
    group_size: int = 2,
    steps: int = 2,
) -> DiffPeakEngine:
    engine = object.__new__(DiffPeakEngine)
    engine.device = torch.device("cpu")
    engine.amp = False
    engine.image_size = 4
    engine.heatmap_size = 4
    engine.seed = 1234
    engine.group_size = int(group_size)
    engine.ppo_clip = 0.2
    engine.kl_weight = float(kl_weight)
    engine.anchor_weight = float(anchor_weight)
    engine.update_epochs = int(update_epochs)
    engine.advantage_std_delta = 1e-4
    engine.grad_clip = 0.0
    engine.global_step = 0
    engine.grpo_groups_processed = 0
    engine.point_cost_warmup_steps = 0
    engine._stage = "D"
    engine.reward_config = RewardConfig(
        iou_weight=0.5, cldice_weight=0.5, path_weight=0.0
    )

    engine.sam = _TinySam(decoder_weight=2.0, decoder_bias=0.0)
    engine.condition_encoder = _TinyConditionEncoder(0.5)
    engine.dit = _TinyDit(1.0)
    engine.ref_dit = _freeze(copy.deepcopy(engine.dit))
    engine.ref_condition_encoder = _freeze(copy.deepcopy(engine.condition_encoder))
    engine.ref_sam_decoder = _freeze(copy.deepcopy(engine.sam.decoder))
    # 参考解码器与当前策略不同，终端 KL 才有非零信号。
    with torch.no_grad():
        engine.ref_sam_decoder.weight.fill_(1.0)
        engine.ref_sam_decoder.bias.fill_(0.0)

    engine.diffusion = _TinyDiffusion(steps=steps, constant=constant)
    engine.optimizer = torch.optim.AdamW(
        engine._trainable_parameters(), lr=1e-2
    )
    engine.scheduler = _RecordingScheduler()
    engine.scaler = _RecordingScaler()
    engine.anchor_calls: list[tuple[str, int]] = []

    def supervised(stage, batch, epoch):
        engine.anchor_calls.append((str(stage), int(epoch)))
        return {"loss": 0.5 * engine.dit.weight.pow(2)}

    engine._supervised_loss = supervised  # type: ignore[method-assign]
    return engine


def _batch(batch_size: int = 1, valid_sizes=None) -> dict[str, object]:
    if valid_sizes is None:
        valid_sizes = [(3, 4)] * batch_size
    images = torch.full((batch_size, 3, 4, 4), 0.5)
    mask = torch.zeros(batch_size, 1, 4, 4)
    mask[:, :, :, :2] = 1.0  # 左半为道路，用于区分全 1/全 0 预测
    return {
        "image": images,
        "mask": mask,
        "skeleton": torch.zeros_like(mask),
        "heatmaps": torch.zeros(batch_size, 2, 4, 4),
        "valid_sizes": list(valid_sizes),
        "ids": [f"image-{index}" for index in range(batch_size)],
    }


def _run_joint(engine, batch, epoch: int = 0):
    """联合测试中固定终端动作为 logits 的硬阈值，消除采样偶然性。"""
    with mock.patch(
        "diffpeak.joint_rl._sample_terminal_actions",
        side_effect=lambda logits, generator: logits > 0,
    ):
        return joint_grpo_group(engine, batch, epoch)


# ---------------------------------------------------------------------------
# Bernoulli 数学
# ---------------------------------------------------------------------------
class BernoulliMathTest(unittest.TestCase):
    def test_log_prob_sums_only_valid_pixels_in_float64(self) -> None:
        logits = torch.zeros(1, 1, 4, 4)
        actions = torch.zeros(1, 1, 4, 4, dtype=torch.bool)

        value = bernoulli_log_prob(logits, actions, [(2, 3)])

        self.assertEqual(value.dtype, torch.float64)
        self.assertAlmostEqual(float(value), 6.0 * math.log(0.5), places=12)

        # padding 区的 logits 即使极端，也不得影响概率。
        dirty = logits.clone()
        dirty[:, :, 2:, :] = 1e4
        dirty[:, :, :, 3:] = -1e4
        self.assertAlmostEqual(
            float(bernoulli_log_prob(dirty, actions, [(2, 3)])),
            float(value),
            places=12,
        )

        # 全动作下联合对数概率是逐像素和：2x3 区域各 6 个像素。
        ones = torch.ones_like(actions)
        self.assertAlmostEqual(
            float(bernoulli_log_prob(logits, ones, [(2, 3)])),
            6.0 * math.log(0.5),
            places=12,
        )

    def test_kl_is_masked_analytic_and_only_current_logits_get_grad(self) -> None:
        logits = torch.full((1, 1, 4, 4), 0.5, requires_grad=True)
        reference = torch.zeros(1, 1, 4, 4, requires_grad=True)

        value = bernoulli_kl_divergence(logits, reference, [(2, 2)])
        self.assertEqual(value.dtype, torch.float64)
        self.assertGreater(float(value.detach()), 0.0)

        # padding 区参考值变化不影响 KL。
        dirty = reference.detach().clone()
        dirty[:, :, 2:, :] = 1e4
        dirty[:, :, :, 2:] = -1e4
        self.assertAlmostEqual(
            float(bernoulli_kl_divergence(logits, dirty, [(2, 2)]).detach()),
            float(value.detach()),
            places=12,
        )

        value.sum().backward()
        self.assertIsNotNone(logits.grad)
        self.assertIsNone(reference.grad)

        # 相同分布 KL 为 0，且不会因舍入变成负数。
        zeros = torch.zeros(1, 1, 4, 4)
        self.assertGreaterEqual(
            float(bernoulli_kl_divergence(zeros, zeros, [(2, 2)])), 0.0
        )

    def test_terminal_sampling_is_seeded_and_handles_cpu_generator(self) -> None:
        logits = torch.zeros(3, 1, 5, 5)
        generator_a = torch.Generator(device="cpu").manual_seed(99)
        generator_b = torch.Generator(device="cpu").manual_seed(99)
        generator_c = torch.Generator(device="cpu").manual_seed(100)

        first = _sample_terminal_actions(logits, generator_a)
        second = _sample_terminal_actions(logits, generator_b)
        third = _sample_terminal_actions(logits, generator_c)

        self.assertTrue(torch.equal(first, second))
        self.assertFalse(torch.equal(first, third))
        self.assertEqual(first.dtype, torch.bool)
        self.assertEqual(tuple(first.shape), (3, 1, 5, 5))

        always = _sample_terminal_actions(
            torch.full((1, 1, 2, 2), 100.0),
            torch.Generator(device="cpu").manual_seed(1),
        )
        never = _sample_terminal_actions(
            torch.full((1, 1, 2, 2), -100.0),
            torch.Generator(device="cpu").manual_seed(1),
        )
        self.assertTrue(bool(always.all()))
        self.assertFalse(bool(never.any()))


# ---------------------------------------------------------------------------
# 联合组
# ---------------------------------------------------------------------------
class JointGroupTest(unittest.TestCase):
    def test_active_group_trains_condition_dit_and_decoder(self) -> None:
        engine = _make_engine(constant=False, kl_weight=0.0, anchor_weight=0.0)
        batch = _batch(1)
        before = {
            "condition": engine.condition_encoder.weight.detach().clone(),
            "dit": engine.dit.weight.detach().clone(),
            "decoder": engine.sam.decoder.weight.detach().clone(),
            "encoder": engine.sam.encoder_scale.detach().clone(),
            "ref_decoder": engine.ref_sam_decoder.weight.detach().clone(),
            "ref_dit": engine.ref_dit.weight.detach().clone(),
        }

        record = _run_joint(engine, batch, epoch=0)

        self.assertEqual(record["updates"], 1.0)
        self.assertEqual(record["skipped_updates"], 0.0)
        self.assertEqual(record["policy_terms"], 6.0)  # G*T + G
        self.assertTrue(math.isfinite(record["grpo"]))
        self.assertTrue(math.isnan(record["kl"]))
        self.assertTrue(math.isnan(record["anchor"]))
        self.assertEqual(record["clip_fraction"], 0.0)
        self.assertEqual(record["logratio_abs_max"], 0.0)
        self.assertEqual(record["point_count"], 0.0)
        self.assertEqual(record["positive_point_count"], 0.0)
        self.assertEqual(record["negative_point_count"], 0.0)
        self.assertEqual(record["point_penalty"], 0.0)
        self.assertEqual(record["active_group_rate"], 1.0)
        self.assertEqual(engine.grpo_groups_processed, 1)
        self.assertEqual(len(engine.scaler.losses), 6)  # G*(T+1)

        # 三个策略模块都收到非零更新；冻结编码器与参考模块保持不变。
        self.assertFalse(
            torch.equal(before["condition"], engine.condition_encoder.weight.detach())
        )
        self.assertFalse(torch.equal(before["dit"], engine.dit.weight.detach()))
        self.assertFalse(torch.equal(before["decoder"], engine.sam.decoder.weight.detach()))
        self.assertTrue(torch.equal(before["encoder"], engine.sam.encoder_scale.detach()))
        self.assertTrue(
            torch.equal(before["ref_decoder"], engine.ref_sam_decoder.weight.detach())
        )
        self.assertTrue(torch.equal(before["ref_dit"], engine.ref_dit.weight.detach()))

        # 每个扩散时间步都带梯度重算条件；rollout 的 transition 在 no_grad 中。
        replay = [row for row in engine.diffusion.transitions if row["grad_enabled"]]
        rollout = [row for row in engine.diffusion.transitions if not row["grad_enabled"]]
        self.assertEqual(len(replay), 4)  # G*T
        self.assertEqual(len(rollout), 4)  # sample 内部同样调用 transition
        self.assertTrue(all(row["conditions_grad_fn"] for row in replay))
        self.assertTrue(all(not row["conditions_requires_grad"] for row in rollout))
        self.assertEqual(engine.condition_encoder.calls, 5)  # 1 rollout + G*T

        # 终端热图不得带梯度；参考解码器只在 KL 开启时使用。
        self.assertTrue(
            all(not call["heatmaps_requires_grad"] for call in engine.sam.decode_calls)
        )
        self.assertFalse(any(call["uses_reference"] for call in engine.sam.decode_calls))
        self.assertEqual(len(engine.diffusion.sample_seeds), 2)
        self.assertEqual(len(set(engine.diffusion.sample_seeds)), 2)

        expected_keys = {
            "reward", "iou", "f1", "cldice", "path", "reward_std",
            "reward_variance", "grpo", "kl", "anchor", "clip_fraction",
            "logratio_abs_mean", "logratio_abs_max", "updates",
            "skipped_updates", "point_count", "point_penalty",
            "quality_reward", "active_group_rate", "inactive_group_rate",
        }
        self.assertTrue(expected_keys.issubset(record.keys()))

    def test_constant_group_without_kl_or_anchor_skips_everything(self) -> None:
        engine = _make_engine(
            constant=True, kl_weight=0.0, anchor_weight=0.0, update_epochs=2
        )
        batch = _batch(2)
        parameter_snapshot = [
            parameter.detach().clone() for parameter in engine._trainable_parameters()
        ]

        record = _run_joint(engine, batch)

        self.assertEqual(record["updates"], 0.0)
        self.assertEqual(record["skipped_updates"], 2.0)
        self.assertEqual(engine.scaler.losses, [])
        self.assertEqual(engine.scheduler.steps, 0)
        self.assertEqual(engine.global_step, 0)
        self.assertTrue(math.isnan(record["grpo"]))
        self.assertTrue(math.isnan(record["kl"]))
        self.assertTrue(math.isnan(record["anchor"]))
        self.assertEqual(record["policy_terms"], 0.0)
        self.assertEqual(record["reward_std"], 0.0)
        self.assertEqual(engine.anchor_calls, [])
        for parameter, snapshot in zip(engine._trainable_parameters(), parameter_snapshot):
            self.assertTrue(torch.equal(parameter.detach(), snapshot))

    def test_constant_group_still_updates_for_terminal_kl(self) -> None:
        engine = _make_engine(
            constant=True, kl_weight=0.1, anchor_weight=0.0, update_epochs=1
        )
        batch = _batch(1)

        record = _run_joint(engine, batch)

        self.assertEqual(record["updates"], 1.0)
        self.assertEqual(record["skipped_updates"], 0.0)
        self.assertEqual(record["policy_terms"], 0.0)
        self.assertTrue(math.isnan(record["grpo"]))
        self.assertTrue(math.isfinite(record["kl"]))
        self.assertGreater(record["terminal_kl"], 0.0)
        self.assertAlmostEqual(record["diffusion_kl"], 0.0, places=12)
        # KL 开启后仍要重放 G*T 个扩散步和 G 个终端项。
        self.assertEqual(len(engine.scaler.losses), 6)
        self.assertTrue(
            any(call["uses_reference"] for call in engine.sam.decode_calls)
        )
        self.assertEqual(engine.ref_condition_encoder.calls, 1)
        self.assertIsNone(engine.ref_sam_decoder.weight.grad)

    def test_anchor_only_constant_group_updates_via_abc_supervision(self) -> None:
        engine = _make_engine(
            constant=True, kl_weight=0.0, anchor_weight=1.0, update_epochs=1
        )
        batch = _batch(1)
        before_condition = engine.condition_encoder.weight.detach().clone()
        before_decoder = engine.sam.decoder.weight.detach().clone()

        record = _run_joint(engine, batch, epoch=3)

        self.assertEqual(engine.anchor_calls, [("ABC", 3)])
        self.assertEqual(record["updates"], 1.0)
        self.assertEqual(record["policy_terms"], 0.0)
        self.assertAlmostEqual(record["anchor"], 0.5, places=12)  # alpha * 0.5*1^2
        self.assertEqual(len(engine.scaler.losses), 1)  # 仅 anchor
        # anchor 只走 DiT；没有策略项时条件/解码器不应被更新。
        self.assertTrue(
            torch.equal(before_condition, engine.condition_encoder.weight.detach())
        )
        self.assertTrue(torch.equal(before_decoder, engine.sam.decoder.weight.detach()))

    def test_anchor_is_backwarded_before_policy_terms(self) -> None:
        engine = _make_engine(
            constant=False, kl_weight=0.0, anchor_weight=0.5, update_epochs=1
        )
        record = _run_joint(engine, _batch(1))

        self.assertEqual(len(engine.scaler.losses), 7)  # 1 anchor + G*(T+1)
        self.assertAlmostEqual(engine.scaler.losses[0], 0.25, places=12)
        self.assertAlmostEqual(record["anchor"], 0.25, places=12)
        self.assertTrue(math.isfinite(record["grpo"]))
        self.assertEqual(record["updates"], 1.0)

    def test_update_epochs_replays_condition_each_step_and_steps_once_per_epoch(self) -> None:
        engine = _make_engine(
            constant=False, kl_weight=0.0, anchor_weight=0.0, update_epochs=2
        )
        record = _run_joint(engine, _batch(1))

        self.assertEqual(record["updates"], 2.0)
        self.assertEqual(record["skipped_updates"], 0.0)
        self.assertEqual(record["policy_terms"], 12.0)  # 2 轮 * (G*T + G)
        self.assertEqual(len(engine.scaler.losses), 12)
        self.assertEqual(engine.scheduler.steps, 2)
        self.assertEqual(engine.global_step, 2)
        # 1 次 rollout 条件 + 每轮每步重算 G*T 次。
        self.assertEqual(engine.condition_encoder.calls, 1 + 2 * 2 * 2)
        # 终端热图只在 rollout 采样一次，第二轮重放复用已保存的热图。
        self.assertEqual(len(engine.diffusion.sample_seeds), 2)
        # 第一轮更新前 ratio 必须恰为 1；step 之后第二轮才出现策略漂移。
        self.assertGreater(record["logratio_abs_max"], 0.0)

    def test_preupdate_ratio_is_exactly_one_and_post_step_ratio_drifts(self) -> None:
        engine = _make_engine(
            constant=False, kl_weight=0.0, anchor_weight=0.0, update_epochs=2
        )
        ratio_errors: list[float] = []
        from diffpeak.joint_rl import grpo_policy_loss as real_policy_loss

        def spy(ratio, advantage, epsilon):
            ratio_errors.append(float((ratio.detach() - 1.0).abs().max().cpu()))
            return real_policy_loss(ratio, advantage, epsilon)

        with mock.patch(
            "diffpeak.joint_rl._sample_terminal_actions",
            side_effect=lambda logits, generator: logits > 0,
        ), mock.patch(
            "diffpeak.joint_rl.grpo_policy_loss", side_effect=spy
        ):
            record = joint_grpo_group(engine, _batch(1), 0)

        # 每轮 G*(T+1) 个策略项；第一轮都发生在 optimizer.step() 之前。
        per_epoch = 2 * (2 + 1)
        self.assertEqual(len(ratio_errors), per_epoch * 2)
        self.assertTrue(all(error < 1e-12 for error in ratio_errors[:per_epoch]))
        self.assertTrue(any(error > 0.0 for error in ratio_errors[per_epoch:]))
        self.assertEqual(record["updates"], 2.0)

    def test_terminal_heatmaps_are_stored_and_reward_uses_sampled_mask(self) -> None:
        engine = _make_engine(
            constant=False, kl_weight=0.0, anchor_weight=0.0, update_epochs=2
        )
        batch = _batch(1, valid_sizes=[(3, 4)])
        captured: list[tuple[torch.Tensor, torch.Tensor, int]] = []
        from diffpeak.inference import compute_image_metrics as real_metrics

        def spy_metrics(prediction, target, point_count, reward_config):
            captured.append(
                (
                    torch.as_tensor(prediction).clone(),
                    torch.as_tensor(target).clone(),
                    int(point_count),
                )
            )
            return real_metrics(prediction, target, point_count, reward_config)

        with mock.patch(
            "diffpeak.joint_rl._sample_terminal_actions",
            side_effect=lambda logits, generator: logits > 0,
        ), mock.patch(
            "diffpeak.joint_rl.compute_image_metrics", side_effect=spy_metrics
        ):
            joint_grpo_group(engine, batch, 0)

        # 每张图每条轨迹一次，奖励只来自采样得到的二值掩码与 GT 比较。
        self.assertEqual(len(captured), 2)
        trajectory0_prediction, target, point_count = captured[0]
        trajectory1_prediction, same_target, second_count = captured[1]
        self.assertTrue(bool(trajectory0_prediction.all()))
        self.assertFalse(bool(trajectory1_prediction.any()))
        self.assertEqual(tuple(trajectory0_prediction.shape), (3, 4))
        self.assertEqual(point_count, 0)
        self.assertEqual(second_count, 0)
        # GT 只作为奖励比较对象；有效区域裁剪正确，未把 padding 计入。
        self.assertTrue(torch.equal(target, same_target))
        self.assertEqual(tuple(target.shape), (3, 4))
        self.assertTrue(bool(target[:, :2].all()))
        self.assertFalse(bool(target[:, 2:].any()))

        # 所有解码调用都收到 valid mask（终端重放也使用保存的热图）。
        self.assertTrue(
            all(call["valid_sizes"] == [(3, 4)] for call in engine.sam.decode_calls)
        )
        live_calls = [call for call in engine.sam.decode_calls if not call["uses_reference"]]
        self.assertEqual(len(live_calls), 6)  # 2 条 rollout + 2 轮 * 2 条 replay
        # update_epochs=2 但 sample 只发生 G 次；重放直接复用已存热图。
        self.assertEqual(len(engine.diffusion.sample_seeds), 2)
        for replay_index in (2, 4):
            self.assertAlmostEqual(
                live_calls[0]["heatmap_sum"],
                live_calls[replay_index]["heatmap_sum"],
            )
            self.assertAlmostEqual(
                live_calls[1]["heatmap_sum"],
                live_calls[replay_index + 1]["heatmap_sum"],
            )

    def test_shared_advantages_and_normalization_denominators(self) -> None:
        engine = _make_engine(
            constant=False, kl_weight=0.0, anchor_weight=0.0, steps=2
        )
        batch = _batch(2)
        # 轨迹优先排列：g0=(b0,b1), g1=(b0,b1)。
        # b0 两轨迹奖励相同 -> 常量组；b1 奖励 [1,3] -> 优势 [-1,1]。
        known_rewards = iter([0.0, 1.0, 0.0, 3.0])

        def fake_metrics(prediction, target, point_count, reward_config):
            reward = next(known_rewards)
            return {
                "reward": reward,
                "iou": reward,
                "f1": reward,
                "cldice": reward,
                "path": 0.0,
                "tp": 0.0,
                "fp": 0.0,
                "fn": 0.0,
                "tn": 0.0,
            }

        with mock.patch(
            "diffpeak.joint_rl.compute_image_metrics", side_effect=fake_metrics
        ):
            record = _run_joint(engine, batch)

        # G=2, T=2, B=2；扩散分母 G*T*B=8，终端分母 G*B=4。
        # g0 优势 [0,-1] -> grpo_loss=0.5；g1 优势 [0,1] -> -0.5。
        expected = [
            0.5 * 2 / 8,   # g0 扩散步 1
            0.5 * 2 / 8,   # g0 扩散步 2
            0.5 * 2 / 4,   # g0 终端
            -0.5 * 2 / 8,  # g1 扩散步 1
            -0.5 * 2 / 8,  # g1 扩散步 2
            -0.5 * 2 / 4,  # g1 终端
        ]
        self.assertEqual(len(engine.scaler.losses), len(expected))
        for actual, wanted in zip(engine.scaler.losses, expected):
            self.assertAlmostEqual(actual, wanted, places=10)

        self.assertEqual(record["policy_terms"], 6.0)  # 只有 b1 组有效
        self.assertEqual(record["active_group_rate"], 0.5)
        self.assertEqual(record["inactive_group_rate"], 0.5)
        self.assertEqual(record["invalid_groups"], 1.0)
        self.assertAlmostEqual(record["diffusion_grpo"], 0.0, places=12)
        self.assertAlmostEqual(record["terminal_grpo"], 0.0, places=12)
        self.assertEqual(record["updates"], 1.0)


if __name__ == "__main__":
    unittest.main()
