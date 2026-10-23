"""扩散轨迹 GRPO（冻结 SAM、确定性评分）的单元/集成测试。

测试不加载 SAM2 或 DeepGlobe 数据，只用微型条件编码器、DiT、扩散替身和
可微密集热图解码器验证新协议：

* 公共 API 只保留 ``joint_grpo_group``，不再导出 Bernoulli/终端采样；
* 终端评分是确定性阈值 ``sigmoid(logits) >= threshold``，无 Bernoulli 动作；
* 同一图像的 G 条扩散轨迹构成一组，优势逐图计算；
* 只有 condition_encoder 与 DiT 收到梯度；SAM 被防御性冻结并保持 eval；
* 参考 KL 只使用参考条件编码器与参考 DiT，不需要参考 SAM 解码器；
* 固定路径奖励采样：每张图/每个候选使用稳定的 ``RewardConfig.seed``；
* 扩散项按 ``G*T*B`` 归一，每轮只 step 一次，重放前 ratio 恰为 1；
* 常量组在无 KL/anchor 时完全跳过 backward/optimizer；
* 记录中不再包含 ``terminal_grpo``/``terminal_kl``。
"""

from __future__ import annotations

import copy
import math
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import diffpeak.joint_rl as joint_rl  # noqa: E402
from diffpeak.inference import compute_image_metrics, stable_seed  # noqa: E402
from diffpeak.joint_rl import joint_grpo_group  # noqa: E402
from diffpeak.rewards import RewardConfig  # noqa: E402


# ---------------------------------------------------------------------------
# 引擎替身
# ---------------------------------------------------------------------------
class _EngineHarness:
    """只实现 joint_rl 依赖的引擎方法，避免耦合正在重构的 engine 模块。"""

    def _trainable_parameters(self):
        parameters = []
        seen = set()
        for module in (self.sam, self.condition_encoder, self.dit):
            for parameter in module.parameters():
                if id(parameter) in seen:
                    continue
                seen.add(id(parameter))
                if parameter.requires_grad:
                    parameters.append(parameter)
        return parameters

    @staticmethod
    def _active_group_mask(reward_tensor):
        if reward_tensor.ndim != 2:
            raise ValueError("reward_tensor 必须是 [G, B]")
        return reward_tensor.amax(dim=0) > reward_tensor.amin(dim=0)

    def _optimizer_step(self) -> bool:
        parameters = [
            p for p in self._trainable_parameters() if p.grad is not None
        ]
        if parameters and self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(parameters, self.grad_clip)
        self.optimizer.step()
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        self.scheduler.step()
        self.global_step += 1
        return True


# ---------------------------------------------------------------------------
# 微型模型（单通道有符号场）
# ---------------------------------------------------------------------------
class _TinyDecoder(nn.Module):
    def __init__(self, weight: float = 4.0, bias: float = 0.0) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(float(weight)))
        self.bias = nn.Parameter(torch.tensor(float(bias)))


class _TinySam(nn.Module):
    """提供 encode 与可微 decode_heatmaps 的最小 SAM 替身。"""

    def __init__(self, decoder_weight: float = 4.0, decoder_bias: float = 0.0) -> None:
        super().__init__()
        # 冻结的图像编码器参数：任何阶段都不应被更新。
        self.encoder_scale = nn.Parameter(torch.tensor(1.0))
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
        decoder: nn.Module | None = None,
    ) -> torch.Tensor:
        module = self.decoder if decoder is None else decoder
        self.decode_calls.append(
            {
                "decoder": module,
                "uses_reference": decoder is not None,
                "heatmaps_requires_grad": bool(heatmaps.requires_grad),
                "heatmap_channels": int(heatmaps.shape[1]),
                "heatmap_sum": float(heatmaps.detach().double().sum().cpu()),
                "grad_enabled": torch.is_grad_enabled(),
                "valid_sizes": valid_sizes,
            }
        )
        # 新协议：单通道有符号山脊场直接作为密集提示。
        dense_mask = heatmaps[:, 0:1]
        logits = dense_mask * module.weight + module.bias
        if tuple(logits.shape[-2:]) != tuple(output_size):
            logits = F.interpolate(logits, size=tuple(output_size), mode="nearest")
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
            1,
            self.height,
            self.width,
            device=images.device,
            dtype=images.dtype,
        )
        condition = base * self.weight
        auxiliary = torch.zeros_like(condition)
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

    def _field(self, batch_size: int, height: int, width: int, device) -> torch.Tensor:
        # 非恒定模式交替 +1/-1 的有符号场，保证两条候选奖励不同。
        if self.constant:
            on = True
        else:
            on = self.sample_calls % 2 == 0
        value = 1.0 if on else -1.0
        return torch.full(
            (batch_size, 1, height, width),
            value,
            device=device,
            dtype=torch.float32,
        )

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
        heatmaps = self._field(batch_size, height, width, conditions[0].device)
        # 新协议直接在有符号场域扩散，不再做 [0,1] <-> [-1,1] 变换。
        state = heatmaps
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
    threshold: float = 0.5,
    decoder_weight: float = 4.0,
    decoder_bias: float = 0.0,
    freeze_sam: bool = True,
    ref_sam_decoder: str = "absent",
    reward_config: RewardConfig | None = None,
) -> _EngineHarness:
    engine = _EngineHarness()
    engine.device = torch.device("cpu")
    engine.amp = False
    engine.image_size = 4
    engine.heatmap_size = 4
    engine.seed = 1234
    engine.threshold = float(threshold)
    engine.group_size = int(group_size)
    engine.ppo_clip = 0.2
    engine.kl_weight = float(kl_weight)
    engine.anchor_weight = float(anchor_weight)
    engine.update_epochs = int(update_epochs)
    engine.advantage_std_delta = 1e-4
    engine.grad_clip = 0.0
    engine.global_step = 0
    engine.grpo_groups_processed = 0
    engine._stage = "D"
    engine.reward_config = reward_config or RewardConfig(
        iou_weight=0.5, cldice_weight=0.5, path_weight=0.0
    )

    engine.sam = _TinySam(decoder_weight=decoder_weight, decoder_bias=decoder_bias)
    if freeze_sam:
        _freeze(engine.sam)
    engine.condition_encoder = _TinyConditionEncoder(0.5)
    engine.dit = _TinyDit(1.0)
    # 参考 DiT 与当前策略不同，KL 才有非零信号。
    engine.ref_dit = copy.deepcopy(engine.dit)
    with torch.no_grad():
        engine.ref_dit.weight.fill_(2.0)
    _freeze(engine.ref_dit)
    engine.ref_condition_encoder = _freeze(copy.deepcopy(engine.condition_encoder))
    engine.ref_sam_decoder = (
        _freeze(copy.deepcopy(engine.sam.decoder))
        if ref_sam_decoder == "present"
        else None
    )

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
        "heatmaps": torch.full((batch_size, 1, 4, 4), -1.0),
        "valid_sizes": list(valid_sizes),
        "ids": [f"image-{index}" for index in range(batch_size)],
    }


def _fake_metrics(reward: float) -> dict[str, float]:
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


# ---------------------------------------------------------------------------
# 公共 API 与确定性阈值
# ---------------------------------------------------------------------------
class DeterministicScoringTest(unittest.TestCase):
    def test_public_api_only_exposes_joint_grpo_group(self) -> None:
        self.assertEqual(joint_rl.__all__, ["joint_grpo_group"])
        for removed in (
            "bernoulli_log_prob",
            "bernoulli_kl_divergence",
            "_sample_terminal_actions",
        ):
            self.assertFalse(hasattr(joint_rl, removed), removed)

    def test_threshold_predictions_are_inclusive_and_mask_padding(self) -> None:
        logits = torch.tensor([[[[0.0, 1.0, -1.0]]]])
        prediction = joint_rl._threshold_predictions(logits, 0.5, [(1, 3)])[0]
        self.assertEqual(prediction.tolist(), [[True, True, False]])
        self.assertEqual(prediction.dtype, np.bool_)

        # padding 区的极端 logits 不进入有效区域。
        dirty = torch.tensor([[[[0.0, 1.0, -1.0], [1e4, 1e4, 1e4]]]])
        self.assertEqual(
            joint_rl._threshold_predictions(dirty, 0.5, [(1, 3)])[0].tolist(),
            [[True, True, False]],
        )

    def test_segmentation_threshold_default_and_validation(self) -> None:
        engine = _EngineHarness()
        self.assertEqual(joint_rl._segmentation_threshold(engine), 0.5)
        engine.threshold = 0.3
        self.assertEqual(joint_rl._segmentation_threshold(engine), 0.3)
        engine.threshold = 1.5
        with self.assertRaises(ValueError):
            joint_rl._segmentation_threshold(engine)

    def test_decode_heatmaps_rejects_legacy_two_channel_prompts(self) -> None:
        engine = _make_engine()
        with self.assertRaises(ValueError):
            joint_rl._decode_heatmaps(
                engine,
                torch.zeros(1, 1),
                torch.zeros(1, 2, 4, 4),
                (4, 4),
                None,
            )

    def test_reward_uses_deterministic_threshold_not_bernoulli_sampling(self) -> None:
        # logits 恰为 0 -> sigmoid = 0.5，>= 阈值 0.5 必须判为道路。
        engine = _make_engine(
            constant=True, decoder_weight=0.0, decoder_bias=0.0, threshold=0.5
        )
        captured: list[np.ndarray] = []

        def spy(prediction, target, point_count, config):
            captured.append(np.asarray(prediction).copy())
            return compute_image_metrics(prediction, target, point_count, config)

        with mock.patch(
            "diffpeak.joint_rl.compute_image_metrics", side_effect=spy
        ):
            joint_grpo_group(engine, _batch(1), 0)

        self.assertEqual(len(captured), 2)  # G=2 条轨迹
        for prediction in captured:
            self.assertEqual(prediction.shape, (3, 4))
            self.assertTrue(bool(prediction.all()))

        # 略低于阈值时必须全为背景。
        engine = _make_engine(
            constant=True, decoder_weight=0.0, decoder_bias=-0.5, threshold=0.5
        )
        captured.clear()
        with mock.patch(
            "diffpeak.joint_rl.compute_image_metrics", side_effect=spy
        ):
            joint_grpo_group(engine, _batch(1), 0)
        for prediction in captured:
            self.assertFalse(bool(prediction.any()))


# ---------------------------------------------------------------------------
# 扩散组更新
# ---------------------------------------------------------------------------
class DiffusionGroupTest(unittest.TestCase):
    def test_active_group_trains_condition_and_dit_only(self) -> None:
        engine = _make_engine(constant=False, kl_weight=0.0, anchor_weight=0.0)
        batch = _batch(1)
        before = {
            "condition": engine.condition_encoder.weight.detach().clone(),
            "dit": engine.dit.weight.detach().clone(),
            "decoder": engine.sam.decoder.weight.detach().clone(),
            "encoder": engine.sam.encoder_scale.detach().clone(),
            "ref_dit": engine.ref_dit.weight.detach().clone(),
            "ref_condition": engine.ref_condition_encoder.weight.detach().clone(),
        }

        record = joint_grpo_group(engine, batch, epoch=0)

        self.assertEqual(record["updates"], 1.0)
        self.assertEqual(record["skipped_updates"], 0.0)
        # G=2, T=2 -> 仅扩散项，共 4 项。
        self.assertEqual(record["policy_terms"], 4.0)
        self.assertTrue(math.isfinite(record["grpo"]))
        self.assertTrue(math.isnan(record["kl"]))
        self.assertTrue(math.isnan(record["anchor"]))
        self.assertEqual(record["clip_fraction"], 0.0)
        self.assertEqual(record["logratio_abs_max"], 0.0)
        self.assertEqual(record["active_group_rate"], 1.0)
        self.assertEqual(engine.grpo_groups_processed, 1)
        self.assertEqual(len(engine.scaler.losses), 4)  # G*T

        # 终端 Bernoulli 策略日志已移除。
        self.assertNotIn("terminal_grpo", record)
        self.assertNotIn("terminal_kl", record)
        self.assertNotIn("terminal_foreground_rate", record)
        self.assertIn("prediction_foreground_rate", record)
        self.assertIn("diffusion_grpo", record)
        self.assertIn("diffusion_kl", record)

        # 只有条件编码器与 DiT 被更新。
        self.assertFalse(
            torch.equal(before["condition"], engine.condition_encoder.weight.detach())
        )
        self.assertFalse(torch.equal(before["dit"], engine.dit.weight.detach()))
        self.assertTrue(
            torch.equal(before["decoder"], engine.sam.decoder.weight.detach())
        )
        self.assertTrue(
            torch.equal(before["encoder"], engine.sam.encoder_scale.detach())
        )
        self.assertTrue(
            torch.equal(before["ref_dit"], engine.ref_dit.weight.detach())
        )
        self.assertTrue(
            torch.equal(
                before["ref_condition"],
                engine.ref_condition_encoder.weight.detach(),
            )
        )

        # SAM 防御性冻结：eval 且全部 requires_grad=False。
        self.assertFalse(engine.sam.training)
        self.assertTrue(
            all(not p.requires_grad for p in engine.sam.parameters())
        )

        # rollout/no_grad 与 replay/带梯度条件各 G*T 次。
        replay = [row for row in engine.diffusion.transitions if row["grad_enabled"]]
        rollout = [
            row for row in engine.diffusion.transitions if not row["grad_enabled"]
        ]
        self.assertEqual(len(replay), 4)
        self.assertEqual(len(rollout), 4)
        self.assertTrue(all(row["conditions_grad_fn"] for row in replay))
        self.assertTrue(all(not row["conditions_requires_grad"] for row in rollout))
        self.assertEqual(engine.condition_encoder.calls, 5)  # 1 rollout + G*T

        # 只在 rollout 阶段用冻结 SAM 做确定性评分；重放不再解码。
        self.assertEqual(len(engine.sam.decode_calls), 2)  # G
        for call in engine.sam.decode_calls:
            self.assertFalse(call["uses_reference"])
            self.assertFalse(call["heatmaps_requires_grad"])
            self.assertFalse(call["grad_enabled"])
            self.assertEqual(call["heatmap_channels"], 1)
            self.assertEqual(call["valid_sizes"], [(3, 4)])
        # 无 KL 时不编码参考条件。
        self.assertEqual(engine.ref_condition_encoder.calls, 0)

        self.assertEqual(len(engine.diffusion.sample_seeds), 2)
        self.assertEqual(len(set(engine.diffusion.sample_seeds)), 2)

        expected_keys = {
            "reward", "iou", "f1", "cldice", "path", "reward_std",
            "reward_variance", "grpo", "kl", "anchor", "clip_fraction",
            "logratio_abs_mean", "logratio_abs_max", "updates",
            "skipped_updates", "quality_reward",
            "active_group_rate", "inactive_group_rate",
        }
        self.assertTrue(expected_keys.issubset(record.keys()))
        for removed in (
            "point_count",
            "positive_point_count",
            "negative_point_count",
            "point_penalty",
            "point_cost",
            "point_cost_scale",
            "quality_reward_mean",
        ):
            self.assertNotIn(removed, record)

    def test_only_condition_and_dit_receive_gradients(self) -> None:
        engine = _make_engine(constant=False, freeze_sam=False)
        engine.sam.train()  # 模拟残留的可训练状态
        with torch.no_grad():
            for parameter in engine.sam.parameters():
                parameter.requires_grad_(True)

        observed: dict[str, list[bool]] = {}
        original_step = engine._optimizer_step

        def step():
            observed["condition"] = [
                p.grad is not None for p in engine.condition_encoder.parameters()
            ]
            observed["dit"] = [p.grad is not None for p in engine.dit.parameters()]
            observed["sam"] = [p.grad is not None for p in engine.sam.parameters()]
            return original_step()

        engine._optimizer_step = step  # type: ignore[method-assign]
        joint_grpo_group(engine, _batch(1), 0)

        self.assertTrue(any(observed["condition"]))
        self.assertTrue(any(observed["dit"]))
        self.assertFalse(any(observed["sam"]))
        self.assertFalse(engine.sam.training)
        self.assertTrue(all(not p.requires_grad for p in engine.sam.parameters()))

    def test_constant_group_without_kl_or_anchor_skips_everything(self) -> None:
        engine = _make_engine(
            constant=True, kl_weight=0.0, anchor_weight=0.0, update_epochs=2
        )
        batch = _batch(2)
        parameter_snapshot = [
            parameter.detach().clone() for parameter in engine._trainable_parameters()
        ]

        record = joint_grpo_group(engine, batch, 0)

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
        # 评分仍然需要 G 次确定性 SAM 解码，只是不 backward/step。
        self.assertEqual(len(engine.sam.decode_calls), 2)
        for parameter, snapshot in zip(
            engine._trainable_parameters(), parameter_snapshot
        ):
            self.assertTrue(torch.equal(parameter.detach(), snapshot))

    def test_constant_group_still_updates_for_diffusion_kl(self) -> None:
        engine = _make_engine(
            constant=True, kl_weight=0.1, anchor_weight=0.0, update_epochs=1
        )
        batch = _batch(1)
        before_decoder = engine.sam.decoder.weight.detach().clone()

        record = joint_grpo_group(engine, batch, 0)

        self.assertEqual(record["updates"], 1.0)
        self.assertEqual(record["skipped_updates"], 0.0)
        self.assertEqual(record["policy_terms"], 0.0)
        self.assertTrue(math.isnan(record["grpo"]))
        self.assertNotIn("terminal_kl", record)
        self.assertTrue(math.isfinite(record["kl"]))
        self.assertGreater(record["diffusion_kl"], 0.0)
        self.assertEqual(len(engine.scaler.losses), 4)  # 只有 G*T 扩散项
        # 参考 SAM 从不参与：即使没设置 ref_sam_decoder 也能做 KL。
        self.assertIsNone(engine.ref_sam_decoder)
        self.assertFalse(
            any(call["uses_reference"] for call in engine.sam.decode_calls)
        )
        self.assertEqual(engine.ref_condition_encoder.calls, 1)
        self.assertTrue(
            torch.equal(before_decoder, engine.sam.decoder.weight.detach())
        )

    def test_anchor_only_constant_group_updates_dit_not_sam_or_condition(self) -> None:
        engine = _make_engine(
            constant=True, kl_weight=0.0, anchor_weight=1.0, update_epochs=1
        )
        batch = _batch(1)
        before_condition = engine.condition_encoder.weight.detach().clone()
        before_decoder = engine.sam.decoder.weight.detach().clone()
        before_dit = engine.dit.weight.detach().clone()

        record = joint_grpo_group(engine, batch, epoch=3)

        self.assertEqual(engine.anchor_calls, [("ABC", 3)])
        self.assertEqual(record["updates"], 1.0)
        self.assertEqual(record["policy_terms"], 0.0)
        self.assertAlmostEqual(record["anchor"], 0.5, places=12)  # alpha * 0.5*1^2
        self.assertEqual(len(engine.scaler.losses), 1)  # 仅 anchor
        self.assertTrue(
            torch.equal(before_condition, engine.condition_encoder.weight.detach())
        )
        self.assertTrue(
            torch.equal(before_decoder, engine.sam.decoder.weight.detach())
        )
        self.assertFalse(torch.equal(before_dit, engine.dit.weight.detach()))

    def test_anchor_is_backwarded_before_policy_terms(self) -> None:
        engine = _make_engine(
            constant=False, kl_weight=0.0, anchor_weight=0.5, update_epochs=1
        )
        record = joint_grpo_group(engine, _batch(1), 0)

        self.assertEqual(len(engine.scaler.losses), 5)  # 1 anchor + G*T
        self.assertAlmostEqual(engine.scaler.losses[0], 0.25, places=12)
        self.assertAlmostEqual(record["anchor"], 0.25, places=12)
        self.assertTrue(math.isfinite(record["grpo"]))
        self.assertEqual(record["updates"], 1.0)

    def test_update_epochs_replays_condition_and_steps_once_per_epoch(self) -> None:
        engine = _make_engine(
            constant=False, kl_weight=0.0, anchor_weight=0.0, update_epochs=2
        )
        record = joint_grpo_group(engine, _batch(1), 0)

        self.assertEqual(record["updates"], 2.0)
        self.assertEqual(record["skipped_updates"], 0.0)
        self.assertEqual(record["policy_terms"], 8.0)  # 2 轮 * (G*T)
        self.assertEqual(len(engine.scaler.losses), 8)
        self.assertEqual(engine.scheduler.steps, 2)
        self.assertEqual(engine.global_step, 2)
        # 1 次 rollout 条件 + 每轮重算 G*T 次。
        self.assertEqual(engine.condition_encoder.calls, 1 + 2 * 2 * 2)
        # 采样只发生 G 次；重放不再调用 SAM 解码。
        self.assertEqual(len(engine.diffusion.sample_seeds), 2)
        self.assertEqual(len(engine.sam.decode_calls), 2)
        # 第一轮更新前 ratio 必须恰为 1；step 之后第二轮才出现策略漂移。
        self.assertGreater(record["logratio_abs_max"], 0.0)

    def test_preupdate_ratio_is_exactly_one_and_post_step_ratio_drifts(self) -> None:
        engine = _make_engine(
            constant=False, kl_weight=0.0, anchor_weight=0.0, update_epochs=2
        )
        ratio_errors: list[float] = []
        real_policy_loss = joint_rl.grpo_policy_loss

        def spy(ratio, advantage, epsilon):
            ratio_errors.append(float((ratio.detach() - 1.0).abs().max().cpu()))
            return real_policy_loss(ratio, advantage, epsilon)

        with mock.patch(
            "diffpeak.joint_rl.grpo_policy_loss", side_effect=spy
        ):
            record = joint_grpo_group(engine, _batch(1), 0)

        per_epoch = 2 * 2  # G * T
        self.assertEqual(len(ratio_errors), per_epoch * 2)
        self.assertTrue(all(error < 1e-12 for error in ratio_errors[:per_epoch]))
        self.assertTrue(any(error > 0.0 for error in ratio_errors[per_epoch:]))
        self.assertEqual(record["updates"], 2.0)

    def test_shared_advantages_and_diffusion_normalization_denominators(self) -> None:
        engine = _make_engine(
            constant=False, kl_weight=0.0, anchor_weight=0.0, steps=2
        )
        batch = _batch(2)
        # 轨迹优先排列：g0=(b0,b1), g1=(b0,b1)。
        # b0 两候选奖励相同 -> 常量组；b1 奖励 [1,3] -> 优势 [-1,1]。
        known_rewards = iter([0.0, 1.0, 0.0, 3.0])

        def fake_metrics(prediction, target, point_count, reward_config):
            return _fake_metrics(next(known_rewards))

        with mock.patch(
            "diffpeak.joint_rl.compute_image_metrics", side_effect=fake_metrics
        ):
            record = joint_grpo_group(engine, batch, 0)

        # G=2, T=2, B=2；扩散分母 G*T*B=8。
        # g0 优势 [0,-1] -> grpo_loss=0.5；g1 优势 [0,1] -> -0.5。
        expected = [
            0.5 * 2 / 8,   # g0 step 1
            0.5 * 2 / 8,   # g0 step 2
            -0.5 * 2 / 8,  # g1 step 1
            -0.5 * 2 / 8,  # g1 step 2
        ]
        self.assertEqual(len(engine.scaler.losses), len(expected))
        for actual, wanted in zip(engine.scaler.losses, expected):
            self.assertAlmostEqual(actual, wanted, places=10)

        self.assertEqual(record["policy_terms"], 4.0)  # 只有 b1 组有效
        self.assertEqual(record["active_group_rate"], 0.5)
        self.assertEqual(record["inactive_group_rate"], 0.5)
        self.assertEqual(record["invalid_groups"], 1.0)
        self.assertAlmostEqual(record["diffusion_grpo"], 0.0, places=12)
        self.assertAlmostEqual(record["grpo"], 0.0, places=12)
        self.assertEqual(record["updates"], 1.0)


# ---------------------------------------------------------------------------
# 固定奖励采样与参考模块
# ---------------------------------------------------------------------------
class RewardSamplingTest(unittest.TestCase):
    def test_path_reward_seed_is_fixed_per_image(self) -> None:
        base = RewardConfig(
            iou_weight=0.5,
            cldice_weight=0.5,
            path_weight=0.25,
            path_samples=4,
            seed=7,
        )
        engine = _make_engine(constant=True, reward_config=base)
        captured: list[RewardConfig] = []

        def spy(prediction, target, point_count, reward_config):
            captured.append(reward_config)
            return _fake_metrics(0.5)

        with mock.patch(
            "diffpeak.joint_rl.compute_image_metrics", side_effect=spy
        ):
            joint_grpo_group(engine, _batch(2), 0)
            first_seeds = [config.seed for config in captured]
            captured.clear()
            # 常量组不更新参数；换 epoch 后奖励采样种子必须仍然相同。
            joint_grpo_group(engine, _batch(2), 1)
            second_seeds = [config.seed for config in captured]

        per_image = [
            stable_seed(engine.seed, "D-reward", f"image-{b}") % (1 << 32)
            for b in range(2)
        ]
        # 轨迹优先排列：每条轨迹的 B 张图依次使用各自的固定种子。
        expected = per_image * engine.group_size
        self.assertEqual(first_seeds, expected)
        self.assertEqual(second_seeds, expected)
        self.assertEqual(len(set(expected)), 2)
        # 同一图像的所有候选共享同一条路径采样规则。
        self.assertEqual(first_seeds[0], first_seeds[2])
        self.assertEqual(first_seeds[1], first_seeds[3])
        for reward_config in captured:
            self.assertIsInstance(reward_config, RewardConfig)
            self.assertEqual(reward_config.path_weight, 0.25)
            self.assertEqual(reward_config.path_samples, 4)
            self.assertEqual(reward_config.iou_weight, 0.5)
        # 原始配置不被就地修改。
        self.assertEqual(base.seed, 7)

    def test_reference_kl_never_uses_or_requires_reference_sam(self) -> None:
        engine = _make_engine(
            constant=False,
            kl_weight=0.1,
            ref_sam_decoder="present",
        )
        # 参考解码器与当前解码器不同，若被误用会改变 logits。
        with torch.no_grad():
            engine.ref_sam_decoder.weight.fill_(0.5)
        before_reference = engine.ref_sam_decoder.weight.detach().clone()

        record = joint_grpo_group(engine, _batch(1), 0)

        self.assertTrue(math.isfinite(record["kl"]))
        self.assertFalse(
            any(call["uses_reference"] for call in engine.sam.decode_calls)
        )
        self.assertEqual(len(engine.sam.decode_calls), 2)  # 只有 G 次 rollout 评分
        self.assertTrue(
            torch.equal(before_reference, engine.ref_sam_decoder.weight.detach())
        )
        self.assertIsNone(engine.ref_sam_decoder.weight.grad)
        self.assertNotIn("terminal_kl", record)


if __name__ == "__main__":
    unittest.main()
