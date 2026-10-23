"""Signed-ridge engine invariants: frozen SAM, diffusion-only GRPO wiring.

These are lightweight engine-level checks. Detailed reward/KL/ratio behaviour
for the diffusion-only group lives in ``tests/test_joint_rl.py`` (owned by the
joint-RL test suite); here we verify the engine boundary:

* ABC and D never train or switch SAM out of eval;
* frozen SAM parameters still pass input gradients from the signed field;
* ``_grpo_group`` delegates to the diffusion-only ``joint_grpo_group``;
* the D group/epoch aggregation and group-relative activity mask stay correct;
* an integrated D run updates only the condition encoder and DiT, records no
  terminal-policy fields and creates only ``ref_dit``/``ref_condition_encoder``.
"""
from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from unittest.mock import patch

import torch
import torch.nn.functional as F
from torch import nn

import diffpeak.engine as engine_module
from diffpeak.engine import DiffPeakEngine
from diffpeak.models.sam import SAM2ImageFeatures
from tests import test_engine_pipeline as pipeline
from tests.test_engine_pipeline import _build_tiny_models


class _TinyModule(nn.Module):
    def __init__(self, value: float = 1.0) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(float(value)))


class _TinySam(nn.Module):
    prompt_scale = 1.0

    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Parameter(torch.tensor(1.0))
        self.sam = nn.Module()
        self.sam.mask_decoder = _TinyModule(2.0)

    def encode(self, images, valid_sizes=None):
        features = F.avg_pool2d(images, 4)
        features = features[:, :1].repeat(1, 4, 1, 1)
        return SAM2ImageFeatures(features, (features, features))

    def decode_heatmaps(
        self, embeddings, heatmaps, output_size=(16, 16), valid_sizes=None
    ):
        dense = heatmaps[:, 0:1] * float(self.prompt_scale)
        return F.interpolate(
            dense, size=tuple(output_size), mode="bilinear", align_corners=False
        )


def _bare_engine() -> DiffPeakEngine:
    engine = object.__new__(DiffPeakEngine)
    engine.device = torch.device("cpu")
    engine.amp = False
    engine.sam = _TinySam()
    engine.condition_encoder = _TinyModule(3.0)
    engine.dit = _TinyModule(4.0)
    engine.ref_dit = None
    engine.ref_condition_encoder = None
    return engine


class FreezingTest(unittest.TestCase):
    def test_abc_and_d_freeze_all_sam_parameters(self) -> None:
        # 新协议全阶段冻结 SAM：ABC 与 D 都没有可训练 SAM 参数。
        for stage in ("ABC", "D"):
            with self.subTest(stage=stage):
                engine = _bare_engine()
                engine._set_requires_grad(stage)

                self.assertTrue(
                    all(
                        not parameter.requires_grad
                        for parameter in engine.sam.parameters()
                    )
                )
                self.assertTrue(
                    all(
                        parameter.requires_grad
                        for parameter in engine.condition_encoder.parameters()
                    )
                )
                self.assertTrue(
                    all(
                        parameter.requires_grad
                        for parameter in engine.dit.parameters()
                    )
                )

    def test_train_modes_keep_sam_eval_and_use_eval_replay_in_d(self) -> None:
        expected = {
            # ABC: SAM 始终 eval；条件编码器与 DiT 训练。
            "ABC": (True, True),
            # D: rollout/replay 对策略使用 eval 保证两次前向一致；梯度仍开启。
            "D": (False, False),
        }
        for stage, (condition, dit) in expected.items():
            with self.subTest(stage=stage):
                engine = _bare_engine()
                engine.ref_dit = nn.Module()
                engine.ref_dit.train()
                engine.ref_condition_encoder = nn.Module()
                engine.ref_condition_encoder.train()
                engine._set_train_modes(stage)

                self.assertFalse(engine.sam.training)
                self.assertEqual(engine.condition_encoder.training, condition)
                self.assertEqual(engine.dit.training, dit)
                self.assertFalse(engine.ref_dit.training)
                self.assertFalse(engine.ref_condition_encoder.training)

    def test_frozen_sam_passes_field_gradients_without_parameter_updates(self) -> None:
        sam = _TinySam()
        for parameter in sam.parameters():
            parameter.requires_grad_(False)
        sam.eval()

        images = torch.rand(1, 3, 16, 16)
        with torch.no_grad():
            embeddings = sam.encode(images)
        heatmaps = torch.randn(1, 1, 16, 16, requires_grad=True)
        logits = sam.decode_heatmaps(
            embeddings, heatmaps, output_size=(16, 16), valid_sizes=[(16, 16)]
        )
        logits.mean().backward()

        self.assertIsNotNone(heatmaps.grad)
        self.assertTrue(torch.isfinite(heatmaps.grad).all())
        self.assertGreater(float(heatmaps.grad.abs().sum()), 0.0)
        # 冻结参数不接收梯度，也不会被修改。
        self.assertTrue(all(p.grad is None for p in sam.parameters()))
        self.assertFalse(sam.training)


class ActiveGroupTest(unittest.TestCase):
    def test_only_exactly_constant_groups_are_inactive(self) -> None:
        rewards = torch.tensor(
            [[0.0, 0.5], [1e-5, 0.5], [2e-5, 0.5]],
            dtype=torch.float64,
        )

        active = DiffPeakEngine._active_group_mask(rewards)

        self.assertEqual(active.tolist(), [True, False])

    def test_low_variance_groups_still_carry_scaled_advantages(self) -> None:
        rewards = torch.tensor([[0.0], [1e-5]], dtype=torch.float64)
        self.assertTrue(bool(DiffPeakEngine._active_group_mask(rewards)[0]))


class GrpoDelegationTest(unittest.TestCase):
    def test_grpo_group_delegates_to_diffusion_only_joint_group(self) -> None:
        engine = _bare_engine()
        batch = {"image": torch.zeros(1, 3, 4, 4)}
        sentinel = {"reward": 1.0}
        # The engine may either import the helper at module scope or lazily.
        if hasattr(engine_module, "joint_grpo_group"):
            target = "diffpeak.engine.joint_grpo_group"
        else:
            target = "diffpeak.joint_rl.joint_grpo_group"
        with patch(target, return_value=sentinel) as mocked:
            result = engine._grpo_group(batch, epoch=3)

        self.assertIs(result, sentinel)
        mocked.assert_called_once_with(engine, batch, 3)


class GrpoEpochStatisticsTest(unittest.TestCase):
    def test_skipped_updates_are_not_counted_as_multiple_groups(self) -> None:
        engine = object.__new__(DiffPeakEngine)
        engine.log_every = 100
        engine._set_train_modes = mock.Mock()
        engine._prepare_batch = lambda batch: batch
        engine._grpo_group = mock.Mock(side_effect=[
            {"updates": 0.0, "skipped_updates": 3.0},
            {"updates": 2.0, "skipped_updates": 1.0},
        ])
        stats = engine._train_epoch_grpo("D", 0, [{}, {}])
        self.assertEqual(stats["optimizer_steps"], 2.0)
        self.assertEqual(stats["skipped_update_steps"], 4.0)
        self.assertEqual(stats["skipped_groups"], 1.0)


class DiffusionOnlyIntegrationTest(unittest.TestCase):
    def test_d_stage_updates_only_condition_and_dit(self) -> None:
        fixture = pipeline.EnginePipelineTest()
        with tempfile.TemporaryDirectory() as temporary, patch(
            "diffpeak.engine.build_models", side_effect=_build_tiny_models
        ):
            config = fixture._make_config(Path(temporary))
            # 锚定保证微型组也产生真实更新，独立于奖励是否恰好相同。
            config["rl"]["anchor_weight"] = 0.1
            engine = fixture._engine(config)

            engine.run_stage("ABC")
            sam_before = copy.deepcopy(engine.sam.state_dict())
            condition_before = {
                name: value.detach().clone()
                for name, value in engine.condition_encoder.state_dict().items()
            }
            dit_before = {
                name: value.detach().clone()
                for name, value in engine.dit.state_dict().items()
            }

            metrics = engine.run_stage("D")
            self.assertIn("reward", metrics)
            self.assertTrue(torch.isfinite(torch.tensor(metrics["iou"])))
            train_stats = engine.history[-1]["train"]
            self.assertIn("reward", train_stats)
            # 终端 Bernoulli 分割策略已移除。
            self.assertNotIn("terminal_grpo", train_stats)
            self.assertNotIn("terminal_kl", train_stats)

            # SAM 状态完全不变；只有条件编码器与 DiT 更新。
            self.assertFalse(engine.sam.training)
            self.assertTrue(all(not p.requires_grad for p in engine.sam.parameters()))
            for name, value in engine.sam.state_dict().items():
                torch.testing.assert_close(value, sam_before[name], rtol=0, atol=0)
            condition_changed = any(
                not torch.equal(condition_before[name], value)
                for name, value in engine.condition_encoder.state_dict().items()
            )
            dit_changed = any(
                not torch.equal(dit_before[name], value)
                for name, value in engine.dit.state_dict().items()
            )
            self.assertTrue(condition_changed)
            self.assertTrue(dit_changed)

            # 固定参考只有条件编码器与 DiT；不存在参考 SAM 解码器。
            self.assertIsNotNone(engine.ref_dit)
            self.assertIsNotNone(engine.ref_condition_encoder)
            self.assertIsNone(getattr(engine, "ref_sam_decoder", None))
            for reference in (engine.ref_dit, engine.ref_condition_encoder):
                self.assertFalse(reference.training)
                self.assertTrue(
                    all(not p.requires_grad for p in reference.parameters())
                )


if __name__ == "__main__":
    unittest.main()
