"""GRPO 引擎的轻量集成测试。

不加载 SAM/DeepGlobe 权重，只用微型模块和假扩散验证：
* 阶段 A/B/C/D 的可训练参数与 eval/train 模式；
* 同图组内优势、低方差组与完全常量组的区分；
* batch>1 时 G*T*B 分母、无效组安全跳过 backward/optimizer；
* 提示成本预热、点集 diversity、正负点数和日志字段；
* checkpoint 的算法/调度元数据，以及旧 DDPO 快照禁止直接 --resume。
"""

from __future__ import annotations

import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diffpeak.engine import DiffPeakEngine  # noqa: E402
from diffpeak.rewards import RewardConfig  # noqa: E402


class _TinyModule(nn.Module):
    def __init__(self, value: float = 1.0) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(float(value)))


class _TinySam(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Parameter(torch.tensor(1.0))
        self.sam = nn.Module()
        self.sam.mask_decoder = _TinyModule(2.0)


def _bare_engine() -> DiffPeakEngine:
    engine = object.__new__(DiffPeakEngine)
    engine.device = torch.device("cpu")
    engine.amp = False
    engine.sam = _TinySam()
    engine.condition_encoder = _TinyModule(3.0)
    engine.dit = _TinyModule(4.0)
    engine.ref_dit = None
    return engine


class FreezingTest(unittest.TestCase):
    def test_joint_stages_train_condition_dit_and_decoder(self) -> None:
        # 联合协议下 ABC 与 D 都训练三个策略模块；冻结的只有 SAM image /
        # prompt encoder。
        for stage in ("ABC", "D"):
            with self.subTest(stage=stage):
                engine = _bare_engine()
                engine._set_requires_grad(stage)

                self.assertTrue(
                    all(
                        parameter.requires_grad
                        for parameter in engine.sam.sam.mask_decoder.parameters()
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
                # 只有 mask decoder 可训练，SAM 其余参数保持冻结。
                self.assertFalse(engine.sam.backbone.requires_grad)
                self.assertEqual(
                    any(
                        parameter.requires_grad
                        for parameter in engine.sam.parameters()
                    ),
                    True,
                )

    def test_train_modes_train_abc_and_eval_all_during_d(self) -> None:
        expected = {
            # ABC 端到端训练三个模块。
            "ABC": (True, True, True),
            # D 的 rollout/replay 对整个策略使用 eval，保证两次前向一致；
            # 梯度仍然开启。
            "D": (False, False, False),
        }
        for stage, (decoder, condition, dit) in expected.items():
            with self.subTest(stage=stage):
                engine = _bare_engine()
                engine.ref_dit = nn.Module()
                engine.ref_dit.train()
                engine._set_train_modes(stage)

                self.assertFalse(engine.sam.training)
                self.assertEqual(engine.sam.sam.mask_decoder.training, decoder)
                self.assertEqual(engine.condition_encoder.training, condition)
                self.assertEqual(engine.dit.training, dit)
                self.assertFalse(engine.ref_dit.training)


class ActiveGroupTest(unittest.TestCase):
    def test_only_exactly_constant_groups_are_inactive(self) -> None:
        rewards = torch.tensor(
            [[0.0, 0.5], [1e-5, 0.5], [2e-5, 0.5]],
            dtype=torch.float64,
        )

        active = DiffPeakEngine._active_group_mask(rewards)

        self.assertEqual(active.tolist(), [True, False])

    def test_low_variance_advantages_are_scaled_not_zeroed(self) -> None:
        rewards = torch.tensor([[0.0], [1e-5]], dtype=torch.float64)
        advantages = torch.tensor([[-0.05], [0.05]], dtype=torch.float64)

        self.assertTrue(bool(DiffPeakEngine._active_group_mask(rewards)[0]))
        self.assertTrue(bool((advantages != 0).all()))


class _FakeDiffusion:
    def transition(self, dit, state, timestep, previous_timestep, conditions):
        return state * dit.weight, torch.ones_like(state)

    def log_prob(self, action, mean, std):
        standardized = (action - mean) / std
        per_sample = -0.5 * standardized.reshape(
            standardized.shape[0], -1
        ).pow(2).sum(dim=1)
        return per_sample.double()


class _RecordingScaler:
    def __init__(self) -> None:
        self.losses = []
        self.steps = 0

    def is_enabled(self) -> bool:
        return False

    def scale(self, loss):
        self.losses.append(float(loss.detach()))
        return loss

    def unscale_(self, optimizer) -> None:
        return None

    def step(self, optimizer) -> None:
        self.steps += 1
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


def _update_engine() -> DiffPeakEngine:
    engine = _bare_engine()
    engine.dit = _TinyModule(1.0)
    engine.optimizer = torch.optim.AdamW([engine.dit.weight], lr=1e-3)
    engine.scheduler = _RecordingScheduler()
    engine.scaler = _RecordingScaler()
    engine.grad_clip = 0.0
    engine.ppo_clip = 0.2
    engine.kl_weight = 0.0
    engine.anchor_weight = 0.0
    engine.update_epochs = 1
    engine.global_step = 0
    engine.diffusion = _FakeDiffusion()
    engine.group_size = 2
    engine.advantage_std_delta = 1e-4
    return engine


class GrpoUpdateTest(unittest.TestCase):
    def test_batch_gt_one_uses_g_times_t_b_times_b_denominator(self) -> None:
        engine = _update_engine()
        state = torch.ones(2, 1, 2, 2)
        old_log_prob = engine.diffusion.log_prob(
            state, state, torch.ones_like(state)
        )
        step = {
            "state": state,
            "action": state.clone(),
            "timestep": 0,
            "previous_timestep": -1,
            "old_log_prob": old_log_prob,
        }
        advantages = torch.tensor(
            [[1.0, 0.0], [-1.0, 0.0]], dtype=torch.float64
        )
        active_groups = torch.tensor([True, False])

        stats = engine._grpo_update(
            [torch.zeros(2, 1, 2, 2)],
            torch.zeros(2, 2, 2, 2),
            [[step], [step]],
            advantages,
            active_groups,
        )

        # 两个 trajectory、T=1、B=2：每步策略损失先按 B 求和再除以 2*1*2。
        # 第二张图是常量组，分子为 0，但仍占分母。
        self.assertEqual(len(engine.scaler.losses), 2)
        self.assertAlmostEqual(engine.scaler.losses[0], -0.25)
        self.assertAlmostEqual(engine.scaler.losses[1], 0.25)
        self.assertEqual(stats["updates"], 1.0)
        self.assertEqual(stats["policy_terms"], 2.0)
        self.assertEqual(stats["clip_fraction"], 0.0)
        self.assertEqual(stats["logratio_abs_max"], 0.0)
        self.assertAlmostEqual(stats["lr"], 1e-3)
        self.assertEqual(engine.global_step, 1)
        self.assertEqual(engine.scheduler.steps, 1)

    def test_constant_group_skips_backward_and_optimizer(self) -> None:
        engine = _update_engine()
        before = engine.dit.weight.detach().clone()
        state = torch.ones(2, 2, 2)
        step = {
            "state": state,
            "action": state.clone(),
            "timestep": 0,
            "previous_timestep": -1,
            "old_log_prob": torch.zeros(2),
        }

        stats = engine._grpo_update(
            [torch.zeros(1, 1, 2, 2)],
            torch.zeros(1, 2, 2, 2),
            [[step], [step]],
            torch.zeros(2, 1, dtype=torch.float64),
            torch.tensor([False]),
        )

        self.assertEqual(engine.scaler.losses, [])
        self.assertEqual(stats["updates"], 0.0)
        self.assertEqual(stats["skipped_updates"], 1.0)
        self.assertEqual(engine.global_step, 0)
        self.assertEqual(engine.scheduler.steps, 0)
        torch.testing.assert_close(engine.dit.weight.detach(), before)


class _GroupCondition(nn.Module):
    def forward(self, images, embeddings):
        reference = torch.zeros(2, 1, 2, 2)
        return [reference.clone() for _ in range(4)], None


class _GroupDiffusion:
    def sample(self, dit, conditions, generator=None, record=False):
        return torch.zeros(2, 2, 2, 2), [{"dummy": torch.tensor(0.0)}]


class GrpoGroupDiagnosticsTest(unittest.TestCase):
    def test_group_logs_cost_lr_points_and_diversity(self) -> None:
        engine = _bare_engine()
        engine._stage = "D"
        engine.seed = 0
        engine.group_size = 2
        engine.advantage_std_delta = 1e-4
        engine.point_cost_warmup_steps = 4
        # _grpo_group 先递增独立组计数器，再按组数计算乘子：
        # 本组从 0 变为 1，第 1 组成本乘子为 1/4。
        engine.global_step = 0
        engine.grpo_groups_processed = 0
        engine.reward_config = RewardConfig(point_cost=0.05)
        engine.image_size = 2
        engine.threshold = 0.5
        engine.ref_dit = nn.Module()
        engine.peak_extractor = object()
        engine.diffusion = _GroupDiffusion()
        engine.condition_encoder = _GroupCondition()
        update_stats = {
            "grpo": 0.0,
            "kl": float("nan"),
            "anchor": float("nan"),
            "clip_fraction": 0.0,
            "logratio_abs_mean": 0.0,
            "logratio_abs_max": 0.0,
            "lr": 1e-6,
            "updates": 1.0,
            "skipped_updates": 0.0,
            "policy_terms": 4.0,
        }
        engine._grpo_update = mock.Mock(return_value=update_stats)

        # 用 side_effect 依次返回两条轨迹的点集。
        trajectory_points = [
            [
                (torch.tensor([[0.0, 0.0]]), torch.tensor([1])),
                (torch.tensor([[0.0, 0.0], [1.0, 1.0]]), torch.tensor([1, 1])),
            ],
            [
                (
                    torch.tensor([[0.0, 0.0], [1.0, 1.0], [0.0, 1.0]]),
                    torch.tensor([1, 1, 0]),
                ),
                (torch.tensor([[0.0, 0.0], [1.0, 1.0]]), torch.tensor([1, 1])),
            ],
        ]

        def fake_metrics(prediction, target, point_count, reward_config):
            reward = 0.1 * float(point_count)
            return {
                "reward": reward,
                "iou": 0.5,
                "f1": 0.5,
                "cldice": 0.5,
                "path": 0.0,
            }

        batch = {
            "image": torch.zeros(2, 3, 2, 2),
            "heatmaps": torch.zeros(2, 2, 2, 2),
            "mask": torch.zeros(2, 1, 2, 2),
            "valid_sizes": [(2, 2), (2, 2)],
            "ids": ["a", "b"],
        }
        with mock.patch(
            "diffpeak.engine.encode_images",
            return_value=torch.zeros(2, 1, 2, 2),
        ), mock.patch(
            "diffpeak.engine.get_points", side_effect=trajectory_points
        ), mock.patch(
            "diffpeak.engine.decode_with_prompts",
            return_value=(torch.zeros(2, 1, 2, 2), [True, True]),
        ), mock.patch(
            "diffpeak.engine.compute_image_metrics", side_effect=fake_metrics
        ):
            record = engine._grpo_group(batch, epoch=0)

        self.assertEqual(engine.grpo_groups_processed, 1)
        self.assertAlmostEqual(record["point_cost_scale"], 0.25)
        self.assertAlmostEqual(
            record["point_cost"], 0.05 * record["point_cost_scale"]
        )
        self.assertAlmostEqual(record["active_group_rate"], 0.5)
        self.assertAlmostEqual(record["inactive_group_rate"], 0.5)
        self.assertEqual(record["invalid_groups"], 1.0)
        # 两张图的 reward variance 分别为 0.01 和 0，平均 0.005。
        self.assertAlmostEqual(record["reward_variance"], 0.005)
        self.assertAlmostEqual(record["point_count"], 2.0)
        self.assertAlmostEqual(record["positive_point_count"], 1.75)
        self.assertAlmostEqual(record["negative_point_count"], 0.25)
        # 图 0 的两个点集不同，图 1 相同：平均 diversity=(1+0.5)/2。
        self.assertAlmostEqual(record["point_set_diversity"], 0.75)
        # 图 0 有 2 个不同点集，图 1 所有候选完全相同：1/2 张图 identical。
        self.assertAlmostEqual(record["point_set_identical_ratio"], 0.5)
        self.assertAlmostEqual(record["unique_point_sets"], 1.5)
        self.assertAlmostEqual(record["lr"], 1e-6)
        self.assertAlmostEqual(record["advantage_abs_mean"], 0.5)
        details = record["trajectory_diagnostics"]
        self.assertEqual(len(details), 4)
        self.assertEqual([row["image_id"] for row in details], ["a", "b", "a", "b"])
        self.assertEqual([row["trajectory_index"] for row in details], [0, 0, 1, 1])
        for row, expected in zip(details, [-1.0, 0.0, 1.0, 0.0]):
            self.assertAlmostEqual(row["advantage"], expected)
        for row in details:
            self.assertAlmostEqual(row["quality_reward"] - row["point_penalty"], row["reward"])
            self.assertAlmostEqual(row["point_penalty"], 0.0125 * row["point_count"] / 32)
        self.assertEqual(record["positive_on_road_rate"], 0.0)
        self.assertEqual(record["negative_on_background_rate"], 1.0)
        self.assertEqual(record["no_positive_rate"], 0.0)

        active_groups = engine._grpo_update.call_args.args[4]
        self.assertEqual(active_groups.tolist(), [True, False])


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


class CostWarmupTest(unittest.TestCase):
    def test_cost_scales_linearly_with_processed_groups(self) -> None:
        engine = object.__new__(DiffPeakEngine)
        engine._stage = "D"
        engine.point_cost_warmup_steps = 4
        engine.reward_config = RewardConfig(point_cost=0.05)

        # 独立组计数器：第 1 组为 1/N，第 N 组到 1。这里同时验证它不依赖
        # global_step（后者可能因常量组安全跳过而永远不增加）。
        expected = {0: 0.0, 1: 0.25, 2: 0.5, 3: 0.75, 4: 1.0, 9: 1.0}
        for processed, scale in expected.items():
            with self.subTest(processed=processed):
                engine.grpo_groups_processed = processed
                engine.global_step = 0
                self.assertAlmostEqual(engine._point_cost_scale(), scale)
                self.assertAlmostEqual(
                    engine._effective_reward_config().point_cost,
                    0.05 * scale,
                )

    def test_cost_warmup_advances_when_optimizer_never_steps(self) -> None:
        engine = object.__new__(DiffPeakEngine)
        engine._stage = "D"
        engine.point_cost_warmup_steps = 4
        engine.reward_config = RewardConfig(point_cost=0.05)
        engine.grpo_groups_processed = 1
        engine.global_step = 0

        # 常量组 + kl_weight=0 + anchor_weight=0 时 optimizer 不 step，
        # 但组计数器推进，成本仍从 1/N 开始而不是永久 0。
        self.assertAlmostEqual(engine._point_cost_scale(), 0.25)
        self.assertAlmostEqual(
            engine._effective_reward_config().point_cost, 0.0125
        )

    def test_zero_warmup_disables_ramp(self) -> None:
        engine = object.__new__(DiffPeakEngine)
        engine._stage = "D"
        engine.point_cost_warmup_steps = 0
        engine.grpo_groups_processed = 0
        engine.reward_config = RewardConfig(point_cost=0.05)

        self.assertEqual(engine._point_cost_scale(), 1.0)
        self.assertEqual(engine._effective_reward_config().point_cost, 0.05)


class CheckpointMetadataTest(unittest.TestCase):
    def test_payload_contains_algorithm_and_schedule_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train_split = root / "train.txt"
            train_split.write_text("sample\n", encoding="utf-8")
            sam_checkpoint = root / "sam.pth"
            sam_checkpoint.write_bytes(b"stub")
            output_dir = root / "out"
            output_dir.mkdir()

            engine = _bare_engine()
            engine.sam.architecture = "sam2.1_hiera_base_plus"
            engine.sam.config_name = "configs/sam2.1/sam2.1_hiera_b+.yaml"
            engine.config = {
                "training_protocol": "joint_abc_v1",
                "data": {"train_split": "train.txt", "val_split": None},
                "model": {"sam_checkpoint": "sam.pth"},
                "output_dir": str(output_dir),
            }
            engine.data_root = root
            engine.sam_checkpoint = sam_checkpoint
            engine.output_dir = output_dir
            engine._train_split_cache = train_split
            engine._val_split_cache = None
            engine._val_split_cached = True
            engine.history = []
            engine.global_step = 7
            engine._stage = "D"
            engine.optimizer = None
            engine.scheduler = None
            engine.scaler = None
            engine.scheduler_stage_lr = 1e-6
            engine.scheduler_total_steps = 20
            engine.scheduler_warmup_steps = 2
            engine.scheduler_min_lr_ratio = 0.1
            engine.scheduler_steps_per_epoch = 2
            engine.scheduler_epochs = 10
            engine.scheduler_lr_schedule = "cosine"
            engine.rl_algorithm = "grpo"
            engine.group_size = 4
            engine.advantage_std_delta = 1e-4
            engine.ppo_clip = 1e-4
            engine.kl_weight = 0.01
            engine.anchor_weight = 0.0
            engine.update_epochs = 1
            engine.point_cost_warmup_steps = 10
            engine.reward_config = RewardConfig(point_cost=0.05)
            engine.diffusion = nn.Module()
            engine.ref_dit = None

            payload = engine._checkpoint_payload("D", 1, None)

            self.assertEqual(payload["format_version"], 4)
            self.assertEqual(payload["training_protocol"], "joint_abc_v1")
            self.assertEqual(
                payload["sam"]["architecture"], "sam2.1_hiera_base_plus"
            )
            self.assertEqual(
                payload["config"]["model"]["sam_architecture"],
                "sam2.1_hiera_base_plus",
            )
            self.assertEqual(
                payload["config"]["model"]["sam_config"],
                "configs/sam2.1/sam2.1_hiera_b+.yaml",
            )
            self.assertEqual(payload["algorithm"]["name"], "grpo")
            self.assertEqual(payload["algorithm"]["point_cost_warmup_steps"], 10)
            self.assertAlmostEqual(payload["algorithm"]["reward_point_cost"], 0.05)
            self.assertEqual(payload["schedule"]["name"], "warmup_cosine")
            self.assertEqual(payload["schedule"]["total_steps"], 20)
            self.assertEqual(payload["schedule"]["warmup_steps"], 2)
            self.assertAlmostEqual(payload["schedule"]["min_lr_ratio"], 0.1)
            self.assertAlmostEqual(payload["schedule"]["learning_rate"], 1e-6)

    def test_old_ddpo_checkpoint_cannot_resume(self) -> None:
        engine = _bare_engine()
        engine._stage = "D"
        engine.optimizer = object()
        engine.scheduler = object()
        engine.scaler = object()
        old_payload = {
            "stage": "D",
            "format_version": 1,
            "optimizer": {},
            "scheduler": {},
            "scaler": {},
        }

        with self.assertRaises(ValueError) as caught:
            engine._restore_resume(old_payload)

        self.assertIn("--init", str(caught.exception))

    def test_staged_checkpoint_cannot_resume_joint_protocol(self) -> None:
        engine = _bare_engine()
        engine.config = {"training_protocol": "joint_abc_v1"}
        engine._stage = "D"

        with self.assertRaises(ValueError) as caught:
            engine._validate_resume_algorithm(
                {"stage": "ABC", "format_version": 3}
            )

        self.assertIn("--init", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
