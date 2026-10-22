"""Tiny ABC,D control-flow integration; no pretrained weights or quality claims."""
from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from diffpeak.engine import DiffPeakEngine
from diffpeak.experiment_logging import ExperimentLogger
from diffpeak.inference import DEFAULT_CONFIG, read_checkpoint
from diffpeak.models import ConditionEncoder, ConditionalDiT, GaussianDiffusion
from diffpeak.models.sam import SAM2ImageFeatures
from diffpeak.prompts import PeakExtractor


class _TinySAM(nn.Module):
    no_positive_logit = -20.0

    def __init__(self):
        super().__init__()
        self.sam = nn.Module()
        self.sam.image_encoder = nn.Conv2d(3, 4, 1)
        self.sam.prompt_encoder = nn.Linear(2, 4)
        self.sam.mask_decoder = nn.Conv2d(4, 1, 1)

    def encode(self, images, valid_sizes=None):
        features = self.sam.image_encoder(images)
        # 与 SAM2 一致的 1/16 低分辨率特征，以及 1/4、1/8 两级 FPN 特征。
        return SAM2ImageFeatures(
            F.avg_pool2d(features, 16),
            (F.avg_pool2d(features, 4), F.avg_pool2d(features, 8)),
        )

    def decode(self, embeddings, points, output_size):
        logits = self.sam.mask_decoder(embeddings.low_res)
        # The lightweight environment is prompt-sensitive, without using labels.
        coords, labels = points[0]
        shift = coords[labels == 1].mean() / output_size[0]
        return F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False) + shift

    def decode_heatmaps(
        self,
        embeddings,
        heatmaps,
        output_size=(1024, 1024),
        valid_sizes=None,
        *,
        decoder=None,
    ):
        """可微密集热图提示：解码器参数与热图都必须保留梯度。"""
        module = self.sam.mask_decoder if decoder is None else decoder
        low_res = getattr(embeddings, "low_res", embeddings)
        logits = module(low_res)
        if heatmaps.shape[1] == 2:
            dense = heatmaps[:, 0:1] - heatmaps[:, 1:2]
        else:
            dense = heatmaps[:, 0:1]
        if valid_sizes is not None:
            mask = torch.zeros_like(dense)
            for index, size in enumerate(valid_sizes):
                height, width = (
                    (int(dense.shape[-2]), int(dense.shape[-1]))
                    if size is None
                    else (int(size[0]), int(size[1]))
                )
                mask[index, :, :height, :width] = 1.0
            dense = dense * mask
        dense = F.interpolate(
            dense, size=logits.shape[-2:], mode="bilinear", align_corners=False
        )
        return F.interpolate(
            logits + dense,
            size=(int(output_size[0]), int(output_size[1])),
            mode="bilinear",
            align_corners=False,
        )


def _build_tiny_models(config, device, sam_checkpoint=None):
    models = {
        "sam": _TinySAM(),
        "condition_encoder": ConditionEncoder(
            sam_channels=4, condition_channels=(4, 8, 8, 8)
        ),
        "dit": ConditionalDiT(
            hidden_size=32, depth=2, num_heads=4, patch_size=4,
            condition_channels=(4, 8, 8, 8), gradient_checkpointing=True,
        ),
        "diffusion": GaussianDiffusion(timesteps=8, sampling_steps=2, heatmap_size=16),
    }
    return {name: module.to(device) for name, module in models.items()}, PeakExtractor(
        sigma=0, h=0, threshold=0.3, min_distance=4
    )


def _session_records(directory):
    return [
        json.loads(line)
        for line in (directory / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _assert_module_states_equal(test, expected, actual, names):
    """比较策略与参考模块的完整 state_dict（逐位相等）。"""
    for name in names:
        expected_module = getattr(expected, name)
        actual_module = getattr(actual, name)
        test.assertIsNotNone(actual_module, f"{name} 未恢复")
        expected_state = expected_module.state_dict()
        actual_state = actual_module.state_dict()
        test.assertEqual(set(actual_state), set(expected_state), name)
        for key, value in expected_state.items():
            torch.testing.assert_close(
                actual_state[key], value, rtol=0, atol=0,
                msg=f"{name}.{key} 恢复后不一致",
            )


class EnginePipelineTest(unittest.TestCase):
    def _make_config(self, root, samples=2, batch_size=2):
        data = root / "data"
        (data / "train").mkdir(parents=True)
        entries = []
        for index in range(samples):
            name = f"{index}_sat.jpg"
            image = np.full((64, 64, 3), 80 + 40 * index, dtype=np.uint8)
            mask = np.zeros((64, 64), dtype=np.uint8)
            mask[28:36] = 255
            Image.fromarray(image).save(data / "train" / name)
            Image.fromarray(mask).save(data / "train" / f"{index}_mask.png")
            entries.append(f"train/{name}")
        split = root / "split.json"
        split.write_text(json.dumps(entries), encoding="utf-8")
        config = copy.deepcopy(DEFAULT_CONFIG)
        # 默认配置必须使用联合 ABC,D 协议；分阶段 A/B/C 已移除。
        self.assertEqual(config["training_protocol"], "joint_abc_v1")
        self.assertEqual(tuple(config["stages"]), ("ABC", "D"))
        # 快照配置必须描述测试工厂实际构建的微型 DiT。
        config["model"]["dit"].update(
            hidden_size=32, depth=2, num_heads=4, patch_size=4,
            condition_channels=(4, 8, 8, 8), gradient_checkpointing=True,
        )
        config["data"].update(root=str(data), train_split=str(split), val_split=str(split), num_workers=0)
        config["output_dir"] = str(root / "run")
        checkpoint = root / "unused.pth"
        checkpoint.write_bytes(b"tiny SAM builder does not read pretrained weights")
        config["model"]["sam_checkpoint"] = str(checkpoint)
        config["train"].update(
            amp=False, batch_size=batch_size, accumulation_steps=1, val_limit=1, log_every=100
        )
        config["rl"].update(group_size=2, point_cost_warmup_steps=2)
        for stage in ("ABC", "D"):
            config["stages"][stage].update(epochs=1, lr=1e-4, warmup_steps=1)
        return config

    def _engine(self, config):
        engine = DiffPeakEngine(config, device="cpu")
        engine.image_size = 64
        engine.heatmap_size = 16
        return engine

    def test_joint_abc_d_flow_and_resume_match_uninterrupted_training(self):
        with tempfile.TemporaryDirectory() as temporary, patch(
            "diffpeak.engine.build_models", side_effect=_build_tiny_models
        ):
            root = Path(temporary)
            config = self._make_config(root)
            engine = self._engine(config)

            # ABC：只更新扩散分支，SAM 参数和 buffers 全部保持不变。
            sam_before = copy.deepcopy(engine.sam.state_dict())
            abc_metrics = engine.run_stage("ABC")
            self.assertTrue(np.isfinite(abc_metrics["iou"]))
            self.assertFalse(engine.sam.sam.mask_decoder.training)
            self.assertTrue(all(not p.requires_grad for p in engine.sam.parameters()))
            for name, value in engine.sam.state_dict().items():
                torch.testing.assert_close(value, sam_before[name], rtol=0, atol=0)
            self.assertTrue(engine.condition_encoder.training)
            self.assertTrue(engine.dit.training)
            payload = read_checkpoint(engine.output_dir / "last.pt")
            self.assertEqual(payload["stage"], "ABC")
            self.assertEqual(payload["format_version"], 4)
            self.assertEqual(payload["training_protocol"], "joint_abc_v1")
            self.assertNotIn("ref_dit", payload["models"])
            self.assertTrue((engine.output_dir / "stage_ABC.pth").is_file())
            # D 从 ABC 结束状态初始化；快照不依赖旧分阶段权重。
            abc_state = copy.deepcopy(engine._model_states())

            # D：rollout 与 replay 都使用 eval 模式，但三个模块仍可训练。
            d_metrics = engine.run_stage("D")
            self.assertTrue(np.isfinite(d_metrics["iou"]))
            self.assertFalse(engine.sam.sam.mask_decoder.training)
            self.assertFalse(engine.condition_encoder.training)
            self.assertFalse(engine.dit.training)
            self.assertFalse(engine.ref_dit.training)
            payload = read_checkpoint(engine.output_dir / "last.pt")
            self.assertEqual(payload["stage"], "D")
            self.assertEqual(payload["format_version"], 4)
            self.assertEqual(payload["training_protocol"], "joint_abc_v1")
            for name in ("ref_dit", "ref_condition_encoder", "ref_sam_decoder"):
                self.assertIn(name, payload["models"])
            self.assertTrue((engine.output_dir / "stage_D.pth").is_file())
            self.assertEqual([item["stage"] for item in engine.history], ["ABC", "D"])
            self.assertGreater(engine.global_step, 0)
            # 2 个样本、D 阶段 batch_size=1 -> 2 个 GRPO 组。
            self.assertEqual(engine.grpo_groups_processed, 2)

            # 两轮 D 计划：一轮完整结束后中断，验证恢复与不中断训练的逐位一致。
            config["stages"]["D"]["epochs"] = 2
            config["output_dir"] = str(root / "continuous")
            continuous = self._engine(config)
            continuous.load_models_from_payload(
                {"models": copy.deepcopy(abc_state), "stage": "ABC"}
            )
            uninterrupted_config = copy.deepcopy(config)
            uninterrupted_config["output_dir"] = str(root / "uninterrupted")
            uninterrupted = self._engine(uninterrupted_config)
            uninterrupted.load_models_from_payload(
                {"models": copy.deepcopy(abc_state), "stage": "ABC"}
            )
            uninterrupted.run_stage("D")
            original_builder = continuous._build_train_loader

            def interrupt_second_epoch(epoch):
                if epoch == 1:
                    raise KeyboardInterrupt
                return original_builder(epoch)

            with patch.object(continuous, "_build_train_loader", side_effect=interrupt_second_epoch):
                with self.assertRaises(KeyboardInterrupt):
                    continuous.run_stage("D")
            checkpoint = read_checkpoint(continuous.output_dir / "last.pt")
            self.assertEqual(checkpoint["epoch"], 1)
            # D: 2 个样本、batch_size=1 -> 2 步/epoch，两轮共 4 步。
            self.assertEqual(checkpoint["schedule"]["total_steps"], 4)
            for name in ("ref_dit", "ref_condition_encoder", "ref_sam_decoder"):
                self.assertIn(name, checkpoint["models"])
            continuous.run_stage("D", start_epoch=1, resume_payload=copy.deepcopy(checkpoint))

            config["output_dir"] = str(root / "resumed")
            # A CLI override must not silently replace the checkpoint's LR curve.
            config["stages"]["D"].update(warmup_steps=0, warmup_ratio=0.0, min_lr_ratio=0.9)
            resumed = self._engine(config)
            resumed.run_stage("D", start_epoch=1, resume_payload=copy.deepcopy(checkpoint))
            self.assertEqual(resumed.global_step, continuous.global_step)
            self.assertEqual(resumed.grpo_groups_processed, continuous.grpo_groups_processed)
            self.assertEqual(resumed.scheduler.state_dict(), continuous.scheduler.state_dict())
            self.assertEqual(resumed.optimizer.param_groups[0]["lr"], continuous.optimizer.param_groups[0]["lr"])
            self.assertEqual(uninterrupted.global_step, continuous.global_step)
            # 三个策略模块与三个固定参考模块都必须逐位恢复。
            module_names = (
                "sam", "condition_encoder", "dit",
                "ref_dit", "ref_condition_encoder", "ref_sam_decoder",
            )
            _assert_module_states_equal(self, uninterrupted, continuous, module_names)
            _assert_module_states_equal(self, uninterrupted, resumed, module_names)

    def test_stage_throughput_overrides_apply_before_loader_and_scheduler(self):
        with tempfile.TemporaryDirectory() as temporary, patch(
            "diffpeak.engine.build_models", side_effect=_build_tiny_models
        ):
            root = Path(temporary)
            config = self._make_config(root, samples=5, batch_size=2)
            # 删除 D 的阶段覆盖，验证缺失键回退到全局吞吐配置。
            config["stages"]["D"].pop("batch_size", None)
            config["stages"]["ABC"].update(
                batch_size=1, accumulation_steps=2, num_workers=1
            )
            engine = self._engine(config)
            observed = []
            original_builder = engine._build_train_loader

            def recording_builder(epoch):
                loader = original_builder(epoch)
                # 在 loader 真正构建完成后读取运行时属性与 DataLoader 实际值，
                # 以确认覆盖发生在 loader/scheduler 之前。
                observed.append(
                    (
                        engine.batch_size,
                        engine.accumulation_steps,
                        engine.num_workers,
                        loader.batch_size,
                        loader.num_workers,
                    )
                )
                return loader

            with patch.object(
                engine, "_build_train_loader", side_effect=recording_builder
            ):
                engine.run_stage("ABC")
                abc_observed = list(observed)
                abc_steps = engine.scheduler_steps_per_epoch
                observed.clear()

                # D 没有覆盖：必须完全回退到全局 batch=2 / accum=1 / workers=0，
                # 证明 ABC 的覆盖没有泄漏到下一阶段。
                engine.run_stage("D")
                d_observed = list(observed)
                observed.clear()
                d_batch_size = engine.batch_size
                d_accumulation_steps = engine.accumulation_steps
                d_num_workers = engine.num_workers
                d_steps = engine.scheduler_steps_per_epoch

                # 第二轮 D 使用自己的覆盖，验证阶段内覆盖同样生效。
                engine.stage_cfgs["D"].update(
                    batch_size=3, accumulation_steps=3, num_workers=0
                )
                engine.run_stage("D")
                d_override_observed = list(observed)
                d_override_steps = engine.scheduler_steps_per_epoch

            # ABC：batch 1、累积 2；5 个样本 -> 5 个微批 -> 3 次优化器步。
            self.assertTrue(abc_observed)
            for snapshot in abc_observed:
                self.assertEqual(snapshot, (1, 2, 1, 1, 1))
            self.assertEqual(abc_steps, 3)

            # D 回退全局值：batch 2 -> 3 个 loader 批次，GRPO 每批一组。
            self.assertTrue(d_observed)
            for snapshot in d_observed:
                self.assertEqual(snapshot, (2, 1, 0, 2, 0))
            self.assertEqual((d_batch_size, d_accumulation_steps, d_num_workers), (2, 1, 0))
            self.assertEqual(d_steps, 3)

            # D 自身覆盖：batch 3 -> 2 个 loader 批次。
            self.assertTrue(d_override_observed)
            for snapshot in d_override_observed:
                self.assertEqual(snapshot, (3, 3, 0, 3, 0))
            self.assertEqual(d_override_steps, 2)

            # 覆盖只改变运行时属性，不回写用户配置。
            self.assertEqual(config["train"]["batch_size"], 2)
            self.assertEqual(config["train"]["accumulation_steps"], 1)
            self.assertEqual(config["data"]["num_workers"], 0)
            self.assertNotIn("batch_size", config["stages"]["D"])

    def test_stage_overrides_resume_keeps_checkpoint_schedule(self):
        with tempfile.TemporaryDirectory() as temporary, patch(
            "diffpeak.engine.build_models", side_effect=_build_tiny_models
        ):
            root = Path(temporary)
            config = self._make_config(root, samples=4, batch_size=1)
            config["stages"]["ABC"].update(
                epochs=1, batch_size=1, accumulation_steps=2, num_workers=0
            )
            first = self._engine(config)
            first.run_stage("ABC")
            # 4 个样本、batch=1、累积=2 -> 每个 epoch 2 步。
            self.assertEqual(first.scheduler_steps_per_epoch, 2)
            self.assertEqual(first.scheduler_total_steps, 2)
            checkpoint = read_checkpoint(first.output_dir / "last.pt")
            self.assertEqual(checkpoint["schedule"]["steps_per_epoch"], 2)
            self.assertEqual(checkpoint["schedule"]["total_steps"], 2)
            checkpoint_global_step = int(checkpoint["global_step"])

            # 恢复时改变吞吐覆盖：新 loader 应使用 batch=2/累积=1（按新配置
            # 会得到 2 步/epoch），但调度必须沿用 checkpoint 的 2 步曲线。
            resume_config = copy.deepcopy(config)
            resume_config["output_dir"] = str(root / "resume")
            resume_config["stages"]["ABC"].update(
                epochs=2, batch_size=2, accumulation_steps=1
            )
            resumed = self._engine(resume_config)
            observed = []
            original_builder = resumed._build_train_loader

            def recording_builder(epoch):
                loader = original_builder(epoch)
                observed.append((epoch, loader.batch_size, loader.num_workers))
                return loader

            with patch.object(
                resumed, "_build_train_loader", side_effect=recording_builder
            ):
                resumed.run_stage(
                    "ABC", start_epoch=1,
                    resume_payload=copy.deepcopy(checkpoint),
                )

            # probe loader 与恢复 epoch 的 loader 都采用阶段覆盖。
            self.assertEqual(observed, [(1, 2, 0), (1, 2, 0)])
            self.assertEqual(resumed.scheduler_steps_per_epoch, 2)
            self.assertEqual(resumed.scheduler_total_steps, 2)
            # checkpoint 的 2 步 + 新 loader 的 2 个优化器步。
            self.assertEqual(
                resumed.global_step, checkpoint_global_step + 2
            )

    def test_logging_scopes_epoch_tags_and_resume_session_isolation(self):
        with tempfile.TemporaryDirectory() as temporary, patch(
            "diffpeak.engine.build_models", side_effect=_build_tiny_models
        ):
            root = Path(temporary)
            config = self._make_config(root, samples=4, batch_size=1)
            config["train"]["log_every"] = 2
            config["logging"] = {
                "tensorboard": False, "log_every": 2, "flush_secs": 1
            }
            sessions = root / "sessions"
            engine = self._engine(config)

            with ExperimentLogger(
                sessions, config, metadata={"phase": "all-stages"}
            ) as logger:
                engine.experiment_logger = logger
                with patch.object(
                    logger, "log_scalars", wraps=logger.log_scalars
                ) as scalars_spy, patch.object(
                    logger, "event", wraps=logger.event
                ) as event_spy, patch.object(
                    logger, "flush", wraps=logger.flush
                ) as flush_spy:
                    for stage in ("ABC", "D"):
                        engine.run_stage(stage)
                full_directory = logger.directory
            self.assertGreater(scalars_spy.call_count, 0)
            self.assertGreaterEqual(event_spy.call_count, 2)
            self.assertGreaterEqual(flush_spy.call_count, 2)
            first_call = scalars_spy.call_args_list[0]
            self.assertEqual(set(first_call.kwargs), {"step", "epoch"})

            # 上下文退出后 session_end 必须已落盘，文件句柄已关闭。
            full_records = _session_records(full_directory)
            self.assertEqual(full_records[-1]["event"], "session_end")
            self.assertEqual(full_records[-1]["status"], "completed")
            metadata = json.loads(
                (full_directory / "metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metadata["status"], "completed")
            self.assertEqual(metadata["phase"], "all-stages")

            metrics = [record for record in full_records if record["event"] == "metrics"]

            def scoped(stage, scope):
                return [
                    record for record in metrics
                    if record["stage"] == stage and record["scope"] == scope
                ]

            # 4 个样本、batch_size=1、log_every=2：记录第 1 批、每 2 批、
            # 以及末批；1-based index 为 1/2/4。
            batches = scoped("ABC", "train_batch")
            self.assertEqual([record["step"] for record in batches], [1, 2, 4])
            self.assertEqual({record["epoch"] for record in batches}, {1})
            self.assertIn("loss", batches[0]["metrics"])
            self.assertIn("lr", batches[0]["metrics"])

            epoch_rows = scoped("ABC", "train_epoch")
            self.assertEqual(
                [(record["step"], record["epoch"]) for record in epoch_rows],
                [(1, 1)],
            )
            self.assertIn("loss", epoch_rows[0]["metrics"])
            # 联合 ABC 同时记录分段与采样分割损失。
            self.assertIn("seg", epoch_rows[0]["metrics"])
            self.assertIn("sampled_seg", epoch_rows[0]["metrics"])

            validation = scoped("ABC", "val_epoch")
            self.assertEqual(
                [(record["step"], record["epoch"]) for record in validation],
                [(1, 1)],
            )
            self.assertIn("iou", validation[0]["metrics"])
            self.assertIn("positive_on_road_rate", validation[0]["metrics"])
            self.assertIn("heatmap_above_threshold_rate", validation[0]["metrics"])
            self.assertIn("quality_reward", validation[0]["metrics"])

            system = scoped("ABC", "system")
            self.assertEqual(
                [(record["step"], record["epoch"]) for record in system],
                [(1, 1)],
            )
            self.assertIn("epoch_seconds", system[0]["metrics"])

            groups = scoped("D", "train_group")
            self.assertEqual([record["step"] for record in groups], [1, 2, 4])
            self.assertEqual({record["epoch"] for record in groups}, {1})
            self.assertIn("reward", groups[0]["metrics"])
            details = groups[0]["metrics"]["trajectory_diagnostics"]
            self.assertTrue(details)
            # 联合 D 的逐轨迹日志：密集提示没有离散点，点惩罚恒为 0。
            for key in ("image_id", "advantage", "reward", "quality_reward", "point_penalty"):
                self.assertIn(key, details[0])
            self.assertAlmostEqual(
                details[0]["quality_reward"] - details[0]["point_penalty"],
                details[0]["reward"],
            )
            d_epoch = scoped("D", "train_epoch")
            self.assertEqual(
                [(record["step"], record["epoch"]) for record in d_epoch], [(1, 1)]
            )
            self.assertIn("reward", d_epoch[0]["metrics"])
            self.assertEqual(len(scoped("D", "val_epoch")), 1)
            self.assertIn("epoch_seconds", scoped("D", "system")[0]["metrics"])

            events = [record["event"] for record in full_records]
            self.assertEqual(events.count("stage_start"), 2)
            self.assertEqual(events.count("stage_end"), 2)
            self.assertIn("checkpoint", events)

            # resume 必须创建新 session，旧 session 文件保持不变。
            resume_config = copy.deepcopy(config)
            resume_config["stages"]["D"]["epochs"] = 1
            resume_config["output_dir"] = str(root / "resume_run")
            before_resume = self._engine(resume_config)
            with ExperimentLogger(
                sessions, resume_config, metadata={"phase": "before-resume"}
            ) as before_logger:
                before_resume.experiment_logger = before_logger
                before_resume.run_stage("D")
                before_directory = before_logger.directory
            before_text = (before_directory / "metrics.jsonl").read_text(
                encoding="utf-8"
            )
            checkpoint = read_checkpoint(before_resume.output_dir / "last.pt")
            self.assertEqual(checkpoint["epoch"], 1)

            resume_config["stages"]["D"]["epochs"] = 2
            after_resume = self._engine(resume_config)
            with ExperimentLogger(
                sessions, resume_config, metadata={"phase": "after-resume"}
            ) as after_logger:
                after_resume.experiment_logger = after_logger
                after_resume.run_stage(
                    "D", start_epoch=1,
                    resume_payload=copy.deepcopy(checkpoint),
                )
                after_directory = after_logger.directory

            self.assertNotEqual(before_directory, after_directory)
            self.assertEqual(
                (before_directory / "metrics.jsonl").read_text(encoding="utf-8"),
                before_text,
            )
            before_records = _session_records(before_directory)
            after_records = _session_records(after_directory)
            self.assertEqual(
                [record["phase"] for record in before_records
                 if record["event"] == "session_start"],
                ["before-resume"],
            )
            self.assertEqual(
                [record["phase"] for record in after_records
                 if record["event"] == "session_start"],
                ["after-resume"],
            )
            for records, directory in (
                (before_records, before_directory),
                (after_records, after_directory),
            ):
                self.assertEqual(records[-1]["event"], "session_end")
                self.assertEqual(records[-1]["status"], "completed")
                session_metadata = json.loads(
                    (directory / "metadata.json").read_text(encoding="utf-8")
                )
                self.assertEqual(session_metadata["status"], "completed")

            stage_starts = [
                record for record in after_records
                if record["event"] == "stage_start"
            ]
            self.assertEqual(len(stage_starts), 1)
            self.assertTrue(stage_starts[0]["resumed"])
            self.assertEqual(stage_starts[0]["start_epoch"], 1)

            # 新 session 只包含恢复的第二个 epoch，坐标轴继续且不重叠。
            resumed_groups = [
                record for record in after_records
                if record["event"] == "metrics"
                and record["scope"] == "train_group"
            ]
            self.assertEqual(
                [record["step"] for record in resumed_groups], [5, 6, 8]
            )
            self.assertEqual(
                {record["epoch"] for record in resumed_groups}, {2}
            )
            resumed_epoch = [
                record for record in after_records
                if record["event"] == "metrics"
                and record["scope"] == "train_epoch"
            ]
            self.assertEqual(
                [(record["step"], record["epoch"]) for record in resumed_epoch],
                [(2, 2)],
            )


if __name__ == "__main__":
    unittest.main()
