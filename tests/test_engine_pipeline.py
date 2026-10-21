"""Tiny A–D control-flow integration; no pretrained weights or quality claims."""
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
from diffpeak.models import ConditionEncoder, ConditionalUNet, GaussianDiffusion
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


def _build_tiny_models(config, device, sam_checkpoint=None):
    models = {
        "sam": _TinySAM(),
        "condition_encoder": ConditionEncoder(
            sam_channels=4, condition_channels=(4, 8, 8, 8)
        ),
        "unet": ConditionalUNet(
            base_channels=4, channel_mult=(1, 2, 2, 2),
            condition_channels=(4, 8, 8, 8), num_res_blocks=1,
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
        config["data"].update(root=str(data), train_split=str(split), val_split=str(split), num_workers=0)
        config["output_dir"] = str(root / "run")
        checkpoint = root / "unused.pth"
        checkpoint.write_bytes(b"tiny SAM builder does not read pretrained weights")
        config["model"]["sam_checkpoint"] = str(checkpoint)
        config["train"].update(
            amp=False, batch_size=batch_size, accumulation_steps=1, val_limit=1, log_every=100
        )
        config["rl"].update(group_size=2, point_cost_warmup_steps=2)
        for stage in "ABCD":
            config["stages"][stage].update(epochs=1, lr=1e-4, warmup_steps=1)
        return config

    def _engine(self, config):
        engine = DiffPeakEngine(config, device="cpu")
        engine.image_size = 64
        engine.heatmap_size = 16
        return engine

    def test_all_stages_and_checkpoint_resume_match_uninterrupted_training(self):
        with tempfile.TemporaryDirectory() as temporary, patch(
            "diffpeak.engine.build_models", side_effect=_build_tiny_models
        ):
            config = self._make_config(Path(temporary))
            engine = self._engine(config)
            for stage in "ABCD":
                metrics = engine.run_stage(stage)
                self.assertTrue(np.isfinite(metrics["iou"]))
                payload = read_checkpoint(engine.output_dir / "last.pt")
                self.assertEqual(payload["stage"], stage)
                self.assertEqual(payload["format_version"], 3)
                self.assertTrue((engine.output_dir / f"stage_{stage}.pth").is_file())
            self.assertEqual([item["stage"] for item in engine.history], list("ABCD"))
            self.assertGreater(engine.global_step, 0)
            self.assertEqual(engine.grpo_groups_processed, 1)

            # Build a two-epoch schedule, then stop after the first complete epoch.
            config["stages"]["D"]["epochs"] = 2
            config["output_dir"] = str(Path(temporary) / "continuous")
            continuous = self._engine(config)
            initial = copy.deepcopy(engine._model_states())
            continuous.load_models_from_payload({"models": initial, "stage": "C"})
            uninterrupted_config = copy.deepcopy(config)
            uninterrupted_config["output_dir"] = str(Path(temporary) / "uninterrupted")
            uninterrupted = self._engine(uninterrupted_config)
            uninterrupted.load_models_from_payload({"models": initial, "stage": "C"})
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
            self.assertEqual(checkpoint["schedule"]["total_steps"], 2)
            self.assertIn("ref_unet", checkpoint["models"])
            continuous.run_stage("D", start_epoch=1, resume_payload=copy.deepcopy(checkpoint))

            config["output_dir"] = str(Path(temporary) / "resumed")
            # A CLI override must not silently replace the checkpoint's LR curve.
            config["stages"]["D"].update(warmup_steps=0, warmup_ratio=0.0, min_lr_ratio=0.9)
            resumed = self._engine(config)
            resumed.run_stage("D", start_epoch=1, resume_payload=copy.deepcopy(checkpoint))
            self.assertEqual(resumed.global_step, continuous.global_step)
            self.assertEqual(resumed.grpo_groups_processed, continuous.grpo_groups_processed)
            self.assertEqual(resumed.scheduler.state_dict(), continuous.scheduler.state_dict())
            self.assertEqual(resumed.optimizer.param_groups[0]["lr"], continuous.optimizer.param_groups[0]["lr"])
            self.assertEqual(uninterrupted.global_step, continuous.global_step)
            for name, parameter in continuous.unet.state_dict().items():
                torch.testing.assert_close(uninterrupted.unet.state_dict()[name], parameter, rtol=0, atol=0)
                torch.testing.assert_close(resumed.unet.state_dict()[name], parameter, rtol=0, atol=0)
            for name, parameter in continuous.ref_unet.state_dict().items():
                torch.testing.assert_close(resumed.ref_unet.state_dict()[name], parameter, rtol=0, atol=0)

    def test_stage_throughput_overrides_apply_before_loader_and_scheduler(self):
        with tempfile.TemporaryDirectory() as temporary, patch(
            "diffpeak.engine.build_models", side_effect=_build_tiny_models
        ):
            root = Path(temporary)
            config = self._make_config(root, samples=5, batch_size=2)
            config["stages"]["A"].update(
                batch_size=1, accumulation_steps=2, num_workers=1
            )
            config["stages"]["C"].update(
                batch_size=3, accumulation_steps=3, num_workers=0
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
                engine.run_stage("A")
                a_observed = list(observed)
                a_steps = engine.scheduler_steps_per_epoch
                observed.clear()

                engine.run_stage("B")
                b_observed = list(observed)
                observed.clear()
                b_batch_size = engine.batch_size
                b_accumulation_steps = engine.accumulation_steps
                b_num_workers = engine.num_workers

                engine.run_stage("C")
                c_observed = list(observed)
                c_steps = engine.scheduler_steps_per_epoch

            # A：batch 1、累积 2；5 个样本 -> 5 个微批 -> 3 次优化器步。
            self.assertTrue(a_observed)
            for snapshot in a_observed:
                self.assertEqual(snapshot, (1, 2, 1, 1, 1))
            self.assertEqual(a_steps, 3)

            # B 没有任何覆盖：完全回退到全局 batch=2 / accum=1 / workers=0，
            # 证明 A 的覆盖没有泄漏到下一阶段。
            self.assertTrue(b_observed)
            for snapshot in b_observed:
                self.assertEqual(snapshot, (2, 1, 0, 2, 0))
            self.assertEqual(b_batch_size, 2)
            self.assertEqual(b_accumulation_steps, 1)
            self.assertEqual(b_num_workers, 0)

            # C 使用自己的覆盖：batch 3、累积 3；5 个样本 -> 2 个微批 -> 1 步。
            self.assertTrue(c_observed)
            for snapshot in c_observed:
                self.assertEqual(snapshot, (3, 3, 0, 3, 0))
            self.assertEqual(c_steps, 1)

            # 覆盖只改变运行时属性，不回写用户配置。
            self.assertEqual(config["train"]["batch_size"], 2)
            self.assertEqual(config["train"]["accumulation_steps"], 1)
            self.assertEqual(config["data"]["num_workers"], 0)

    def test_stage_overrides_resume_keeps_checkpoint_schedule(self):
        with tempfile.TemporaryDirectory() as temporary, patch(
            "diffpeak.engine.build_models", side_effect=_build_tiny_models
        ):
            root = Path(temporary)
            config = self._make_config(root, samples=4, batch_size=1)
            config["stages"]["A"].update(
                epochs=1, batch_size=1, accumulation_steps=2, num_workers=0
            )
            first = self._engine(config)
            first.run_stage("A")
            # 4 个样本、batch=1、累积=2 -> 每个 epoch 2 步。
            self.assertEqual(first.scheduler_steps_per_epoch, 2)
            self.assertEqual(first.scheduler_total_steps, 2)
            checkpoint = read_checkpoint(first.output_dir / "last.pt")
            self.assertEqual(checkpoint["schedule"]["steps_per_epoch"], 2)
            self.assertEqual(checkpoint["schedule"]["total_steps"], 2)
            checkpoint_global_step = int(checkpoint["global_step"])

            # 恢复时改变吞吐覆盖：新 loader 应使用 batch=2/累积=1（按新配置
            # 会得到 2 步/epoch、总 4 步），但调度必须沿用 checkpoint 的曲线。
            resume_config = copy.deepcopy(config)
            resume_config["output_dir"] = str(root / "resume")
            resume_config["stages"]["A"].update(
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
                    "A", start_epoch=1,
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
                    for stage in "ABCD":
                        engine.run_stage(stage)
                full_directory = logger.directory
            self.assertGreater(scalars_spy.call_count, 0)
            self.assertGreaterEqual(event_spy.call_count, 4)
            self.assertGreaterEqual(flush_spy.call_count, 4)
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
            for stage in "ABC":
                batches = scoped(stage, "train_batch")
                self.assertEqual([record["step"] for record in batches], [1, 2, 4])
                self.assertEqual({record["epoch"] for record in batches}, {1})
                self.assertIn("loss", batches[0]["metrics"])
                self.assertIn("lr", batches[0]["metrics"])

                epoch_rows = scoped(stage, "train_epoch")
                self.assertEqual(
                    [(record["step"], record["epoch"]) for record in epoch_rows],
                    [(1, 1)],
                )
                self.assertIn("loss", epoch_rows[0]["metrics"])

                validation = scoped(stage, "val_epoch")
                self.assertEqual(
                    [(record["step"], record["epoch"]) for record in validation],
                    [(1, 1)],
                )
                self.assertIn("iou", validation[0]["metrics"])
                self.assertIn("positive_on_road_rate", validation[0]["metrics"])
                self.assertIn("heatmap_above_threshold_rate", validation[0]["metrics"])
                self.assertIn("quality_reward", validation[0]["metrics"])

                system = scoped(stage, "system")
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
            self.assertIn("image_id", details[0])
            self.assertIn("advantage", details[0])
            self.assertIn("positive_on_road_rate", details[0])
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
            self.assertEqual(events.count("stage_start"), 4)
            self.assertEqual(events.count("stage_end"), 4)
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
