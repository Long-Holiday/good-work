"""Regression checks for the signed-ridge joint ABC loss and training entry points."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from diffpeak.engine import DiffPeakEngine, encode_images
from train import _parse_stages
from tests import test_engine_pipeline as pipeline
from tests.test_engine_pipeline import PROTOCOL, _build_tiny_models


def _valid_payload(stage="ABC"):
    payload = {
        "stage": stage,
        "format_version": 5,
        "training_protocol": PROTOCOL,
        "target_representation": PROTOCOL,
        "sam_frozen": True,
        "latent_channels": 1,
        "prompt": {"scale": 1.0, "terminal_mapping": "clamp"},
        # 仅扩散轨迹的 Gaussian GRPO 快照版本。
        "algorithm": {"name": "grpo", "version": 3},
    }
    return payload


class JointTrainingTest(unittest.TestCase):
    def test_only_two_public_stages(self):
        self.assertEqual(_parse_stages("all"), ["ABC", "D"])
        self.assertEqual(_parse_stages("D,ABC"), ["ABC", "D"])
        for stage in ("A", "B", "C"):
            with self.assertRaises(ValueError):
                _parse_stages(stage)

    def test_batch_carries_single_channel_signed_field_and_regions(self):
        fixture = pipeline.EnginePipelineTest()
        with tempfile.TemporaryDirectory() as temporary, patch(
            "diffpeak.engine.build_models", side_effect=_build_tiny_models
        ):
            engine = fixture._engine(
                fixture._make_config(Path(temporary))
            )
            batch = engine._prepare_batch(
                next(iter(engine._build_train_loader(0)))
            )

            heatmaps = batch["heatmaps"]
            valid = batch["field_valid_mask"]
            regions = batch["field_regions"]

            self.assertEqual(tuple(heatmaps.shape[1:2]), (1,))
            self.assertEqual(tuple(valid.shape), tuple(heatmaps.shape))
            self.assertEqual(int(regions.shape[1]), 4)
            self.assertEqual(
                tuple(regions.shape[-2:]), tuple(heatmaps.shape[-2:])
            )
            self.assertEqual(regions.shape[0], heatmaps.shape[0])
            # 直接以有符号山脊场扩散：不经过旧的 [0, 1] 仿射目标变换。
            self.assertGreaterEqual(float(heatmaps.min()), -1.0)
            self.assertLessEqual(float(heatmaps.max()), 1.0)
            valid_bool = valid > 0.5
            self.assertTrue(bool(valid_bool.any()))
            # 道路中心为正、有效背景为负，且四个区域互斥并落在有效区内。
            self.assertTrue(bool((heatmaps[valid_bool] > 0).any()))
            self.assertTrue(bool((heatmaps[valid_bool] < 0).any()))
            regions_bool = regions > 0.5
            self.assertTrue(bool((regions_bool.sum(dim=1) <= 1).all()))
            self.assertFalse(
                bool((regions_bool & ~valid_bool.unsqueeze(1)).any())
            )

    def test_supervised_loss_passes_valid_mask_and_regions(self):
        fixture = pipeline.EnginePipelineTest()
        with tempfile.TemporaryDirectory() as temporary, patch(
            "diffpeak.engine.build_models", side_effect=_build_tiny_models
        ):
            config = fixture._make_config(Path(temporary))
            engine = fixture._engine(config)
            batch = engine._prepare_batch(
                next(iter(engine._build_train_loader(0)))
            )
            captured = {}
            original = engine.diffusion.training_losses

            def recording_losses(*args, **kwargs):
                captured["args"] = args
                captured["kwargs"] = kwargs
                return original(*args, **kwargs)

            with patch.object(
                engine.diffusion, "training_losses", side_effect=recording_losses
            ):
                losses = engine._supervised_loss("ABC", batch, 0)

            kwargs = captured["kwargs"]
            self.assertIn("valid_mask", kwargs)
            self.assertIn("field_regions", kwargs)
            self.assertEqual(
                tuple(kwargs["valid_mask"].shape), tuple(batch["heatmaps"].shape)
            )
            self.assertEqual(
                tuple(kwargs["field_regions"].shape),
                tuple(batch["field_regions"].shape),
            )
            self.assertEqual(set(losses), {"loss", "diff", "heat", "aux"})
            self.assertFalse(kwargs.get("return_prediction", False))
            expected = (
                losses["diff"]
                + engine.heat_weight * losses["heat"]
                + engine.aux_weight * losses["aux"]
            )
            torch.testing.assert_close(losses["loss"], expected)

    def test_field_loss_reaches_trainable_modules_without_sam_decoding(self):
        fixture = pipeline.EnginePipelineTest()
        with tempfile.TemporaryDirectory() as temporary, patch(
            "diffpeak.engine.build_models", side_effect=_build_tiny_models
        ):
            config = fixture._make_config(Path(temporary))
            engine = fixture._engine(config)
            engine._set_requires_grad("ABC")
            engine._set_train_modes("ABC")
            batch = engine._prepare_batch(
                next(iter(engine._build_train_loader(0)))
            )
            # DiT's zero output head blocks upstream derivatives on the very first
            # forward. Take one ordinary joint update before testing the pathway.
            sam_before = {k: v.clone() for k, v in engine.sam.state_dict().items()}
            self.assertFalse(engine.sam.training)
            self.assertTrue(all(not p.requires_grad for p in engine.sam.parameters()))
            trainable = engine._trainable_parameters()
            self.assertFalse(
                {id(p) for p in engine.sam.parameters()}
                & {id(p) for p in trainable}
            )
            optimizer = torch.optim.AdamW(trainable, lr=1e-3)
            def frozen_encode(*args, **kwargs):
                self.assertFalse(torch.is_grad_enabled())
                return encode_images(*args, **kwargs)

            with patch("diffpeak.engine.encode_images", side_effect=frozen_encode) as encode, patch.object(
                engine.sam, "decode_heatmaps", side_effect=AssertionError("ABC must not decode SAM masks")
            ), patch.object(
                engine.diffusion, "sample_differentiable",
                side_effect=AssertionError("ABC must not sample full trajectories"),
            ):
                engine._supervised_loss("ABC", batch, 0)["loss"].backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                losses = engine._supervised_loss("ABC", batch, 1)
                # Diffusion supervision alone must reach the condition encoder too.
                (losses["diff"] + engine.heat_weight * losses["heat"]).backward()
                self.assertEqual(encode.call_count, 2)
            for module in (engine.condition_encoder, engine.dit):
                gradients = [p.grad for p in module.parameters() if p.grad is not None]
                self.assertTrue(gradients)
                self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
                self.assertGreater(sum(float(g.abs().sum()) for g in gradients), 0)
            optimizer.step()
            self.assertTrue(all(p.grad is None for p in engine.sam.parameters()))
            for name, value in engine.sam.state_dict().items():
                torch.testing.assert_close(value, sam_before[name], rtol=0, atol=0)

    def test_old_segmentation_objective_cannot_resume(self):
        engine = object.__new__(DiffPeakEngine)
        engine.config = {"training_protocol": PROTOCOL, "target_representation": PROTOCOL}
        for stage in ("ABC", "D"):
            for key in ("seg_weight", "sampled_seg_weight"):
                payload = _valid_payload(stage)
                payload["config"] = {"train": {key: 1.0}}
                with self.assertRaisesRegex(ValueError, "旧 ABC 分割监督目标"):
                    engine._validate_resume_algorithm(payload)

    def test_old_loss_options_are_removed_from_saved_config(self):
        fixture = pipeline.EnginePipelineTest()
        with tempfile.TemporaryDirectory() as temporary, patch(
            "diffpeak.engine.build_models", side_effect=_build_tiny_models
        ):
            config = fixture._make_config(Path(temporary))
            legacy = {"seg_weight": 1.0, "sampled_seg_weight": 1.0, "cldice_weight": 0.2}
            config["train"].update(legacy)
            with self.assertWarnsRegex(RuntimeWarning, "ABC 已移除 SAM 分割监督"):
                engine = fixture._engine(config)
            for key in legacy:
                self.assertNotIn(key, engine.train_cfg)
                self.assertNotIn(key, engine._resolved_config()["train"])
                self.assertIn(key, config["train"])

    def test_legacy_protocols_are_rejected_before_optimizer_loading(self):
        engine = object.__new__(DiffPeakEngine)
        engine.config = {
            "training_protocol": PROTOCOL,
            "target_representation": PROTOCOL,
        }
        # 新协议完整快照可以恢复。
        engine._validate_resume_algorithm(_valid_payload("ABC"))
        engine._validate_resume_algorithm(_valid_payload("D"))

        # 旧联合 ABC 双通道/SAM 可训练协议不能直接 --resume。
        legacy = _valid_payload("ABC")
        legacy["training_protocol"] = "joint_abc_v1"
        legacy["target_representation"] = "joint_abc_v1"
        legacy["format_version"] = 4
        with self.assertRaises(ValueError):
            engine._validate_resume_algorithm(legacy)

        # 协议标记正确但缺少全冻结声明同样拒绝。
        unfrozen = _valid_payload("ABC")
        unfrozen["sam_frozen"] = False
        with self.assertRaises(ValueError):
            engine._validate_resume_algorithm(unfrozen)

        stale_representation = _valid_payload("ABC")
        stale_representation["target_representation"] = "peaks_v1"
        with self.assertRaises(ValueError):
            engine._validate_resume_algorithm(stale_representation)

        # 旧 DDPO 快照的算法状态语义与 GRPO 不同。
        with self.assertRaises(ValueError):
            engine._validate_resume_algorithm(
                {"stage": "D", "format_version": 1}
            )


if __name__ == "__main__":
    unittest.main()
