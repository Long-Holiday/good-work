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
from diffpeak.inference import DEFAULT_CONFIG, read_checkpoint
from diffpeak.models import ConditionEncoder, ConditionalUNet, GaussianDiffusion
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
        return self.sam.image_encoder(F.avg_pool2d(images, 4))

    def decode(self, embeddings, points, output_size):
        logits = self.sam.mask_decoder(embeddings)
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


class EnginePipelineTest(unittest.TestCase):
    def _make_config(self, root):
        data = root / "data"
        (data / "train").mkdir(parents=True)
        entries = []
        for index in range(2):
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
        config["train"].update(amp=False, batch_size=2, accumulation_steps=1, val_limit=1, log_every=100)
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
                self.assertEqual(payload["format_version"], 2)
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


if __name__ == "__main__":
    unittest.main()
