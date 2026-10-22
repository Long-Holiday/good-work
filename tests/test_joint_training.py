"""Regression checks for the unified loss and supported training entry points."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from train import _parse_stages
from tests import test_engine_pipeline as pipeline
from tests.test_engine_pipeline import _build_tiny_models


class JointTrainingTest(unittest.TestCase):
    def test_only_two_public_stages(self):
        self.assertEqual(_parse_stages("all"), ["ABC", "D"])
        self.assertEqual(_parse_stages("D,ABC"), ["ABC", "D"])
        for stage in ("A", "B", "C"):
            with self.assertRaises(ValueError):
                _parse_stages(stage)

    def test_segmentation_alone_reaches_all_trainable_modules(self):
        fixture = pipeline.EnginePipelineTest()
        with tempfile.TemporaryDirectory() as temporary, patch(
            "diffpeak.engine.build_models", side_effect=_build_tiny_models
        ):
            config = fixture._make_config(Path(temporary))
            engine = fixture._engine(config)
            engine._set_requires_grad("ABC")
            engine._set_train_modes("ABC")
            batch = engine._prepare_batch(next(iter(engine._build_train_loader(0))))
            # DiT's zero output head blocks upstream derivatives on the very first
            # forward. Take one ordinary joint update before testing the pathway.
            optimizer = torch.optim.AdamW(engine._trainable_parameters(), lr=1e-3)
            engine._supervised_loss("ABC", batch, 0)["loss"].backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            losses = engine._supervised_loss("ABC", batch, 1)
            expected = (
                losses["diff"] + engine.heat_weight * losses["heat"]
                + engine.aux_weight * losses["aux"] + losses["seg"] + losses["sampled_seg"]
            )
            torch.testing.assert_close(losses["loss"], expected)
            # Check cross-module coupling, not just gradients from independent losses.
            (losses["seg"] + losses["sampled_seg"]).backward()
            for module in (engine.condition_encoder, engine.dit, engine.sam.sam.mask_decoder):
                gradients = [p.grad for p in module.parameters() if p.grad is not None]
                self.assertTrue(gradients)
                self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
                self.assertGreater(sum(float(g.abs().sum()) for g in gradients), 0)
            self.assertTrue(all(p.grad is None for p in engine.sam.sam.image_encoder.parameters()))

    def test_legacy_resume_is_rejected_before_optimizer_loading(self):
        fixture = pipeline.EnginePipelineTest()
        with tempfile.TemporaryDirectory() as temporary, patch(
            "diffpeak.engine.build_models", side_effect=_build_tiny_models
        ):
            engine = fixture._engine(fixture._make_config(Path(temporary)))
            for stage in ("A", "B", "C", "D"):
                with self.assertRaisesRegex(ValueError, "--init"):
                    engine._validate_resume_algorithm({"stage": stage, "format_version": 3})
