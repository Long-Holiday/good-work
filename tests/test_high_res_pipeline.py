"""High-resolution conditioning is trainable throughout ABC and D."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from tests import test_engine_pipeline as pipeline
from tests.test_engine_pipeline import _build_tiny_models


class HighResPipelineTest(unittest.TestCase):
    def test_fusion_trains_in_joint_pretraining_and_posttraining(self):
        fixture = pipeline.EnginePipelineTest()
        with tempfile.TemporaryDirectory() as temporary, patch(
            "diffpeak.engine.build_models", side_effect=_build_tiny_models
        ):
            config = fixture._make_config(Path(temporary))
            # Anchor guarantees a signal even if this tiny group's rewards coincide.
            config["rl"]["anchor_weight"] = 0.1
            engine = fixture._engine(config)
            for stage in ("ABC", "D"):
                before = {
                    name: value.detach().clone()
                    for name, value in engine.condition_encoder.named_parameters()
                    if name.startswith("high_res_")
                }
                self.assertTrue(before)
                metrics = engine.run_stage(stage)
                self.assertIn("positive_on_road_rate", metrics)
                after = dict(engine.condition_encoder.named_parameters())
                self.assertTrue(any(not torch.equal(before[name], after[name]) for name in before))
                self.assertTrue(all(after[name].requires_grad for name in before))
            self.assertTrue(all(p.requires_grad for p in engine.sam.sam.mask_decoder.parameters()))
            self.assertTrue(all(not p.requires_grad for p in engine.sam.sam.image_encoder.parameters()))
