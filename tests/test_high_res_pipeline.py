"""High-resolution conditioning through the real A–D engine with a tiny SAM."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from tests import test_engine_pipeline as pipeline
from tests.test_engine_pipeline import _build_tiny_models


class HighResPipelineTest(unittest.TestCase):
    def test_fusion_trains_in_b_and_is_frozen_in_c_d(self):
        fixture = pipeline.EnginePipelineTest()
        with tempfile.TemporaryDirectory() as temporary, patch(
            "diffpeak.engine.build_models", side_effect=_build_tiny_models
        ):
            config = fixture._make_config(Path(temporary))
            engine = fixture._engine(config)
            engine.run_stage("A")
            before = {
                name: value.detach().clone()
                for name, value in engine.condition_encoder.named_parameters()
                if name.startswith("high_res_")
            }
            self.assertTrue(before)
            engine.run_stage("B")
            after = {
                name: value.detach().clone()
                for name, value in engine.condition_encoder.named_parameters()
                if name.startswith("high_res_")
            }
            self.assertTrue(any(not torch.equal(before[name], value) for name, value in after.items()))
            for stage in ("C", "D"):
                metrics = engine.run_stage(stage)
                self.assertIn("positive_on_road_rate", metrics)
                for name, value in engine.condition_encoder.named_parameters():
                    if name in after:
                        torch.testing.assert_close(value, after[name], rtol=0, atol=0)
                        self.assertFalse(value.requires_grad)
            self.assertTrue(all(not p.requires_grad for p in engine.sam.parameters()))
