"""High-resolution conditioning stays trainable in ABC and D with SAM frozen."""
from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from tests import test_engine_pipeline as pipeline
from tests.test_engine_pipeline import _build_tiny_models


class HighResPipelineTest(unittest.TestCase):
    def test_fusion_trains_in_abc_and_d_while_sam_stays_frozen(self):
        fixture = pipeline.EnginePipelineTest()
        with tempfile.TemporaryDirectory() as temporary, patch(
            "diffpeak.engine.build_models", side_effect=_build_tiny_models
        ):
            config = fixture._make_config(Path(temporary))
            # Anchor guarantees a signal even if this tiny group's rewards coincide.
            config["rl"]["anchor_weight"] = 0.1
            engine = fixture._engine(config)
            sam_before = copy.deepcopy(engine.sam.state_dict())

            for stage in ("ABC", "D"):
                with self.subTest(stage=stage):
                    before = {
                        name: value.detach().clone()
                        for name, value in engine.condition_encoder.named_parameters()
                        if name.startswith("high_res_")
                    }
                    self.assertTrue(before)
                    metrics = engine.run_stage(stage)
                    self.assertTrue(torch.isfinite(torch.tensor(metrics["iou"])))
                    after = dict(engine.condition_encoder.named_parameters())
                    self.assertTrue(
                        any(
                            not torch.equal(before[name], after[name])
                            for name in before
                        )
                    )
                    self.assertTrue(
                        all(after[name].requires_grad for name in before)
                    )
                    # 高分辨率投影属于条件编码器，不属于 SAM 解码器。
                    self.assertTrue(
                        all(
                            not p.requires_grad for p in engine.sam.parameters()
                        )
                    )
                    self.assertFalse(engine.sam.training)
                    for name, value in engine.sam.state_dict().items():
                        torch.testing.assert_close(
                            value, sam_before[name], rtol=0, atol=0
                        )


if __name__ == "__main__":
    unittest.main()
