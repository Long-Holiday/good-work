"""条件编码器与条件扩散 U-Net 的静态形状测试。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diffpeak.models.conditioning import ConditionEncoder  # noqa: E402
from diffpeak.models.unet import (  # noqa: E402
    ConditionalUNet,
    timestep_embedding,
)


class ConditionEncoderTest(unittest.TestCase):
    def test_output_shapes_and_pyramid(self) -> None:
        model = ConditionEncoder(
            condition_channels=(8, 16, 24, 32),
            sam_channels=16,
            auxiliary_classes=2,
        )
        model.eval()

        images = torch.randn(1, 3, 32, 32)
        embeddings = torch.randn(1, 16, 8, 8)

        with torch.no_grad():
            conditions, auxiliary_logits = model(images, embeddings)

        self.assertIsInstance(conditions, list)
        self.assertEqual(len(conditions), 4)
        expected = [(8, 8, 8), (16, 4, 4), (24, 2, 2), (32, 1, 1)]
        for feature, (channels, height, width) in zip(conditions, expected):
            self.assertEqual(
                tuple(feature.shape), (1, channels, height, width)
            )
        self.assertEqual(tuple(auxiliary_logits.shape), (1, 2, 8, 8))

    def test_invalid_inputs_raise(self) -> None:
        model = ConditionEncoder(sam_channels=8)
        with self.assertRaises(ValueError):
            model(torch.randn(1, 1, 32, 32), torch.randn(1, 8, 8, 8))
        with self.assertRaises(ValueError):
            model(torch.randn(2, 3, 32, 32), torch.randn(1, 8, 8, 8))
        with self.assertRaises(ValueError):
            model(torch.randn(1, 3, 32, 32), torch.randn(1, 8, 8))

    def test_requires_four_condition_scales(self) -> None:
        with self.assertRaises(ValueError):
            ConditionEncoder(condition_channels=(8, 16, 32))


class ConditionalUNetTest(unittest.TestCase):
    def _small_model(self) -> ConditionalUNet:
        return ConditionalUNet(
            in_channels=2,
            out_channels=2,
            base_channels=8,
            channel_mult=(1, 2, 4, 4),
            condition_channels=(4, 8, 16, 16),
            num_res_blocks=1,
            gradient_checkpointing=False,
        )

    def test_forward_shape_and_zero_initialized_output(self) -> None:
        model = self._small_model()
        model.eval()

        z = torch.randn(2, 2, 16, 16)
        timesteps = torch.tensor([0, 5])
        conditions = [
            torch.randn(2, 4, 16, 16),
            torch.randn(2, 8, 8, 8),
            torch.randn(2, 16, 4, 4),
            torch.randn(2, 16, 2, 2),
        ]

        with torch.no_grad():
            output = model(z, timesteps, conditions)

        self.assertEqual(tuple(output.shape), tuple(z.shape))
        # 输出卷积零初始化：未训练时噪声预测恒为零。
        self.assertTrue(torch.equal(output, torch.zeros_like(output)))

    def test_invalid_condition_count_raises(self) -> None:
        model = self._small_model()
        z = torch.randn(1, 2, 16, 16)
        with self.assertRaises(ValueError):
            model(z, 0, [torch.randn(1, 4, 16, 16)])

    def test_timestep_embedding_shape(self) -> None:
        timesteps = torch.tensor([0, 1, 999])
        even = timestep_embedding(timesteps, dim=8)
        odd = timestep_embedding(timesteps, dim=5)

        self.assertEqual(tuple(even.shape), (3, 8))
        self.assertEqual(tuple(odd.shape), (3, 5))
        self.assertTrue(torch.isfinite(even).all())
        self.assertTrue(torch.isfinite(odd).all())
        with self.assertRaises(ValueError):
            timestep_embedding(timesteps, dim=0)

    def test_as_batch_timesteps(self) -> None:
        device = torch.device("cpu")
        scalar = ConditionalUNet._as_batch_timesteps(7, 3, device)
        vector = ConditionalUNet._as_batch_timesteps(
            torch.tensor([1, 2, 3]), 3, device
        )
        single = ConditionalUNet._as_batch_timesteps(torch.tensor([4]), 3, device)

        self.assertTrue(torch.equal(scalar, torch.full((3,), 7)))
        self.assertTrue(torch.equal(vector, torch.tensor([1, 2, 3])))
        self.assertTrue(torch.equal(single, torch.full((3,), 4)))
        with self.assertRaises(ValueError):
            ConditionalUNet._as_batch_timesteps(torch.tensor([1, 2]), 3, device)


if __name__ == "__main__":
    unittest.main()
