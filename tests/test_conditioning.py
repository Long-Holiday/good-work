"""条件编码器与时间步编码的静态形状测试。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diffpeak.models.conditioning import ConditionEncoder  # noqa: E402
from diffpeak.models.sam import SAM2ImageFeatures  # noqa: E402
from diffpeak.models.dit import timestep_embedding  # noqa: E402


class ConditionEncoderTest(unittest.TestCase):
    def test_output_shapes_and_pyramid(self) -> None:
        model = ConditionEncoder(
            condition_channels=(8, 16, 24, 32),
            sam_channels=16,
            auxiliary_classes=2,
        )
        model.eval()

        images = torch.randn(1, 3, 32, 32)
        embeddings = SAM2ImageFeatures(
            torch.randn(1, 16, 8, 8),
            (torch.randn(1, 16, 8, 8), torch.randn(1, 16, 4, 4)),
        )

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
        high = (torch.randn(1, 8, 8, 8), torch.randn(1, 8, 4, 4))
        with self.assertRaises(ValueError):
            model(torch.randn(1, 1, 32, 32), SAM2ImageFeatures(high[0], high))
        with self.assertRaises(ValueError):
            model(torch.randn(2, 3, 32, 32), SAM2ImageFeatures(high[0], high))
        with self.assertRaises(ValueError):
            model(
                torch.randn(1, 3, 32, 32),
                SAM2ImageFeatures(torch.randn(1, 8, 8), high),
            )
        with self.assertRaises(TypeError):
            model(torch.randn(1, 3, 32, 32), torch.randn(1, 8, 8, 8))

    def test_requires_four_condition_scales(self) -> None:
        with self.assertRaises(ValueError):
            ConditionEncoder(condition_channels=(8, 16, 32))


class ConditionEncoderHighResTest(unittest.TestCase):
    """SAM2 原始 FPN 高分辨率特征的融合契约。"""

    def _model(self) -> ConditionEncoder:
        return ConditionEncoder(
            condition_channels=(8, 16, 24, 32),
            sam_channels=16,
        )

    def _features(self, batch: int = 1) -> SAM2ImageFeatures:
        return SAM2ImageFeatures(
            torch.randn(batch, 16, 8, 8),
            (torch.randn(batch, 16, 8, 8), torch.randn(batch, 16, 4, 4)),
        )

    def test_high_res_forward_shapes_and_fusion_effect(self) -> None:
        model = self._model()
        model.eval()
        images = torch.randn(1, 3, 32, 32)
        features = self._features()

        with torch.no_grad():
            conditions, auxiliary_logits = model(images, features)
            changed_256, _ = model(
                images,
                SAM2ImageFeatures(
                    features.low_res, (features.high_res[0] + 1.0, features.high_res[1])
                ),
            )
            changed_128, _ = model(
                images,
                SAM2ImageFeatures(
                    features.low_res, (features.high_res[0], features.high_res[1] + 1.0)
                ),
            )

        expected = [(8, 8, 8), (16, 4, 4), (24, 2, 2), (32, 1, 1)]
        for feature, (channels, height, width) in zip(conditions, expected):
            self.assertEqual(tuple(feature.shape), (1, channels, height, width))
        self.assertEqual(tuple(auxiliary_logits.shape), (1, 2, 8, 8))

        # high_res[0] 融合进 C256 并随金字塔下传；high_res[1] 从 C128
        # 开始生效，不改变 C256。
        self.assertFalse(torch.allclose(conditions[0], changed_256[0]))
        self.assertFalse(torch.allclose(conditions[1], changed_256[1]))
        self.assertTrue(torch.allclose(conditions[0], changed_128[0]))
        self.assertFalse(torch.allclose(conditions[1], changed_128[1]))

    def test_high_res_projections_receive_gradients(self) -> None:
        model = self._model()
        conditions, auxiliary_logits = model(
            torch.randn(1, 3, 32, 32), self._features()
        )
        (conditions[0].mean() + conditions[1].mean() + auxiliary_logits.mean()).backward()

        for name in (
            "high_res_project_256.weight",
            "high_res_project_128.weight",
            "high_res_fuse_256.conv.weight",
            "high_res_fuse_128.conv.weight",
        ):
            gradient = dict(model.named_parameters())[name].grad
            self.assertIsNotNone(gradient, name)
            self.assertTrue(bool(torch.count_nonzero(gradient)), name)

    def test_state_dict_strict_roundtrip_and_rejects_incomplete_state(self) -> None:
        source = self._model()
        target = self._model()
        target.load_state_dict(source.state_dict(), strict=True)

        # 旧快照缺少高分辨率投影参数，strict 加载必须报错。
        incomplete = {
            key: value
            for key, value in source.state_dict().items()
            if not key.startswith("high_res_")
        }
        self.assertLess(len(incomplete), len(source.state_dict()))
        with self.assertRaises(RuntimeError):
            target.load_state_dict(incomplete, strict=True)

    def test_requires_sam2imagefeatures_with_two_level_high_res(self) -> None:
        model = self._model()
        images = torch.randn(1, 3, 32, 32)

        with self.assertRaises(TypeError):
            model(images, torch.randn(1, 16, 8, 8))
        with self.assertRaises(ValueError) as caught:
            model(images, SAM2ImageFeatures(torch.randn(1, 16, 8, 8), None))
        self.assertIn("high_res", str(caught.exception))
        with self.assertRaises(ValueError):
            model(
                images,
                SAM2ImageFeatures(
                    torch.randn(1, 16, 8, 8), (torch.randn(1, 16, 8, 8),)
                ),
            )

    def test_high_res_validation_rejects_wrong_batch_channels_or_order(self) -> None:
        model = self._model()
        images = torch.randn(2, 3, 32, 32)
        low = torch.randn(2, 16, 8, 8)

        cases = {
            "batch": (torch.randn(1, 16, 8, 8), torch.randn(1, 16, 4, 4)),
            "channels": (torch.randn(2, 8, 8, 8), torch.randn(2, 16, 4, 4)),
            "order": (torch.randn(2, 16, 4, 4), torch.randn(2, 16, 8, 8)),
        }
        for name, high in cases.items():
            with self.subTest(name=name), self.assertRaises(ValueError):
                model(images, SAM2ImageFeatures(low, high))


class TimestepEmbeddingTest(unittest.TestCase):
    def test_shape_and_finite_values(self) -> None:
        timesteps = torch.tensor([0, 1, 999])
        for dim in (1, 5, 8):
            embedding = timestep_embedding(timesteps, dim=dim)
            self.assertEqual(tuple(embedding.shape), (3, dim))
            self.assertTrue(torch.isfinite(embedding).all())
        with self.assertRaises(ValueError):
            timestep_embedding(timesteps, dim=0)


if __name__ == "__main__":
    unittest.main()
