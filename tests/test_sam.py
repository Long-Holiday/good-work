"""RoadSAM 的无需权重测试。

SAM 依赖在构建模型时才导入，因此这里可以在未安装 segment-anything 的环境中
测试纯张量逻辑：
* 变长点集的补齐与填充标签（-1）；
* 无正点样本返回常量负 logits（空掩码边界）；
* ``valid_sizes`` 校验（与推理几何约定一致）。
"""

from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diffpeak.models.sam import RoadSAM  # noqa: E402


class RoadSAMPointHandlingTest(unittest.TestCase):
    def _bare_road_sam(self):
        model = object.__new__(RoadSAM)
        model.no_positive_logit = -20.0
        return model

    def test_prepare_points_pads_with_not_a_point_labels(self) -> None:
        model = self._bare_road_sam()
        coords, labels = RoadSAM._prepare_points(
            model,
            [
                (torch.tensor([[10.0, 20.0], [30.0, 40.0]]), torch.tensor([1, 0])),
                (torch.empty(0, 2), torch.empty(0, dtype=torch.long)),
            ],
            batch_size=2,
            device=torch.device("cpu"),
        )

        self.assertEqual(tuple(coords.shape), (2, 2, 2))
        self.assertEqual(tuple(labels.shape), (2, 2))
        self.assertTrue(torch.equal(coords[0], torch.tensor([[10.0, 20.0], [30.0, 40.0]])))
        self.assertTrue(torch.equal(labels[0], torch.tensor([1, 0])))
        self.assertTrue(torch.equal(coords[1], torch.zeros(2, 2)))
        self.assertTrue(torch.equal(labels[1], torch.tensor([-1, -1])))

    def test_prepare_points_rejects_inconsistent_inputs(self) -> None:
        model = self._bare_road_sam()
        with self.assertRaises(ValueError):
            RoadSAM._prepare_points(
                model,
                [(torch.zeros(2, 2), torch.zeros(3, dtype=torch.long))],
                batch_size=1,
                device=torch.device("cpu"),
            )
        with self.assertRaises(ValueError):
            RoadSAM._prepare_points(
                model,
                [(torch.zeros(1, 2), torch.ones(1, dtype=torch.long))],
                batch_size=2,
                device=torch.device("cpu"),
            )
        with self.assertRaises(ValueError):
            RoadSAM._prepare_points(
                model,
                [torch.zeros(1, 2)],
                batch_size=1,
                device=torch.device("cpu"),
            )

    def test_empty_logits_are_constant_and_detached(self) -> None:
        model = self._bare_road_sam()
        embeddings = torch.zeros(2, 256, 4, 4)

        logits = RoadSAM._empty_logits(model, embeddings, (8, 6))

        self.assertEqual(tuple(logits.shape), (2, 1, 8, 6))
        self.assertTrue(torch.equal(logits, torch.full_like(logits, -20.0)))
        self.assertFalse(logits.requires_grad)

    def test_empty_logits_accept_explicit_dtype(self) -> None:
        model = self._bare_road_sam()
        embeddings = torch.zeros(1, 256, 4, 4)

        logits = RoadSAM._empty_logits(
            model, embeddings, (4, 4), dtype=torch.float16
        )

        self.assertEqual(logits.dtype, torch.float16)
        self.assertTrue(torch.equal(logits, torch.full_like(logits, -20.0)))

    def test_decode_without_positive_points_returns_empty_mask_logits(self) -> None:
        """没有任何正点时不应强行补全局最大点，而是返回空掩码常量。"""
        model = self._bare_road_sam()
        embeddings = torch.zeros(2, 256, 4, 4)
        points = [
            (torch.tensor([[5.0, 6.0]]), torch.tensor([0])),
            (torch.empty(0, 2), torch.empty(0, dtype=torch.long)),
        ]

        logits = RoadSAM.decode(model, embeddings, points, output_size=(16, 16))

        self.assertEqual(tuple(logits.shape), (2, 1, 16, 16))
        self.assertTrue(torch.equal(logits, torch.full_like(logits, -20.0)))


class RoadSAMValidSizesTest(unittest.TestCase):
    def test_accepts_single_pair_and_per_sample_sequence(self) -> None:
        single = RoadSAM._normalise_valid_sizes((512, 1024), 1, image_size=1024)
        batch = RoadSAM._normalise_valid_sizes(
            [(512, 1024), (1024, 256)], 2, image_size=1024
        )
        tensor = RoadSAM._normalise_valid_sizes(
            torch.tensor([[512, 1024], [1024, 256]]), 2, image_size=1024
        )

        self.assertEqual(single, [(512, 1024)])
        self.assertEqual(batch, [(512, 1024), (1024, 256)])
        self.assertEqual(tensor, [(512, 1024), (1024, 256)])

    def test_invalid_sizes_raise(self) -> None:
        with self.assertRaises(ValueError):
            RoadSAM._normalise_valid_sizes([(512, 1024)], 2, image_size=1024)
        with self.assertRaises(ValueError):
            RoadSAM._normalise_valid_sizes((0, 1024), 1, image_size=1024)
        with self.assertRaises(ValueError):
            RoadSAM._normalise_valid_sizes((1025, 1024), 1, image_size=1024)
        with self.assertRaises(TypeError):
            RoadSAM._normalise_valid_sizes(123, 1, image_size=1024)


class _CapturingEncoder:
    def __init__(self) -> None:
        self.input = None

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        self.input = x
        return x[:, :1]


class RoadSAMEncodePaddingTest(unittest.TestCase):
    """padding 必须在标准化之后置零，而不是变成 (-mean/std)。"""

    def _bare_encoder(self):
        model = object.__new__(RoadSAM)
        model.pixel_mean = torch.tensor([1.0, 2.0, 3.0]).view(1, 3, 1, 1)
        model.pixel_std = torch.ones(1, 3, 1, 1)
        encoder = _CapturingEncoder()
        model.sam = types.SimpleNamespace(image_encoder=encoder)
        return model, encoder

    def test_normalized_padding_is_exactly_zero(self) -> None:
        model, encoder = self._bare_encoder()
        images = torch.ones(1, 3, 1024, 1024)

        model.encode(images, valid_sizes=[(512, 768)])

        normalized = (255.0 - model.pixel_mean[0]) / model.pixel_std[0]
        seen = encoder.input
        self.assertIsNotNone(seen)
        self.assertTrue(
            torch.allclose(seen[0, :, :512, :768], normalized, atol=1e-6)
        )
        self.assertTrue(torch.equal(seen[0, :, 512:, :], torch.zeros(3, 512, 1024)))
        self.assertTrue(torch.equal(seen[0, :, :, 768:], torch.zeros(3, 1024, 256)))

    def test_encode_rejects_wrong_input_shape(self) -> None:
        model, _ = self._bare_encoder()
        with self.assertRaises(ValueError):
            model.encode(torch.ones(1, 3, 512, 512))
        with self.assertRaises(TypeError):
            model.encode(torch.ones(1, 3, 1024, 1024, dtype=torch.uint8))


if __name__ == "__main__":
    unittest.main()
