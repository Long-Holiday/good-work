"""RoadSAM 的无需权重测试。

SAM2 依赖在构建模型时才导入；本文件用轻量替身测试纯张量逻辑：
* 变长点集的补齐与填充标签（-1）；
* 无正点样本返回常量负 logits（空掩码边界）；
* ``valid_sizes`` 校验（与推理几何约定一致）。
"""

from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diffpeak.models.sam import RoadSAM, SAM2ImageFeatures  # noqa: E402


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
    def test_per_sample_none_means_full_image(self) -> None:
        self.assertEqual(
            RoadSAM._normalise_valid_sizes([None, (512, 1024)], 2, image_size=1024),
            [(1024, 1024), (512, 1024)],
        )
        self.assertEqual(
            RoadSAM._normalise_valid_sizes([None], 1, image_size=64),
            [(64, 64)],
        )

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


class RoadSAMValidRegionMaskTest(unittest.TestCase):
    """SAM padding 掩码必须与山脊场数据侧共享同一像素中心判据。"""

    def test_matches_ridge_valid_heatmap_mask_on_odd_rectangles(self) -> None:
        import numpy as np

        from diffpeak.data.geometry import ImageGeometry
        from diffpeak.data.ridge import _valid_heatmap_mask

        cases = [
            (64, 16, (10, 38)),  # 10/64*16=2.5：banker's round 会得到 2。
            (64, 16, (38, 10)),
            (64, 16, (37, 37)),
            (1024, 256, (512, 300)),
            (1024, 256, (1, 1)),
            (68, 17, (33, 9)),
        ]
        for image_size, heatmap_size, resized in cases:
            with self.subTest(image_size=image_size, resized=resized):
                geometry = ImageGeometry(
                    source_size=resized,
                    resized_size=resized,
                    image_size=image_size,
                    heatmap_size=heatmap_size,
                )
                expected = _valid_heatmap_mask(geometry)
                actual = RoadSAM._valid_region_mask(
                    [resized],
                    1,
                    (heatmap_size, heatmap_size),
                    (image_size, image_size),
                    torch.device("cpu"),
                    torch.float32,
                )
                np.testing.assert_array_equal(
                    actual[0, 0].to(torch.bool).numpy(), expected
                )

    def test_center_criterion_handles_ties_and_zero_cell_regions(self) -> None:
        # 16 热图 -> 64 画布，valid=10：中心判据保留 3 个像素，
        # round(10 * 16 / 64)=round(2.5) 会错误地得到 2。
        tie = RoadSAM._valid_region_mask(
            [(10, 10)], 1, (16, 16), (64, 64), torch.device("cpu"), torch.float32
        )
        self.assertEqual(float(tie.sum()), 9.0)
        self.assertTrue(torch.all(tie[0, 0, :3, :3] == 1.0))
        self.assertTrue(torch.all(tie[0, 0, 3:, :] == 0.0))

        # 极细有效区域可能映射到零个目标像素；不得强制至少保留一个。
        zero = RoadSAM._valid_region_mask(
            [(1, 1)], 1, (4, 4), (64, 64), torch.device("cpu"), torch.float32
        )
        self.assertEqual(float(zero.sum()), 0.0)

        # 非方形画布也使用同一判据：field 8x6, output 16x12。
        rectangle = RoadSAM._valid_region_mask(
            [(5, 10)], 1, (8, 6), (16, 12), torch.device("cpu"), torch.float32
        )
        self.assertTrue(torch.all(rectangle[0, 0, :3, :5] == 1.0))
        self.assertTrue(torch.all(rectangle[0, 0, 3:, :] == 0.0))
        self.assertTrue(torch.all(rectangle[0, 0, :, 5:] == 0.0))


class RoadSAMCheckpointErrorTest(unittest.TestCase):
    def test_legacy_state_dict_without_model_key_raises_value_error(self) -> None:
        """旧 SAM1 原始 state dict 缺少 ``model`` 字段时应给出明确错误。"""
        fake_sam2 = types.ModuleType("sam2")
        fake_sam2.__path__ = []  # type: ignore[attr-defined]
        fake_build = types.ModuleType("sam2.build_sam")

        def _raise_key_error(**kwargs):
            raise KeyError("model")

        fake_build.build_sam2 = _raise_key_error  # type: ignore[attr-defined]
        with mock.patch.dict(
            sys.modules, {"sam2": fake_sam2, "sam2.build_sam": fake_build}
        ):
            with self.assertRaises(ValueError) as caught:
                RoadSAM(checkpoint="checkpoints/sam_vit_b_01ec64.pth")

        message = str(caught.exception)
        self.assertIn("SAM1", message)
        self.assertIn("SAM2", message)


class _CapturingEncoder:
    def __init__(self) -> None:
        self.input = None

    def __call__(self, x: torch.Tensor) -> dict:
        self.input = x
        return {"vision_features": x[:, :1]}


class _FakeSAM2Encoder:
    """返回固定形状的 SAM2 FPN 输出，不运行真实 Hiera。"""

    def __init__(self) -> None:
        self.channels = 5
        self.vision = None
        self.low = torch.arange(2 * 5 * 2 * 3, dtype=torch.float32).reshape(2, 5, 2, 3)
        self.high = (
            torch.arange(2 * 5 * 8 * 8, dtype=torch.float32).reshape(2, 5, 8, 8),
            torch.arange(2 * 5 * 4 * 4, dtype=torch.float32).reshape(2, 5, 4, 4),
        )

    def __call__(self, x: torch.Tensor) -> dict:
        self.vision = x
        return {
            "vision_features": self.low,
            "backbone_fpn": list(self.high),
        }


class RoadSAMEncodeSAM2FeaturesTest(unittest.TestCase):
    """CPU 假 SAM2：检查 no_mem_embed 广播与原始高分辨率特征契约。"""

    def _bare_model(self):
        model = object.__new__(RoadSAM)
        model.pixel_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        model.pixel_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        encoder = _FakeSAM2Encoder()
        # 官方 no_mem_embed 形状为 (1, 1, C)，必须按通道广播。
        model.sam = types.SimpleNamespace(
            image_encoder=encoder,
            directly_add_no_mem_embed=True,
            use_high_res_features_in_sam=True,
            no_mem_embed=torch.nn.Parameter(
                torch.arange(encoder.channels, dtype=torch.float32).view(1, 1, -1)
            ),
        )
        return model, encoder

    def test_no_mem_embed_broadcasts_over_channels_not_spatial_dims(self) -> None:
        model, encoder = self._bare_model()

        features = model.encode(torch.zeros(2, 3, 1024, 1024))

        self.assertEqual(tuple(features.low_res.shape), (2, 5, 2, 3))
        expected = encoder.low + model.sam.no_mem_embed.view(1, 5, 1, 1)
        self.assertTrue(torch.equal(features.low_res, expected))
        self.assertFalse(features.low_res.requires_grad)

    def test_high_res_features_are_raw_fpn_levels_and_detached(self) -> None:
        model, encoder = self._bare_model()

        features = model.encode(torch.zeros(2, 3, 1024, 1024))

        self.assertIsNotNone(features.high_res)
        for actual, expected in zip(features.high_res, encoder.high):
            self.assertTrue(torch.equal(actual, expected))
            self.assertFalse(actual.requires_grad)


class RoadSAMEncodePaddingTest(unittest.TestCase):
    """padding 必须在标准化之后置零，而不是变成 (-mean/std)。"""

    def _bare_encoder(self):
        model = object.__new__(RoadSAM)
        model.pixel_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        model.pixel_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        encoder = _CapturingEncoder()
        model.sam = types.SimpleNamespace(
            image_encoder=encoder,
            directly_add_no_mem_embed=False,
            use_high_res_features_in_sam=False,
        )
        return model, encoder

    def test_normalized_padding_is_exactly_zero(self) -> None:
        model, encoder = self._bare_encoder()
        images = torch.ones(1, 3, 1024, 1024)

        model.encode(images, valid_sizes=[(512, 768)])

        normalized = (1.0 - model.pixel_mean[0]) / model.pixel_std[0]
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


class _FakeSAM2ImageEncoder(nn.Module):
    """返回固定 SAM2 形状的原始 FPN 特征，不运行真实 Hiera。"""

    def __init__(self) -> None:
        super().__init__()
        self.input = None

    def forward(self, x: torch.Tensor) -> dict:
        self.input = x
        batch = x.shape[0]
        return {
            "vision_features": torch.full((batch, 256, 4, 4), 2.0),
            "backbone_fpn": [
                torch.full((batch, 256, 8, 8), 3.0),
                torch.full((batch, 256, 4, 4), 4.0),
            ],
        }


class _FakeSAM2PromptEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # 官方 PromptEncoder 的掩码输入是其图像嵌入网格的 4 倍。
        self.mask_input_size = (8, 8)
        self.last_masks = None

    def get_dense_pe(self) -> torch.Tensor:
        return torch.zeros(1, 256, 4, 4)

    def forward(self, points=None, boxes=None, masks=None):
        if masks is not None:
            self.last_masks = masks
            batch = masks.shape[0]
            # 让 dense embedding 直接依赖掩码提示，验证输入侧梯度路径。
            dense = masks.repeat(1, 256, 1, 1)
            return torch.zeros(batch, 0, 256), dense
        coords, _ = points
        batch = coords.shape[0]
        return (
            torch.zeros(batch, 1, 256),
            torch.zeros(batch, 256, 4, 4),
        )


class _FakeSAM2MaskDecoder(nn.Module):
    """记录调用参数，并让输出依赖 conv_s0/conv_s1 以验证梯度路径。"""

    def __init__(self) -> None:
        super().__init__()
        self.conv_s0 = nn.Conv2d(256, 32, 1)
        self.conv_s1 = nn.Conv2d(256, 64, 1)
        self.calls = []

    def forward(
        self,
        image_embeddings,
        image_pe,
        sparse_prompt_embeddings,
        dense_prompt_embeddings,
        multimask_output,
        repeat_image,
        high_res_features=None,
    ):
        self.calls.append(
            {
                "multimask_output": bool(multimask_output),
                "repeat_image": bool(repeat_image),
                "sparse_tokens": tuple(sparse_prompt_embeddings.shape),
                "high_res_shapes": (
                    None
                    if high_res_features is None
                    else [tuple(item.shape) for item in high_res_features]
                ),
            }
        )
        mask = image_embeddings[:, :1]
        if dense_prompt_embeddings is not None:
            # 让输出依赖 dense 提示，便于验证热图 -> 冻结提示编码器 -> 解码器。
            mask = mask + F.interpolate(
                dense_prompt_embeddings[:, :1],
                size=mask.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        if high_res_features is not None:
            for level in high_res_features:
                mask = mask + F.interpolate(
                    level[:, :1],
                    size=mask.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
        batch = image_embeddings.shape[0]
        return (
            mask,
            torch.ones(batch, 1),
            torch.zeros(batch, 1, 256),
            torch.zeros(batch, 1),
        )


class _FakeSAM2(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.image_encoder = _FakeSAM2ImageEncoder()
        self.sam_prompt_encoder = _FakeSAM2PromptEncoder()
        self.sam_mask_decoder = _FakeSAM2MaskDecoder()
        # 官方形状为 1 x 1 x 256；错误 reshape 会在 encode 中直接报错。
        self.no_mem_embed = nn.Parameter(torch.full((1, 1, 256), 0.5))
        self.directly_add_no_mem_embed = True
        self.use_high_res_features_in_sam = True


def _fake_road_sam(prompt_scale: float = 1.0) -> RoadSAM:
    """构造不依赖官方 sam2 包的 RoadSAM，复用真实 encode/decode 逻辑。"""
    model = object.__new__(RoadSAM)
    nn.Module.__init__(model)
    model.sam = _FakeSAM2()
    model.pixel_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    model.pixel_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    model.no_positive_logit = -20.0
    model.config_name = "fake"
    model.architecture = "fake"
    model.apply_postprocessing = False
    model.prompt_scale = float(prompt_scale)
    # train(False) 同时切换到 eval 并施加全参数冻结。
    model.train(False)
    return model


class FakeSAM2FeatureContractTest(unittest.TestCase):
    """无需安装 sam2 的 CPU 假后端测试：形状、冻结与梯度契约。"""

    def _positive_prompt(self):
        return (
            torch.tensor([[16.0, 16.0], [32.0, 8.0]]),
            torch.tensor([1, 0]),
        )

    def test_encode_reshapes_no_mem_embed_and_returns_raw_high_res(self) -> None:
        model = _fake_road_sam()
        features = model.encode(torch.ones(2, 3, 1024, 1024))

        self.assertIsInstance(features, SAM2ImageFeatures)
        self.assertEqual(tuple(features.low_res.shape), (2, 256, 4, 4))
        # vision_features=2.0，no_mem_embed=0.5，逐通道广播。
        self.assertTrue(torch.allclose(features.low_res, torch.full_like(features.low_res, 2.5)))
        self.assertIsNotNone(features.high_res)
        self.assertEqual(tuple(features.high_res[0].shape), (2, 256, 8, 8))
        self.assertEqual(tuple(features.high_res[1].shape), (2, 256, 4, 4))
        # 编码结果必须是原始 FPN 通道，而不是 conv_s0/conv_s1 投影。
        self.assertTrue(torch.equal(features.high_res[0], torch.full((2, 256, 8, 8), 3.0)))
        self.assertTrue(torch.equal(features.high_res[1], torch.full((2, 256, 4, 4), 4.0)))
        self.assertFalse(features.low_res.requires_grad)
        self.assertFalse(features.high_res[0].requires_grad)
        self.assertFalse(features.high_res[1].requires_grad)

    def test_sparse_decode_uses_sam2_signature_with_all_parameters_frozen(self) -> None:
        model = _fake_road_sam()
        features = model.encode(torch.ones(2, 3, 1024, 1024))
        prompts = [self._positive_prompt(), self._positive_prompt()]

        logits = model.decode(features, prompts, output_size=(8, 6))

        self.assertEqual(tuple(logits.shape), (2, 1, 8, 6))
        # 全冻结且输入特征 detach：稀疏工具路径不构建计算图。
        self.assertFalse(logits.requires_grad)
        decoder = model.sam.sam_mask_decoder
        self.assertEqual(len(decoder.calls), 2)
        for call in decoder.calls:
            self.assertFalse(call["multimask_output"])
            self.assertFalse(call["repeat_image"])
            self.assertEqual(call["high_res_shapes"], [(1, 32, 8, 8), (1, 64, 4, 4)])
        self.assertTrue(
            all(not parameter.requires_grad for parameter in model.parameters())
        )
        self.assertIsNone(model.sam.no_mem_embed.grad)

    def test_decode_without_positive_points_skips_decoder(self) -> None:
        model = _fake_road_sam()
        features = model.encode(torch.ones(1, 3, 1024, 1024))
        prompts = [(torch.tensor([[5.0, 6.0]]), torch.tensor([0]))]

        logits = model.decode(features, prompts, output_size=(16, 16))

        self.assertEqual(tuple(logits.shape), (1, 1, 16, 16))
        self.assertFalse(logits.requires_grad)
        self.assertEqual(model.sam.sam_mask_decoder.calls, [])
        self.assertTrue(torch.equal(logits, torch.full_like(logits, -20.0)))

    def test_decode_rejects_bare_low_res_when_high_res_required(self) -> None:
        model = _fake_road_sam()
        features = model.encode(torch.ones(1, 3, 1024, 1024))
        with self.assertRaises(ValueError):
            model.decode(features.low_res, [self._positive_prompt()])

    def test_decode_heatmaps_uses_scaled_signed_field_and_backpropagates(self) -> None:
        model = _fake_road_sam(prompt_scale=2.0)
        features = model.encode(torch.ones(2, 3, 1024, 1024))
        field = (torch.rand(2, 1, 4, 4) * 2.0 - 1.0).detach().requires_grad_(True)

        logits = model.decode_heatmaps(features, field, output_size=(8, 6))

        self.assertEqual(tuple(logits.shape), (2, 1, 8, 6))
        self.assertTrue(logits.requires_grad)
        # 固定映射 U = prompt_scale * H，并缩放到 mask_input_size。
        expected = F.interpolate(
            2.0 * field, size=(8, 8), mode="bilinear", align_corners=False
        )
        torch.testing.assert_close(model.sam.sam_prompt_encoder.last_masks, expected)
        # 不做跨样本补齐，也没有任何稀疏点 token。
        decoder = model.sam.sam_mask_decoder
        self.assertEqual(len(decoder.calls), 1)
        self.assertEqual(decoder.calls[0]["sparse_tokens"], (2, 0, 256))
        self.assertFalse(decoder.calls[0]["multimask_output"])
        self.assertFalse(decoder.calls[0]["repeat_image"])
        self.assertEqual(
            decoder.calls[0]["high_res_shapes"], [(2, 32, 8, 8), (2, 64, 4, 4)]
        )

        logits.sum().backward()
        # SAM 参数全冻结，但输入有符号场必须获得非零梯度。
        self.assertIsNotNone(field.grad)
        self.assertTrue(bool(torch.count_nonzero(field.grad)))
        self.assertTrue(
            all(parameter.grad is None for parameter in decoder.parameters())
        )
        self.assertIsNone(model.sam.no_mem_embed.grad)

    def test_decode_heatmaps_decodes_all_samples_including_empty_maps(self) -> None:
        model = _fake_road_sam()
        features = model.encode(torch.ones(2, 3, 1024, 1024))
        field = torch.zeros(2, 1, 4, 4, requires_grad=True)

        logits = model.decode_heatmaps(features, field, output_size=(8, 8))

        self.assertEqual(tuple(logits.shape), (2, 1, 8, 8))
        self.assertTrue(torch.isfinite(logits).all())
        self.assertTrue(logits.requires_grad)
        # 空场不能被短路跳过：整批一次解码。
        self.assertEqual(len(model.sam.sam_mask_decoder.calls), 1)
        self.assertEqual(
            model.sam.sam_mask_decoder.calls[0]["sparse_tokens"], (2, 0, 256)
        )
        logits.sum().backward()
        self.assertIsNotNone(field.grad)
        self.assertTrue(torch.isfinite(field.grad).all())

    def test_decode_heatmaps_masks_padding_from_valid_sizes(self) -> None:
        model = _fake_road_sam()
        features = model.encode(torch.ones(1, 3, 1024, 1024))
        field = torch.ones(1, 1, 4, 4)

        model.decode_heatmaps(
            features,
            field,
            output_size=(4, 4),
            valid_sizes=[(2, 3)],
        )

        mask = model.sam.sam_prompt_encoder.last_masks
        self.assertEqual(tuple(mask.shape), (1, 1, 8, 8))
        # valid=2x3 在 4x4 画布上按比例缩放为 4x6（mask_input_size=8）。
        self.assertTrue(torch.all(mask[:, :, :4, :6] > 0))
        self.assertTrue(torch.equal(mask[:, :, 4:, :], torch.zeros_like(mask[:, :, 4:, :])))
        self.assertTrue(torch.equal(mask[:, :, :, 6:], torch.zeros_like(mask[:, :, :, 6:])))

    def test_decode_heatmaps_scales_only_the_valid_prompt_region(self) -> None:
        model = _fake_road_sam(prompt_scale=3.0)
        features = model.encode(torch.ones(1, 3, 1024, 1024))
        # 与 mask_input_size 同分辨率，避免双线性边界影响断言。
        field = torch.ones(1, 1, 8, 8)

        model.decode_heatmaps(
            features,
            field,
            output_size=(8, 8),
            valid_sizes=[(8, 4)],
        )

        mask = model.sam.sam_prompt_encoder.last_masks
        # 有效区域为 1 * 3.0；右半 padding 严格为 0。
        self.assertTrue(
            torch.allclose(
                mask[:, :, :, :4], torch.full_like(mask[:, :, :, :4], 3.0)
            )
        )
        self.assertTrue(torch.equal(mask[:, :, :, 4:], torch.zeros_like(mask[:, :, :, 4:])))

    def test_decode_heatmaps_rejects_invalid_shapes(self) -> None:
        model = _fake_road_sam()
        features = model.encode(torch.ones(2, 3, 1024, 1024))
        with self.assertRaises(ValueError):
            model.decode_heatmaps(features, torch.zeros(2, 3, 4, 4))
        with self.assertRaises(ValueError):
            # 新协议只接受单通道有符号场。
            model.decode_heatmaps(features, torch.zeros(2, 2, 4, 4))
        with self.assertRaises(ValueError):
            model.decode_heatmaps(features, torch.zeros(1, 1, 4, 4))
        with self.assertRaises(TypeError):
            model.decode_heatmaps(features, torch.zeros(2, 1, 4, 4, dtype=torch.uint8))
        with self.assertRaises(ValueError):
            model.decode_heatmaps(
                features,
                torch.zeros(2, 1, 4, 4),
                output_size=(8, 8),
                valid_sizes=[(8, 8), (0, 4)],
            )

    def test_decode_heatmaps_rejects_invalid_prompt_scale(self) -> None:
        model = _fake_road_sam()
        features = model.encode(torch.ones(1, 3, 1024, 1024))
        for scale in (0.0, -1.0, float("nan"), float("inf")):
            with self.subTest(prompt_scale=scale):
                model.prompt_scale = scale
                with self.assertRaises(ValueError):
                    model.decode_heatmaps(features, torch.zeros(1, 1, 4, 4))

    def test_train_mode_keeps_all_sam_modules_eval_and_frozen(self) -> None:
        model = _fake_road_sam()
        model.train(True)

        self.assertFalse(model.training)
        self.assertFalse(model.sam.image_encoder.training)
        self.assertFalse(model.sam.sam_prompt_encoder.training)
        self.assertFalse(model.sam.sam_mask_decoder.training)
        self.assertTrue(
            all(not parameter.requires_grad for parameter in model.parameters())
        )
        self.assertTrue(all(not buffer.requires_grad for buffer in model.buffers()))
        self.assertFalse(model.sam.no_mem_embed.requires_grad)

        model.train(False)
        self.assertFalse(model.sam.sam_mask_decoder.training)

    def test_constructor_validates_prompt_scale_without_official_sam2(self) -> None:
        fake_sam = _FakeSAM2()
        with mock.patch("diffpeak.models.sam._build_sam2", return_value=fake_sam):
            model = RoadSAM(prompt_scale=2.5)
            self.assertEqual(model.prompt_scale, 2.5)
            self.assertFalse(model.training)
            for scale in (0.0, -1.0, float("nan"), float("inf")):
                with self.subTest(prompt_scale=scale), self.assertRaises(ValueError):
                    RoadSAM(prompt_scale=scale)


if __name__ == "__main__":
    unittest.main()
