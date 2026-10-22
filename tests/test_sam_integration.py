"""RoadSAM 与官方 ``sam2`` 的随机权重架构集成测试。

本文件验证的是**架构、接口与梯度路径**，不是预训练效果：

* 使用 ``checkpoint=None`` 构建官方 SAM2.1 Hiera-B+，随机初始化，不下载权重；
* 不运行 1024x1024 图像编码器（CPU 上过慢），而是把正确形状的低/高分辨率
  特征直接送入真实的官方提示编码器与掩码解码器；
* 因此通过这里的测试只说明 RoadSAM 能正确驱动官方模块，不代表 SAM2 的
  预训练分割质量、最终道路精度或完整 ``encode`` 前向的耗时/显存。

仅当 ``sam2`` 包确实未安装时跳过；安装存在但导入失败、或架构接口变化导致
失败时，测试必须报错而不是静默跳过。
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from typing import Any, List, Tuple

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if importlib.util.find_spec("sam2") is None:  # pragma: no cover
    raise unittest.SkipTest("未安装 sam2；随机权重架构集成测试跳过")

from sam2.modeling.backbones.image_encoder import ImageEncoder  # noqa: E402
from sam2.modeling.sam.mask_decoder import MaskDecoder  # noqa: E402
from sam2.modeling.sam.prompt_encoder import PromptEncoder  # noqa: E402
from sam2.modeling.sam2_base import SAM2Base  # noqa: E402

from diffpeak.models.sam import RoadSAM, SAM2ImageFeatures  # noqa: E402

EMBEDDING_SHAPE = (256, 64, 64)
HIGH_RES_SHAPES = ((256, 256, 256), (256, 128, 128))
OUTPUT_SIZE = (96, 64)  # 非方形输出，顺带验证插值尺寸。


def _prompt(
    coords: List[Tuple[float, float]], labels: List[int]
) -> Tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.tensor(coords, dtype=torch.float32),
        torch.tensor(labels, dtype=torch.long),
    )


class RoadSAMArchitectureIntegrationTest(unittest.TestCase):
    """真实 SAM2 模块 + 随机权重的最小 CPU 集成测试。"""

    @classmethod
    def setUpClass(cls) -> None:
        torch.manual_seed(0)
        # checkpoint=None 是官方 SAM2 构建器的“随机初始化、不加载权重”路径。
        cls.road_sam = RoadSAM(checkpoint=None)
        generator = torch.Generator().manual_seed(1)
        low = torch.randn(2, *EMBEDDING_SHAPE, generator=generator) * 0.1
        high = tuple(
            torch.randn(2, *shape, generator=generator) * 0.1
            for shape in HIGH_RES_SHAPES
        )
        cls.features = SAM2ImageFeatures(low, (high[0], high[1]))

    def setUp(self) -> None:
        self.road_sam.zero_grad(set_to_none=True)
        self.road_sam.train()
        for tensor in (
            self.features.low_res,
            self.features.high_res[0],
            self.features.high_res[1],
        ):
            tensor.requires_grad_(False)

    def _record_decoder_calls(self) -> List[Tuple[bool, bool, bool]]:
        """记录每次 mask decoder 调用的 multimask/repeat/high-res 参数。"""
        decoder = self.road_sam.sam.sam_mask_decoder
        original_forward = decoder.forward
        calls: List[Tuple[bool, bool, bool]] = []

        def recording_forward(*args: Any, **kwargs: Any):
            calls.append(
                (
                    bool(kwargs.get("multimask_output")),
                    bool(kwargs.get("repeat_image")),
                    kwargs.get("high_res_features") is not None,
                )
            )
            return original_forward(*args, **kwargs)

        decoder.forward = recording_forward  # type: ignore[method-assign]

        def restore() -> None:
            decoder.__dict__.pop("forward", None)

        self.addCleanup(restore)
        return calls

    def test_builds_real_sam2_modules_with_frozen_encoders(self) -> None:
        """RoadSAM 注入的是官方 SAM2 模块，且只训练 mask decoder。"""
        self.assertIsInstance(self.road_sam.sam, SAM2Base)
        self.assertIsInstance(self.road_sam.sam.image_encoder, ImageEncoder)
        self.assertIsInstance(self.road_sam.sam.sam_prompt_encoder, PromptEncoder)
        self.assertIsInstance(self.road_sam.sam.sam_mask_decoder, MaskDecoder)
        self.assertEqual(
            self.road_sam.config_name, "configs/sam2.1/sam2.1_hiera_b+.yaml"
        )

        image_params = list(self.road_sam.sam.image_encoder.parameters())
        prompt_params = list(self.road_sam.sam.sam_prompt_encoder.parameters())
        decoder_params = list(self.road_sam.sam.sam_mask_decoder.parameters())
        self.assertGreater(len(image_params), 0)
        self.assertGreater(len(prompt_params), 0)
        self.assertGreater(len(decoder_params), 0)

        self.assertTrue(all(not p.requires_grad for p in image_params))
        self.assertTrue(all(not p.requires_grad for p in prompt_params))
        self.assertTrue(all(p.requires_grad for p in decoder_params))
        self.assertFalse(self.road_sam.sam.image_encoder.training)
        self.assertFalse(self.road_sam.sam.sam_prompt_encoder.training)
        self.assertFalse(self.road_sam.sam.memory_attention.training)
        self.assertTrue(self.road_sam.sam.sam_mask_decoder.training)

        # 训练脚本整体调用 train() 后，冻结模块仍必须保持 eval。
        self.road_sam.train(True)
        self.assertFalse(self.road_sam.sam.image_encoder.training)
        self.assertFalse(self.road_sam.sam.sam_prompt_encoder.training)
        self.assertFalse(self.road_sam.sam.memory_attention.training)
        self.assertTrue(self.road_sam.sam.sam_mask_decoder.training)

    def test_single_mask_interface_and_output_shape(self) -> None:
        """multimask_output=False、repeat_image=False，每图只返回一个掩码。"""
        calls = self._record_decoder_calls()
        prompts = [
            _prompt([[300.0, 300.0], [700.0, 700.0]], [1, 0]),
            _prompt([[100.0, 200.0]], [1]),
        ]

        logits = self.road_sam.decode(
            self.features, prompts, output_size=OUTPUT_SIZE
        )

        self.assertEqual(tuple(logits.shape), (2, 1, *OUTPUT_SIZE))
        self.assertEqual(calls, [(False, False, True), (False, False, True)])
        self.assertTrue(bool(torch.isfinite(logits).all()))
        # 掩码解码器参数可训练，输出应连接计算图。
        self.assertTrue(logits.requires_grad)

        # 不同的稀疏提示应产生不同的掩码，排除“提示被忽略”的接线错误。
        with torch.no_grad():
            first = self.road_sam.decode(
                self.features.select(0), [prompts[0]], output_size=OUTPUT_SIZE
            )
            second = self.road_sam.decode(
                self.features.select(0), [prompts[1]], output_size=OUTPUT_SIZE
            )
        self.assertGreater(float((first - second).abs().max()), 1e-6)

    def test_removing_positive_prompt_yields_constant_empty_mask(self) -> None:
        """移除正点的样本输出常量空掩码，同批其他样本仍正常解码。"""
        calls = self._record_decoder_calls()
        prompts = [
            _prompt([[320.0, 320.0]], [1]),
            _prompt([[512.0, 512.0]], [0]),  # 只有负点 = 移除正点提示
        ]

        logits = self.road_sam.decode(
            self.features, prompts, output_size=OUTPUT_SIZE
        )

        self.assertEqual(len(calls), 1)  # 无正点样本不调用解码器
        empty = logits[1]
        self.assertTrue(
            torch.equal(
                empty,
                torch.full_like(empty, self.road_sam.no_positive_logit),
            )
        )
        # 空行没有指向解码器的梯度路径。
        self.road_sam.zero_grad(set_to_none=True)
        empty.sum().backward()
        self.assertTrue(
            all(
                parameter.grad is None
                or not bool(torch.count_nonzero(parameter.grad))
                for parameter in self.road_sam.sam.sam_mask_decoder.parameters()
            )
        )
        self.assertFalse(torch.equal(logits[0], empty))
        self.assertTrue(logits[0].requires_grad)

    def test_all_samples_without_positive_prompts_short_circuit(self) -> None:
        """整批都没有正点时直接返回常量，不构造解码计算图。"""
        decoder = self.road_sam.sam.sam_mask_decoder

        def fail_if_called(*args: Any, **kwargs: Any):
            raise AssertionError("无正点批次不应调用 mask decoder")

        decoder.forward = fail_if_called  # type: ignore[method-assign]
        self.addCleanup(lambda: decoder.__dict__.pop("forward", None))

        prompts = [
            _prompt([], []),
            _prompt([[10.0, 20.0]], [0]),
        ]
        logits = self.road_sam.decode(
            self.features, prompts, output_size=OUTPUT_SIZE
        )

        self.assertEqual(tuple(logits.shape), (2, 1, *OUTPUT_SIZE))
        self.assertFalse(logits.requires_grad)
        self.assertTrue(
            torch.equal(
                logits,
                torch.full_like(logits, self.road_sam.no_positive_logit),
            )
        )

    def test_prompt_order_within_a_sample_is_invariant(self) -> None:
        """同一图内的提示顺序可打乱，SAM2 输出应保持不变。"""
        ordered = _prompt(
            [[300.0, 300.0], [700.0, 700.0], [200.0, 800.0]], [1, 0, 1]
        )
        shuffled = _prompt(
            [[200.0, 800.0], [300.0, 300.0], [700.0, 700.0]], [1, 1, 0]
        )

        with torch.no_grad():
            first = self.road_sam.decode(
                self.features.select(0), [ordered], output_size=OUTPUT_SIZE
            )
            second = self.road_sam.decode(
                self.features.select(0), [shuffled], output_size=OUTPUT_SIZE
            )

        torch.testing.assert_close(second, first, atol=1e-6, rtol=1e-5)

    def test_prompts_are_paired_per_image_not_cross_shuffled(self) -> None:
        """批量提示必须与各自图像配对，不能出现 B x B 交叉组合。"""
        prompt_a = _prompt([[128.0, 256.0]], [1])
        prompt_b = _prompt([[800.0, 640.0]], [1])
        features = self.features
        self.assertFalse(torch.equal(features.low_res[0], features.low_res[1]))

        with torch.no_grad():
            batch = self.road_sam.decode(
                features, [prompt_a, prompt_b], output_size=OUTPUT_SIZE
            )
            row_a0 = self.road_sam.decode(
                features.select(0), [prompt_a], output_size=OUTPUT_SIZE
            )
            row_b1 = self.road_sam.decode(
                features.select(1), [prompt_b], output_size=OUTPUT_SIZE
            )

        torch.testing.assert_close(batch[0:1], row_a0, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(batch[1:2], row_b1, atol=1e-6, rtol=1e-5)

        with torch.no_grad():
            swapped = self.road_sam.decode(
                features, [prompt_b, prompt_a], output_size=OUTPUT_SIZE
            )
            row_b0 = self.road_sam.decode(
                features.select(0), [prompt_b], output_size=OUTPUT_SIZE
            )
            row_a1 = self.road_sam.decode(
                features.select(1), [prompt_a], output_size=OUTPUT_SIZE
            )

        torch.testing.assert_close(swapped[0:1], row_b0, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(swapped[1:2], row_a1, atol=1e-6, rtol=1e-5)

    def test_variable_point_counts_match_single_image_decoding(self) -> None:
        """批量解码不能因跨样本补齐改变任一图的提示 token 数。

        官方提示编码器在 ``boxes=None`` 时自行追加一个 ``not_a_point``
        token；若再按 batch 最长点集补齐，短提示样本会多一个 token，输出
        与单图解码不一致。
        """
        prompt_many = _prompt(
            [[300.0, 300.0], [700.0, 700.0]], [1, 0]
        )
        prompt_one = _prompt([[512.0, 128.0]], [1])

        with torch.no_grad():
            batch = self.road_sam.decode(
                self.features,
                [prompt_many, prompt_one],
                output_size=OUTPUT_SIZE,
            )
            single_many = self.road_sam.decode(
                self.features.select(0), [prompt_many], output_size=OUTPUT_SIZE
            )
            single_one = self.road_sam.decode(
                self.features.select(1), [prompt_one], output_size=OUTPUT_SIZE
            )

        torch.testing.assert_close(batch[0:1], single_many, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(batch[1:2], single_one, atol=1e-6, rtol=1e-5)

    def test_backward_updates_only_trainable_mask_decoder(self) -> None:
        """解码输出可反向，梯度只落到 mask decoder（含高分辨率投影）。"""
        self.road_sam.zero_grad(set_to_none=True)
        prompts = [_prompt([[400.0, 500.0]], [1])]

        logits = self.road_sam.decode(
            self.features.select(0), prompts, output_size=OUTPUT_SIZE
        )
        logits.mean().backward()

        decoder_grads = [
            (name, p.grad)
            for name, p in self.road_sam.sam.sam_mask_decoder.named_parameters()
            if p.grad is not None
        ]
        self.assertGreater(len(decoder_grads), 0)
        self.assertTrue(all(bool(torch.isfinite(g).all()) for _, g in decoder_grads))
        self.assertTrue(
            any(float(g.abs().sum()) > 0.0 for _, g in decoder_grads),
            "mask decoder 没有获得非零梯度",
        )
        # 高分辨率投影属于解码器，必须参与训练。
        self.assertTrue(
            any(
                name.startswith("conv_s0") and float(g.abs().sum()) > 0.0
                for name, g in decoder_grads
            )
        )
        self.assertTrue(
            any(
                name.startswith("conv_s1") and float(g.abs().sum()) > 0.0
                for name, g in decoder_grads
            )
        )
        self.assertTrue(
            all(p.grad is None for p in self.road_sam.sam.image_encoder.parameters())
        )
        self.assertTrue(
            all(
                p.grad is None
                for p in self.road_sam.sam.sam_prompt_encoder.parameters()
            )
        )

    def test_decode_heatmaps_gradient_passes_through_fully_frozen_sam(self) -> None:
        parameters = list(self.road_sam.parameters())
        flags = [parameter.requires_grad for parameter in parameters]
        self.addCleanup(lambda: [p.requires_grad_(flag) for p, flag in zip(parameters, flags)])
        self.road_sam.requires_grad_(False).eval()
        self.road_sam.zero_grad(set_to_none=True)
        heatmaps = torch.rand(2, 2, 16, 16, requires_grad=True)
        logits = self.road_sam.decode_heatmaps(
            self.features, heatmaps, output_size=OUTPUT_SIZE
        )
        logits.square().mean().backward()
        self.assertIsNotNone(heatmaps.grad)
        self.assertTrue(bool(torch.isfinite(heatmaps.grad).all()))
        self.assertGreater(float(heatmaps.grad.abs().sum()), 0)
        self.assertTrue(all(p.grad is None for p in parameters))

    def test_decode_heatmaps_gradient_passes_through_frozen_prompt_encoder(self) -> None:
        """密集热图提示可微：冻结提示编码器只把梯度传回输入热图。"""
        self.road_sam.zero_grad(set_to_none=True)
        generator = torch.Generator().manual_seed(7)
        heatmaps = torch.rand(2, 2, 16, 16, generator=generator, requires_grad=True)

        logits = self.road_sam.decode_heatmaps(
            self.features, heatmaps, output_size=OUTPUT_SIZE
        )

        self.assertEqual(tuple(logits.shape), (2, 1, *OUTPUT_SIZE))
        self.assertTrue(logits.requires_grad)
        logits.mean().backward()

        self.assertIsNotNone(heatmaps.grad)
        self.assertTrue(bool(torch.count_nonzero(heatmaps.grad)))
        # 提示编码器权重冻结，不能累积梯度。
        self.assertTrue(
            all(
                p.grad is None
                for p in self.road_sam.sam.sam_prompt_encoder.parameters()
            )
        )
        # 但掩码解码器（含高分辨率投影）仍获得可训练梯度。
        decoder = self.road_sam.sam.sam_mask_decoder
        self.assertTrue(
            any(
                p.grad is not None and bool(torch.count_nonzero(p.grad))
                for p in decoder.parameters()
            )
        )
        self.assertTrue(
            all(bool(torch.isfinite(p.grad).all()) for p in decoder.parameters() if p.grad is not None)
        )

    def test_decode_heatmaps_decodes_empty_maps_without_sparse_points(self) -> None:
        """空热图也整批解码，且不依赖任何稀疏点提示。"""
        self.road_sam.zero_grad(set_to_none=True)
        heatmaps = torch.zeros(2, 2, 16, 16)

        logits = self.road_sam.decode_heatmaps(
            self.features, heatmaps, output_size=OUTPUT_SIZE
        )

        self.assertEqual(tuple(logits.shape), (2, 1, *OUTPUT_SIZE))
        self.assertTrue(bool(torch.isfinite(logits).all()))
        self.assertTrue(logits.requires_grad)
        logits.mean().backward()
        self.assertTrue(
            any(
                p.grad is not None and bool(torch.count_nonzero(p.grad))
                for p in self.road_sam.sam.sam_mask_decoder.parameters()
            )
        )

    def test_random_weights_are_not_pretrained(self) -> None:
        """显式记录：本测试使用随机初始化，不是预训练效果验证。

        同一 seed 下两次独立构建的权重应完全一致；这只能证明构建是确定性的
        随机初始化，不能证明任何分割质量。
        """
        torch.manual_seed(123)
        first = RoadSAM(checkpoint=None, config_name="configs/sam2.1/sam2.1_hiera_t.yaml")
        torch.manual_seed(123)
        second = RoadSAM(checkpoint=None, config_name="configs/sam2.1/sam2.1_hiera_t.yaml")

        first_state = first.sam.sam_mask_decoder.state_dict()
        second_state = second.sam.sam_mask_decoder.state_dict()
        self.assertEqual(first_state.keys(), second_state.keys())
        for key in first_state:
            self.assertTrue(
                torch.equal(first_state[key], second_state[key]),
                f"随机初始化在同 seed 下不一致: {key}",
            )


if __name__ == "__main__":
    unittest.main()
