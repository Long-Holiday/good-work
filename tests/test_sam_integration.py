"""RoadSAM 与官方 ``segment_anything`` 的随机权重架构集成测试。

本文件验证的是**架构、接口与梯度路径**，不是预训练效果：

* 使用 ``checkpoint=None`` 构建官方 SAM ViT-B，随机初始化，不下载任何权重；
* 不运行 1024x1024 图像编码器（CPU 上过慢），而是把正确形状的
  ``B x 256 x 64 x 64`` 图像特征直接送入真实的官方提示编码器与掩码解码器；
* 因此通过这里的测试只说明 RoadSAM 能正确驱动官方模块，不代表 SAM 的
  预训练分割质量、最终道路精度或完整 ``encode`` 前向的耗时/显存。

仅当 ``segment_anything`` 包确实未安装时跳过；安装存在但导入失败、或
架构接口变化导致失败时，测试必须报错而不是静默跳过。
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

if importlib.util.find_spec("segment_anything") is None:  # pragma: no cover
    raise unittest.SkipTest(
        "未安装 segment-anything；随机权重架构集成测试跳过"
    )

from segment_anything.modeling import (  # noqa: E402
    ImageEncoderViT,
    MaskDecoder,
    PromptEncoder,
)

from diffpeak.models.sam import RoadSAM  # noqa: E402

IMAGE_EMBEDDING_SIZE = 64
EMBEDDING_SHAPE = (256, IMAGE_EMBEDDING_SIZE, IMAGE_EMBEDDING_SIZE)
OUTPUT_SIZE = (96, 64)  # 非方形输出，顺带验证插值尺寸。


def _prompt(
    coords: List[Tuple[float, float]], labels: List[int]
) -> Tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.tensor(coords, dtype=torch.float32),
        torch.tensor(labels, dtype=torch.long),
    )


class RoadSAMArchitectureIntegrationTest(unittest.TestCase):
    """真实 SAM 模块 + 随机权重的最小 CPU 集成测试。"""

    @classmethod
    def setUpClass(cls) -> None:
        torch.manual_seed(0)
        # checkpoint=None 是官方 SAM 构建器的“随机初始化、不加载权重”路径。
        cls.road_sam = RoadSAM(checkpoint=None, model_type="vit_b")
        generator = torch.Generator().manual_seed(1)
        cls.embeddings = (
            torch.randn(2, *EMBEDDING_SHAPE, generator=generator) * 0.1
        )

    def setUp(self) -> None:
        self.road_sam.zero_grad(set_to_none=True)
        self.road_sam.train()
        self.embeddings.requires_grad_(False)

    def _record_decoder_calls(self) -> List[bool]:
        """记录每次 mask decoder 调用的 ``multimask_output`` 参数。"""
        decoder = self.road_sam.sam.mask_decoder
        original_forward = decoder.forward
        calls: List[bool] = []

        def recording_forward(*args: Any, **kwargs: Any):
            calls.append(bool(kwargs.get("multimask_output")))
            return original_forward(*args, **kwargs)

        decoder.forward = recording_forward  # type: ignore[method-assign]

        def restore() -> None:
            decoder.__dict__.pop("forward", None)

        self.addCleanup(restore)
        return calls

    def test_builds_real_sam_modules_with_frozen_encoders(self) -> None:
        """RoadSAM 注入的是官方模块，且只训练 mask decoder。"""
        self.assertIsInstance(self.road_sam.sam.image_encoder, ImageEncoderViT)
        self.assertIsInstance(self.road_sam.sam.prompt_encoder, PromptEncoder)
        self.assertIsInstance(self.road_sam.sam.mask_decoder, MaskDecoder)
        self.assertEqual(self.road_sam.model_type, "vit_b")

        image_params = list(self.road_sam.sam.image_encoder.parameters())
        prompt_params = list(self.road_sam.sam.prompt_encoder.parameters())
        decoder_params = list(self.road_sam.sam.mask_decoder.parameters())
        self.assertGreater(len(image_params), 0)
        self.assertGreater(len(prompt_params), 0)
        self.assertGreater(len(decoder_params), 0)

        self.assertTrue(all(not p.requires_grad for p in image_params))
        self.assertTrue(all(not p.requires_grad for p in prompt_params))
        self.assertTrue(all(p.requires_grad for p in decoder_params))
        self.assertFalse(self.road_sam.sam.image_encoder.training)
        self.assertFalse(self.road_sam.sam.prompt_encoder.training)
        self.assertTrue(self.road_sam.sam.mask_decoder.training)

        # 训练脚本整体调用 train() 后，冻结模块仍必须保持 eval。
        self.road_sam.train(True)
        self.assertFalse(self.road_sam.sam.image_encoder.training)
        self.assertFalse(self.road_sam.sam.prompt_encoder.training)
        self.assertTrue(self.road_sam.sam.mask_decoder.training)

    def test_single_mask_interface_and_output_shape(self) -> None:
        """multimask_output=False，每张图只返回一个掩码。"""
        calls = self._record_decoder_calls()
        prompts = [
            _prompt([[300.0, 300.0], [700.0, 700.0]], [1, 0]),
            _prompt([[100.0, 200.0]], [1]),
        ]

        logits = self.road_sam.decode(
            self.embeddings, prompts, output_size=OUTPUT_SIZE
        )

        self.assertEqual(tuple(logits.shape), (2, 1, *OUTPUT_SIZE))
        self.assertEqual(calls, [False, False])
        self.assertTrue(bool(torch.isfinite(logits).all()))
        # 掩码解码器参数可训练，输出应连接计算图。
        self.assertTrue(logits.requires_grad)

        # 不同的稀疏提示应产生不同的掩码，排除“提示被忽略”的接线错误。
        with torch.no_grad():
            first = self.road_sam.decode(
                self.embeddings[:1], [prompts[0]], output_size=OUTPUT_SIZE
            )
            second = self.road_sam.decode(
                self.embeddings[:1], [prompts[1]], output_size=OUTPUT_SIZE
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
            self.embeddings, prompts, output_size=OUTPUT_SIZE
        )

        self.assertEqual(calls, [False])  # 无正点样本不调用解码器
        empty = logits[1]
        self.assertTrue(
            torch.equal(
                empty,
                torch.full_like(empty, self.road_sam.no_positive_logit),
            )
        )
        # 单个 cat 输出无法只对某一行关闭 requires_grad；真正需要保证的是
        # 空行没有指向解码器的梯度路径。仅对空行反传，解码器应保持无梯度。
        self.road_sam.zero_grad(set_to_none=True)
        empty.sum().backward()
        # cat 的反向会经过另一行并留下零梯度，因此检查“无梯度或全零”，
        # 而不是要求 .grad 必须为 None。
        self.assertTrue(
            all(
                parameter.grad is None
                or not bool(torch.count_nonzero(parameter.grad))
                for parameter in self.road_sam.sam.mask_decoder.parameters()
            )
        )
        self.assertFalse(torch.equal(logits[0], empty))
        self.assertTrue(logits[0].requires_grad)

    def test_all_samples_without_positive_prompts_short_circuit(self) -> None:
        """整批都没有正点时直接返回常量，不构造解码计算图。"""
        decoder = self.road_sam.sam.mask_decoder

        def fail_if_called(*args: Any, **kwargs: Any):
            raise AssertionError("无正点批次不应调用 mask decoder")

        decoder.forward = fail_if_called  # type: ignore[method-assign]
        self.addCleanup(lambda: decoder.__dict__.pop("forward", None))

        prompts = [
            _prompt([], []),
            _prompt([[10.0, 20.0]], [0]),
        ]
        logits = self.road_sam.decode(
            self.embeddings, prompts, output_size=OUTPUT_SIZE
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
        """同一图内的提示顺序可打乱，SAM 输出应保持不变。"""
        ordered = _prompt(
            [[300.0, 300.0], [700.0, 700.0], [200.0, 800.0]], [1, 0, 1]
        )
        shuffled = _prompt(
            [[200.0, 800.0], [300.0, 300.0], [700.0, 700.0]], [1, 1, 0]
        )

        with torch.no_grad():
            first = self.road_sam.decode(
                self.embeddings[:1], [ordered], output_size=OUTPUT_SIZE
            )
            second = self.road_sam.decode(
                self.embeddings[:1], [shuffled], output_size=OUTPUT_SIZE
            )

        torch.testing.assert_close(second, first, atol=1e-6, rtol=1e-5)

    def test_prompts_are_paired_per_image_not_cross_shuffled(self) -> None:
        """批量提示必须与各自图像配对，不能出现 B x B 交叉组合。"""
        prompt_a = _prompt([[128.0, 256.0]], [1])
        prompt_b = _prompt([[800.0, 640.0]], [1])
        embeddings = self.embeddings
        self.assertFalse(torch.equal(embeddings[0], embeddings[1]))

        with torch.no_grad():
            batch = self.road_sam.decode(
                embeddings, [prompt_a, prompt_b], output_size=OUTPUT_SIZE
            )
            row_a0 = self.road_sam.decode(
                embeddings[:1], [prompt_a], output_size=OUTPUT_SIZE
            )
            row_b1 = self.road_sam.decode(
                embeddings[1:], [prompt_b], output_size=OUTPUT_SIZE
            )

        torch.testing.assert_close(batch[0:1], row_a0, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(batch[1:2], row_b1, atol=1e-6, rtol=1e-5)

        # 交换提示列表时，每一行应跟随对应图像，而不是沿用旧行。
        with torch.no_grad():
            swapped = self.road_sam.decode(
                embeddings, [prompt_b, prompt_a], output_size=OUTPUT_SIZE
            )
            row_b0 = self.road_sam.decode(
                embeddings[:1], [prompt_b], output_size=OUTPUT_SIZE
            )
            row_a1 = self.road_sam.decode(
                embeddings[1:], [prompt_a], output_size=OUTPUT_SIZE
            )

        torch.testing.assert_close(swapped[0:1], row_b0, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(swapped[1:2], row_a1, atol=1e-6, rtol=1e-5)

    def test_backward_updates_only_trainable_mask_decoder(self) -> None:
        """解码输出可反向，梯度只落到 mask decoder。"""
        self.road_sam.zero_grad(set_to_none=True)
        prompts = [_prompt([[400.0, 500.0]], [1])]

        logits = self.road_sam.decode(
            self.embeddings[:1], prompts, output_size=OUTPUT_SIZE
        )
        logits.mean().backward()

        decoder_grads = [
            p.grad
            for p in self.road_sam.sam.mask_decoder.parameters()
            if p.grad is not None
        ]
        self.assertGreater(len(decoder_grads), 0)
        self.assertTrue(all(bool(torch.isfinite(g).all()) for g in decoder_grads))
        self.assertTrue(
            any(float(g.abs().sum()) > 0.0 for g in decoder_grads),
            "mask decoder 没有获得非零梯度",
        )
        self.assertTrue(
            all(p.grad is None for p in self.road_sam.sam.image_encoder.parameters())
        )
        self.assertTrue(
            all(p.grad is None for p in self.road_sam.sam.prompt_encoder.parameters())
        )

    def test_random_weights_are_not_pretrained(self) -> None:
        """显式记录：本测试使用随机初始化，不是预训练效果验证。

        同一 seed 下两次独立构建的权重应完全一致；这只能证明构建是确定性的
        随机初始化，不能证明任何分割质量。
        """
        torch.manual_seed(123)
        first = RoadSAM(checkpoint=None, model_type="vit_b")
        torch.manual_seed(123)
        second = RoadSAM(checkpoint=None, model_type="vit_b")

        first_state = first.sam.mask_decoder.state_dict()
        second_state = second.sam.mask_decoder.state_dict()
        self.assertEqual(first_state.keys(), second_state.keys())
        for key in first_state:
            self.assertTrue(
                torch.equal(first_state[key], second_state[key]),
                f"随机初始化在同 seed 下不一致: {key}",
            )


if __name__ == "__main__":
    unittest.main()
