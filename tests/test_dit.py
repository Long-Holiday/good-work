"""条件 DiT 的形状、条件/时间步语义与梯度检查点测试。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diffpeak.models.dit import (  # noqa: E402
    ConditionalDiT,
    DiT,
    build_2d_sincos_pos_embed,
)


def _scale_sizes(height: int, width: int):
    """与 256/128/64/32 金字塔一致的四个尺度尺寸（支持奇数/矩形）。"""
    return [
        (height, width),
        ((height + 1) // 2, (width + 1) // 2),
        ((height + 3) // 4, (width + 3) // 4),
        (max(1, (height + 7) // 8), max(1, (width + 7) // 8)),
    ]


def _conditions(
    batch: int,
    height: int,
    width: int,
    channels=(4, 8, 8, 8),
) -> list[torch.Tensor]:
    return [
        torch.randn(batch, channel, h, w)
        for channel, (h, w) in zip(channels, _scale_sizes(height, width))
    ]


def _small_model(**overrides) -> ConditionalDiT:
    config = dict(
        in_channels=2,
        out_channels=2,
        hidden_size=32,
        depth=2,
        num_heads=4,
        patch_size=4,
        mlp_ratio=2.0,
        condition_channels=(4, 8, 8, 8),
        gradient_checkpointing=False,
    )
    config.update(overrides)
    return ConditionalDiT(**config)


def _warm_up(model: ConditionalDiT) -> ConditionalDiT:
    """解除 adaLN-Zero 与零输出头，使条件/时间步信号可被观测。

    真实训练中这些参数由损失更新；测试里显式写入非零权重即可。
    """
    generator = torch.Generator().manual_seed(2026)
    with torch.no_grad():
        for block in model.blocks:
            modulation = block.adaLN_modulation[-1]
            modulation.weight.normal_(0.0, 0.1, generator=generator)
            modulation.bias.normal_(0.0, 0.1, generator=generator)
        final_modulation = model.final_layer.adaLN_modulation[-1]
        final_modulation.weight.normal_(0.0, 0.1, generator=generator)
        final_modulation.bias.normal_(0.0, 0.1, generator=generator)
        model.final_layer.linear.weight.normal_(0.0, 0.1, generator=generator)
        model.final_layer.linear.bias.zero_()
    return model


class DiTConfigurationTest(unittest.TestCase):
    def test_defaults_match_practical_256_heatmap_configuration(self) -> None:
        model = ConditionalDiT()

        self.assertEqual(model.in_channels, 2)
        self.assertEqual(model.out_channels, 2)
        self.assertEqual(model.hidden_size, 256)
        self.assertEqual(model.depth, 6)
        self.assertEqual(model.num_heads, 8)
        self.assertEqual(model.patch_size, 8)
        self.assertEqual(model.mlp_ratio, 4.0)
        self.assertEqual(model.condition_channels, (64, 128, 256, 256))
        self.assertTrue(model.gradient_checkpointing)
        self.assertIs(DiT, ConditionalDiT)

    def test_invalid_configurations_raise(self) -> None:
        cases = {
            "hidden_not_divisible": dict(hidden_size=30, num_heads=4),
            "zero_depth": dict(depth=0),
            "zero_heads": dict(num_heads=0),
            "zero_patch": dict(patch_size=0),
            "zero_ratio": dict(mlp_ratio=0.0),
            "nan_ratio": dict(mlp_ratio=float("nan")),
            "inf_ratio": dict(mlp_ratio=float("inf")),
            "three_scales": dict(condition_channels=(4, 8, 16)),
            "zero_condition_channel": dict(condition_channels=(4, 8, 16, 0)),
            "zero_in_channels": dict(in_channels=0),
            "heads_do_not_divide": dict(hidden_size=32, num_heads=3),
        }
        for name, overrides in cases.items():
            with self.subTest(name=name), self.assertRaises(ValueError):
                _small_model(**overrides)

    def test_default_model_forward_on_rectangular_input(self) -> None:
        torch.manual_seed(0)
        model = ConditionalDiT().eval()
        # 20x36 不是 8 的整数倍：覆盖默认配置下的补零/裁剪路径。
        height, width = 20, 36
        x = torch.randn(1, 2, height, width)
        conditions = [
            torch.randn(1, channel, h, w)
            for channel, (h, w) in zip(
                (64, 128, 256, 256), _scale_sizes(height, width)
            )
        ]

        with torch.no_grad():
            output = model(x, torch.tensor([0]), conditions)

        self.assertEqual(tuple(output.shape), (1, 2, height, width))
        self.assertTrue(torch.isfinite(output).all())


class DiTForwardTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)

    def test_forward_shape_and_zero_initialized_output(self) -> None:
        model = _small_model().eval()
        height, width = 16, 20
        x = torch.randn(2, 2, height, width)
        conditions = _conditions(2, height, width)

        with torch.no_grad():
            output = model(x, torch.tensor([0, 5]), conditions)

        self.assertEqual(tuple(output.shape), tuple(x.shape))
        # 最终线性头零初始化：未训练时任何输入都预测零噪声。
        self.assertTrue(torch.equal(output, torch.zeros_like(output)))

    def test_rectangular_padding_and_cropping_preserves_shape(self) -> None:
        model = _small_model().eval()
        for height, width in ((18, 26), (1, 3), (4, 4), (13, 7)):
            with self.subTest(height=height, width=width):
                x = torch.randn(1, 2, height, width)
                with torch.no_grad():
                    output = model(x, 3, _conditions(1, height, width))
                self.assertEqual(tuple(output.shape), (1, 2, height, width))
                self.assertTrue(torch.isfinite(output).all())

    def test_accepts_scalar_and_per_sample_timesteps(self) -> None:
        model = _warm_up(_small_model()).eval()
        x = torch.randn(2, 2, 8, 12)
        conditions = _conditions(2, 8, 12)

        with torch.no_grad():
            scalar = model(x, 4, conditions)
            per_sample = model(x, torch.tensor([4, 4]), conditions)
            single = model(x, torch.tensor([4]), conditions)
            mixed = model(x, torch.tensor([1, 4]), conditions)

        self.assertEqual(tuple(scalar.shape), tuple(x.shape))
        self.assertEqual(tuple(per_sample.shape), tuple(x.shape))
        self.assertTrue(torch.equal(scalar, single))
        self.assertTrue(torch.equal(scalar, per_sample))
        # 逐样本时间步确实独立生效。
        self.assertFalse(torch.allclose(scalar, mixed))

    def test_invalid_inputs_raise(self) -> None:
        model = _small_model().eval()
        x = torch.randn(2, 2, 16, 16)
        conditions = _conditions(2, 16, 16)

        with self.assertRaises(ValueError):
            model(torch.randn(2, 2, 16), 0, conditions)
        with self.assertRaises(ValueError):
            model(torch.randn(2, 1, 16, 16), 0, conditions)
        with self.assertRaises(ValueError):
            model(x, 0, conditions[:3])
        with self.assertRaises(ValueError):
            model(x, 0, torch.randn(4, 4, 16, 16))
        with self.assertRaises(ValueError):
            model(x, 0, conditions[:3] + [conditions[3][:, :3]])
        with self.assertRaises(ValueError):
            model(x, 0, [c[:1] for c in conditions])
        with self.assertRaises(ValueError):
            model(x, 0, [c.unsqueeze(0) for c in conditions])
        with self.assertRaises(ValueError):
            model(x, torch.tensor([1, 2, 3]), conditions)

    def test_non_tensor_condition_raises_type_error(self) -> None:
        model = _small_model().eval()
        x = torch.randn(1, 2, 8, 8)
        conditions = _conditions(1, 8, 8)

        with self.assertRaises(TypeError):
            model(x, 0, [object()] + conditions[1:])
        with self.assertRaises(TypeError):
            model(x, 0, [conditions[0], conditions[1], conditions[2], 1.0])

    def test_condition_scales_influence_output_after_zero_head_warmup(self) -> None:
        model = _warm_up(_small_model()).eval()
        x = torch.randn(2, 2, 16, 16)
        timestep = torch.tensor([2, 8])
        conditions = _conditions(2, 16, 16)

        with torch.no_grad():
            reference = model(x, timestep, conditions)
            for index in range(4):
                changed = [condition.clone() for condition in conditions]
                changed[index] = changed[index] + 1.0
                output = model(x, timestep, changed)
                self.assertFalse(
                    torch.allclose(reference, output),
                    f"第 {index} 个尺度的条件未影响输出",
                )

    def test_timestep_influences_output_after_zero_head_warmup(self) -> None:
        model = _warm_up(_small_model()).eval()
        x = torch.randn(2, 2, 16, 16)
        conditions = _conditions(2, 16, 16)

        with torch.no_grad():
            early = model(x, torch.tensor([0, 0]), conditions)
            late = model(x, torch.tensor([9, 9]), conditions)
            mixed = model(x, torch.tensor([0, 9]), conditions)

        self.assertFalse(torch.allclose(early, late))
        self.assertFalse(torch.allclose(early, mixed))
        # 不同 batch 元素使用不同时间步时，被调制的 token 应分别响应。
        self.assertFalse(torch.allclose(mixed[0], mixed[1]))

    def test_attention_dropout_is_zero_in_training_mode(self) -> None:
        model = _warm_up(_small_model()).train()
        self.assertEqual(model.blocks[0].attn.attention_dropout, 0.0)
        x = torch.randn(1, 2, 8, 8)
        conditions = _conditions(1, 8, 8)

        first = model(x, 1, conditions)
        second = model(x, 1, conditions)

        self.assertTrue(torch.allclose(first, second, atol=0.0, rtol=0.0))


class DiTBackwardTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(11)

    def test_forward_backward_updates_all_major_components(self) -> None:
        model = _warm_up(_small_model()).train()
        x = torch.randn(2, 2, 16, 16, requires_grad=True)
        conditions = _conditions(2, 16, 16)

        output = model(x, torch.tensor([1, 6]), conditions)
        self.assertFalse(torch.allclose(output, torch.zeros_like(output)))
        output.square().mean().backward()

        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertGreater(float(x.grad.abs().sum()), 0.0)

        named_parameters = dict(model.named_parameters())
        for name in (
            "patch_embed.weight",
            "cond_proj.0.weight",
            "cond_proj.3.weight",
            "cond_patch_embed.weight",
            "time_embed.0.weight",
            "blocks.0.attn.qkv.weight",
            "blocks.0.attn.proj.weight",
            "blocks.0.mlp.0.weight",
            "final_layer.linear.weight",
        ):
            gradient = named_parameters[name].grad
            self.assertIsNotNone(gradient, name)
            self.assertTrue(torch.isfinite(gradient).all(), name)
            self.assertGreater(
                int(torch.count_nonzero(gradient)), 0, f"{name} 梯度为零"
            )

    def test_gradient_checkpointing_eval_with_detached_inputs(self) -> None:
        """GRPO 场景：eval + 梯度开启 + x/条件均 detach，参数仍需梯度。"""
        model = _warm_up(_small_model(gradient_checkpointing=True)).eval()
        x = torch.randn(2, 2, 16, 16)
        conditions = _conditions(2, 16, 16)
        self.assertFalse(x.requires_grad)
        self.assertTrue(all(not condition.requires_grad for condition in conditions))

        output = model(x, 4, conditions)
        output.square().mean().backward()

        named_parameters = dict(model.named_parameters())
        for name in (
            "patch_embed.weight",
            "cond_proj.0.weight",
            "blocks.0.attn.qkv.weight",
            "blocks.1.mlp.0.weight",
            "final_layer.linear.weight",
        ):
            gradient = named_parameters[name].grad
            self.assertIsNotNone(gradient, name)
            self.assertGreater(
                int(torch.count_nonzero(gradient)), 0, f"{name} 梯度为零"
            )

    def test_gradient_checkpointing_skipped_under_no_grad(self) -> None:
        model = _small_model(gradient_checkpointing=True).eval()
        x = torch.randn(1, 2, 8, 8)

        with torch.no_grad():
            output = model(x, 0, _conditions(1, 8, 8))

        self.assertEqual(tuple(output.shape), tuple(x.shape))
        self.assertFalse(output.requires_grad)


class DiTPositionEmbeddingTest(unittest.TestCase):
    def test_rectangular_2d_embedding_shape_and_variation(self) -> None:
        embedding = build_2d_sincos_pos_embed(3, 5, 32)

        self.assertEqual(tuple(embedding.shape), (15, 32))
        self.assertTrue(torch.isfinite(embedding).all())
        # 不同位置必须有不同的编码（首行内、列间均不同）。
        self.assertFalse(torch.allclose(embedding[0], embedding[1]))
        self.assertFalse(torch.allclose(embedding[0], embedding[5]))

    def test_odd_embedding_dimension_and_invalid_sizes(self) -> None:
        embedding = build_2d_sincos_pos_embed(2, 2, 31, dtype=torch.float64)

        self.assertEqual(tuple(embedding.shape), (4, 31))
        self.assertEqual(embedding.dtype, torch.float64)
        with self.assertRaises(ValueError):
            build_2d_sincos_pos_embed(0, 4, 8)
        with self.assertRaises(ValueError):
            build_2d_sincos_pos_embed(4, 0, 8)
        with self.assertRaises(ValueError):
            build_2d_sincos_pos_embed(4, 4, 0)


if __name__ == "__main__":
    unittest.main()
