"""多尺度 U-Net 的形状、条件、梯度与扩散接口回归。"""

import unittest

import torch
from torch import nn

from diffpeak.models.unet import ConditionalUNet, SpatialAttention
from diffpeak.models.flow_matching import ConditionalFlowMatching


def tiny_unet(checkpointing=False):
    return ConditionalUNet(
        base_channels=8, channel_mults=(1, 2, 2, 4),
        condition_channels=(4, 8, 8, 8), num_res_blocks=1,
        num_heads=2, gradient_checkpointing=checkpointing,
    )


def conditions(batch=2):
    return [torch.randn(batch, c, h, w) for c, h, w in (
        (4, 17, 19), (8, 9, 10), (8, 5, 5), (8, 3, 3),
    )]


class TestConditionalUNet(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_rectangular_shape_and_zero_head(self):
        model = tiny_unet()
        for height, width in ((16, 16), (17, 19), (8, 11)):
            x = torch.randn(2, 1, height, width)
            output = model(x, 0.42, conditions())
            self.assertEqual(output.shape, x.shape)
            self.assertEqual(output.count_nonzero().item(), 0)

    def test_attention_only_at_lowest_scale(self):
        model = tiny_unet()
        for level in range(4):
            self.assertEqual(isinstance(model.encoders[level].attention, SpatialAttention), level == 3)
            self.assertEqual(isinstance(model.decoders[3 - level].attention, SpatialAttention), level == 3)

    def test_each_condition_and_time_affect_output(self):
        torch.manual_seed(7)
        model = tiny_unet().eval()
        nn.init.normal_(model.output_conv.weight, std=0.02)
        x = torch.randn(2, 1, 17, 19)
        features = conditions()
        baseline = model(x, 0.1, features)
        for level in range(4):
            changed = list(features)
            changed[level] = changed[level] + torch.randn_like(changed[level])
            self.assertGreater((model(x, 0.1, changed) - baseline).abs().max().item(), 1e-6)
        self.assertGreater((model(x, 0.9, features) - baseline).abs().max().item(), 1e-6)

    def test_gradients_with_frozen_inputs_and_eval_checkpointing(self):
        model = tiny_unet(checkpointing=True).eval()
        nn.init.normal_(model.output_conv.weight, std=0.02)
        features = [feature.requires_grad_() for feature in conditions()]
        output = model(torch.randn(2, 1, 17, 19), torch.tensor([0.01, 0.5]), features)
        output.square().mean().backward()
        for feature in features:
            self.assertIsNotNone(feature.grad)
            self.assertGreater(feature.grad.abs().sum().item(), 0)
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        model.zero_grad(set_to_none=True)
        model(torch.randn(2, 1, 17, 19), 0.01, conditions()).sum().backward()
        self.assertGreater(model.output_conv.weight.grad.abs().sum().item(), 0)

    def test_checkpoint_matches_forward_and_backward(self):
        eager = tiny_unet()
        nn.init.normal_(eager.output_conv.weight, std=0.02)
        checkpointed = tiny_unet(checkpointing=True)
        checkpointed.load_state_dict(eager.state_dict())
        x, features = torch.randn(2, 1, 17, 19), conditions()
        expected, actual = eager(x, 0.2, features), checkpointed(x, 0.2, features)
        torch.testing.assert_close(expected, actual)
        expected.square().mean().backward()
        actual.square().mean().backward()
        for first, second in zip(eager.parameters(), checkpointed.parameters()):
            torch.testing.assert_close(first.grad, second.grad)

    def test_diffusion_training_and_differentiable_sampling(self):
        model = tiny_unet(checkpointing=True)
        diffusion = ConditionalFlowMatching(sampling_steps=2)
        features = conditions()
        target = torch.randn(2, 1, 17, 19).clamp(-1, 1)
        losses = diffusion.training_losses(model, features, target)
        losses["flow"].backward()
        self.assertGreater(model.output_conv.weight.grad.abs().sum().item(), 0)
        model.zero_grad(set_to_none=True)
        sampled = diffusion.sample_differentiable(model, features)
        self.assertEqual(sampled.shape, target.shape)
        sampled.square().mean().backward()
        self.assertTrue(torch.isfinite(model.output_conv.weight.grad).all())

    def test_invalid_configuration_and_inputs(self):
        for kwargs in (
            {"base_channels": 0}, {"channel_mults": (1, 2)},
            {"condition_channels": (4, 8)}, {"num_res_blocks": 0},
            {"attention_levels": (0,)}, {"attention_levels": (1,)},
            {"attention_levels": (2.0,)}, {"attention_levels": (True,)},
            {"num_heads": 3},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                ConditionalUNet(**kwargs)
        model = tiny_unet()
        x, features = torch.randn(2, 1, 17, 19), conditions()
        with self.assertRaises(ValueError):
            model(x, torch.tensor([1, 2, 3]), features)
        with self.assertRaises(ValueError):
            model(x, 1.1, features)
        with self.assertRaises(ValueError):
            model(x, 1, features[:3])
        with self.assertRaises(ValueError):
            model(x, 1, [features[0][:1], *features[1:]])


if __name__ == "__main__":
    unittest.main()
