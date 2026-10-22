"""ConditionalDiT 与工厂、配置和 checkpoint 信封的集成测试。

这里只覆盖跨模块契约，不重复 ``tests/test_dit.py`` 的单元测试：

* ``diffpeak.models`` 只导出 ``ConditionalDiT`` 去噪器；
* ``build_models`` 在 ``models["dit"]`` 下构造 DiT，并把 DiT 的
  ``condition_channels`` 同步给 ConditionEncoder；
* 旧 U-Net 快照被显式拒绝，形状/键不匹配时也不会静默使用随机权重；
* DiT 能直接接入 ``GaussianDiffusion`` 的训练与采样接口。
"""

from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path
from typing import Any, Mapping
from unittest.mock import patch

import torch
from torch import nn

from diffpeak.inference import (
    DEFAULT_CONFIG,
    build_models,
    load_checkpoint_models,
    load_config,
)
from diffpeak.models import ConditionalDiT, GaussianDiffusion

# 这些集成测试会构造随机初始化的模型。保存/恢复全局 RNG，避免改变同一
# pytest 进程中后续测试（例如 ConditionEncoder 的高分辨率融合断言）的
# 随机权重，造成与测试顺序相关的宽容差误判。
_MODULE_RNG_STATE = None


def setUpModule() -> None:
    global _MODULE_RNG_STATE
    _MODULE_RNG_STATE = torch.random.get_rng_state()


def tearDownModule() -> None:
    if _MODULE_RNG_STATE is not None:
        torch.random.set_rng_state(_MODULE_RNG_STATE)


class _DummySAM(nn.Module):
    """build_models 只要求可 ``to(device)``；SAM 权重由真实测试覆盖。"""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self.args = args
        self.kwargs = kwargs


def _tiny_dit(**overrides: Any) -> ConditionalDiT:
    kwargs = {
        "patch_size": 4,
        "hidden_size": 32,
        "depth": 1,
        "num_heads": 4,
        "mlp_ratio": 2.0,
        "condition_channels": (4, 8, 8, 8),
        "gradient_checkpointing": False,
    }
    kwargs.update(overrides)
    return ConditionalDiT(**kwargs)


def _tiny_conditions(batch: int = 2, size: int = 16) -> list[torch.Tensor]:
    return [
        torch.randn(batch, 4, size, size),
        torch.randn(batch, 8, size // 2, size // 2),
        torch.randn(batch, 8, size // 4, size // 4),
        torch.randn(batch, 8, max(size // 8, 1), max(size // 8, 1)),
    ]


class ExportAndConfigTest(unittest.TestCase):
    def test_models_package_exports_only_dit(self) -> None:
        import diffpeak.models as models

        self.assertIs(models.ConditionalDiT, ConditionalDiT)
        self.assertIn("ConditionalDiT", models.__all__)
        self.assertFalse(hasattr(models, "ConditionalUNet"))

    def test_default_config_carries_dit_fields(self) -> None:
        dit = DEFAULT_CONFIG["model"]["dit"]
        self.assertEqual(dit["patch_size"], 8)
        self.assertEqual(dit["hidden_size"], 256)
        self.assertEqual(dit["depth"], 6)
        self.assertEqual(dit["num_heads"], 8)
        self.assertEqual(dit["mlp_ratio"], 4.0)
        self.assertEqual(dit["condition_channels"], (64, 128, 256, 256))
        self.assertTrue(dit["gradient_checkpointing"])

    def test_yaml_override_is_not_shadowed_by_flat_legacy_key(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text(
                "model:\n"
                "  dit:\n"
                "    hidden_size: 128\n"
                "    gradient_checkpointing: false\n",
                encoding="utf-8",
            )
            config = load_config(path)

        self.assertEqual(config["model"]["dit"]["hidden_size"], 128)
        self.assertFalse(config["model"]["dit"]["gradient_checkpointing"])
        # 默认值仍补齐其他 DiT 字段。
        self.assertEqual(config["model"]["dit"]["patch_size"], 8)
        # 已移除容易误导的旧扁平开关。
        self.assertNotIn("gradient_checkpointing", config["model"])


class BuildModelsTest(unittest.TestCase):
    def _config(self, directory: Path, **dit_overrides: Any) -> dict:
        config = copy.deepcopy(DEFAULT_CONFIG)
        config["model"]["dit"].update(
            {
                "patch_size": 4,
                "hidden_size": 32,
                "depth": 2,
                "num_heads": 4,
                "mlp_ratio": 2.0,
                "condition_channels": (4, 8, 8, 8),
                "gradient_checkpointing": False,
            }
        )
        config["model"]["dit"].update(dit_overrides)
        checkpoint = directory / "unused_sam.pth"
        checkpoint.write_bytes(b"dummy")
        config["model"]["sam_checkpoint"] = str(checkpoint)
        return config

    def test_dit_key_holds_dit_and_shares_condition_channels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = self._config(Path(directory))
            with patch("diffpeak.inference.RoadSAM", _DummySAM):
                models, _ = build_models(config, "cpu")

        self.assertEqual(
            set(models), {"sam", "condition_encoder", "dit", "diffusion"}
        )
        denoiser = models["dit"]
        self.assertIsInstance(denoiser, ConditionalDiT)
        self.assertEqual(denoiser.condition_channels, (4, 8, 8, 8))
        self.assertFalse(denoiser.gradient_checkpointing)
        # ConditionEncoder 必须与 DiT 使用同一组尺度，否则 forward 会因通道
        # 不匹配而失败。
        self.assertEqual(
            models["condition_encoder"].condition_channels, (4, 8, 8, 8)
        )

    def test_nondefault_channels_keep_encoder_and_dit_aligned(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = self._config(
                Path(directory), condition_channels=(8, 8, 8, 8)
            )
            with patch("diffpeak.inference.RoadSAM", _DummySAM):
                models, _ = build_models(config, "cpu")

        self.assertEqual(models["dit"].condition_channels, (8, 8, 8, 8))
        self.assertEqual(
            models["condition_encoder"].condition_channels, (8, 8, 8, 8)
        )


class LegacyCheckpointTest(unittest.TestCase):
    def _payload(self, dit_state: Mapping[str, Any]) -> dict:
        return {
            "models": {
                "sam": {},
                "condition_encoder": {},
                "dit": dict(dit_state),
                "diffusion": {},
            }
        }

    def _targets(self) -> dict:
        return {
            "sam": nn.Identity(),
            "condition_encoder": nn.Identity(),
            "dit": _tiny_dit(),
            "diffusion": nn.Identity(),
        }

    def test_old_checkpoint_keys_are_not_aliased(self) -> None:
        payload = self._payload({})
        payload["models"]["unet"] = payload["models"].pop("dit")
        with self.assertRaisesRegex(ValueError, "缺少必需模块：dit"):
            load_checkpoint_models(self._targets(), payload)

    def test_attention_head_count_must_match_checkpoint(self) -> None:
        source = _tiny_dit(num_heads=2)
        payload = self._payload(source.state_dict())
        payload["config"] = {"model": {"dit": {"num_heads": 2}}}
        # 头数不同的模型仍可 strict 加载同形状参数，必须额外检查配置。
        targets = self._targets()
        targets["dit"].load_state_dict(source.state_dict(), strict=True)
        with self.assertRaisesRegex(ValueError, "num_heads"):
            load_checkpoint_models(targets, payload)
        payload["config"]["model"]["dit"]["num_heads"] = targets["dit"].num_heads
        load_checkpoint_models(targets, payload)

    def test_mismatched_dit_state_fails_clearly_not_silently(self) -> None:
        payload = self._payload({"bogus.weight": torch.zeros(1)})
        with self.assertRaises(ValueError) as caught:
            load_checkpoint_models(self._targets(), payload)

        message = str(caught.exception)
        self.assertIn("ConditionalDiT", message)
        self.assertIn("不兼容", message)


class DiffusionIntegrationTest(unittest.TestCase):
    def test_training_losses_and_sampling_accept_dit(self) -> None:
        torch.manual_seed(0)
        denoiser = _tiny_dit()
        diffusion = GaussianDiffusion(
            timesteps=8, sampling_steps=2, heatmap_size=16
        )
        conditions = _tiny_conditions()
        target = torch.rand(2, 2, 16, 16)

        losses = diffusion.training_losses(denoiser, conditions, target)
        self.assertEqual(set(losses), {"diff", "heat"})
        self.assertTrue(torch.isfinite(losses["diff"]))
        self.assertTrue(torch.isfinite(losses["heat"]))
        self.assertTrue(losses["diff"].requires_grad)

        with torch.no_grad():
            heatmaps = diffusion.sample(denoiser, conditions)
        self.assertEqual(tuple(heatmaps.shape), (2, 2, 16, 16))
        self.assertTrue(torch.isfinite(heatmaps).all())
        self.assertGreaterEqual(float(heatmaps.min()), 0.0)
        self.assertLessEqual(float(heatmaps.max()), 1.0)


if __name__ == "__main__":
    unittest.main()
