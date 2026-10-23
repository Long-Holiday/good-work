"""ConditionalFlowMatching / ConditionalUNet 与工厂、配置和 checkpoint 信封的集成测试。

这里只覆盖跨模块契约，不重复 ``tests/test_unet.py`` / ``tests/test_flow_matching.py``
的模型单元测试：

* ``diffpeak.models`` 导出 ``ConditionalUNet`` 与 ``ConditionalFlowMatching``；
  旧 ``ConditionalDiT`` 仍可从 ``diffpeak.models.dit`` 直接导入；
* ``build_models`` 在 ``models["diffusion"]`` 通用槽位构造
  ``ConditionalFlowMatching``，并保持 UNet/ConditionEncoder 的 condition 通道对齐；
* 旧 ``diffusion`` 配置键不会被默认 ``flow_matching`` 静默遮蔽；
* 旧 DiT 模块键、旧 GaussianDiffusion 状态、形状/注意力头数不匹配均被显式拒绝；
* Flow Matching 容器能直接接入 ``training_losses`` / 采样接口。
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
    DENOISER_ARCHITECTURE,
    DEFAULT_CONFIG,
    GENERATIVE_PROCESS,
    build_models,
    load_checkpoint_models,
    load_config,
)
from diffpeak.models import ConditionalFlowMatching, ConditionalUNet
from diffpeak.models.diffusion import GaussianDiffusion
from diffpeak.models.dit import ConditionalDiT

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


def _tiny_unet(**overrides: Any) -> ConditionalUNet:
    kwargs = {
        "in_channels": 1,
        "out_channels": 1,
        "base_channels": 8,
        "channel_mults": (1, 2, 2, 4),
        "num_res_blocks": 1,
        "attention_levels": (3,),
        "num_heads": 2,
        "condition_channels": (4, 8, 8, 8),
        "gradient_checkpointing": False,
        "time_scale": 1000.0,
    }
    kwargs.update(overrides)
    return ConditionalUNet(**kwargs)


def _tiny_flow(**overrides: Any) -> ConditionalFlowMatching:
    kwargs = {
        "sampling_steps": 2,
        "latent_channels": 1,
        "heatmap_size": 16,
        "time_scale": 1000.0,
        "field_loss_min_time": 0.5,
        "solver": "heun",
        "policy_std": 0.02,
    }
    kwargs.update(overrides)
    return ConditionalFlowMatching(**kwargs)


def _tiny_conditions(batch: int = 2, size: int = 16) -> list[torch.Tensor]:
    return [
        torch.randn(batch, 4, size, size),
        torch.randn(batch, 8, size // 2, size // 2),
        torch.randn(batch, 8, size // 4, size // 4),
        torch.randn(batch, 8, max(size // 8, 1), max(size // 8, 1)),
    ]


class ExportAndConfigTest(unittest.TestCase):
    def test_models_package_exports_fm_unet_and_keeps_legacy_dit_import(self) -> None:
        import diffpeak.models as models

        self.assertIs(models.ConditionalUNet, ConditionalUNet)
        self.assertIs(models.ConditionalFlowMatching, ConditionalFlowMatching)
        self.assertIn("ConditionalUNet", models.__all__)
        self.assertIn("ConditionalFlowMatching", models.__all__)
        self.assertNotIn("ConditionalDiT", models.__all__)
        # 历史实现仍可直接从子模块导入，供 tests/test_dit.py 使用。
        self.assertEqual(ConditionalDiT.__name__, "ConditionalDiT")

    def test_default_config_carries_flow_matching_fields(self) -> None:
        model = DEFAULT_CONFIG["model"]
        self.assertEqual(model["denoiser_architecture"], DENOISER_ARCHITECTURE)
        self.assertEqual(DEFAULT_CONFIG["generative_process"], GENERATIVE_PROCESS)
        self.assertNotIn("dit", model)
        self.assertNotIn("diffusion", DEFAULT_CONFIG)
        unet = model["unet"]
        self.assertEqual(unet["base_channels"], 64)
        self.assertEqual(unet["channel_mults"], (1, 2, 4, 4))
        self.assertEqual(unet["num_res_blocks"], 2)
        self.assertEqual(unet["attention_levels"], (3,))
        self.assertEqual(unet["num_heads"], 8)
        self.assertEqual(unet["condition_channels"], (64, 128, 256, 256))
        self.assertEqual(unet["time_scale"], 1000.0)
        self.assertTrue(unet["gradient_checkpointing"])
        flow = DEFAULT_CONFIG["flow_matching"]
        self.assertEqual(flow["sampling_steps"], 20)
        self.assertEqual(flow["latent_channels"], 1)
        self.assertEqual(flow["policy_std"], 0.02)
        self.assertEqual(flow["solver"], "heun")
        self.assertEqual(flow["time_scale"], 1000.0)
        self.assertEqual(flow["field_loss_min_time"], 0.5)

    def test_yaml_flow_override_is_not_shadowed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text(
                "model:\n"
                "  unet:\n"
                "    base_channels: 32\n"
                "flow_matching:\n"
                "  sampling_steps: 8\n"
                "  policy_std: 0.05\n",
                encoding="utf-8",
            )
            config = load_config(path)

        self.assertEqual(config["model"]["unet"]["base_channels"], 32)
        self.assertEqual(config["flow_matching"]["sampling_steps"], 8)
        self.assertEqual(config["flow_matching"]["policy_std"], 0.05)
        # 默认值仍补齐其他字段。
        self.assertEqual(config["flow_matching"]["solver"], "heun")
        self.assertNotIn("diffusion", config)

    def test_legacy_diffusion_yaml_is_rejected_not_silently_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text(
                "diffusion:\n"
                "  timesteps: 1000\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "diffusion"):
                load_config(path)


class BuildModelsTest(unittest.TestCase):
    def _config(self, directory: Path, **unet_overrides: Any) -> dict:
        config = copy.deepcopy(DEFAULT_CONFIG)
        config["model"]["unet"].update(
            {
                "base_channels": 8,
                "channel_mults": (1, 2, 2, 4),
                "num_res_blocks": 1,
                "attention_levels": (3,),
                "num_heads": 2,
                "condition_channels": (4, 8, 8, 8),
                "gradient_checkpointing": False,
                "time_scale": 1000.0,
            }
        )
        config["model"]["unet"].update(unet_overrides)
        config["flow_matching"].update(
            {
                "sampling_steps": 4,
                "heatmap_size": 16,
                "policy_std": 0.02,
                "solver": "heun",
            }
        )
        checkpoint = directory / "unused_sam.pth"
        checkpoint.write_bytes(b"dummy")
        config["model"]["sam_checkpoint"] = str(checkpoint)
        return config

    def test_diffusion_slot_holds_flow_matching_and_shares_channels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = self._config(Path(directory))
            with patch("diffpeak.inference.RoadSAM", _DummySAM):
                models, _ = build_models(config, "cpu")

        self.assertEqual(
            set(models), {"sam", "condition_encoder", "unet", "diffusion"}
        )
        denoiser = models["unet"]
        self.assertIsInstance(denoiser, ConditionalUNet)
        self.assertNotIsInstance(denoiser, ConditionalDiT)
        self.assertEqual(denoiser.condition_channels, (4, 8, 8, 8))
        self.assertEqual(denoiser.num_heads, 2)
        self.assertEqual(denoiser.time_scale, 1000.0)
        flow = models["diffusion"]
        self.assertIsInstance(flow, ConditionalFlowMatching)
        self.assertEqual(flow.sampling_steps, 4)
        self.assertEqual(flow.policy_std, 0.02)
        self.assertEqual(flow.time_scale, 1000.0)
        self.assertIn("flow_signature", flow.state_dict())
        self.assertEqual(
            models["condition_encoder"].condition_channels, (4, 8, 8, 8)
        )

    def test_nondefault_channels_keep_encoder_and_unet_aligned(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = self._config(
                Path(directory), condition_channels=(8, 8, 8, 8)
            )
            with patch("diffpeak.inference.RoadSAM", _DummySAM):
                models, _ = build_models(config, "cpu")

        self.assertEqual(models["unet"].condition_channels, (8, 8, 8, 8))
        self.assertEqual(
            models["condition_encoder"].condition_channels, (8, 8, 8, 8)
        )

    def test_legacy_diffusion_model_config_cannot_build_fm(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = self._config(Path(directory))
            config["diffusion"] = {"timesteps": 1000}
            with patch("diffpeak.inference.RoadSAM", _DummySAM):
                with self.assertRaisesRegex(ValueError, "diffusion"):
                    build_models(config, "cpu")


class LegacyCheckpointTest(unittest.TestCase):
    def _payload(self, unet_state: Mapping[str, Any]) -> dict:
        return {
            "format_version": 7,
            "training_protocol": "signed_ridge_v1",
            "target_representation": "signed_ridge_v1",
            "denoiser_architecture": DENOISER_ARCHITECTURE,
            "generative_process": GENERATIVE_PROCESS,
            "latent_channels": 1,
            "sam_frozen": True,
            "prompt": {"scale": 1.0, "terminal_mapping": "clamp"},
            "config": {
                "generative_process": GENERATIVE_PROCESS,
                "model": {
                    "denoiser_architecture": DENOISER_ARCHITECTURE,
                    "unet": {"num_heads": 2},
                },
            },
            "models": {
                "sam": {},
                "condition_encoder": {},
                "unet": dict(unet_state),
                # 与实际目标容器具有相同元数据 buffers。
                "diffusion": _tiny_flow().state_dict(),
            },
        }

    def _targets(self) -> dict:
        return {
            "sam": nn.Identity(),
            "condition_encoder": nn.Identity(),
            "unet": _tiny_unet(),
            "diffusion": _tiny_flow(),
        }

    def test_legacy_dit_checkpoint_modules_are_rejected(self) -> None:
        payload = self._payload({})
        payload["models"]["dit"] = payload["models"].pop("unet")
        with self.assertRaisesRegex(ValueError, "旧 DiT"):
            load_checkpoint_models(self._targets(), payload)

        payload = self._payload({})
        payload["models"].pop("unet")
        with self.assertRaisesRegex(ValueError, "缺少必需模块：unet"):
            load_checkpoint_models(self._targets(), payload)

    def test_attention_head_count_must_match_checkpoint(self) -> None:
        source = _tiny_unet(num_heads=4)
        payload = self._payload(source.state_dict())
        payload["config"]["model"]["unet"]["num_heads"] = 4
        targets = self._targets()
        targets["unet"].load_state_dict(source.state_dict(), strict=True)
        with self.assertRaisesRegex(ValueError, "num_heads"):
            load_checkpoint_models(targets, payload)
        payload["config"]["model"]["unet"]["num_heads"] = targets["unet"].num_heads
        load_checkpoint_models(targets, payload)

    def test_mismatched_unet_state_fails_clearly_not_silently(self) -> None:
        payload = self._payload({"bogus.weight": torch.zeros(1)})
        with self.assertRaises(ValueError) as caught:
            load_checkpoint_models(self._targets(), payload)

        message = str(caught.exception)
        self.assertIn("ConditionalUNet", message)
        self.assertIn("不兼容", message)

    def test_old_gaussian_diffusion_state_is_rejected(self) -> None:
        old_flow = GaussianDiffusion(
            timesteps=8, sampling_steps=2, heatmap_size=16
        )
        payload = self._payload(_tiny_unet().state_dict())
        payload["models"]["diffusion"] = old_flow.state_dict()
        targets = self._targets()
        targets["diffusion"] = _tiny_flow()
        with self.assertRaises(ValueError) as caught:
            load_checkpoint_models(targets, payload)
        message = str(caught.exception)
        self.assertIn("ConditionalFlowMatching", message)


class FlowMatchingIntegrationTest(unittest.TestCase):
    def test_training_losses_and_sampling_accept_fm_unet(self) -> None:
        torch.manual_seed(0)
        denoiser = _tiny_unet()
        flow = _tiny_flow()
        conditions = _tiny_conditions()
        target = torch.rand(2, 1, 16, 16) * 2.0 - 1.0

        losses = flow.training_losses(
            denoiser, conditions, target, return_prediction=True
        )
        self.assertEqual(set(losses), {"flow", "heat", "heat_pred"})
        self.assertTrue(torch.isfinite(losses["flow"]))
        self.assertTrue(torch.isfinite(losses["heat"]))
        self.assertTrue(losses["flow"].requires_grad)
        self.assertEqual(tuple(losses["heat_pred"].shape), (2, 1, 16, 16))
        self.assertGreaterEqual(float(losses["heat_pred"].min()), -1.0)
        self.assertLessEqual(float(losses["heat_pred"].max()), 1.0)

        with torch.no_grad():
            heatmaps = flow.sample(denoiser, conditions)
        self.assertEqual(tuple(heatmaps.shape), (2, 1, 16, 16))
        self.assertTrue(torch.isfinite(heatmaps).all())
        self.assertGreaterEqual(float(heatmaps.min()), -1.0)
        self.assertLessEqual(float(heatmaps.max()), 1.0)


if __name__ == "__main__":
    unittest.main()
