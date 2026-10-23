"""diffpeak.inference 的 signed_ridge_v1 推理、协议与迁移边界测试。"""

from __future__ import annotations

import copy
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diffpeak.inference import (  # noqa: E402
    CHECKPOINT_FORMAT_VERSION,
    CheckpointProtocolError,
    DEFAULT_CONFIG,
    DENOISER_ARCHITECTURE,
    GENERATIVE_PROCESS,
    JOINT_ABC_PROTOCOL,
    PROMPT_TERMINAL_MAPPINGS,
    SIGNED_RIDGE_PROTOCOL,
    TARGET_REPRESENTATION,
    Predictor,
    _validate_sam_architecture,
    aggregate_metrics,
    apply_terminal_mapping,
    build_models,
    load_checkpoint_models,
    load_compatible_weights,
    load_labeled_split,
    load_mask_file,
    make_generator,
    normalize_valid_sizes,
    prepare_image,
    prepare_image_file,
    resolve_prompt_mapping,
    save_mask_file,
    stable_seed,
    uses_joint_abc_protocol,
    uses_signed_ridge_protocol,
    validate_checkpoint_protocol,
    validate_signed_ridge_config,
)


def _metadata(**overrides):
    """构造完整 signed_ridge_v1 快照元数据。"""
    prompt = {"scale": 1.0, "terminal_mapping": "clamp"}
    prompt.update(overrides.pop("prompt", {}))
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["prompt"] = dict(prompt)
    config["model"]["denoiser_architecture"] = DENOISER_ARCHITECTURE
    config.update(overrides.pop("config", {}))
    payload = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "training_protocol": SIGNED_RIDGE_PROTOCOL,
        "target_representation": TARGET_REPRESENTATION,
        "denoiser_architecture": DENOISER_ARCHITECTURE,
        "generative_process": GENERATIVE_PROCESS,
        "latent_channels": 1,
        "sam_frozen": True,
        "prompt": prompt,
        "config": config,
    }
    payload.update(overrides)
    return payload


def _payload_with_models(models, **overrides):
    payload = _metadata(**overrides)
    payload["models"] = models
    return payload


class PrepareImageTest(unittest.TestCase):
    def test_uint8_image_is_resized_and_padded(self) -> None:
        image = np.full((100, 200, 3), 100, dtype=np.uint8)

        tensor, original_size, resized_size = prepare_image(image, image_size=64)

        self.assertEqual(original_size, (100, 200))
        self.assertEqual(resized_size, (32, 64))
        self.assertEqual(tuple(tensor.shape), (1, 3, 64, 64))
        self.assertEqual(tensor.dtype, torch.float32)
        self.assertTrue(torch.all(tensor[0, :, 32:, :] == 0.0))

    def test_right_padding_for_portrait_image(self) -> None:
        image = np.full((200, 100, 3), 255, dtype=np.uint8)

        tensor, original_size, resized_size = prepare_image(image, image_size=64)

        self.assertEqual(original_size, (200, 100))
        self.assertEqual(resized_size, (64, 32))
        self.assertTrue(torch.all(tensor[0, :, :, 32:] == 0.0))

    def test_grayscale_and_torch_chw_inputs(self) -> None:
        gray = np.full((10, 20), 100, dtype=np.uint8)
        gray_tensor, _, _ = prepare_image(gray, image_size=32)
        self.assertEqual(tuple(gray_tensor.shape), (1, 3, 32, 32))

        chw = torch.full((3, 10, 20), 100, dtype=torch.uint8)
        chw_tensor, original, _ = prepare_image(chw, image_size=32)
        self.assertEqual(original, (10, 20))
        self.assertEqual(tuple(chw_tensor.shape), (1, 3, 32, 32))

    def test_float_image_in_unit_range_is_scaled(self) -> None:
        image = np.full((8, 8, 3), 0.5, dtype=np.float32)

        tensor, _, _ = prepare_image(image, image_size=16)

        self.assertGreaterEqual(float(tensor.min()), 0.0)
        self.assertLessEqual(float(tensor.max()), 1.0)
        self.assertAlmostEqual(
            float(tensor[0, :, :8, :8].mean()), 127.0 / 255.0, places=3
        )

    def test_invalid_ndim_raises(self) -> None:
        with self.assertRaises(ValueError):
            prepare_image(np.zeros((2, 2, 2, 3), dtype=np.uint8), image_size=16)


class ValidSizesTest(unittest.TestCase):
    def test_none_and_sequence(self) -> None:
        self.assertEqual(normalize_valid_sizes(None, 2), [None, None])
        self.assertEqual(
            normalize_valid_sizes([(10, 20), (30, 40)], 2),
            [(10, 20), (30, 40)],
        )
        self.assertEqual(normalize_valid_sizes((10, 20), 1), [(10, 20)])

    def test_tensor_forms(self) -> None:
        pair = normalize_valid_sizes(torch.tensor([10, 20]), 1)
        batch = normalize_valid_sizes(torch.tensor([[10, 20], [30, 40]]), 2)
        collated = normalize_valid_sizes(
            [torch.tensor([10, 30]), torch.tensor([20, 40])], 2
        )

        self.assertEqual(pair, [(10, 20)])
        self.assertEqual(batch, [(10, 20), (30, 40)])
        self.assertEqual(collated, [(10, 20), (30, 40)])

    def test_invalid_value_raises(self) -> None:
        with self.assertRaises(ValueError):
            normalize_valid_sizes(123, 1)
        with self.assertRaises(ValueError):
            normalize_valid_sizes([(10, 20)], 2)


class ProtocolConfigTest(unittest.TestCase):
    def test_constants_and_aliases_are_one_protocol(self) -> None:
        self.assertEqual(SIGNED_RIDGE_PROTOCOL, "signed_ridge_v1")
        self.assertEqual(TARGET_REPRESENTATION, "signed_ridge_v1")
        self.assertEqual(JOINT_ABC_PROTOCOL, SIGNED_RIDGE_PROTOCOL)
        self.assertEqual(PROMPT_TERMINAL_MAPPINGS, ("clamp",))

    def test_default_config_carries_signed_ridge_fields(self) -> None:
        config = DEFAULT_CONFIG
        self.assertEqual(config["training_protocol"], SIGNED_RIDGE_PROTOCOL)
        self.assertEqual(
            config["target_representation"], TARGET_REPRESENTATION
        )
        self.assertTrue(config["sam_frozen"])
        self.assertTrue(config["model"]["sam_frozen"])
        self.assertEqual(config["model"]["denoiser_architecture"], DENOISER_ARCHITECTURE)
        self.assertEqual(config["model"]["unet"]["time_scale"], 1000.0)
        self.assertEqual(config["generative_process"], GENERATIVE_PROCESS)
        self.assertEqual(config["model"]["unet"]["in_channels"], 1)
        self.assertEqual(config["model"]["unet"]["out_channels"], 1)
        self.assertEqual(config["model"]["unet"]["base_channels"], 64)
        self.assertEqual(config["model"]["unet"]["channel_mults"], (1, 2, 4, 4))
        self.assertEqual(config["model"]["unet"]["attention_levels"], (3,))
        self.assertNotIn("dit", config["model"])
        self.assertNotIn("diffusion", config)
        self.assertEqual(config["flow_matching"]["latent_channels"], 1)
        self.assertEqual(config["flow_matching"]["sampling_steps"], 20)
        self.assertEqual(config["flow_matching"]["policy_std"], 0.02)
        self.assertEqual(config["flow_matching"]["solver"], "heun")
        self.assertEqual(config["flow_matching"]["field_loss_min_time"], 0.5)
        self.assertEqual(config["prompt"], {"scale": 1.0, "terminal_mapping": "clamp"})
        self.assertEqual(config["reward"]["point_cost"], 0.0)
        self.assertEqual(config["rl"]["anchor_weight"], 0.01)

    def test_protocol_detection_is_exact_and_case_insensitive(self) -> None:
        self.assertTrue(
            uses_signed_ridge_protocol(
                {"training_protocol": " Signed_Ridge_V1 "}
            )
        )
        self.assertTrue(
            uses_joint_abc_protocol({"training_protocol": SIGNED_RIDGE_PROTOCOL})
        )
        for config in (
            {},
            {"training_protocol": None},
            {"training_protocol": "joint_abc_v1"},
            {"training_protocol": 1},
        ):
            self.assertFalse(uses_signed_ridge_protocol(config))

    def test_resolve_prompt_mapping_validates_finite_positive_scale(self) -> None:
        self.assertEqual(
            resolve_prompt_mapping({"prompt": {"scale": 2, "terminal_mapping": "clamp"}}),
            (2.0, "clamp"),
        )
        for scale in (0, -1, float("inf"), float("nan"), "x", None):
            with self.subTest(scale=scale):
                with self.assertRaises(ValueError):
                    resolve_prompt_mapping({"prompt": {"scale": scale}})
        # 第一版只允许 clamp；identity 会造成训练/推理映射不一致。
        with self.assertRaises(ValueError):
            resolve_prompt_mapping(
                {"prompt": {"scale": 1.0, "terminal_mapping": "identity"}}
            )

    def test_config_validation_rejects_wrong_protocol_and_channels(self) -> None:
        validate_signed_ridge_config(DEFAULT_CONFIG)
        bad = copy.deepcopy(DEFAULT_CONFIG)
        bad["training_protocol"] = "joint_abc_v1"
        with self.assertRaises(ValueError):
            validate_signed_ridge_config(bad)
        # 显式迁移旧配置时可以关闭协议要求。
        validate_signed_ridge_config(bad, require_protocol=False)

        legacy_diffusion = copy.deepcopy(DEFAULT_CONFIG)
        legacy_diffusion["diffusion"] = {"latent_channels": 1}
        with self.assertRaisesRegex(ValueError, "diffusion"):
            validate_signed_ridge_config(legacy_diffusion)

        for path, value in (
            (("model", "unet", "in_channels"), 2),
            (("model", "unet", "out_channels"), 2),
            (("model", "unet", "time_scale"), 500.0),
            (("flow_matching", "latent_channels"), 2),
            (("flow_matching", "solver"), "ddim"),
            (("flow_matching", "policy_std"), 0.0),
            (("flow_matching", "field_loss_min_time"), 1.5),
            (("sam_frozen",), False),
            (("prompt", "scale"), 0.0),
        ):
            config = copy.deepcopy(DEFAULT_CONFIG)
            node = config
            for key in path[:-1]:
                node = node[key]
            node[path[-1]] = value
            with self.subTest(path=path):
                with self.assertRaises(ValueError):
                    validate_signed_ridge_config(config)

    def test_terminal_mapping_only_clamps(self) -> None:
        field = torch.tensor([[[[-2.0, 0.5, 2.0]]]])
        clamped = apply_terminal_mapping(field, "clamp")
        self.assertTrue(
            torch.equal(clamped, torch.tensor([[[[-1.0, 0.5, 1.0]]]]))
        )
        with self.assertRaises(ValueError):
            apply_terminal_mapping(field, "identity")


class BuildModelsTest(unittest.TestCase):
    class _DummySAM(nn.Module):
        def __init__(self, *args, prompt_scale: float = 1.0, **kwargs) -> None:
            super().__init__()
            self.prompt_scale = float(prompt_scale)
            self.kwargs = kwargs

    def _config(self, directory: Path) -> dict:
        config = copy.deepcopy(DEFAULT_CONFIG)
        config["model"]["unet"].update(
            {
                "base_channels": 8,
                "channel_mults": (1, 2, 4, 4),
                "num_res_blocks": 1,
                "attention_levels": (3,),
                "num_heads": 4,
                "condition_channels": (8, 16, 32, 32),
                "gradient_checkpointing": False,
            }
        )
        checkpoint = directory / "unused_sam.pth"
        checkpoint.write_bytes(b"dummy")
        config["model"]["sam_checkpoint"] = str(checkpoint)
        config["prompt"]["scale"] = 2.5
        return config

    def test_build_models_is_single_channel_and_has_no_peak_extractor(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = self._config(Path(directory))
            with patch("diffpeak.inference.RoadSAM", self._DummySAM):
                models, second = build_models(config, "cpu")

        self.assertIsNone(second)
        self.assertEqual(
            set(models), {"sam", "condition_encoder", "unet", "diffusion"}
        )
        self.assertEqual(models["unet"].in_channels, 1)
        self.assertEqual(models["unet"].out_channels, 1)
        self.assertEqual(models["unet"].condition_channels, (8, 16, 32, 32))
        self.assertFalse(models["unet"].gradient_checkpointing)
        self.assertEqual(models["diffusion"].latent_channels, 1)
        # diffusion 槽位现在是 Flow Matching 容器，且携带签名 buffer。
        self.assertEqual(models["diffusion"].policy_std, 0.02)
        self.assertIn("flow_signature", models["diffusion"].state_dict())
        self.assertEqual(models["sam"].prompt_scale, 2.5)

    def test_build_models_rejects_explicit_multi_channel(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = self._config(Path(directory))
            config["flow_matching"]["latent_channels"] = 2
            with patch("diffpeak.inference.RoadSAM", self._DummySAM):
                with self.assertRaises(ValueError):
                    build_models(config, "cpu")


class CheckpointProtocolTest(unittest.TestCase):
    def test_complete_metadata_passes_and_is_returned(self) -> None:
        meta = validate_checkpoint_protocol(_metadata())
        self.assertEqual(meta["training_protocol"], SIGNED_RIDGE_PROTOCOL)
        self.assertEqual(meta["target_representation"], TARGET_REPRESENTATION)
        self.assertEqual(meta["latent_channels"], 1)
        self.assertTrue(meta["sam_frozen"])
        self.assertEqual(meta["denoiser_architecture"], DENOISER_ARCHITECTURE)
        self.assertEqual(meta["generative_process"], GENERATIVE_PROCESS)
        self.assertEqual(meta["prompt_scale"], 1.0)
        self.assertEqual(meta["terminal_mapping"], "clamp")

    def test_old_protocol_or_missing_metadata_is_rejected(self) -> None:
        old = _metadata(
            format_version=4,
            training_protocol="joint_abc_v1",
            target_representation="point_heatmap_v1",
        )
        with self.assertRaisesRegex(CheckpointProtocolError, "format_version"):
            validate_checkpoint_protocol(old)

        missing = _metadata()
        missing.pop("target_representation")
        missing["config"].pop("target_representation")
        with self.assertRaisesRegex(CheckpointProtocolError, "target_representation"):
            validate_checkpoint_protocol(missing)

        # 旧 DiT 快照即使其它协议字段齐全，也因缺少 UNet 架构标识被拒绝。
        legacy_dit = _metadata(denoiser_architecture=None)
        legacy_dit["config"]["model"].pop("denoiser_architecture", None)
        with self.assertRaisesRegex(CheckpointProtocolError, "denoiser_architecture"):
            validate_checkpoint_protocol(legacy_dit)

        # 旧扩散快照缺 generative_process 同样拒绝。
        legacy_diffusion = _metadata(generative_process=None)
        legacy_diffusion["config"].pop("generative_process", None)
        with self.assertRaisesRegex(CheckpointProtocolError, "generative_process"):
            validate_checkpoint_protocol(legacy_diffusion)

    def test_root_and_config_contradictions_are_rejected(self) -> None:
        payload = _metadata()
        payload["config"]["training_protocol"] = "joint_abc_v1"
        with self.assertRaisesRegex(CheckpointProtocolError, "不一致"):
            validate_checkpoint_protocol(payload)

        payload = _metadata()
        payload["prompt"]["scale"] = 2.0
        with self.assertRaisesRegex(CheckpointProtocolError, "prompt 在根级"):
            validate_checkpoint_protocol(payload)

        payload = _metadata()
        payload["config"]["sam_frozen"] = False
        with self.assertRaisesRegex(CheckpointProtocolError, "sam_frozen"):
            validate_checkpoint_protocol(payload)

        payload = _metadata()
        payload["denoiser_architecture"] = "conditional_dit_v1"
        with self.assertRaisesRegex(CheckpointProtocolError, "不一致"):
            validate_checkpoint_protocol(payload)

        payload = _metadata()
        payload["generative_process"] = "ddpm_v1"
        with self.assertRaisesRegex(CheckpointProtocolError, "不一致"):
            validate_checkpoint_protocol(payload)

    def test_channel_freeze_and_mapping_are_validated(self) -> None:
        cases = [
            (_metadata(latent_channels=2), "latent_channels"),
            (_metadata(sam_frozen=False), "sam_frozen"),
            (
                _metadata(prompt={"scale": 1.0, "terminal_mapping": "identity"}),
                "terminal_mapping",
            ),
            (
                _metadata(prompt={"scale": 0.0, "terminal_mapping": "clamp"}),
                "prompt.scale",
            ),
        ]
        for payload, pattern in cases:
            with self.subTest(pattern=pattern):
                with self.assertRaisesRegex(CheckpointProtocolError, pattern):
                    validate_checkpoint_protocol(payload)

    def test_current_config_must_match_saved_training_protocol(self) -> None:
        payload = _metadata()
        current = copy.deepcopy(DEFAULT_CONFIG)
        current["prompt"]["scale"] = 3.0
        with self.assertRaisesRegex(CheckpointProtocolError, "prompt"):
            validate_checkpoint_protocol(payload, current)
        current = copy.deepcopy(DEFAULT_CONFIG)
        current["rl"]["kl_weight"] = 0.5
        with self.assertRaisesRegex(CheckpointProtocolError, "rl.kl_weight"):
            validate_checkpoint_protocol(payload, current)

    def test_load_checkpoint_models_rejects_unmarked_or_old_payloads(self) -> None:
        with self.assertRaises(CheckpointProtocolError):
            load_checkpoint_models({}, {"models": {}})
        old = _payload_with_models({}, format_version=4)
        with self.assertRaises(CheckpointProtocolError):
            load_checkpoint_models({}, old)
        # v6 扩散 UNet 快照同样低于 v7，不能加载到 Flow Matching。
        diffusion_v6 = _payload_with_models({}, format_version=6)
        with self.assertRaisesRegex(CheckpointProtocolError, "format_version"):
            load_checkpoint_models({}, diffusion_v6)


class CheckpointSAM2GuardTest(unittest.TestCase):
    def test_old_sam1_snapshot_is_rejected_with_clear_message(self) -> None:
        target = types.SimpleNamespace(
            sam=types.SimpleNamespace(
                sam_mask_decoder=torch.nn.Module(),
                sam_prompt_encoder=torch.nn.Module(),
            )
        )
        payload = _payload_with_models(
            {
                "sam": {
                    "sam.prompt_encoder.embedding.weight": torch.zeros(1),
                    "sam.mask_decoder.iou_token.weight": torch.zeros(1),
                },
                "condition_encoder": {},
                "unet": {},
                "diffusion": {},
            }
        )
        with self.assertRaises(ValueError) as caught:
            load_checkpoint_models({"sam": target}, payload)
        message = str(caught.exception)
        self.assertIn("旧 SAM1", message)
        self.assertIn("SAM2", message)

    def test_mismatched_sam_architecture_is_rejected(self) -> None:
        target = types.SimpleNamespace(
            sam=types.SimpleNamespace(
                sam_mask_decoder=torch.nn.Module(),
                sam_prompt_encoder=torch.nn.Module(),
            ),
            architecture="sam2.1_hiera_base_plus",
        )
        payload = _payload_with_models(
            {
                "sam": {"sam.sam_mask_decoder.conv_s0.weight": torch.zeros(1)},
                "condition_encoder": {},
                "unet": {},
                "diffusion": {},
            },
            config={"model": {"sam_architecture": "sam2.1_hiera_large"}},
        )
        with self.assertRaises(ValueError) as caught:
            load_checkpoint_models({"sam": target}, payload)
        self.assertIn("骨干", str(caught.exception))

    def test_config_and_architecture_must_match(self) -> None:
        _validate_sam_architecture(
            "configs/sam2.1/sam2.1_hiera_b+.yaml", "sam2.1_hiera_base_plus"
        )
        with self.assertRaises(ValueError) as caught:
            _validate_sam_architecture(
                "configs/sam2.1/sam2.1_hiera_l.yaml", "sam2.1_hiera_base_plus"
            )
        self.assertIn("sam2.1_hiera_large", str(caught.exception))


class MigrationTest(unittest.TestCase):
    class _TinyLegacyDit(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.patch_embed = nn.Conv2d(1, 2, 3, padding=1)
            self.blocks = nn.Linear(2, 2)
            self.final_layer = nn.Linear(2, 1)

    def test_migration_loads_only_condition_encoder_and_never_old_dit(self) -> None:
        condition = nn.Linear(2, 2)
        unet = nn.Linear(2, 2)
        legacy_dit = self._TinyLegacyDit()
        sam = nn.Linear(2, 2)
        models = {
            "condition_encoder": condition,
            "unet": unet,
            "sam": sam,
            "diffusion": nn.Linear(1, 1),
        }
        sam_before = {key: value.clone() for key, value in sam.state_dict().items()}
        unet_before = {key: value.clone() for key, value in unet.state_dict().items()}
        payload = {
            "models": {
                "sam": {"weight": torch.full((2, 2), 9.0), "bias": torch.ones(2)},
                "condition_encoder": {
                    "weight": torch.full((2, 2), 3.0),
                    "bias": torch.ones(2),
                    "legacy_only": torch.zeros(1),
                },
                # 旧 DiT 状态即使部分形状与目标 UNet 偶然匹配，也不得迁移。
                "dit": {
                    "patch_embed.weight": torch.zeros_like(
                        legacy_dit.patch_embed.weight
                    ),
                    "blocks.weight": torch.full((2, 2), 4.0),
                    "final_layer.weight": torch.zeros_like(
                        legacy_dit.final_layer.weight
                    ),
                },
                "ref_dit": {"blocks.weight": torch.full((2, 2), 5.0)},
            }
        }

        report = load_compatible_weights(models, payload)

        self.assertTrue(report["migration"])
        self.assertFalse(report["optimizer_restored"])
        self.assertEqual(report["target_architecture"], DENOISER_ARCHITECTURE)
        # 只加载条件编码器兼容权重。
        self.assertIn("weight", report["loaded"]["condition_encoder"])
        self.assertEqual(report["loaded"]["dit"], [])
        self.assertEqual(report["loaded"]["unet"], [])
        self.assertEqual(report["loaded"]["ref_dit"], [])
        self.assertIn("dit", report["skipped_modules"])
        self.assertIn("ref_dit", report["skipped_modules"])
        self.assertEqual(
            report["skipped"]["dit"]["blocks.weight"],
            "legacy_dit_not_migrated",
        )
        self.assertEqual(
            report["skipped"]["ref_dit"]["blocks.weight"],
            "legacy_dit_not_migrated",
        )
        self.assertEqual(report["reinitialized"]["dit"], [])
        self.assertEqual(
            set(report["reinitialized"]["unet"]), set(unet_before)
        )
        self.assertTrue(torch.all(condition.weight == 3.0))
        # 目标 UNet 保持构建时的初始化，绝不会接收旧 DiT 权重。
        for key, value in unet.state_dict().items():
            self.assertTrue(torch.equal(value, unet_before[key]))
        # 冻结 SAM 必须保持构建时的官方权重，不能被旧快照覆盖。
        for key, value in sam.state_dict().items():
            self.assertTrue(torch.equal(value, sam_before[key]))


class PredictorSignedRidgeTest(unittest.TestCase):
    class _DummySam:
        no_positive_logit = -20.0

        def __init__(self, prompt_scale=None) -> None:
            self.heatmap_calls = []
            if prompt_scale is not None:
                self.prompt_scale = prompt_scale

        def encode(self, images, valid_sizes=None):
            return torch.zeros(images.shape[0], 256, 4, 4)

        def decode_heatmaps(
            self, embeddings, heatmaps, output_size=(1024, 1024), valid_sizes=None
        ):
            self.heatmap_calls.append(
                {
                    "heatmaps": heatmaps.detach().clone(),
                    "output_size": output_size,
                    "valid_sizes": valid_sizes,
                }
            )
            return torch.zeros(embeddings.shape[0], 1, *output_size)

    class _DummyConditionEncoder:
        def __call__(self, images, embeddings):
            batch = images.shape[0]
            return [torch.zeros(batch, 4, 4, 4) for _ in range(4)], []

    class _DummyDiffusion:
        def __init__(self, field: torch.Tensor) -> None:
            self.field = field

        def sample(self, dit, conditions, generator=None, record=False):
            return self.field.clone()

    def _predictor(self, *, sam_prompt_scale=None, field=None):
        predictor = object.__new__(Predictor)
        predictor.config = copy.deepcopy(DEFAULT_CONFIG)
        predictor.config_dir = None
        predictor.device = torch.device("cpu")
        predictor.seed = 7
        predictor.image_size = 1024
        predictor.amp = False
        predictor.threshold = 0.5
        predictor.prompt_scale = 2.0
        predictor.terminal_mapping = "clamp"
        predictor.prompt_mode = "signed_ridge"
        predictor.joint_abc = True
        predictor.sam = self._DummySam(prompt_scale=sam_prompt_scale)
        predictor.sam_applies_prompt_scale = sam_prompt_scale is not None
        predictor.condition_encoder = self._DummyConditionEncoder()
        predictor.unet = object()
        if field is None:
            field = torch.zeros(1, 1, 4, 4)
            field[0, 0, 1:3, 1:3] = 0.75
        predictor.diffusion = self._DummyDiffusion(field)
        return predictor

    def test_predict_array_decodes_signed_field_without_peak_extraction(self) -> None:
        predictor = self._predictor()

        result = Predictor.predict_array(
            predictor, np.zeros((8, 8, 3), dtype=np.uint8), image_id="scene"
        )

        self.assertEqual(result["prompt_mode"], "signed_ridge")
        self.assertEqual(result["target_representation"], TARGET_REPRESENTATION)
        self.assertNotIn("points", result)
        self.assertNotIn("point_count", result)
        self.assertNotIn("has_positive", result)
        self.assertFalse(hasattr(predictor, "peak_extractor"))
        self.assertEqual(len(predictor.sam.heatmap_calls), 1)
        call = predictor.sam.heatmap_calls[0]
        # 映射固定为 U = scale * clamp(H)，不是 clamp(scale * H)。
        expected = (
            torch.clamp(predictor.diffusion.field, -1.0, 1.0)
            * predictor.prompt_scale
        )
        self.assertTrue(torch.equal(call["heatmaps"], expected))
        self.assertEqual(call["valid_sizes"], [tuple(result["resized_size"])])
        self.assertEqual(tuple(result["field"].shape), (1, 4, 4))
        self.assertEqual(tuple(result["prompt_field"].shape), (1, 4, 4))
        # 概率恰为 threshold 时按 >= 判定为前景（与 engine/D 一致）。
        self.assertTrue(np.all(result["mask"] > 0))

    def test_sam_prompt_scale_is_not_applied_twice(self) -> None:
        predictor = self._predictor(sam_prompt_scale=2.0)

        Predictor.predict_array(
            predictor, np.zeros((8, 8, 3), dtype=np.uint8), image_id="scene"
        )

        expected = torch.clamp(predictor.diffusion.field, -1.0, 1.0)
        self.assertTrue(
            torch.equal(predictor.sam.heatmap_calls[0]["heatmaps"], expected)
        )

    def test_predictor_rejects_old_checkpoint_before_building_models(self) -> None:
        config = copy.deepcopy(DEFAULT_CONFIG)
        old_payload = _metadata(
            format_version=4, training_protocol="joint_abc_v1"
        )
        with patch(
            "diffpeak.inference.resolve_sam_checkpoint",
            return_value=Path("unused_sam.pth"),
        ), patch(
            "diffpeak.inference.build_models",
            side_effect=AssertionError("不应为旧快照构建模型"),
        ):
            with self.assertRaises(CheckpointProtocolError):
                Predictor(config, payload=old_payload, device="cpu")


class GeometryAndFilesTest(unittest.TestCase):
    def test_crop_resize_removes_padding(self) -> None:
        from diffpeak.inference import crop_resize_probability

        probability = torch.zeros(8, 8)
        probability[4:, 4:] = 1.0
        result = crop_resize_probability(
            probability, original_size=(4, 4), resized_size=(4, 4)
        )
        self.assertEqual(result.shape, (4, 4))
        self.assertFalse(result.any())

    def test_predict_file_does_not_read_same_name_mask(self) -> None:
        class _StubPredictor:
            def predict_array(
                self, image, image_id=None, seed=None, return_field=True
            ):
                return {
                    "image_id": image_id,
                    "shape": tuple(image.shape),
                    "return_field": return_field,
                }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "scene_sat.jpg"
            Image.fromarray(
                np.full((12, 20, 3), 128, dtype=np.uint8), mode="RGB"
            ).save(image_path)
            # 同名掩码位置故意放一个目录：一旦推理读取掩码就会失败。
            (root / "scene_mask.png").mkdir()

            result = Predictor.predict_file(_StubPredictor(), image_path)

        self.assertEqual(result["image_id"], "scene")
        self.assertEqual(result["shape"], (12, 20, 3))
        self.assertEqual(result["source_path"], str(image_path))

    def test_load_labeled_split_does_not_require_mask_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "valid").mkdir()
            Image.fromarray(
                np.zeros((8, 8, 3), dtype=np.uint8), mode="RGB"
            ).save(root / "valid" / "a_sat.jpg")

            entries = load_labeled_split(["valid/a_sat.jpg"], root)

        self.assertEqual(entries[0]["id"], "a")
        self.assertEqual(entries[0]["image_path"], root / "valid" / "a_sat.jpg")
        self.assertEqual(entries[0]["mask_path"], root / "valid" / "a_mask.png")

    def test_prepare_image_file_needs_image_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "unlabeled_sat.jpg"
            Image.fromarray(
                np.zeros((10, 16, 3), dtype=np.uint8), mode="RGB"
            ).save(path)

            tensor, original, resized = prepare_image_file(path, image_size=32)

        self.assertEqual(original, (10, 16))
        self.assertEqual(resized, (20, 32))
        self.assertEqual(tuple(tensor.shape), (1, 3, 32, 32))

    def test_mask_roundtrip(self) -> None:
        mask = np.zeros((6, 8), dtype=bool)
        mask[2, 3:6] = True

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "pred.png"
            save_mask_file(mask, path)
            loaded = load_mask_file(path)

        self.assertTrue(np.array_equal(loaded, mask))

    def test_stable_seed_is_reproducible_and_identifier_dependent(self) -> None:
        first = stable_seed(42, "infer", "scene")
        self.assertEqual(first, stable_seed(42, "infer", "scene"))
        self.assertNotEqual(first, stable_seed(42, "infer", "other"))
        self.assertNotEqual(first, stable_seed(43, "infer", "scene"))
        self.assertLess(first, 1 << 63)

    def test_generators_with_same_seed_match(self) -> None:
        first = make_generator(123)
        second = make_generator(123)
        self.assertTrue(
            torch.equal(
                torch.randn(8, generator=first), torch.randn(8, generator=second)
            )
        )

    def test_aggregate_metrics_ignores_nan(self) -> None:
        summary = aggregate_metrics(
            [
                {"reward": 1.0, "iou": 1.0, "f1": 1.0, "cldice": 1.0, "path": 0.0},
                {"reward": 0.0, "iou": 0.0, "f1": 0.0, "cldice": float("nan"), "path": 0.0},
            ]
        )
        self.assertAlmostEqual(summary["iou"], 0.5)
        self.assertAlmostEqual(summary["cldice"], 1.0)
        self.assertAlmostEqual(summary["score"], 0.5)


if __name__ == "__main__":
    unittest.main()
