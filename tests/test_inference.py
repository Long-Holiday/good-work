"""diffpeak.inference 的无标注推理边界与几何工具测试。

重点：
* 推理入口只读影像，不要求同名 GT 掩码存在；
* 没有正提示点时返回空掩码 logits（不强行补点）；
* 图像缩放/填充与点坐标恢复遵循固定几何。
"""

from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diffpeak.inference import (  # noqa: E402
    Predictor,
    aggregate_metrics,
    decode_with_prompts,
    _validate_sam_architecture,
    load_checkpoint_models,
    load_image_file,
    load_labeled_split,
    load_mask_file,
    make_generator,
    map_points_to_original,
    normalize_valid_sizes,
    points_to_prompt,
    prepare_image,
    prepare_image_file,
    save_mask_file,
    stable_seed,
)


class PrepareImageTest(unittest.TestCase):
    def test_uint8_image_is_resized_and_padded(self) -> None:
        image = np.full((100, 200, 3), 100, dtype=np.uint8)

        tensor, original_size, resized_size = prepare_image(image, image_size=64)

        self.assertEqual(original_size, (100, 200))
        self.assertEqual(resized_size, (32, 64))
        self.assertEqual(tuple(tensor.shape), (1, 3, 64, 64))
        self.assertEqual(tensor.dtype, torch.float32)
        self.assertTrue(torch.allclose(tensor[0, :, :32, :], torch.full_like(tensor[0, :, :32, :], 100.0 / 255.0)))
        self.assertTrue(torch.all(tensor[0, :, 32:, :] == 0.0))

    def test_right_padding_for_portrait_image(self) -> None:
        image = np.full((200, 100, 3), 255, dtype=np.uint8)

        tensor, original_size, resized_size = prepare_image(image, image_size=64)

        self.assertEqual(original_size, (200, 100))
        self.assertEqual(resized_size, (64, 32))
        self.assertEqual(tuple(tensor.shape), (1, 3, 64, 64))
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
        self.assertAlmostEqual(float(tensor[0, :, :8, :8].mean()), 127.0 / 255.0, places=3)

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


class PromptBoundaryTest(unittest.TestCase):
    def test_no_points_or_no_positive_returns_none(self) -> None:
        empty = (torch.empty(0, 2), torch.empty(0, dtype=torch.long))
        negative_only = (
            torch.tensor([[4.0, 5.0]]),
            torch.tensor([0], dtype=torch.long),
        )

        self.assertIsNone(points_to_prompt(empty, "cpu"))
        self.assertIsNone(points_to_prompt(negative_only, "cpu"))

    def test_mixed_points_are_converted(self) -> None:
        points = (
            torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            torch.tensor([1, 0], dtype=torch.long),
        )

        prompt = points_to_prompt(points, "cpu")

        self.assertIsNotNone(prompt)
        coords, labels = prompt
        self.assertEqual(coords.dtype, torch.float32)
        self.assertEqual(labels.dtype, torch.int64)
        self.assertTrue(torch.equal(labels, torch.tensor([1, 0])))

    def test_decode_without_positive_prompt_returns_empty_logits(self) -> None:
        class _DummySam:
            no_positive_logit = -20.0

            def __init__(self) -> None:
                self.calls = 0

            def decode(self, embeddings, points, output_size=(1024, 1024)):  # pragma: no cover
                self.calls += 1
                raise AssertionError("没有正点时不应调用 SAM 解码器")

        sam = _DummySam()
        embeddings = torch.zeros(2, 256, 4, 4)
        prompts = [
            None,
            (torch.empty(0, 2), torch.empty(0, dtype=torch.long)),
        ]

        logits, active = decode_with_prompts(
            sam, embeddings, prompts, output_size=(8, 8)
        )

        self.assertEqual(tuple(logits.shape), (2, 1, 8, 8))
        self.assertTrue(torch.equal(logits, torch.full_like(logits, -20.0)))
        self.assertEqual(active, [False, False])
        self.assertEqual(sam.calls, 0)

    def test_decode_skips_empty_samples_but_decodes_valid_ones(self) -> None:
        class _DummySam:
            no_positive_logit = -20.0

            def __init__(self) -> None:
                self.embeddings_shapes = []

            def decode(self, embeddings, points, output_size=(1024, 1024)):
                self.embeddings_shapes.append(tuple(embeddings.shape))
                return torch.zeros(
                    (embeddings.shape[0], 1, output_size[0], output_size[1])
                )

        sam = _DummySam()
        embeddings = torch.zeros(2, 256, 4, 4)
        prompts = [
            (torch.tensor([[4.0, 4.0]]), torch.tensor([1], dtype=torch.long)),
            None,
        ]

        logits, active = decode_with_prompts(
            sam, embeddings, prompts, output_size=(8, 8)
        )

        self.assertEqual(active, [True, False])
        self.assertEqual(sam.embeddings_shapes, [(1, 256, 4, 4)])
        self.assertTrue(torch.all(logits[0] == 0.0))
        self.assertTrue(torch.all(logits[1] == -20.0))

    def test_prompt_count_mismatch_raises(self) -> None:
        embeddings = torch.zeros(2, 256, 4, 4)
        with self.assertRaises(ValueError):
            decode_with_prompts(
                type("Sam", (), {"no_positive_logit": -20.0})(),
                embeddings,
                [None],
                output_size=(8, 8),
            )


class CheckpointSAM2GuardTest(unittest.TestCase):
    def test_old_sam1_snapshot_is_rejected_with_clear_message(self) -> None:
        # RoadSAM/SAM2 目标模型按 ``sam.sam_mask_decoder`` 命名权重。
        target = types.SimpleNamespace(
            sam=types.SimpleNamespace(
                sam_mask_decoder=torch.nn.Module(),
                sam_prompt_encoder=torch.nn.Module(),
            )
        )
        payload = {
            "models": {
                # 故意使用 SAM1 ViT-B 的键名。
                "sam": {
                    "sam.prompt_encoder.embedding.weight": torch.zeros(1),
                    "sam.mask_decoder.iou_token.weight": torch.zeros(1),
                },
                "condition_encoder": {},
                "dit": {},
                "diffusion": {},
            }
        }

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
        payload = {
            "config": {"model": {"sam_architecture": "sam2.1_hiera_large"}},
            "models": {
                "sam": {"sam.sam_mask_decoder.conv_s0.weight": torch.zeros(1)},
                "condition_encoder": {},
                "dit": {},
                "diffusion": {},
            },
        }

        with self.assertRaises(ValueError) as caught:
            load_checkpoint_models({"sam": target}, payload)

        self.assertIn("骨干", str(caught.exception))
        self.assertIn("sam2.1_hiera_large", str(caught.exception))

    def test_config_and_architecture_must_match(self) -> None:
        _validate_sam_architecture(
            "configs/sam2.1/sam2.1_hiera_b+.yaml", "sam2.1_hiera_base_plus"
        )
        with self.assertRaises(ValueError) as caught:
            _validate_sam_architecture(
                "configs/sam2.1/sam2.1_hiera_l.yaml", "sam2.1_hiera_base_plus"
            )
        self.assertIn("sam2.1_hiera_large", str(caught.exception))


class GeometryRecoveryTest(unittest.TestCase):
    def test_map_points_with_unit_scale(self) -> None:
        coords = torch.tensor([[10.0, 20.0], [30.0, 40.0]])
        labels = torch.tensor([1, 0])

        payload = map_points_to_original(
            (coords, labels), original_size=(100, 200), resized_size=(100, 200)
        )

        self.assertEqual(payload["coordinate_space"], "original_image_xy")
        self.assertEqual(payload["positive"], [[10.0, 20.0]])
        self.assertEqual(payload["negative"], [[30.0, 40.0]])
        self.assertEqual(payload["point_count"], 2)

    def test_map_points_scales_and_clips(self) -> None:
        coords = torch.tensor([[0.0, 0.0], [1000.0, 1000.0]])
        labels = torch.tensor([1, 0])

        payload = map_points_to_original(
            (coords, labels), original_size=(100, 200), resized_size=(50, 100)
        )

        self.assertEqual(payload["positive"], [[0.5, 0.5]])
        # 最后一个点被裁剪到原图边界内。
        self.assertEqual(payload["negative"], [[199.0, 99.0]])

    def test_crop_resize_removes_padding(self) -> None:
        probability = torch.zeros(8, 8)
        probability[4:, 4:] = 1.0

        from diffpeak.inference import crop_resize_probability

        result = crop_resize_probability(
            probability, original_size=(4, 4), resized_size=(4, 4)
        )

        self.assertEqual(result.shape, (4, 4))
        self.assertFalse(result.any())


class InferenceWithoutGroundTruthTest(unittest.TestCase):
    def test_predict_file_does_not_read_same_name_mask(self) -> None:
        class _StubPredictor:
            def predict_array(
                self, image, image_id=None, seed=None, return_heatmaps=False
            ):
                return {
                    "image_id": image_id,
                    "shape": tuple(image.shape),
                    "return_heatmaps": return_heatmaps,
                }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = np.full((12, 20, 3), 128, dtype=np.uint8)
            image_path = root / "scene_sat.jpg"
            Image.fromarray(image, mode="RGB").save(image_path)
            # 同名掩码位置故意放一个目录：一旦推理读取掩码就会失败。
            (root / "scene_mask.png").mkdir()

            result = Predictor.predict_file(
                _StubPredictor(), image_path, return_heatmaps=False
            )

        self.assertEqual(result["image_id"], "scene")
        self.assertEqual(result["shape"], (12, 20, 3))
        self.assertEqual(result["source_path"], str(image_path))

    def test_load_labeled_split_does_not_require_mask_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "valid").mkdir()
            image = np.zeros((8, 8, 3), dtype=np.uint8)
            Image.fromarray(image, mode="RGB").save(
                root / "valid" / "a_sat.jpg"
            )

            entries = load_labeled_split(["valid/a_sat.jpg"], root)

        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["id"], "a")
        self.assertEqual(entries[0]["image_path"], root / "valid" / "a_sat.jpg")
        self.assertEqual(entries[0]["mask_path"], root / "valid" / "a_mask.png")
        self.assertEqual(entries[0]["relative_path"], "valid/a_sat.jpg")

    def test_prepare_image_file_needs_image_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "unlabeled_sat.jpg"
            Image.fromarray(np.zeros((10, 16, 3), dtype=np.uint8), mode="RGB").save(
                path
            )

            tensor, original, resized = prepare_image_file(path, image_size=32)

        self.assertEqual(original, (10, 16))
        self.assertEqual(tuple(tensor.shape), (1, 3, 32, 32))


class FileAndSeedTest(unittest.TestCase):
    def test_mask_roundtrip(self) -> None:
        mask = np.zeros((6, 8), dtype=bool)
        mask[2, 3:6] = True

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "pred.png"
            save_mask_file(mask, path)
            loaded = load_mask_file(path)

        self.assertTrue(np.array_equal(loaded, mask))

    def test_load_image_file_returns_rgb_uint8(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.jpg"
            Image.fromarray(
                np.full((5, 7, 3), 120, dtype=np.uint8), mode="RGB"
            ).save(path)

            array = load_image_file(path)

        self.assertEqual(array.shape, (5, 7, 3))
        self.assertEqual(array.dtype, np.uint8)

    def test_stable_seed_is_reproducible_and_identifier_dependent(self) -> None:
        first = stable_seed(42, "infer", "scene")
        second = stable_seed(42, "infer", "scene")
        other = stable_seed(42, "infer", "other")
        other_base = stable_seed(43, "infer", "scene")

        self.assertEqual(first, second)
        self.assertNotEqual(first, other)
        self.assertNotEqual(first, other_base)
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

        self.assertEqual(summary["count"], 2)
        self.assertAlmostEqual(summary["iou"], 0.5)
        self.assertAlmostEqual(summary["cldice"], 1.0)
        self.assertAlmostEqual(summary["score"], 0.5)


if __name__ == "__main__":
    unittest.main()
