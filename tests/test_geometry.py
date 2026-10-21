"""ImageGeometry 的等比缩放、右下填充与坐标映射测试。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diffpeak.data.geometry import (  # noqa: E402
    ImageGeometry,
    build_geometry,
    resize_and_pad,
)


class BuildGeometryTest(unittest.TestCase):
    def test_square_image_fills_whole_canvas(self) -> None:
        geometry = build_geometry(512, 512, image_size=1024, heatmap_size=256)

        self.assertEqual(geometry.resized_size, (1024, 1024))
        self.assertEqual(geometry.valid_size, (1024, 1024))

    def test_landscape_image_keeps_aspect_ratio(self) -> None:
        geometry = build_geometry(100, 200, image_size=1024, heatmap_size=256)

        self.assertEqual(geometry.resized_size, (512, 1024))

    def test_invalid_sizes_raise(self) -> None:
        with self.assertRaises(ValueError):
            build_geometry(0, 10)
        with self.assertRaises(ValueError):
            build_geometry(10, 10, image_size=0)
        with self.assertRaises(ValueError):
            ImageGeometry(
                source_size=(10, 10),
                resized_size=(11, 10),
                image_size=10,
                heatmap_size=5,
            )


class CoordinateMappingTest(unittest.TestCase):
    def test_original_roundtrip_for_same_size_source(self) -> None:
        geometry = ImageGeometry(
            source_size=(256, 256),
            resized_size=(256, 256),
            image_size=1024,
            heatmap_size=256,
        )
        points_yx = np.array([[20.0, 10.0], [0.0, 0.0]])

        heatmap_yx = geometry.original_yx_to_heatmap_yx(points_yx)
        image_xy = geometry.heatmap_yx_to_image_xy(heatmap_yx)
        original_xy = geometry.original_yx_to_image_xy(points_yx)

        self.assertTrue(np.allclose(image_xy, original_xy))
        self.assertTrue(np.allclose(image_xy, np.array([[10.0, 20.0], [0.0, 0.0]])))

    def test_first_heatmap_pixel_maps_to_image_center(self) -> None:
        geometry = ImageGeometry(
            source_size=(256, 256),
            resized_size=(256, 256),
            image_size=1024,
            heatmap_size=256,
        )

        image_xy = geometry.heatmap_yx_to_image_xy(np.array([[0.0, 0.0]]))

        self.assertTrue(np.allclose(image_xy, np.array([[1.5, 1.5]])))

    def test_empty_points_keep_shape(self) -> None:
        geometry = build_geometry(10, 10)
        empty = np.empty((0, 2))

        self.assertEqual(geometry.original_yx_to_heatmap_yx(empty).shape, (0, 2))
        self.assertEqual(geometry.heatmap_yx_to_image_xy(empty).shape, (0, 2))
        self.assertEqual(geometry.original_yx_to_image_xy(empty).shape, (0, 2))


class ResizeAndPadTest(unittest.TestCase):
    def test_bottom_right_padding_for_bool_mask(self) -> None:
        geometry = build_geometry(2, 4, image_size=8, heatmap_size=4)
        mask = np.ones((2, 4), dtype=bool)

        padded = resize_and_pad(mask, geometry, resample="nearest")

        self.assertEqual(padded.shape, (8, 8))
        self.assertEqual(padded.dtype, np.bool_)
        self.assertTrue(padded[:4, :].all())
        self.assertFalse(padded[4:, :].any())
        self.assertEqual(int(padded.sum()), 32)

    def test_uint8_image_has_three_channels(self) -> None:
        geometry = build_geometry(2, 4, image_size=8, heatmap_size=4)
        image = np.full((2, 4, 3), 200, dtype=np.uint8)

        padded = resize_and_pad(image, geometry, resample="bilinear")

        self.assertEqual(padded.shape, (8, 8, 3))
        self.assertTrue((padded[:4, :, :] == 200).all())
        self.assertTrue((padded[4:, :, :] == 0).all())

    def test_invalid_arguments_raise(self) -> None:
        geometry = build_geometry(4, 4, image_size=8, heatmap_size=4)
        with self.assertRaises(ValueError):
            resize_and_pad(np.zeros((4, 4)), geometry, resample="cubic")
        with self.assertRaises(ValueError):
            resize_and_pad(np.zeros((5, 4)), geometry)
        with self.assertRaises(ValueError):
            resize_and_pad(np.zeros((1, 4, 4, 3)), geometry)


if __name__ == "__main__":
    unittest.main()
