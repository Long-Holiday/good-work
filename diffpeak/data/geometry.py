"""DeepGlobe 数据的几何变换工具。

设计约定
--------
* 图像按 ``min(image_size / H, image_size / W)`` 等比缩放；
* 缩放后的内容放在左上角，右侧和底部用零填充（bottom-right padding）；
* 标注（道路掩码、骨架、提示点）始终先在原始标注分辨率下构造，
  再通过同一个 :class:`ImageGeometry` 映射到热图坐标；
* 对外使用的点坐标有两种约定：

  - Numpy 内部数组使用 ``(y, x)``，即 ``(row, col)``；
  - 交给 SAM / 峰值提取器的坐标使用 ``(x, y)``。

本模块只做纯几何计算与重采样，不读取文件，也不依赖数据集对象。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Union

import numpy as np
from PIL import Image

__all__ = ["ImageGeometry", "build_geometry", "resize_and_pad"]


def _as_points(points: np.ndarray, name: str = "points") -> np.ndarray:
    """把输入整理为 ``(N, 2)`` 的 float64 数组，不复制已是 float64 的数组。"""
    array = np.asarray(points, dtype=np.float64)
    if array.size == 0:
        return array.reshape(0, 2)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError(f"{name} 必须是 (N, 2)，实际为 {array.shape}")
    return array


@dataclass(frozen=True)
class ImageGeometry:
    """一次样本从标注分辨率到网络输入的完整几何映射。

    Attributes:
        source_size: 当前标注（可能已做几何增广）的 ``(H, W)``。
        resized_size: 等比缩放后、填充前的 ``(H, W)``，位于 ``image_size`` 坐标系。
        image_size: SAM 输入边长，默认 1024。
        heatmap_size: 提示热图边长，默认 256。
    """

    source_size: Tuple[int, int]
    resized_size: Tuple[int, int]
    image_size: int
    heatmap_size: int

    def __post_init__(self) -> None:
        height, width = self.source_size
        resized_height, resized_width = self.resized_size
        if height <= 0 or width <= 0:
            raise ValueError(f"source_size 必须为正，实际为 {self.source_size}")
        if not (0 < resized_height <= self.image_size):
            raise ValueError("resized_height 必须在 (0, image_size] 内")
        if not (0 < resized_width <= self.image_size):
            raise ValueError("resized_width 必须在 (0, image_size] 内")
        if self.image_size <= 0 or self.heatmap_size <= 0:
            raise ValueError("image_size 与 heatmap_size 必须为正")

    @property
    def source_height(self) -> int:
        return int(self.source_size[0])

    @property
    def source_width(self) -> int:
        return int(self.source_size[1])

    @property
    def resized_height(self) -> int:
        return int(self.resized_size[0])

    @property
    def resized_width(self) -> int:
        return int(self.resized_size[1])

    @property
    def valid_size(self) -> Tuple[int, int]:
        """填充前的有效区域 ``(height, width)``，可直接作为 ``valid_sizes``。"""
        return (self.resized_height, self.resized_width)

    def original_yx_to_resized_yx(self, points: np.ndarray) -> np.ndarray:
        """标注分辨率 ``(y, x)`` -> 缩放后（未填充）图像坐标 ``(y, x)``。"""
        points = _as_points(points)
        if points.shape[0] == 0:
            return points
        scale_y = self.resized_height / self.source_height
        scale_x = self.resized_width / self.source_width
        resized = np.empty_like(points)
        # 使用像素中心对齐，和 PIL 的 resize 约定一致。
        resized[:, 0] = (points[:, 0] + 0.5) * scale_y - 0.5
        resized[:, 1] = (points[:, 1] + 0.5) * scale_x - 0.5
        return resized

    def original_yx_to_heatmap_yx(self, points: np.ndarray) -> np.ndarray:
        """标注分辨率 ``(y, x)`` -> 热图 ``(y, x)``。

        热图与 SAM 输入按像素中心对齐，和峰值提取器的逆映射严格互逆。
        """
        resized = self.original_yx_to_resized_yx(points)
        if resized.shape[0] == 0:
            return resized
        scale = self.heatmap_size / float(self.image_size)
        heatmap = np.empty_like(resized)
        heatmap[:, 0] = (resized[:, 0] + 0.5) * scale - 0.5
        heatmap[:, 1] = (resized[:, 1] + 0.5) * scale - 0.5
        return heatmap

    def heatmap_yx_to_image_xy(self, points: np.ndarray) -> np.ndarray:
        """热图 ``(y, x)`` -> SAM 图像坐标 ``(x, y)``。

        返回坐标位于完整的 ``image_size`` 坐标系；调用方需要再按
        :attr:`valid_size` 过滤右下填充区域。
        """
        points = _as_points(points)
        if points.shape[0] == 0:
            return np.empty((0, 2), dtype=np.float64)
        scale = self.image_size / float(self.heatmap_size)
        image_yx = (points + 0.5) * scale - 0.5
        return image_yx[:, ::-1]

    def original_yx_to_image_xy(self, points: np.ndarray) -> np.ndarray:
        """标注分辨率 ``(y, x)`` -> SAM 图像坐标 ``(x, y)``。"""
        return self.heatmap_yx_to_image_xy(self.original_yx_to_heatmap_yx(points))


def build_geometry(
    source_height: int,
    source_width: int,
    image_size: int = 1024,
    heatmap_size: int = 256,
) -> ImageGeometry:
    """按保持纵横比、右下填充的规则计算几何映射。"""
    if source_height <= 0 or source_width <= 0:
        raise ValueError("source_height 与 source_width 必须为正")
    if image_size <= 0 or heatmap_size <= 0:
        raise ValueError("image_size 与 heatmap_size 必须为正")

    scale = min(image_size / float(source_height), image_size / float(source_width))
    # round 后再 clamp，避免极端长宽比下因舍入超过 image_size。
    resized_height = int(min(image_size, max(1, round(source_height * scale))))
    resized_width = int(min(image_size, max(1, round(source_width * scale))))
    return ImageGeometry(
        source_size=(int(source_height), int(source_width)),
        resized_size=(resized_height, resized_width),
        image_size=int(image_size),
        heatmap_size=int(heatmap_size),
    )


def resize_and_pad(
    array: np.ndarray,
    geometry: ImageGeometry,
    resample: str = "bilinear",
    fill: Union[int, float] = 0,
) -> np.ndarray:
    """把 ``H x W`` 或 ``H x W x C`` 数组等比缩放并做右下零填充。

    Args:
        array: 图像 / 掩码 / 骨架。布尔数组会先转成 0/255。
        geometry: 缩放与填充参数。
        resample: ``"bilinear"`` 用于 RGB 图像，``"nearest"`` 用于类别图。
        fill: 填充值。图像与掩码都使用 0。

    Returns:
        形状为 ``image_size x image_size``（RGB 为 ``x 3``）的数组。
    """
    if resample not in {"bilinear", "nearest"}:
        raise ValueError("resample 必须是 'bilinear' 或 'nearest'")
    if array.ndim not in (2, 3):
        raise ValueError(f"array 必须是 HxW 或 HxWxC，实际为 {array.shape}")
    if tuple(array.shape[:2]) != geometry.source_size:
        raise ValueError(
            "array 的空间尺寸与 geometry.source_size 不一致："
            f"{array.shape[:2]} vs {geometry.source_size}"
        )

    is_bool = array.dtype == np.bool_
    if is_bool:
        work = (array.astype(np.uint8) * 255)
    elif array.dtype == np.uint8:
        work = array
    else:
        # 这里只处理常规图像/掩码；浮点数组请调用方先量化。
        work = np.asarray(array)

    if work.dtype == np.uint8:
        image = Image.fromarray(work)
    else:
        image = Image.fromarray(np.asarray(work))
    method = Image.BILINEAR if resample == "bilinear" else Image.NEAREST
    resized = np.asarray(
        image.resize((geometry.resized_width, geometry.resized_height), method)
    )
    if is_bool:
        resized = resized > 127
        fill_value = bool(fill)
    else:
        fill_value = np.asarray(fill, dtype=resized.dtype).item()

    canvas_shape = (geometry.image_size, geometry.image_size) + resized.shape[2:]
    canvas = np.full(canvas_shape, fill_value, dtype=resized.dtype)
    canvas[: geometry.resized_height, : geometry.resized_width] = resized
    return canvas
