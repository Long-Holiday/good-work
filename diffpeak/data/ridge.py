"""单通道有符号道路山脊场目标。

数值语义
--------
目标场 ``H*`` 为 ``1 x heatmap_size x heatmap_size``，取值范围 ``[-1, 1]``：

* 道路中心线（骨架）显式赋值为 ``+1``，形成连续山脊；
* 道路内部按到边界与到中心线的距离比值从中心向边界下降；
* 有效背景固定为 ``-1``，形成平底；
* 右下 padding 不是真实背景，目标值固定为 ``0``，并由独立的
  ``field_valid_mask`` 标记。这个 ``0`` 约定与有效背景的 ``-1`` 相互独立。

构造与降采样顺序
----------------
1. 在**原始标注分辨率**下由道路掩码计算骨架、到边界的距离与到中心线的
   距离，得到源分辨率山脊场；
2. 用与 :class:`~diffpeak.data.geometry.ImageGeometry` 完全一致的像素中心
   映射把源场降到热图分辨率。降采样对每个热图像素覆盖的源像素区域取
   **最大池化**，因此只要一个热图像素有任意道路覆盖，就不会被平均成背景；
   极细道路也不会因热图缩小而整体消失。

四个互斥监督区域
----------------
热图分辨率下的每个有效像素恰好属于以下一个区域（padding 为 ``-1``）：

* ``REGION_CENTER``：中心线邻域（骨架膨胀 ``center_radius`` 个热图像素，
  且仍位于道路上）；
* ``REGION_REST_ROAD``：其余道路；
* ``REGION_OUTSIDE_BAND``：道路外侧 ``boundary_band`` 个热图像素内的窄带；
* ``REGION_FAR_BACKGROUND``：更远的有效背景。

空道路样本没有中心和道路区域，也没有“边界外窄带”（不存在道路边界），
所有有效像素都属于远端背景，目标场全部为 ``-1``。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize

from .geometry import ImageGeometry

__all__ = [
    "BACKGROUND_VALUE",
    "RIDGE_VALUE",
    "PADDING_VALUE",
    "FIELD_TARGET_VERSION",
    "REGION_CENTER",
    "REGION_REST_ROAD",
    "REGION_OUTSIDE_BAND",
    "REGION_FAR_BACKGROUND",
    "REGION_PADDING",
    "REGION_NAMES",
    "RidgeFieldConfig",
    "RidgeField",
    "coerce_ridge_config",
    "build_source_ridge_field",
    "build_ridge_field",
    "regions_to_one_hot",
    "regions_from_one_hot",
    "source_retention_statistics",
    "field_statistics",
]

BACKGROUND_VALUE = -1.0
RIDGE_VALUE = 1.0
PADDING_VALUE = 0.0
FIELD_TARGET_VERSION = "signed_ridge_v1"

REGION_CENTER = 0
REGION_REST_ROAD = 1
REGION_OUTSIDE_BAND = 2
REGION_FAR_BACKGROUND = 3
REGION_PADDING = -1

REGION_NAMES: Dict[int, str] = {
    REGION_CENTER: "center_neighborhood",
    REGION_REST_ROAD: "rest_road",
    REGION_OUTSIDE_BAND: "outside_boundary_band",
    REGION_FAR_BACKGROUND: "far_background",
}


@dataclass(frozen=True)
class RidgeFieldConfig:
    """山脊场构造参数。

    Args:
        center_radius: 中心线邻域半径，单位为**热图像素**。
        boundary_band: 道路外侧窄带宽度，单位为**热图像素**。
        epsilon: 距离比值分母中的数值稳定项，避免中心线处除零。
    """

    center_radius: float = 1.0
    boundary_band: float = 2.0
    epsilon: float = 1e-6

    def __post_init__(self) -> None:
        for name in ("center_radius", "boundary_band", "epsilon"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} 必须是有限数值")
        if self.center_radius < 0:
            raise ValueError("center_radius 不能为负")
        if self.boundary_band < 0:
            raise ValueError("boundary_band 不能为负")
        if self.epsilon <= 0:
            raise ValueError("epsilon 必须为正")


@dataclass(frozen=True)
class RidgeField:
    """一张影像的山脊场目标。

    Attributes:
        field: ``heatmap_size x heatmap_size`` 的 ``float32`` 有符号场，
            有效背景为 ``-1``、中心线为 ``+1``、padding 为 ``0``。
        valid_mask: 同尺寸的 ``bool`` 掩码，``True`` 表示非 padding 的有效
            图像内容。
        regions: 同尺寸的 ``int64`` 区域标签；取值见 ``REGION_*``，padding
            为 ``REGION_PADDING``。
        road_mask: 热图分辨率下的道路覆盖掩码；任一点道路覆盖即保留，用于
            诊断细道路降采样保留率。
        skeleton_mask: 热图分辨率下的中心线覆盖掩码。
        source_skeleton: 原始标注分辨率下的骨架，供数据集复用缩放结果。
        source_road_mask: 原始标注分辨率下的道路掩码，供离线的细道路
            保留率诊断使用；训练主路径不额外复制。
        geometry: 本次构造使用的几何映射。
        config: 本次构造使用的参数。
    """

    field: np.ndarray
    valid_mask: np.ndarray
    regions: np.ndarray
    road_mask: np.ndarray
    skeleton_mask: np.ndarray
    source_skeleton: np.ndarray
    source_road_mask: np.ndarray
    geometry: ImageGeometry
    config: RidgeFieldConfig


def coerce_ridge_config(
    value: Optional[Any],
) -> RidgeFieldConfig:
    """把 ``None`` / mapping / 配置对象统一成 :class:`RidgeFieldConfig`。"""
    if value is None:
        return RidgeFieldConfig()
    if isinstance(value, RidgeFieldConfig):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("ridge_config 必须是 RidgeFieldConfig、dict 或 None")
    allowed = {field.name for field in fields(RidgeFieldConfig)}
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"ridge_config 包含未知字段: {sorted(unknown)}")
    return RidgeFieldConfig(**dict(value))


def build_source_ridge_field(
    road_mask: np.ndarray,
    *,
    epsilon: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray]:
    """在原始标注分辨率下构造山脊场。

    公式（``a = 0``）::

        H*(x) = d_boundary(x) / (d_boundary(x) + d_center(x) + epsilon), x in M
        H*(x) = -1,                                                   x not in M

    其中 ``d_boundary`` 是道路内部到最近背景（含图像外）的距离减一，
    因此紧贴背景的道路边界取 0；``d_center`` 是到骨架的欧氏距离。
    骨架像素再显式赋值为 ``+1``，稳定离散网格上的山脊高度。

    退化行为：

    * 空道路：整张图为 ``-1``，骨架为空；
    * 骨架提取失败但道路非空：把整个道路区域视为中心线（全为 ``+1``）；
    * 道路接触图像边缘或铺满整张图：图像外按背景处理，距离变换仍然确定。

    Args:
        road_mask: 原始标注分辨率下的布尔道路掩码。
        epsilon: 数值稳定项。

    Returns:
        ``(field, skeleton)``，均为 ``float32`` / ``bool`` 的 ``H x W`` 数组。
    """
    road = np.asarray(road_mask, dtype=bool)
    if road.ndim != 2:
        raise ValueError(f"road_mask 必须是 HxW，实际为 {road.shape}")
    if not math.isfinite(float(epsilon)) or epsilon <= 0:
        raise ValueError("epsilon 必须是有限正数")

    field = np.full(road.shape, BACKGROUND_VALUE, dtype=np.float32)
    if not road.any():
        return field, np.zeros(road.shape, dtype=bool)

    # Lee 方法对直路给出单像素、居中的骨架；默认 Zhang 方法在奇数宽度
    # 道路上会产生两条偏置骨架行，导致山脊不居中。
    skeleton = np.asarray(skeletonize(road, method="lee"), dtype=bool)
    if not skeleton.any():
        # 退化骨架：无法提取中心线时，用整个道路区域兜底，避免全背景。
        skeleton = road.copy()

    # 图像外按背景处理；减一后紧贴边界的道路像素到边界距离为 0。
    padded = np.pad(road, 1, mode="constant", constant_values=False)
    d_boundary = ndimage.distance_transform_edt(padded)[1:-1, 1:-1] - 1.0
    d_boundary = np.maximum(d_boundary, 0.0)

    d_center = ndimage.distance_transform_edt(~skeleton)
    ratio = d_boundary / (d_boundary + d_center + float(epsilon))

    field[road] = ratio[road].astype(np.float32)
    field[skeleton] = RIDGE_VALUE
    np.clip(field, BACKGROUND_VALUE, RIDGE_VALUE, out=field)
    return field, skeleton


def _pool_axis(
    data: np.ndarray,
    *,
    source_length: int,
    heatmap_length: int,
    resized_length: int,
    image_size: int,
    axis: int,
) -> np.ndarray:
    """把一个轴从源分辨率映射到热图分辨率并取最大。

    映射与 ``original_yx_to_heatmap_yx`` 完全一致：
    ``heat = (source + 0.5) * k - 0.5``，
    ``k = (resized / source) * (heatmap / image_size)``。
    每个热图像素覆盖源区间 ``[i / k - 0.5, (i + 1) / k - 0.5)``，对该区间
    取最大。源分辨率低于热图分辨率（``1 / k < 1``）时退化为最近源像素。
    """
    scale = (float(resized_length) / float(source_length)) * (
        float(heatmap_length) / float(image_size)
    )
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("无效的热图/源分辨率比例")
    span = 1.0 / scale
    if span >= 1.0 - 1e-12:
        # lo_d = d * span - 0.5。减去一个远大于浮点误差、远小于像素尺度的
        # 容差，避免 26.000000000000004 这类边界被 ceil 推到下一个源像素。
        starts = np.ceil(
            np.arange(heatmap_length, dtype=np.float64) * span - 0.5 - 1e-9
        ).astype(np.int64)
        # 只对起点仍落在源数组内的单元做 reduceat。若把越界起点直接截断到
        # ``source_length - 1``，会改变前一个有效单元的区间终点，导致最后
        # 一个有效单元漏掉部分源像素。
        effective = int(np.searchsorted(starts, source_length, side="left"))
        if effective <= 0:
            reduced = np.take(data, [0], axis=axis)
            reduced = np.repeat(
                reduced, heatmap_length, axis=axis
            )
            return reduced
        reduced = np.maximum.reduceat(data, starts[:effective], axis=axis)
        if effective < heatmap_length:
            # 越界单元只可能是 padding；用最后一个有效单元填充，调用方会
            # 通过 valid_mask 覆盖为 padding 值。
            last = np.take(reduced, [effective - 1], axis=axis)
            missing = heatmap_length - effective
            reduced = np.concatenate(
                [reduced, np.repeat(last, missing, axis=axis)], axis=axis
            )
        return reduced

    indices = np.rint(
        (np.arange(heatmap_length, dtype=np.float64) + 0.5) / scale - 0.5
    ).astype(np.int64)
    indices = np.clip(indices, 0, source_length - 1)
    return np.take(data, indices, axis=axis)


def _pool_to_heatmap(data: np.ndarray, geometry: ImageGeometry) -> np.ndarray:
    """把 ``source_size`` 数组降到 ``heatmap_size``，逐单元取最大。"""
    if tuple(data.shape[:2]) != geometry.source_size:
        raise ValueError(
            "data 的空间尺寸与 geometry.source_size 不一致："
            f"{data.shape[:2]} vs {geometry.source_size}"
        )
    pooled = _pool_axis(
        data,
        source_length=geometry.source_height,
        heatmap_length=geometry.heatmap_size,
        resized_length=geometry.resized_height,
        image_size=geometry.image_size,
        axis=0,
    )
    return _pool_axis(
        pooled,
        source_length=geometry.source_width,
        heatmap_length=geometry.heatmap_size,
        resized_length=geometry.resized_width,
        image_size=geometry.image_size,
        axis=1,
    )


def _valid_heatmap_mask(geometry: ImageGeometry) -> np.ndarray:
    """标记热图空间中属于缩放图像内容的像素。

    与峰值提取器的 ``valid_sizes`` 约定一致：像素中心映射到 SAM 输入坐标
    ``x_image = (x_heat + 0.5) * image_size / heatmap_size - 0.5``，只有
    ``-0.5 <= x_image <= valid_size - 0.5`` 的像素才是有效内容。
    """
    scale = float(geometry.image_size) / float(geometry.heatmap_size)
    centers = (
        np.arange(geometry.heatmap_size, dtype=np.float64) + 0.5
    ) * scale - 0.5
    valid_y = (centers >= -0.5) & (centers <= geometry.resized_height - 0.5)
    valid_x = (centers >= -0.5) & (centers <= geometry.resized_width - 0.5)
    return valid_y[:, None] & valid_x[None, :]


def _assign_regions(
    road_mask: np.ndarray,
    skeleton_mask: np.ndarray,
    valid_mask: np.ndarray,
    config: RidgeFieldConfig,
) -> np.ndarray:
    """在热图分辨率下分配四个互斥区域；padding 为 ``REGION_PADDING``。"""
    regions = np.full(road_mask.shape, REGION_PADDING, dtype=np.int64)
    if not valid_mask.any():
        return regions
    if not road_mask.any():
        # 空道路：不存在道路边界，全部有效像素都是远端背景。
        regions[valid_mask] = REGION_FAR_BACKGROUND
        return regions

    center_cells = skeleton_mask & road_mask
    if not center_cells.any():
        center_cells = road_mask.copy()
    if center_cells.all():
        center_distance = np.zeros(road_mask.shape, dtype=np.float64)
    else:
        center_distance = ndimage.distance_transform_edt(~center_cells)
    center_region = road_mask & (center_distance <= float(config.center_radius))
    rest_road = road_mask & ~center_region

    if road_mask.all():
        road_distance = np.zeros(road_mask.shape, dtype=np.float64)
    else:
        road_distance = ndimage.distance_transform_edt(~road_mask)
    band = (~road_mask) & (road_distance <= float(config.boundary_band))
    far_background = ~road_mask & ~band

    regions[valid_mask & center_region] = REGION_CENTER
    regions[valid_mask & rest_road] = REGION_REST_ROAD
    regions[valid_mask & band] = REGION_OUTSIDE_BAND
    regions[valid_mask & far_background] = REGION_FAR_BACKGROUND
    return regions


def build_ridge_field(
    road_mask: np.ndarray,
    geometry: ImageGeometry,
    config: Optional[Any] = None,
) -> RidgeField:
    """由原始分辨率道路掩码构造热图分辨率的单通道山脊场。

    Args:
        road_mask: 原始标注分辨率（几何增广后）的布尔道路掩码。
        geometry: 已按同一影像构造的 :class:`ImageGeometry`。
        config: :class:`RidgeFieldConfig`、mapping 或 ``None``。

    Returns:
        :class:`RidgeField`。``field`` 与 ``valid_mask`` 均为热图分辨率。
    """
    road = np.asarray(road_mask, dtype=bool)
    if road.ndim != 2:
        raise ValueError(f"road_mask 必须是 HxW，实际为 {road.shape}")
    if tuple(road.shape) != geometry.source_size:
        raise ValueError(
            "road_mask 尺寸与 geometry.source_size 不一致："
            f"{road.shape} vs {geometry.source_size}"
        )
    resolved = coerce_ridge_config(config)

    source_field, source_skeleton = build_source_ridge_field(
        road, epsilon=resolved.epsilon
    )

    valid_mask = _valid_heatmap_mask(geometry)
    field = _pool_to_heatmap(source_field, geometry).astype(np.float32, copy=True)
    road_heatmap = _pool_to_heatmap(road.astype(np.uint8), geometry) > 0
    skeleton_heatmap = (
        _pool_to_heatmap(source_skeleton.astype(np.uint8), geometry) > 0
    )
    road_heatmap &= valid_mask
    skeleton_heatmap &= valid_mask

    # padding 使用独立的 0 约定；有效背景仍固定为 -1。
    field[~valid_mask] = PADDING_VALUE
    field[valid_mask & ~road_heatmap] = BACKGROUND_VALUE

    regions = _assign_regions(
        road_heatmap, skeleton_heatmap, valid_mask, resolved
    )
    return RidgeField(
        field=field,
        valid_mask=valid_mask,
        regions=regions,
        road_mask=road_heatmap,
        skeleton_mask=skeleton_heatmap,
        source_skeleton=source_skeleton,
        source_road_mask=road,
        geometry=geometry,
        config=resolved,
    )


def regions_to_one_hot(regions: np.ndarray) -> np.ndarray:
    """把 ``H x W`` 区域标签转成 ``4 x H x W`` 的互斥 0/1 掩码。

    顺序固定为 :data:`REGION_CENTER`、:data:`REGION_REST_ROAD`、
    :data:`REGION_OUTSIDE_BAND`、:data:`REGION_FAR_BACKGROUND`。padding
    （``REGION_PADDING``）在四个通道中全为 0，与 ``field_valid_mask`` 配合
    使用。引擎与扩散损失消费的 ``field_regions`` 即此形状。
    """
    labels = np.asarray(regions)
    if labels.ndim != 2:
        raise ValueError(f"regions 必须是 HxW，实际为 {labels.shape}")
    return np.stack(
        [
            (labels == region_id).astype(np.float32)
            for region_id in range(4)
        ],
        axis=0,
    )


def regions_from_one_hot(
    one_hot: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """把 ``4 x H x W`` 互斥掩码还原为区域标签；padding 记为 ``-1``。

    Args:
        one_hot: ``4 x H x W``，每个有效像素最多一个通道为正。
        valid_mask: 可选的 ``H x W`` 有效区域掩码。为 ``None`` 时由
            ``one_hot.sum(axis=0) > 0`` 推断；显式传入可避免全零通道的歧义。
    """
    array = np.asarray(one_hot)
    if array.ndim != 3 or array.shape[0] != 4:
        raise ValueError(f"one_hot 必须是 4xHxW，实际为 {array.shape}")
    if valid_mask is None:
        valid = array.sum(axis=0) > 0
    else:
        valid = np.asarray(valid_mask, dtype=bool)
        if valid.shape != array.shape[1:]:
            raise ValueError("valid_mask 与 one_hot 空间尺寸不一致")
    labels = np.argmax(array, axis=0).astype(np.int64)
    labels = np.where(valid, labels, REGION_PADDING)
    return labels


def _nearest_source_indices(geometry: ImageGeometry) -> Tuple[np.ndarray, np.ndarray]:
    """每个热图单元中心对应的最近源像素索引。

    与 ``_pool_axis`` 的上采样/最近邻分支使用同一像素中心映射。用于量化
    “单元格中心采样”相对“覆盖最大值”会漏掉多少细道路，而不是复用最终
    热图掩码，避免诊断自我循环。
    """
    ym = geometry.heatmap_size
    scale_y = (geometry.resized_height / geometry.source_height) * (
        ym / geometry.image_size
    )
    scale_x = (geometry.resized_width / geometry.source_width) * (
        ym / geometry.image_size
    )
    cells = np.arange(ym, dtype=np.float64) + 0.5
    index_y = np.rint(cells / scale_y - 0.5).astype(np.int64)
    index_x = np.rint(cells / scale_x - 0.5).astype(np.int64)
    index_y = np.clip(index_y, 0, geometry.source_height - 1)
    index_x = np.clip(index_x, 0, geometry.source_width - 1)
    return index_y, index_x


def _project_source_mask(
    mask: np.ndarray,
    geometry: ImageGeometry,
) -> Tuple[np.ndarray, int]:
    """把源分辨率像素按像素中心映射投影到最近的热图单元。

    Returns:
        ``(cells, total)``；``cells`` 为投影后落在热图内的 ``(y, x)`` 索引，
        ``total`` 是投影前源像素数（包含落到热图外者，用于保守的召回率
        分母）。
    """
    points = np.argwhere(np.asarray(mask, dtype=bool)).astype(np.float64)
    total = int(len(points))
    if total == 0:
        return np.empty((0, 2), dtype=np.int64), 0
    heatmap = geometry.original_yx_to_heatmap_yx(points)
    cells = np.rint(heatmap).astype(np.int64)
    inside = (
        (cells[:, 0] >= 0)
        & (cells[:, 0] < geometry.heatmap_size)
        & (cells[:, 1] >= 0)
        & (cells[:, 1] < geometry.heatmap_size)
    )
    return cells[inside], total


def source_retention_statistics(ridge: RidgeField) -> Dict[str, Any]:
    """诊断细道路/中心线在降采样中的保留情况。

    两个维度都基于**源分辨率像素与几何映射**，不复用最终热图掩码作为
    输入：

    * ``projected_*_recall``：源道路/骨架像素投影到最近热图单元后，该单元
      是否被覆盖最大值标记为道路/骨架。保守覆盖下该值应接近 1，用于发现
      半像素/边界映射错误。
    * ``*_coverage_retention_vs_nearest``：仅在热图单元中心采样的最近邻
      下采样会覆盖多少单元；与覆盖最大值得到的单元数相比，比值大于 1
      表示保守降采样额外保住了本会丢失的道路/骨架单元。这是细道路保留
      的直接证据，而不是对同一掩码的自证。
    """
    geometry = ridge.geometry
    road = np.asarray(ridge.source_road_mask, dtype=bool)
    skeleton = np.asarray(ridge.source_skeleton, dtype=bool)
    valid = ridge.valid_mask

    road_cells, road_total = _project_source_mask(road, geometry)
    skeleton_cells, skeleton_total = _project_source_mask(skeleton, geometry)

    def recall(cells: np.ndarray, total: int, heatmap_mask: np.ndarray) -> Optional[float]:
        if total == 0:
            return None
        if len(cells) == 0:
            return 0.0
        hits = int(
            np.count_nonzero(
                valid[cells[:, 0], cells[:, 1]]
                & heatmap_mask[cells[:, 0], cells[:, 1]]
            )
        )
        return float(hits) / float(total)

    center_hit = np.zeros_like(valid)
    center_hit[valid] = ridge.regions[valid] == REGION_CENTER

    index_y, index_x = _nearest_source_indices(geometry)
    nearest_road = road[np.ix_(index_y, index_x)] & valid
    nearest_skeleton = skeleton[np.ix_(index_y, index_x)] & valid
    conservative_road_cells = int(np.count_nonzero(ridge.road_mask & valid))
    conservative_skeleton_cells = int(
        np.count_nonzero(ridge.skeleton_mask & valid)
    )
    nearest_road_cells = int(np.count_nonzero(nearest_road))
    nearest_skeleton_cells = int(np.count_nonzero(nearest_skeleton))

    def ratio(numerator: int, denominator: int) -> Optional[float]:
        if denominator == 0:
            # 最近邻一个单元都没覆盖而保守覆盖仍保留单元时，比值无穷大，
            # 用 None 表示并在 *_extra_cells 中给出绝对差值。
            return None
        return float(numerator) / float(denominator)

    return {
        "source_road_pixels": road_total,
        "source_skeleton_pixels": skeleton_total,
        "projected_road_recall": recall(road_cells, road_total, ridge.road_mask),
        "projected_skeleton_recall": recall(
            skeleton_cells, skeleton_total, ridge.skeleton_mask
        ),
        "projected_skeleton_center_recall": recall(
            skeleton_cells, skeleton_total, center_hit
        ),
        "conservative_road_cells": conservative_road_cells,
        "nearest_road_cells": nearest_road_cells,
        "road_coverage_retention_vs_nearest": ratio(
            conservative_road_cells, nearest_road_cells
        ),
        "conservative_road_extra_cells": conservative_road_cells
        - nearest_road_cells,
        "conservative_skeleton_cells": conservative_skeleton_cells,
        "nearest_skeleton_cells": nearest_skeleton_cells,
        "skeleton_coverage_retention_vs_nearest": ratio(
            conservative_skeleton_cells, nearest_skeleton_cells
        ),
        "conservative_skeleton_extra_cells": conservative_skeleton_cells
        - nearest_skeleton_cells,
    }


def field_statistics(
    field: np.ndarray,
    valid_mask: np.ndarray,
    regions: np.ndarray,
) -> Dict[str, Any]:
    """汇总一张山脊场目标的标量统计，供检查 CLI 与诊断使用。"""
    field = np.asarray(field, dtype=np.float64)
    valid = np.asarray(valid_mask, dtype=bool)
    regions = np.asarray(regions)
    if field.shape != valid.shape or field.shape != regions.shape:
        raise ValueError("field / valid_mask / regions 形状必须一致")

    total = int(field.size)
    valid_count = int(valid.sum())
    valid_values = field[valid]
    background_fraction = (
        float(np.count_nonzero(valid_values <= BACKGROUND_VALUE + 1e-6))
        / float(valid_count)
        if valid_count
        else 0.0
    )
    stats: Dict[str, Any] = {
        "field_target_version": FIELD_TARGET_VERSION,
        "total_pixels": total,
        "valid_pixels": valid_count,
        "padding_pixels": total - valid_count,
        "valid_fraction": float(valid_count) / float(total) if total else 0.0,
        "background_fraction_of_valid": background_fraction,
        "ridge_fraction_of_valid": (
            float(np.count_nonzero(valid_values >= RIDGE_VALUE - 1e-6))
            / float(valid_count)
            if valid_count
            else 0.0
        ),
        "positive_fraction_of_valid": (
            float(np.count_nonzero(valid_values > 0.0)) / float(valid_count)
            if valid_count
            else 0.0
        ),
        # 只要热图像素含道路覆盖，其最大值就不会是 -1；该比例直接反映
        # 降采样后的道路（含极细道路）保留率。
        "road_coverage_fraction_of_valid": (
            float(np.count_nonzero(valid_values > BACKGROUND_VALUE + 1e-6))
            / float(valid_count)
            if valid_count
            else 0.0
        ),
    }
    if valid_count:
        stats.update(
            {
                "field_min": float(valid_values.min()),
                "field_max": float(valid_values.max()),
                "field_mean": float(valid_values.mean()),
                "field_std": float(valid_values.std()),
            }
        )
    else:
        stats.update(
            {
                "field_min": None,
                "field_max": None,
                "field_mean": None,
                "field_std": None,
            }
        )

    region_counts: Dict[str, int] = {}
    for region_id, name in REGION_NAMES.items():
        count = int(np.count_nonzero(valid & (regions == region_id)))
        region_counts[name] = count
        stats[f"{name}_pixels"] = count
        stats[f"{name}_fraction_of_valid"] = (
            float(count) / float(valid_count) if valid_count else 0.0
        )
    stats["region_counts"] = region_counts
    stats["unassigned_valid_pixels"] = int(
        valid_count - sum(region_counts.values())
    )
    return stats
