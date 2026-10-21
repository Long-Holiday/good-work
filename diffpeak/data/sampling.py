"""从道路标注构造正/负提示点目标。

实现遵循 ``docs/data.md`` 的数据约定：

* 骨架与采样都在**原始标注分辨率**下完成，之后才由
  :class:`~diffpeak.data.geometry.ImageGeometry` 映射到热图；
* 正点沿中心线按随机间距采样，并额外加入一部分路口和端点；
* 负点来自道路边界外侧的带状区域以及普通背景；
* 困难负点只通过可选的回调接口注入，本模块不声称能自动挖掘困难负点；
* 无道路影像的双通道目标热图都为零；
* 点转热图时逐点取高斯响应的 **max**，而不是求和。

本模块不依赖 ``torch`` 或数据集类，便于单独测试和复用。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Callable, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage

__all__ = [
    "PositiveSamplingConfig",
    "NegativeSamplingConfig",
    "RoadTargetSampler",
    "SampledTargets",
    "HardNegativeProvider",
    "render_gaussian_heatmap",
    "select_spread_points",
]


_NEIGHBOR_OFFSETS: Tuple[Tuple[int, int], ...] = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)

# 回调签名：``(image, road_mask, rng) -> (N, 2) 的 (y, x) 坐标``。
# image 为当前（可能已增广）的 HxWx3 uint8 图像，road_mask 为 HxW bool。
# 这是由调用方提供的困难负点来源，例如阶段性模型的假阳性坐标；
# RoadTargetSampler 不会自动训练或运行任何挖掘模型。
HardNegativeProvider = Callable[[np.ndarray, np.ndarray, np.random.Generator], np.ndarray]


def _merge_config(config_type, value, name: str):
    """把 dict / dataclass / None 统一成冻结 dataclass 配置。"""
    if value is None:
        return config_type()
    if isinstance(value, config_type):
        return value
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} 必须是 {config_type.__name__}、dict 或 None")
    allowed = {field.name for field in fields(config_type)}
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"{name} 包含未知字段: {sorted(unknown)}")
    return config_type(**dict(value))


@dataclass(frozen=True)
class PositiveSamplingConfig:
    """正点采样参数。

    Args:
        min_spacing: 沿骨架的最小采样间距（原始标注像素）。
        max_spacing: 沿骨架的最大采样间距（原始标注像素）。
        junction_probability: 每个路口像素被额外采样的概率。
        endpoint_probability: 每个端点像素被额外采样的概率。
        min_branch_length: 短于此长度的骨架分支不参与规则采样，以抑制毛刺。
    """

    min_spacing: float = 24.0
    max_spacing: float = 64.0
    junction_probability: float = 0.5
    endpoint_probability: float = 0.5
    min_branch_length: float = 4.0

    def __post_init__(self) -> None:
        if self.min_spacing <= 0 or self.max_spacing <= 0:
            raise ValueError("采样间距必须为正")
        if self.max_spacing < self.min_spacing:
            raise ValueError("max_spacing 不能小于 min_spacing")
        if not 0.0 <= self.junction_probability <= 1.0:
            raise ValueError("junction_probability 必须在 [0, 1] 内")
        if not 0.0 <= self.endpoint_probability <= 1.0:
            raise ValueError("endpoint_probability 必须在 [0, 1] 内")
        if self.min_branch_length < 0:
            raise ValueError("min_branch_length 不能为负")


@dataclass(frozen=True)
class NegativeSamplingConfig:
    """负点采样参数。

    Args:
        boundary_margin: 与道路的最小距离（原始像素），用于避开不确定边界。
        boundary_band: 边界外侧带宽（原始像素）。
        boundary_count: 边界负点数量的均匀采样区间 ``(min, max)``（含端点）。
        background_count: 普通背景负点数量的均匀采样区间。
        min_separation: 同一张图内负点之间的最小间距（原始像素）。
    """

    boundary_margin: float = 2.0
    boundary_band: float = 12.0
    boundary_count: Tuple[int, int] = (8, 20)
    background_count: Tuple[int, int] = (4, 12)
    min_separation: float = 12.0

    def __post_init__(self) -> None:
        if self.boundary_margin < 0:
            raise ValueError("boundary_margin 不能为负")
        if self.boundary_band < self.boundary_margin:
            raise ValueError("boundary_band 不能小于 boundary_margin")
        for name in ("boundary_count", "background_count"):
            low, high = getattr(self, name)
            if low < 0 or high < low:
                raise ValueError(f"{name} 必须满足 0 <= min <= max")
        if self.min_separation < 0:
            raise ValueError("min_separation 不能为负")


@dataclass(frozen=True)
class SampledTargets:
    """一张图的点目标。所有坐标均为原始标注分辨率下的 ``(y, x)``。"""

    positive_yx: np.ndarray
    positive_priorities: np.ndarray
    negative_yx: np.ndarray

    def __post_init__(self) -> None:
        positive = np.asarray(self.positive_yx, dtype=np.float64).reshape(-1, 2)
        priorities = np.asarray(self.positive_priorities, dtype=np.int64).reshape(-1)
        negative = np.asarray(self.negative_yx, dtype=np.float64).reshape(-1, 2)
        if len(priorities) != len(positive):
            raise ValueError("positive_priorities 与 positive_yx 数量不一致")
        object.__setattr__(self, "positive_yx", positive)
        object.__setattr__(self, "positive_priorities", priorities)
        object.__setattr__(self, "negative_yx", negative)


def _as_yx(points: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(points, dtype=np.float64)
    if array.size == 0:
        return array.reshape(0, 2)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError(f"{name} 必须是 (N, 2) 的 (y, x)，实际为 {array.shape}")
    return array


def _neighbors(y: int, x: int, height: int, width: int):
    for dy, dx in _NEIGHBOR_OFFSETS:
        ny, nx = y + dy, x + dx
        if 0 <= ny < height and 0 <= nx < width:
            yield ny, nx


def _edge_key(first: Tuple[int, int], second: Tuple[int, int]):
    return (first, second) if first <= second else (second, first)


def _walk_branch(
    skeleton: np.ndarray,
    degree: np.ndarray,
    start: Tuple[int, int],
    first: Tuple[int, int],
    visited_edges: set,
):
    """沿骨架行走，直到遇到非度 2 像素、走过的边或回到起点。"""
    height, width = skeleton.shape
    path = [start, first]
    visited_edges.add(_edge_key(start, first))
    previous, current = start, first
    while True:
        # 路口、端点或孤立点都终止当前分支。
        if degree[current] != 2:
            break
        next_pixel = None
        for candidate in _neighbors(current[0], current[1], height, width):
            if candidate == previous or not skeleton[candidate]:
                continue
            if _edge_key(current, candidate) not in visited_edges:
                next_pixel = candidate
                break
        if next_pixel is None:
            break
        visited_edges.add(_edge_key(current, next_pixel))
        path.append(next_pixel)
        previous, current = current, next_pixel
        if current == start:
            break
    return path


def extract_skeleton_branches(skeleton: np.ndarray):
    """把 8 连通骨架拆成折线分支。

    路口/端点作为分支端点；纯环状骨架也会被完整遍历一次。返回值中每条
    折线均为 ``[(y, x), ...]``。该函数只用于采样，不改变骨架本身。
    """
    skeleton = np.asarray(skeleton, dtype=bool)
    if not skeleton.any():
        return []
    height, width = skeleton.shape
    kernel = np.ones((3, 3), dtype=np.int16)
    kernel[1, 1] = 0
    degree = ndimage.convolve(skeleton.astype(np.int16), kernel, mode="constant", cval=0)
    coords = np.argwhere(skeleton)
    # ``degree`` 是二维数组；``coords`` 是 (N, 2) 的 (y, x) 行数组。
    # 必须用两个列向量做逐点高级索引，写成 ``degree[coords]`` 会按第一维
    # 展开成 (N, 2, width) 的布尔数组，无法再索引 ``coords``。
    coords_degree = degree[coords[:, 0], coords[:, 1]]
    visited_edges: set = set()
    branches = []

    # 先从所有非度 2 像素出发，覆盖连接路口/端点的普通分支。
    key_coords = [
        tuple(map(int, coord)) for coord in coords[coords_degree != 2]
    ]
    for y, x in key_coords:
        if degree[y, x] == 0:
            branches.append([(y, x)])
            continue
        for neighbor in _neighbors(y, x, height, width):
            if not skeleton[neighbor]:
                continue
            if _edge_key((y, x), neighbor) in visited_edges:
                continue
            branches.append(
                _walk_branch(skeleton, degree, (y, x), neighbor, visited_edges)
            )

    # 再处理不与任何路口相连的纯环（全部像素度数为 2）。
    for (y, x), pixel_degree in zip(coords.tolist(), coords_degree):
        if pixel_degree != 2:
            continue
        for neighbor in _neighbors(y, x, height, width):
            if not skeleton[neighbor]:
                continue
            if _edge_key((y, x), neighbor) in visited_edges:
                continue
            branches.append(
                _walk_branch(skeleton, degree, (y, x), neighbor, visited_edges)
            )
    return branches


def _polyline_length(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def _interpolate(points: np.ndarray, cumulative: np.ndarray, position: float) -> np.ndarray:
    if len(points) == 1:
        return points[0]
    index = int(np.searchsorted(cumulative, position, side="right") - 1)
    index = max(0, min(index, len(points) - 2))
    segment = cumulative[index + 1] - cumulative[index]
    if segment <= 1e-12:
        return points[index]
    fraction = (position - cumulative[index]) / segment
    return points[index] * (1.0 - fraction) + points[index + 1] * fraction


def _sample_polyline(
    points: np.ndarray,
    rng: np.random.Generator,
    min_spacing: float,
    max_spacing: float,
):
    """在折线上按随机间距取点；起点偏移也随机。"""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(points) <= 1:
        return [points[0]] if len(points) == 1 else []
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    total = float(cumulative[-1])
    if total <= 1e-9:
        return [points[0]]

    # 随机起始偏移避免每条分支总在固定位置落点；间距逐次独立采样。
    position = float(rng.uniform(0.0, min(max_spacing, total)))
    sampled = []
    while position <= total + 1e-9:
        sampled.append(_interpolate(points, cumulative, position))
        position += float(rng.uniform(min_spacing, max_spacing))
    return sampled


def _sample_positive_points(
    skeleton: np.ndarray,
    rng: np.random.Generator,
    config: PositiveSamplingConfig,
):
    """返回 ``(points_yx, priorities)``；priority=0 为结构点，1 为普通线点。"""
    skeleton = np.asarray(skeleton, dtype=bool)
    if not skeleton.any():
        return np.empty((0, 2), dtype=np.float64), np.empty(0, dtype=np.int64)

    branches = extract_skeleton_branches(skeleton)
    points = []
    priorities = []

    # 结构点优先保留，后续在热图空间做最小间距筛选时不会被普通线点挤掉。
    kernel = np.ones((3, 3), dtype=np.int16)
    kernel[1, 1] = 0
    degree = ndimage.convolve(skeleton.astype(np.int16), kernel, mode="constant", cval=0)
    coords = np.argwhere(skeleton)
    # 同 ``extract_skeleton_branches``：逐点取 (y, x) 处的度数，
    # 不能写 ``degree[coords]``（那会退化为按第一维取整行）。
    coords_degree = degree[coords[:, 0], coords[:, 1]]
    endpoints = coords[coords_degree == 1]
    junctions = coords[coords_degree >= 3]
    for pool, probability in (
        (junctions, config.junction_probability),
        (endpoints, config.endpoint_probability),
    ):
        for y, x in pool:
            if rng.random() < probability:
                points.append((float(y), float(x)))
                priorities.append(0)

    for branch in branches:
        branch_points = np.asarray(branch, dtype=np.float64).reshape(-1, 2)
        if _polyline_length(branch_points) < config.min_branch_length:
            continue
        for point in _sample_polyline(
            branch_points, rng, config.min_spacing, config.max_spacing
        ):
            points.append(point)
            priorities.append(1)

    if not points:
        # 有骨架但所有分支都太短时，至少保留一个正点，避免把有道路样本
        # 误标成“无道路”。位置仍由样本随机状态选择，可复现。
        index = int(rng.integers(0, len(coords)))
        points.append(coords[index].astype(np.float64))
        priorities.append(0)

    return np.asarray(points, dtype=np.float64), np.asarray(priorities, dtype=np.int64)


def _random_count(rng: np.random.Generator, bounds: Sequence[int]) -> int:
    low, high = int(bounds[0]), int(bounds[1])
    return int(rng.integers(low, high + 1))


def _sample_from_candidates(
    indices: np.ndarray,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if count <= 0 or len(indices) == 0:
        return np.empty(0, dtype=np.int64)
    count = min(int(count), len(indices))
    return rng.choice(indices, size=count, replace=False)


def _sample_negative_points(
    road_mask: np.ndarray,
    rng: np.random.Generator,
    config: NegativeSamplingConfig,
    image: Optional[np.ndarray] = None,
    hard_negative_provider: Optional[HardNegativeProvider] = None,
) -> np.ndarray:
    road_mask = np.asarray(road_mask, dtype=bool)
    background = ~road_mask
    if not background.any():
        return np.empty((0, 2), dtype=np.float64)

    # 背景像素到最近道路像素的欧氏距离；道路像素处为 0。
    distance = ndimage.distance_transform_edt(background)
    ys, xs = np.nonzero(background)
    distances = distance[ys, xs]

    boundary_pool = np.flatnonzero(
        (distances >= config.boundary_margin) & (distances <= config.boundary_band)
    )
    background_pool = np.flatnonzero(distances > config.boundary_band)

    chosen = []
    boundary_indices = _sample_from_candidates(
        boundary_pool, _random_count(rng, config.boundary_count), rng
    )
    background_indices = _sample_from_candidates(
        background_pool, _random_count(rng, config.background_count), rng
    )
    for selected in (boundary_indices, background_indices):
        if len(selected):
            chosen.append(np.stack([ys[selected], xs[selected]], axis=1))

    # 可选困难负点：仅使用调用方提供的坐标，本类不做任何自动挖掘。
    if hard_negative_provider is not None:
        provider_image = (
            image
            if image is not None
            else np.repeat(road_mask[..., None].astype(np.uint8) * 255, 3, axis=2)
        )
        hard = hard_negative_provider(provider_image, road_mask, rng)
        if hard is not None:
            hard = _as_yx(hard, "hard_negative_provider 返回值")
            if len(hard):
                height, width = road_mask.shape
                hard_int = np.rint(hard).astype(np.int64)
                inside = (
                    (hard_int[:, 0] >= 0)
                    & (hard_int[:, 0] < height)
                    & (hard_int[:, 1] >= 0)
                    & (hard_int[:, 1] < width)
                )
                if not inside.all():
                    raise ValueError("困难负点回调返回了图像范围外的坐标")
                if road_mask[hard_int[:, 0], hard_int[:, 1]].any():
                    raise ValueError("困难负点回调返回了道路上的坐标")
                chosen.append(hard.astype(np.float64))

    if not chosen:
        return np.empty((0, 2), dtype=np.float64)
    points = np.concatenate(chosen, axis=0)
    if len(points) == 0:
        return points.reshape(0, 2)

    # 负点数量本身也随机变化；再按原始像素最小间距做一次确定性筛选。
    return select_spread_points(
        points,
        min_distance=config.min_separation,
        rng=rng,
    )


class RoadTargetSampler:
    """把道路掩码/骨架转成点目标。"""

    def __init__(
        self,
        positive: Optional[Mapping] = None,
        negative: Optional[Mapping] = None,
    ) -> None:
        self.positive_config = _merge_config(
            PositiveSamplingConfig, positive, "positive"
        )
        self.negative_config = _merge_config(
            NegativeSamplingConfig, negative, "negative"
        )

    def sample(
        self,
        skeleton: np.ndarray,
        road_mask: np.ndarray,
        rng: np.random.Generator,
        image: Optional[np.ndarray] = None,
        hard_negative_provider: Optional[HardNegativeProvider] = None,
    ) -> SampledTargets:
        """构造一张图的点目标。

        Args:
            skeleton: 原始标注分辨率下的 bool 骨架。
            road_mask: 同分辨率的 bool 道路掩码。
            rng: 该样本专属的 ``numpy.random.Generator``。
            image: 当前图像（HxWx3 uint8），只传给可选的困难负点回调。
            hard_negative_provider: 可选的困难负点来源；默认 ``None``。

        Returns:
            :class:`SampledTargets`。无道路时正负点均为空。
        """
        skeleton = np.asarray(skeleton, dtype=bool)
        road_mask = np.asarray(road_mask, dtype=bool)
        if skeleton.shape != road_mask.shape:
            raise ValueError("skeleton 与 road_mask 尺寸不一致")

        # 无道路样本：双通道热图必须为零，因此这里也不返回任何负点。
        if not road_mask.any():
            return SampledTargets(
                positive_yx=np.empty((0, 2), dtype=np.float64),
                positive_priorities=np.empty(0, dtype=np.int64),
                negative_yx=np.empty((0, 2), dtype=np.float64),
            )

        positive_yx, priorities = _sample_positive_points(
            skeleton, rng, self.positive_config
        )
        negative_yx = _sample_negative_points(
            road_mask,
            rng,
            self.negative_config,
            image=image,
            hard_negative_provider=hard_negative_provider,
        )
        return SampledTargets(
            positive_yx=positive_yx,
            positive_priorities=priorities,
            negative_yx=negative_yx,
        )


def select_spread_points(
    points_yx: np.ndarray,
    min_distance: float,
    rng: np.random.Generator,
    priorities: Optional[np.ndarray] = None,
) -> np.ndarray:
    """按最小间距贪心筛选点；同优先级内的顺序由 ``rng`` 随机决定。

    该函数用于让**目标点本身**满足最终峰值提取器的 NMS 间距，降低
    “热图拟合正确但回提时被合并”的风险。坐标应已在热图或原始像素空间，
    两种空间均可，只要 ``min_distance`` 使用同一单位。
    """
    points = _as_yx(points_yx, "points_yx")
    if len(points) <= 1 or min_distance <= 0:
        return points
    if priorities is None:
        priorities = np.zeros(len(points), dtype=np.int64)
    else:
        priorities = np.asarray(priorities, dtype=np.int64).reshape(-1)
        if len(priorities) != len(points):
            raise ValueError("priorities 与 points 数量不一致")

    # lexsort 的最后一个 key 是主键：先按 priority，再按随机 key。
    random_keys = rng.random(len(points))
    order = np.lexsort((random_keys, priorities))
    kept = []
    min_distance_sq = float(min_distance) ** 2
    for index in order:
        point = points[index]
        if all(
            float(np.sum((point - kept_point) ** 2)) >= min_distance_sq
            for kept_point in kept
        ):
            kept.append(point)
    return np.asarray(kept, dtype=np.float64).reshape(-1, 2)


def render_gaussian_heatmap(
    points_yx: np.ndarray,
    size: int,
    sigma: float,
) -> np.ndarray:
    """把点渲染成单通道热图，逐点取 **max**（不是 sum）。

    Args:
        points_yx: 热图坐标下的 ``(y, x)``，允许浮点数。
        size: 热图边长。
        sigma: 高斯宽度（热图像素）。``sigma <= 0`` 时在最近像素置 1。

    Returns:
        ``float32`` 的 ``size x size`` 热图，取值范围 ``[0, 1]``。
    """
    if size <= 0:
        raise ValueError("size 必须为正")
    heatmap = np.zeros((int(size), int(size)), dtype=np.float32)
    points = _as_yx(points_yx, "points_yx")
    if len(points) == 0:
        return heatmap
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) == 0:
        return heatmap

    if sigma <= 0:
        for y, x in points:
            iy = int(np.clip(round(y), 0, size - 1))
            ix = int(np.clip(round(x), 0, size - 1))
            heatmap[iy, ix] = 1.0
        return heatmap

    radius = max(1, int(math.ceil(3.0 * float(sigma))))
    denominator = 2.0 * float(sigma) ** 2
    for y, x in points:
        if not (-radius <= x < size + radius and -radius <= y < size + radius):
            continue
        y0 = max(0, int(math.floor(y)) - radius)
        y1 = min(size, int(math.floor(y)) + radius + 1)
        x0 = max(0, int(math.floor(x)) - radius)
        x1 = min(size, int(math.floor(x)) + radius + 1)
        if y0 >= y1 or x0 >= x1:
            continue
        yy = np.arange(y0, y1, dtype=np.float32)
        xx = np.arange(x0, x1, dtype=np.float32)
        patch = np.exp(
            -((yy[:, None] - y) ** 2 + (xx[None, :] - x) ** 2) / denominator
        ).astype(np.float32)
        # np.maximum 明确表达“max 而非 sum”的监督约定。
        np.maximum(heatmap[y0:y1, x0:x1], patch, out=heatmap[y0:y1, x0:x1])
    return heatmap
