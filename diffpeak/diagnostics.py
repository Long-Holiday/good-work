"""提示质量与 GRPO 组内诊断的纯只读指标。

本模块只做诊断，不参与训练目标，也不修改任何输入张量/数组：

* :func:`prompt_quality`：单张影像的提示点质量与热图稀疏度；
* :func:`reward_group_diagnostics`：同一影像 G 条轨迹奖励的组内统计，
  以及优势与点数的**逐图**相关性。

坐标约定
--------
* 提示点 ``coords`` 为 ``(x, y)``，位于 ``image_size x image_size`` 的
  SAM 输入画布；
* ``target`` 是**已裁掉右下填充**的二值标注，``H x W``，其中
  ``H, W <= image_size``，与 ``coords`` 使用同一画布尺度；
* ``heatmap`` 为 ``2 x h x w``，通道 0 为正提示、通道 1 为负提示。
  热图覆盖完整画布，因此诊断热图指标时会按像素中心映射排除右下填充。

所有随机性相关的模块都不在这里：两个函数均无随机操作，同一输入产生
同一输出，便于离线复核和单元测试。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Dict, Tuple, Union

import numpy as np
import torch
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize
from torch import Tensor

__all__ = [
    "PROMPT_METRIC_KEYS",
    "REWARD_GROUP_METRIC_KEYS",
    "prompt_quality",
    "reward_group_diagnostics",
]


#: :func:`prompt_quality` 返回的全部指标名，按固定顺序排列。
PROMPT_METRIC_KEYS: Tuple[str, ...] = (
    "positive_on_road_rate",
    "negative_on_background_rate",
    "no_positive_rate",
    "point_outside_rate",
    "heatmap_above_threshold_rate",
    "heatmap_mean",
    "heatmap_max",
    "skeleton_coverage",
)

#: :func:`reward_group_diagnostics` 返回的全部指标名，按固定顺序排列。
REWARD_GROUP_METRIC_KEYS: Tuple[str, ...] = (
    "quality_reward_mean",
    "quality_reward_group_std",
    "point_penalty_mean",
    "point_penalty_group_std",
    "reward_mean",
    "reward_group_std",
    "point_count_mean",
    "point_count_group_std",
    "no_point_rate",
    "advantage_abs_mean",
    "advantage_group_std",
    "advantage_point_count_corr",
    "reward_point_count_corr",
    "quality_reward_point_count_corr",
    "constant_reward_group_rate",
    "undefined_advantage_corr_rate",
    "n_images",
    "n_trajectories",
)


# ---------------------------------------------------------------------------
# 输入整理
# ---------------------------------------------------------------------------


def _as_numpy(value: Any, name: str) -> np.ndarray:
    """只读地把张量/数组转成 numpy；张量先 detach 并移到 CPU。"""
    if torch.is_tensor(value):
        array = value.detach().cpu().numpy()
    else:
        try:
            array = np.asarray(value)
        except Exception as exc:  # pragma: no cover - 取决于第三方输入类型
            raise TypeError(f"{name} 无法转换为 numpy 数组") from exc
    return array


def _as_binary_target(target: np.ndarray) -> np.ndarray:
    """校验并返回 ``H x W`` 的 bool 标注。"""
    array = np.asarray(target)
    if array.ndim != 2:
        raise ValueError(f"target 必须是 HxW，实际为 {array.shape}")
    if array.shape[0] <= 0 or array.shape[1] <= 0:
        raise ValueError(f"target 的 H/W 必须为正，实际为 {array.shape}")
    if array.dtype == np.bool_:
        return array
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError("target 必须是 bool 或 0/1 数值数组")
    if not np.isfinite(array).all():
        raise ValueError("target 不能包含 NaN 或 Inf")
    if not np.isin(array, (0, 1)).all():
        raise ValueError("target 必须是二值（仅含 0/1）")
    return array.astype(bool)


def _as_points(
    points: Tuple[Union[Tensor, np.ndarray], Union[Tensor, np.ndarray]],
) -> Tuple[np.ndarray, np.ndarray]:
    """整理 ``(coords Nx2, labels N)``，失败时给出明确错误。"""
    if not isinstance(points, (tuple, list)) or len(points) != 2:
        raise TypeError("points 必须是 (coords Nx2, labels N) 二元组")
    coords = _as_numpy(points[0], "coords")
    labels = _as_numpy(points[1], "labels")

    if coords.size == 0:
        coords = np.empty((0, 2), dtype=np.float64)
    else:
        if coords.ndim != 2 or coords.shape[1] != 2:
            raise ValueError(f"coords 必须是 Nx2，实际为 {coords.shape}")
        try:
            coords = coords.astype(np.float64, copy=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("coords 必须可转换为浮点数") from exc
        if not np.isfinite(coords).all():
            raise ValueError("coords 不能包含 NaN 或 Inf")

    labels = np.asarray(labels).reshape(-1)
    if labels.shape[0] != coords.shape[0]:
        raise ValueError(
            "coords 与 labels 数量不一致："
            f"{coords.shape[0]} vs {labels.shape[0]}"
        )
    if labels.size:
        try:
            is_binary = np.isin(labels, (0, 1)).all()
        except TypeError as exc:
            raise ValueError("labels 必须只含 0/1") from exc
        if not is_binary:
            raise ValueError("labels 必须只含 0（负）与 1（正）")
    return coords, labels.astype(np.int64, copy=False)


def _heatmap_valid_mask(
    height: int,
    width: int,
    image_size: int,
    heatmap_height: int,
    heatmap_width: int,
) -> np.ndarray:
    """返回热图上属于有效图像内容的像素（排除右下填充）。

    热图与 SAM 输入按像素中心对齐：热图第 ``j`` 个像素的中心映射到图像
    坐标 ``(j + 0.5) * image_size / heatmap_size - 0.5``。有效内容像素的
    中心范围为 ``[0, W-1]``，外边界为 ``[-0.5, W-0.5)``，因此用中心是否
    落在该区间内判定，而不是简单截断 ``round(W * h / image_size)``。
    """
    x_image = (
        (np.arange(heatmap_width, dtype=np.float64) + 0.5)
        * (float(image_size) / float(heatmap_width))
        - 0.5
    )
    y_image = (
        (np.arange(heatmap_height, dtype=np.float64) + 0.5)
        * (float(image_size) / float(heatmap_height))
        - 0.5
    )
    valid_x = (x_image >= -0.5) & (x_image < float(width) - 0.5)
    valid_y = (y_image >= -0.5) & (y_image < float(height) - 0.5)
    return valid_y[:, None] & valid_x[None, :]


def _point_pixels(
    coords: np.ndarray,
    height: int,
    width: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """把连续图像坐标映射到最近像素，返回 ``(px, py, inside)``。

    像素中心位于整数坐标，使用 half-up 四舍五入。落在有效内容之外的
    点（右下填充、负坐标或越界）标记为 outside，不参与 target 采样。
    """
    count = coords.shape[0]
    if count == 0:
        empty_int = np.empty(0, dtype=np.int64)
        return empty_int, empty_int, np.empty(0, dtype=bool)

    px_float = np.floor(coords[:, 0] + 0.5)
    py_float = np.floor(coords[:, 1] + 0.5)
    inside_x = (px_float >= 0.0) & (px_float < float(width))
    inside_y = (py_float >= 0.0) & (py_float < float(height))
    inside = inside_x & inside_y
    # 只在 inside 时转换为整数，避免极大坐标在 astype 时溢出。
    px = np.where(inside, px_float, 0.0).astype(np.int64)
    py = np.where(inside, py_float, 0.0).astype(np.int64)
    return px, py, inside


def _pearson_by_group(
    x: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """逐列计算 Pearson 相关；常量列返回 NaN，不跨列混合。"""
    if x.ndim != 2 or y.ndim != 2 or x.shape != y.shape:
        raise ValueError("相关性输入必须是同形状的二维数组")
    groups = x.shape[1]
    if x.shape[0] < 2 or groups == 0:
        return np.full(groups, np.nan, dtype=np.float64)

    x_centered = x - x.mean(axis=0, keepdims=True)
    y_centered = y - y.mean(axis=0, keepdims=True)
    numerator = np.sum(x_centered * y_centered, axis=0)
    denominator = np.sqrt(
        np.sum(x_centered * x_centered, axis=0)
        * np.sum(y_centered * y_centered, axis=0)
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        correlation = numerator / denominator
    correlation = np.asarray(correlation, dtype=np.float64)
    correlation[~np.isfinite(correlation)] = np.nan
    return correlation


def _mean_finite(values: np.ndarray) -> float:
    """只对有限值求均值；没有有限值时返回 NaN。"""
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan")
    return float(finite.mean())


# ---------------------------------------------------------------------------
# 单图提示质量
# ---------------------------------------------------------------------------


def _threshold_pair(
    threshold: Union[float, Sequence[float]],
) -> Tuple[float, float]:
    """把标量或 ``(positive, negative)`` 阈值归一化为二元组。"""
    if isinstance(threshold, (str, bytes)):
        raise ValueError("threshold 必须是标量或 (positive, negative) 二元组")
    if torch.is_tensor(threshold):
        array = threshold.detach().cpu().numpy()
    else:
        array = np.asarray(threshold)
    if array.ndim == 0:
        try:
            value = float(array)
        except (TypeError, ValueError) as exc:
            raise ValueError("threshold 必须是有限数") from exc
        pair = (value, value)
    else:
        values = array.reshape(-1)
        if values.size != 2:
            raise ValueError(
                "threshold 序列必须是 (positive, negative) 两个值"
            )
        try:
            pair = (float(values[0]), float(values[1]))
        except (TypeError, ValueError) as exc:
            raise ValueError("threshold 必须是有限数") from exc
    if not all(np.isfinite(value) for value in pair):
        raise ValueError("threshold 必须是有限数")
    return pair


def prompt_quality(
    points: Tuple[Union[Tensor, np.ndarray], Union[Tensor, np.ndarray]],
    target: np.ndarray,
    heatmap: Union[Tensor, np.ndarray],
    image_size: int,
    threshold: Union[float, Sequence[float]],
    radius: float = 8.0,
) -> Dict[str, float]:
    """计算单张影像的提示点质量与热图诊断。

    Args:
        points: ``(coords Nx2, labels N)``。``coords`` 为 SAM 输入画布上的
            ``(x, y)`` 像素坐标，``labels`` 中 1 为正提示、0 为负提示。
        target: 已裁掉右下填充的二值道路标注，``H x W``；必须与 ``coords``
            处于同一 ``image_size`` 画布尺度。
        heatmap: ``2 x h x w`` 正/负双通道提示热图。通道 0 为正，通道 1
            为负；热图覆盖完整画布。
        image_size: SAM 输入画布边长。
        threshold: 热图“高响应”阈值。可以是单个标量（正负通道共用），
            也可以是 ``(positive, negative)`` 二元组，分别对应通道 0/1；
            严格大于对应阈值才计入。逐通道阈值先各自判定，再在像素级
            合并为“任一通道超过其阈值”。
        radius: :func:`~diffpeak.diagnostics.prompt_quality` 的骨架覆盖半径
            （图像像素），默认 8。仅用于 ``skeleton_coverage``。

    Returns:
        dict，键与 :data:`PROMPT_METRIC_KEYS` 完全一致：

        ``positive_on_road_rate``
            正提示中落在道路标注上的比例；没有正提示时为 NaN。
        ``negative_on_background_rate``
            负提示中落在有效背景上的比例；没有负提示时为 NaN。
        ``no_positive_rate``
            该图没有正提示时为 1，否则为 0。
        ``point_outside_rate``
            落在有效内容（含右下填充）之外的点比例；没有点时 NaN。
        ``heatmap_above_threshold_rate``
            有效热图像素中，正/负任一通道响应严格大于其对应阈值的比例。
        ``heatmap_mean`` / ``heatmap_max``
            有效区域、两个通道合并后的均值/最大值。
        ``skeleton_coverage``
            标注骨架像素中，距任一**域内**正提示不超过 ``radius`` 的比例。
            GT 有骨架但没有域内正提示时为 0（空提示必须作为 0 参与宏平均，
            不能返回 NaN 被忽略）；GT 没有骨架时为 NaN。

    Notes:
        ``skeleton_coverage`` 只是**局部提示覆盖**：它衡量骨架附近是否有
        正提示，不等价于分支级/拓扑覆盖，也不能替代 APLS 等结构指标。
    """
    try:
        image_size_int = int(image_size)
    except (TypeError, ValueError) as exc:
        raise ValueError("image_size 必须是正整数") from exc
    if image_size_int <= 0 or image_size_int != image_size:
        raise ValueError("image_size 必须是正整数")

    threshold_pair = _threshold_pair(threshold)

    try:
        radius_value = float(radius)
    except (TypeError, ValueError) as exc:
        raise ValueError("radius 必须是有限非负数") from exc
    if not np.isfinite(radius_value) or radius_value < 0.0:
        raise ValueError("radius 必须是有限非负数")

    target_bool = _as_binary_target(target)
    height, width = int(target_bool.shape[0]), int(target_bool.shape[1])
    if height > image_size_int or width > image_size_int:
        raise ValueError(
            "target 必须是已裁掉填充、位于 image_size 画布内的标注："
            f"target={target_bool.shape}, image_size={image_size_int}"
        )

    coords, labels = _as_points(points)
    positive = labels == 1
    negative = labels == 0
    point_count = int(labels.size)
    positive_count = int(positive.sum())
    negative_count = int(negative.sum())

    px, py, inside = _point_pixels(coords, height, width)
    on_target = np.zeros(point_count, dtype=bool)
    if point_count:
        on_target[inside] = target_bool[py[inside], px[inside]]

    if positive_count:
        positive_on_road_rate = float(
            np.count_nonzero(positive & inside & on_target) / positive_count
        )
    else:
        positive_on_road_rate = float("nan")

    if negative_count:
        negative_on_background_rate = float(
            np.count_nonzero(negative & inside & ~on_target) / negative_count
        )
    else:
        negative_on_background_rate = float("nan")

    no_positive_rate = 1.0 if positive_count == 0 else 0.0
    point_outside_rate = (
        float(np.count_nonzero(~inside) / point_count)
        if point_count
        else float("nan")
    )

    heatmap_array = _as_numpy(heatmap, "heatmap")
    if heatmap_array.ndim != 3 or heatmap_array.shape[0] != 2:
        raise ValueError(
            f"heatmap 必须是 2 x h x w，实际为 {heatmap_array.shape}"
        )
    heatmap_height, heatmap_width = int(heatmap_array.shape[1]), int(
        heatmap_array.shape[2]
    )
    if heatmap_height <= 0 or heatmap_width <= 0:
        raise ValueError("heatmap 的 h/w 必须为正")
    if not np.issubdtype(heatmap_array.dtype, np.number):
        raise ValueError("heatmap 必须是数值数组")
    heatmap_array = heatmap_array.astype(np.float64, copy=False)
    if not np.isfinite(heatmap_array).all():
        raise ValueError("heatmap 不能包含 NaN 或 Inf")

    valid_heatmap = _heatmap_valid_mask(
        height,
        width,
        image_size_int,
        heatmap_height,
        heatmap_width,
    )
    valid_count = int(np.count_nonzero(valid_heatmap))
    if valid_count:
        valid_values = heatmap_array[:, valid_heatmap]
        heatmap_mean = float(valid_values.mean())
        heatmap_max = float(valid_values.max())
        threshold_column = np.asarray(threshold_pair, dtype=np.float64)[:, None]
        above_by_channel = valid_values > threshold_column
        heatmap_above_threshold_rate = float(
            np.count_nonzero(above_by_channel.any(axis=0)) / valid_count
        )
    else:
        heatmap_mean = float("nan")
        heatmap_max = float("nan")
        heatmap_above_threshold_rate = float("nan")

    skeleton_coverage = float("nan")
    if bool(target_bool.any()):
        skeleton = skeletonize(target_bool)
        if bool(skeleton.any()):
            # GT 有骨架但没有任何正提示（或正提示全在域外）时，覆盖率
            # 明确为 0，而不是 NaN：否则空提示轨迹会被宏平均静默忽略。
            inside_positive = positive & inside
            if bool(inside_positive.any()):
                skeleton_y, skeleton_x = np.nonzero(skeleton)
                skeleton_xy = np.column_stack(
                    (
                        skeleton_x.astype(np.float64),
                        skeleton_y.astype(np.float64),
                    )
                )
                positive_xy = coords[inside_positive]
                # 只建域内正点的 KDTree（点数少），再查询骨架点，避免
                # N*K 距离矩阵；域外点不能贡献覆盖。
                tree = cKDTree(positive_xy)
                distances, _ = tree.query(skeleton_xy, k=1)
                skeleton_coverage = float(
                    np.count_nonzero(distances <= radius_value)
                    / skeleton_xy.shape[0]
                )
            else:
                skeleton_coverage = 0.0

    return {
        "positive_on_road_rate": float(positive_on_road_rate),
        "negative_on_background_rate": float(negative_on_background_rate),
        "no_positive_rate": float(no_positive_rate),
        "point_outside_rate": float(point_outside_rate),
        "heatmap_above_threshold_rate": float(heatmap_above_threshold_rate),
        "heatmap_mean": float(heatmap_mean),
        "heatmap_max": float(heatmap_max),
        "skeleton_coverage": float(skeleton_coverage),
    }


# ---------------------------------------------------------------------------
# GRPO 组内诊断
# ---------------------------------------------------------------------------


def _empty_group_metrics(n_images: int, n_trajectories: int) -> Dict[str, float]:
    metrics = {key: float("nan") for key in REWARD_GROUP_METRIC_KEYS}
    metrics["n_images"] = float(n_images)
    metrics["n_trajectories"] = float(n_trajectories)
    return metrics


def reward_group_diagnostics(
    rows: Sequence[Mapping[str, Any]],
    advantages: Union[Tensor, np.ndarray],
) -> Dict[str, float]:
    """汇总同一影像 G 条轨迹的奖励与优势诊断。

    Args:
        rows: 按**轨迹优先**排列的 ``G * B`` 行，即第 ``g*B + b`` 行属于
            第 ``b`` 张影像的第 ``g`` 条轨迹。每行至少包含
            ``quality_reward``、``point_penalty`` 和 ``point_count``。
            ``quality_reward`` 是不含点成本的奖励，总奖励为
            ``quality_reward - point_penalty``。
        advantages: ``[G, B]`` 的 GRPO 优势张量，第 0 维是同一影像的
            轨迹组。

    Returns:
        dict，键与 :data:`REWARD_GROUP_METRIC_KEYS` 一致。所有组内标准差
        都是“逐图计算 std 后再对各图取平均”，不会把不同影像的轨迹混在
        一起；相关性同样逐图计算后只对有限值取平均。

    Notes:
        某张影像内优势或点数完全常量时，该图的 Pearson 相关为 NaN，并且
        不会用其它影像的数据“补足”。``undefined_advantage_corr_rate``
        记录这种图的比例。
    """
    row_list = list(rows)
    advantage_array = _as_numpy(advantages, "advantages")
    if advantage_array.ndim != 2:
        raise ValueError(
            f"advantages 必须是 [G, B]，实际为 {advantage_array.shape}"
        )
    try:
        advantage_array = advantage_array.astype(np.float64, copy=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("advantages 必须可转换为浮点数") from exc
    if not np.isfinite(advantage_array).all():
        raise ValueError("advantages 不能包含 NaN 或 Inf")

    group_size, batch_size = (
        int(advantage_array.shape[0]),
        int(advantage_array.shape[1]),
    )
    if len(row_list) != group_size * batch_size:
        raise ValueError(
            "rows 数量必须等于 G*B（轨迹优先）："
            f"{len(row_list)} vs {group_size}*{batch_size}"
        )
    if not row_list:
        return _empty_group_metrics(batch_size, group_size)

    required = ("quality_reward", "point_penalty", "point_count")
    quality = np.empty((group_size, batch_size), dtype=np.float64)
    penalty = np.empty((group_size, batch_size), dtype=np.float64)
    point_count = np.empty((group_size, batch_size), dtype=np.float64)
    for position, row in enumerate(row_list):
        if not isinstance(row, Mapping):
            raise TypeError(f"rows[{position}] 必须是 dict-like 映射")
        missing = [key for key in required if key not in row]
        if missing:
            raise KeyError(f"rows[{position}] 缺少字段: {missing}")
        trajectory_index, image_index = divmod(position, batch_size)
        for key, destination in (
            ("quality_reward", quality),
            ("point_penalty", penalty),
            ("point_count", point_count),
        ):
            try:
                value = float(row[key])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"rows[{position}][{key!r}] 必须是标量"
                ) from exc
            if not np.isfinite(value):
                raise ValueError(
                    f"rows[{position}][{key!r}] 不能是 NaN 或 Inf"
                )
            destination[trajectory_index, image_index] = value

    reward = quality - penalty
    advantage = advantage_array

    quality_group_std = np.std(quality, axis=0, ddof=0)
    penalty_group_std = np.std(penalty, axis=0, ddof=0)
    reward_group_std = np.std(reward, axis=0, ddof=0)
    point_count_group_std = np.std(point_count, axis=0, ddof=0)
    advantage_group_std = np.std(advantage, axis=0, ddof=0)

    advantage_count_corr = _pearson_by_group(advantage, point_count)
    reward_count_corr = _pearson_by_group(reward, point_count)
    quality_count_corr = _pearson_by_group(quality, point_count)

    if batch_size:
        constant_reward_group_rate = float(
            np.count_nonzero(np.ptp(reward, axis=0) == 0.0) / batch_size
        )
        undefined_advantage_corr_rate = float(
            np.count_nonzero(~np.isfinite(advantage_count_corr)) / batch_size
        )
    else:  # pragma: no cover - G*B=0 时已在上面返回
        constant_reward_group_rate = float("nan")
        undefined_advantage_corr_rate = float("nan")

    metrics = {
        "quality_reward_mean": float(quality.mean()),
        "quality_reward_group_std": _mean_finite(quality_group_std),
        "point_penalty_mean": float(penalty.mean()),
        "point_penalty_group_std": _mean_finite(penalty_group_std),
        "reward_mean": float(reward.mean()),
        "reward_group_std": _mean_finite(reward_group_std),
        "point_count_mean": float(point_count.mean()),
        "point_count_group_std": _mean_finite(point_count_group_std),
        "no_point_rate": float(np.count_nonzero(point_count == 0.0) / point_count.size),
        "advantage_abs_mean": float(np.abs(advantage).mean()),
        "advantage_group_std": _mean_finite(advantage_group_std),
        "advantage_point_count_corr": _mean_finite(advantage_count_corr),
        "reward_point_count_corr": _mean_finite(reward_count_corr),
        "quality_reward_point_count_corr": _mean_finite(quality_count_corr),
        "constant_reward_group_rate": constant_reward_group_rate,
        "undefined_advantage_corr_rate": undefined_advantage_corr_rate,
        "n_images": float(batch_size),
        "n_trajectories": float(group_size),
    }
    return metrics
