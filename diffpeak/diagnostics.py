"""场、提示接口与 GRPO 组内诊断的纯只读指标。

本模块只做诊断，不参与训练目标，也不修改任何输入张量/数组：

* :func:`field_quality`：signed_ridge_v1 单通道有符号场的分布、有效区域、
  前景/背景响应与 collapse 指标（engine 验证直接消费）；
* :func:`prompt_quality`：旧双通道点提示质量（历史纯函数，主链路不再使用）；
* :func:`reward_group_diagnostics`：同一影像 G 条轨迹奖励的组内统计，
  以及优势与点数的**逐图**相关性；
* :func:`build_oracle_prompt_fields` / :func:`oracle_prompt_diagnostics`：
  标注 oracle 山脊场/二值提示在不同固定尺度、零/打乱/局部扰动下的 SAM
  接口敏感性诊断（decode 可注入 CPU mock，不依赖真实权重）。

坐标约定
--------
* ``target`` 是**已裁掉右下填充**的二值标注，``H x W``，其中
  ``H, W <= image_size``；
* ``field`` 为 ``1 x h x w``（或 ``h x w``）的单通道有符号场，覆盖完整
  ``image_size`` 画布；有效区域按像素中心映射排除右下 padding；
* 旧 ``heatmap`` 为 ``2 x h x w``，通道 0 为正提示、通道 1 为负提示。

除显式接受 ``seed`` 的扰动函数外，本模块的诊断函数无随机操作，同一输入
产生同一输出，便于离线复核和单元测试。
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize
from torch import Tensor

__all__ = [
    "FIELD_METRIC_KEYS",
    "ORACLE_ROW_KEYS",
    "PERTURBATION_MODES",
    "PROMPT_METRIC_KEYS",
    "PROMPT_FIELD_KINDS",
    "REWARD_GROUP_METRIC_KEYS",
    "build_binary_prompt_field",
    "build_oracle_prompt_fields",
    "build_ridge_prompt_field",
    "field_quality",
    "oracle_prompt_diagnostics",
    "perturb_prompt_field",
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

#: :func:`field_quality` 返回的全部指标名，按固定顺序排列。
FIELD_METRIC_KEYS: Tuple[str, ...] = (
    "field_mean",
    "field_std",
    "field_min",
    "field_max",
    "field_positive_rate",
    "field_negative_rate",
    "field_zero_rate",
    "field_saturation_rate",
    "valid_pixels",
    "valid_field_mean",
    "valid_field_std",
    "valid_field_min",
    "valid_field_max",
    "valid_positive_rate",
    "valid_negative_rate",
    "valid_background_rate",
    "foreground_rate",
    "target_foreground_rate",
    "foreground_field_mean",
    "background_field_mean",
    "foreground_positive_rate",
    "background_negative_rate",
    "foreground_collapse_rate",
    "background_collapse_rate",
    "sign_agreement",
    "collapsed",
)

#: oracle 接口诊断允许的提示场类型与扰动模式。
PROMPT_FIELD_KINDS: Tuple[str, ...] = ("ridge", "binary")
PERTURBATION_MODES: Tuple[str, ...] = ("none", "zero", "shuffled", "local")

#: :func:`oracle_prompt_diagnostics` 每行返回的指标名。
ORACLE_ROW_KEYS: Tuple[str, ...] = (
    "prompt_kind",
    "scale",
    "perturbation",
    "iou",
    "dice",
    "f1",
    "precision",
    "recall",
    "prediction_positive_rate",
    "target_positive_rate",
    "prompt_mean",
    "prompt_std",
    "prompt_positive_rate",
    "output_mean",
    "output_std",
    "mean_abs_delta",
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
# signed_ridge_v1 场质量
# ---------------------------------------------------------------------------


def _as_2d_field(field: Any) -> np.ndarray:
    """把 ``h x w`` / ``1 x h x w`` / ``B x 1 x h x w`` 整理成 float64 数组。"""
    array = _as_numpy(field, "field")
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError("field 必须是数值数组")
    array = array.astype(np.float64, copy=False)
    if not np.isfinite(array).all():
        raise ValueError("field 不能包含 NaN 或 Inf")
    return array


def _normalise_field_batch(field: Any) -> List[np.ndarray]:
    """返回逐样本的 ``h x w`` 单通道场。"""
    array = _as_2d_field(field)
    if array.ndim == 2:
        samples = [array]
    elif array.ndim == 3:
        if int(array.shape[0]) == 1:
            samples = [array[0]]
        else:
            samples = [array[index] for index in range(int(array.shape[0]))]
    elif array.ndim == 4:
        if int(array.shape[1]) != 1:
            raise ValueError(
                "field 通道维必须为 1，实际为 " f"{array.shape}"
            )
        samples = [array[index, 0] for index in range(int(array.shape[0]))]
    else:
        raise ValueError(f"field 必须是 2D/3D/4D，实际为 {array.shape}")
    for sample in samples:
        if sample.shape[0] <= 0 or sample.shape[1] <= 0:
            raise ValueError("field 的 H/W 必须为正")
    return samples


def _normalise_field_targets(
    target: Any,
    batch_size: int,
) -> List[Optional[np.ndarray]]:
    """返回逐样本二值 ``h x w`` 目标；``None`` 表示只做无监督分布诊断。"""
    if target is None:
        return [None] * batch_size
    array = np.asarray(target)
    if array.ndim == 4 and int(array.shape[1]) == 1:
        array = array[:, 0]
    if array.ndim == 2:
        binary = _as_binary_target(array)
        return [binary] * batch_size
    if array.ndim == 3:
        if int(array.shape[0]) != batch_size:
            raise ValueError(
                f"target batch {int(array.shape[0])} 与 field batch "
                f"{batch_size} 不一致"
            )
        return [_as_binary_target(array[index]) for index in range(batch_size)]
    raise ValueError(f"target 必须是 HxW 或 BxHxW，实际为 {array.shape}")


def _normalise_field_valid_sizes(
    valid_sizes: Any,
    batch_size: int,
) -> List[Optional[Tuple[int, int]]]:
    if valid_sizes is None:
        return [None] * batch_size
    if (
        isinstance(valid_sizes, (tuple, list))
        and len(valid_sizes) == 2
        and all(np.isscalar(item) for item in valid_sizes)
    ):
        pair = (int(valid_sizes[0]), int(valid_sizes[1]))
        return [pair] * batch_size
    try:
        items = list(valid_sizes)
    except TypeError as exc:
        raise ValueError("valid_sizes 必须是 (h, w) 或序列") from exc
    if len(items) != batch_size:
        raise ValueError(
            f"valid_sizes 长度 {len(items)} 与 field batch {batch_size} 不一致"
        )
    result: List[Optional[Tuple[int, int]]] = []
    for item in items:
        if item is None:
            result.append(None)
            continue
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            raise ValueError(f"无效的 valid_size：{item!r}")
        result.append((int(item[0]), int(item[1])))
    return result


def _field_valid_mask(
    field_shape: Tuple[int, int],
    target_shape: Optional[Tuple[int, int]],
    valid_size: Optional[Tuple[int, int]],
    image_size: Optional[int],
) -> np.ndarray:
    """计算完整画布场上的有效像素（排除右下 padding）。"""
    height, width = int(field_shape[0]), int(field_shape[1])
    crop = target_shape if target_shape is not None else valid_size
    if crop is None:
        return np.ones((height, width), dtype=bool)
    crop_height, crop_width = int(crop[0]), int(crop[1])
    if (height, width) == (crop_height, crop_width):
        return np.ones((height, width), dtype=bool)
    if image_size is None:
        raise ValueError(
            "field 分辨率与有效区域不同时必须提供 image_size 画布尺寸"
        )
    size = int(image_size)
    if size <= 0:
        raise ValueError("image_size 必须是正整数")
    if crop_height > size or crop_width > size:
        raise ValueError(
            "target/valid_size 必须位于 image_size 画布内："
            f"crop={(crop_height, crop_width)}, image_size={size}"
        )
    return _heatmap_valid_mask(
        crop_height, crop_width, size, height, width
    )


def _field_distribution(values: np.ndarray, prefix: str) -> Dict[str, float]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return {
            f"{prefix}_mean": float("nan"),
            f"{prefix}_std": float("nan"),
            f"{prefix}_min": float("nan"),
            f"{prefix}_max": float("nan"),
        }
    return {
        f"{prefix}_mean": float(values.mean()),
        f"{prefix}_std": float(values.std(ddof=0)),
        f"{prefix}_min": float(values.min()),
        f"{prefix}_max": float(values.max()),
    }


def _rate(mask: np.ndarray, denominator: int) -> float:
    if denominator <= 0:
        return float("nan")
    return float(np.count_nonzero(mask) / denominator)


def field_quality(
    field: Any,
    target: Any = None,
    *,
    image_size: Optional[int] = None,
    valid_sizes: Any = None,
    background_value: float = -1.0,
    background_tolerance: float = 0.05,
    saturation_threshold: float = 0.99,
    collapse_std: float = 0.01,
) -> Dict[str, float]:
    """诊断单通道有符号场的分布、有效区域与 collapse/前景响应。

    Engine 验证的调用约定为
    ``field_quality(fields[i], target_numpy_valid_crop, image_size=self.image_size)``：

    Args:
        field: ``h x w``、``1 x h x w`` 或 ``B x 1 x h x w`` 有符号场。完整
            画布上的 padding 不参与 ``valid_*`` / 前景/背景指标。
        target: 可选二值标注，``H x W``（已裁掉 padding）或 ``B x H x W``。
            与 ``field`` 同分辨率，或通过 ``image_size`` 按像素中心映射。
        image_size: SAM 输入画布边长；当 field 分辨率与 target 不同时必需。
        valid_sizes: 可选 ``(h, w)`` 或逐样本序列；没有 target 时用于排除
            padding。单位与 target 相同（画布像素）。
        background_value: 山脊场背景基准，默认 -1。
        background_tolerance: 判定“落在背景值附近”的容差。
        saturation_threshold: ``|field| >= threshold`` 记为饱和。
        collapse_std: 有效区域标准差低于该值视为常量/塌缩。

    Returns:
        dict，键与 :data:`FIELD_METRIC_KEYS` 完全一致。无定义的值（例如没有
        target 时的前景指标、空有效区域）返回 ``NaN``；``collapsed`` 为 0/1。
    """
    try:
        background = float(background_value)
        tolerance = float(background_tolerance)
        saturation = float(saturation_threshold)
        std_threshold = float(collapse_std)
    except (TypeError, ValueError) as exc:
        raise ValueError("背景/饱和/collapse 参数必须是有限数") from exc
    if not all(
        np.isfinite(value)
        for value in (background, tolerance, saturation, std_threshold)
    ):
        raise ValueError("背景/饱和/collapse 参数必须是有限数")
    if tolerance < 0.0 or std_threshold < 0.0:
        raise ValueError("background_tolerance / collapse_std 必须非负")
    if not 0.0 <= saturation <= 1.0:
        raise ValueError("saturation_threshold 必须在 [0, 1] 内")

    samples = _normalise_field_batch(field)
    batch_size = len(samples)
    targets = _normalise_field_targets(target, batch_size)
    sizes = _normalise_field_valid_sizes(valid_sizes, batch_size)
    if any(item is not None for item in targets) and any(
        item is None for item in targets
    ):
        raise ValueError("target 要么全部提供，要么全部为 None")

    all_values: List[np.ndarray] = []
    valid_values: List[np.ndarray] = []
    # 逐样本只保存“有效像素内”的前景标记，长度与 valid_values 一致。
    foreground_flags: List[np.ndarray] = []
    have_target = True

    for index, sample in enumerate(samples):
        sample_target = targets[index]
        target_shape = (
            None
            if sample_target is None
            else (int(sample_target.shape[0]), int(sample_target.shape[1]))
        )
        valid = _field_valid_mask(
            (int(sample.shape[0]), int(sample.shape[1])),
            target_shape,
            sizes[index],
            image_size,
        )
        values = sample.reshape(-1)
        all_values.append(values)
        valid_flat = valid.reshape(-1)
        valid_values.append(sample[valid])
        if sample_target is None:
            have_target = False
            continue
        if sample_target.shape != sample.shape:
            # 有效 crop 与热图分辨率不同：最近邻对齐到热图网格后，再与
            # 像素中心有效掩码相交，避免把 padding 当作背景。
            target_tensor = torch.from_numpy(
                sample_target.astype(np.float32)
            )[None, None]
            resized = F.interpolate(
                target_tensor,
                size=(int(sample.shape[0]), int(sample.shape[1])),
                mode="nearest",
            )[0, 0]
            sample_target = resized.numpy() > 0.5
        # 只保留有效像素内的前景标记，与 valid_concat 一一对应。
        foreground_flags.append(
            (sample_target.reshape(-1) & valid_flat)[valid_flat]
        )

    all_concat = np.concatenate(all_values) if all_values else np.empty(0)
    valid_concat = np.concatenate(valid_values) if valid_values else np.empty(0)
    valid_count = int(valid_concat.size)

    metrics = _field_distribution(all_concat, "field")
    metrics.update(_field_distribution(valid_concat, "valid_field"))
    metrics["valid_pixels"] = float(valid_count)

    all_count = int(all_concat.size)
    metrics["field_positive_rate"] = _rate(all_concat > 0.0, all_count)
    metrics["field_negative_rate"] = _rate(all_concat < 0.0, all_count)
    metrics["field_zero_rate"] = _rate(all_concat == 0.0, all_count)
    metrics["field_saturation_rate"] = _rate(
        np.abs(all_concat) >= saturation, all_count
    )

    metrics["valid_positive_rate"] = _rate(valid_concat > 0.0, valid_count)
    metrics["valid_negative_rate"] = _rate(valid_concat < 0.0, valid_count)
    metrics["valid_background_rate"] = _rate(
        np.abs(valid_concat - background) <= tolerance, valid_count
    )
    metrics["foreground_rate"] = metrics["valid_positive_rate"]

    valid_std = metrics["valid_field_std"]
    valid_mean = metrics["valid_field_mean"]
    collapsed = 0.0
    if valid_count and np.isfinite(valid_std) and valid_std <= std_threshold:
        collapsed = 1.0

    if have_target:
        foreground = (
            np.concatenate(foreground_flags)
            if foreground_flags
            else np.zeros(0, dtype=bool)
        )
        field_valid = valid_concat
        background = ~foreground
        target_foreground_count = int(np.count_nonzero(foreground))
        target_background_count = int(np.count_nonzero(background))

        metrics["target_foreground_rate"] = _rate(
            foreground, valid_count
        )
        if target_foreground_count:
            metrics["foreground_field_mean"] = float(
                field_valid[foreground].mean()
            )
            metrics["foreground_positive_rate"] = _rate(
                field_valid[foreground] > 0.0, target_foreground_count
            )
            metrics["foreground_collapse_rate"] = _rate(
                np.abs(field_valid[foreground] - background_value) <= tolerance,
                target_foreground_count,
            )
        else:
            metrics["foreground_field_mean"] = float("nan")
            metrics["foreground_positive_rate"] = float("nan")
            metrics["foreground_collapse_rate"] = float("nan")
        if target_background_count:
            metrics["background_field_mean"] = float(
                field_valid[background].mean()
            )
            metrics["background_negative_rate"] = _rate(
                field_valid[background] < 0.0, target_background_count
            )
            metrics["background_collapse_rate"] = _rate(
                np.abs(field_valid[background] - background_value) <= tolerance,
                target_background_count,
            )
        else:
            metrics["background_field_mean"] = float("nan")
            metrics["background_negative_rate"] = float("nan")
            metrics["background_collapse_rate"] = float("nan")

        target_concat = np.zeros(valid_count, dtype=bool)
        target_concat[foreground] = True
        metrics["sign_agreement"] = _rate(
            (field_valid > 0.0) == target_concat, valid_count
        )
        if (
            target_foreground_count > 0
            and np.isfinite(metrics["foreground_positive_rate"])
            and metrics["foreground_positive_rate"] <= 0.0
        ):
            collapsed = 1.0
    else:
        for key in (
            "target_foreground_rate",
            "foreground_field_mean",
            "background_field_mean",
            "foreground_positive_rate",
            "background_negative_rate",
            "foreground_collapse_rate",
            "background_collapse_rate",
            "sign_agreement",
        ):
            metrics[key] = float("nan")

    metrics["collapsed"] = float(collapsed)
    # 固定键顺序，便于 engine 直接按 FIELD_METRIC_KEYS 记录。
    return {key: float(metrics[key]) for key in FIELD_METRIC_KEYS}


# ---------------------------------------------------------------------------
# Oracle 提示接口诊断
# ---------------------------------------------------------------------------


def _oracle_mask(mask: Any) -> np.ndarray:
    array = np.asarray(mask)
    if array.ndim != 2:
        raise ValueError(f"mask 必须是 HxW，实际为 {array.shape}")
    return _as_binary_target(array)


def build_binary_prompt_field(
    mask: Any,
    *,
    foreground: float = 1.0,
    background: float = -1.0,
) -> np.ndarray:
    """由标注 mask 构造二值 oracle 提示：道路 +1、背景 -1。"""
    binary = _oracle_mask(mask)
    field = np.where(binary, float(foreground), float(background))
    return field.astype(np.float32)


def build_ridge_prompt_field(
    mask: Any,
    *,
    ridge_height: float = 1.0,
    background: float = -1.0,
    epsilon: float = 1e-6,
) -> np.ndarray:
    """由标注 mask 按方案公式构造 oracle 单通道山脊场。

    ``H*(x) = d_boundary / (d_boundary + d_center + eps)``（道路内），
    中心线显式赋值为 ``ridge_height``，道路外为 ``background``。距离变换在
    四周补一圈背景，因此全道路图和接触图像边缘的道路也有确定边界。
    """
    binary = _oracle_mask(mask)
    if not np.isfinite(ridge_height) or not np.isfinite(background):
        raise ValueError("ridge_height / background 必须是有限数")
    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon 必须是正有限数")
    field = np.full(binary.shape, float(background), dtype=np.float64)
    if not bool(binary.any()):
        return field.astype(np.float32)

    padded = np.pad(binary, 1, mode="constant", constant_values=False)
    d_boundary = distance_transform_edt(padded)[1:-1, 1:-1]
    centerline = skeletonize(binary)
    if not bool(centerline.any()):
        centerline = binary.copy()
    d_center = distance_transform_edt(~centerline)
    ratio = d_boundary / (d_boundary + d_center + float(epsilon))
    field[binary] = ratio[binary]
    field[centerline] = float(ridge_height)
    return np.clip(field, -1.0, 1.0).astype(np.float32)


def build_oracle_prompt_fields(
    mask: Any,
    kinds: Sequence[str] = PROMPT_FIELD_KINDS,
) -> Dict[str, np.ndarray]:
    """构造 oracle ``ridge`` / ``binary`` 提示场，供接口诊断比较。"""
    requested = [str(kind).strip().lower() for kind in kinds]
    unknown = [kind for kind in requested if kind not in PROMPT_FIELD_KINDS]
    if unknown:
        raise ValueError(
            "未知 oracle 提示类型：" f"{unknown}；可选 {PROMPT_FIELD_KINDS}"
        )
    fields: Dict[str, np.ndarray] = {}
    for kind in requested:
        if kind == "ridge":
            fields[kind] = build_ridge_prompt_field(mask)
        else:
            fields[kind] = build_binary_prompt_field(mask)
    return fields


def _as_field_2d(field: Any, name: str = "field") -> np.ndarray:
    array = _as_numpy(field, name)
    if array.ndim == 3 and int(array.shape[0]) == 1:
        array = array[0]
    if array.ndim != 2:
        raise ValueError(f"{name} 必须是 HxW 或 1xHxW，实际为 {array.shape}")
    array = array.astype(np.float64, copy=False)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} 不能包含 NaN 或 Inf")
    return array


def perturb_prompt_field(
    field: Any,
    mode: str = "none",
    *,
    seed: int = 0,
    strength: float = 0.1,
    valid_mask: Any = None,
) -> np.ndarray:
    """对提示场施加确定性扰动，用于检查 SAM 是否真正使用该提示。

    Args:
        field: ``H x W`` 或 ``1 x H x W`` 的已映射有符号场。
        mode: ``none`` / ``zero`` / ``shuffled`` / ``local``。``zero`` 表示
            零提示；``shuffled`` 在有效区域内打乱像素；``local`` 只在随机
            小方块内加入高斯扰动。
        seed: 扰动随机种子；同一 ``(field, mode, seed)`` 结果完全一致。
        strength: ``local`` 扰动标准差。
        valid_mask: 可选 ``H x W`` bool，限制 shuffled/local 的作用范围。

    Returns:
        ``float32`` 的 ``H x W`` 场。
    """
    array = _as_field_2d(field)
    normalised = str(mode).strip().lower()
    if normalised not in PERTURBATION_MODES:
        raise ValueError(
            f"未知扰动模式 {mode!r}；可选 {PERTURBATION_MODES}"
        )
    if not np.isfinite(strength) or strength < 0.0:
        raise ValueError("strength 必须是有限非负数")
    rng = np.random.default_rng(int(seed))

    if normalised == "none":
        return array.astype(np.float32).copy()
    if normalised == "zero":
        return np.zeros(array.shape, dtype=np.float32)

    if valid_mask is None:
        valid = np.ones(array.shape, dtype=bool)
    else:
        valid = np.asarray(valid_mask)
        if valid.shape != array.shape:
            raise ValueError(
                f"valid_mask 形状 {valid.shape} 与 field {array.shape} 不一致"
            )
        valid = valid.astype(bool)

    perturbed = array.copy()
    if normalised == "shuffled":
        indices = np.flatnonzero(valid)
        if indices.size > 1:
            perturbed[valid] = array.reshape(-1)[indices][
                rng.permutation(indices.size)
            ]
        return perturbed.astype(np.float32)

    # local：在有效区域内随机选择一个约 1/8 边长的小方块加噪声。
    height, width = array.shape
    valid_y, valid_x = np.nonzero(valid)
    if valid_y.size:
        center_y = int(valid_y[rng.integers(valid_y.size)])
        center_x = int(valid_x[rng.integers(valid_x.size)])
        half_h = max(1, height // 16)
        half_w = max(1, width // 16)
        y0, y1 = max(0, center_y - half_h), min(height, center_y + half_h + 1)
        x0, x1 = max(0, center_x - half_w), min(width, center_x + half_w + 1)
        noise = rng.normal(0.0, float(strength), size=(y1 - y0, x1 - x0))
        window = perturbed[y0:y1, x0:x1]
        perturbed[y0:y1, x0:x1] = np.clip(window + noise, -1.0, 1.0)
    return perturbed.astype(np.float32)


def _oracle_logits_to_2d(logits: Any) -> np.ndarray:
    if torch.is_tensor(logits):
        array = logits.detach().float().cpu().numpy()
    else:
        array = np.asarray(logits)
    array = array.astype(np.float64, copy=False)
    if array.ndim == 4:
        if int(array.shape[0]) != 1 or int(array.shape[1]) != 1:
            raise ValueError(
                "decode 输出必须是单样本单通道 B=1,C=1，实际为 "
                f"{array.shape}"
            )
        array = array[0, 0]
    elif array.ndim == 3:
        if int(array.shape[0]) != 1:
            raise ValueError(f"decode 输出 batch 必须为 1，实际为 {array.shape}")
        array = array[0]
    if array.ndim != 2:
        raise ValueError(f"decode 输出无法解释为 HxW：{array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("decode 输出不能包含 NaN 或 Inf")
    return array


def _resize_oracle_output(array: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    if array.shape == shape:
        return array
    tensor = torch.from_numpy(array).float()[None, None]
    resized = F.interpolate(
        tensor, size=shape, mode="bilinear", align_corners=False
    )
    return resized[0, 0].numpy().astype(np.float64)


def _binary_metrics(
    prediction: np.ndarray, target: np.ndarray
) -> Dict[str, float]:
    prediction = np.asarray(prediction).astype(bool)
    target = np.asarray(target).astype(bool)
    tp = int(np.count_nonzero(prediction & target))
    fp = int(np.count_nonzero(prediction & ~target))
    fn = int(np.count_nonzero(~prediction & target))
    union = tp + fp + fn
    denominator = 2 * tp + fp + fn
    iou = 1.0 if union == 0 else tp / float(union)
    dice = 1.0 if denominator == 0 else 2.0 * tp / float(denominator)
    precision = float("nan") if tp + fp == 0 else tp / float(tp + fp)
    recall = float("nan") if tp + fn == 0 else tp / float(tp + fn)
    return {
        "iou": float(iou),
        "dice": float(dice),
        "f1": float(dice),
        "precision": float(precision),
        "recall": float(recall),
        "prediction_positive_rate": float(prediction.mean()),
        "target_positive_rate": float(target.mean()),
    }


def oracle_prompt_diagnostics(
    prompt_fields: Mapping[str, Any],
    decode: Callable[[Tensor], Any],
    target: Any,
    *,
    scales: Sequence[float] = (1.0,),
    perturbations: Sequence[str] = PERTURBATION_MODES,
    terminal_mapping: str = "clamp",
    threshold: float = 0.5,
    seed: int = 0,
    valid_mask: Any = None,
    prompt_valid_mask: Any = None,
) -> Dict[str, Any]:
    """比较 oracle 山脊/二值提示在不同尺度与扰动下的分割响应。

    该函数是接口诊断而非训练目标：
    ``标注 mask -> 目标山脊场/二值场 -> 固定尺度映射 -> decode -> 指标``。
    ``decode`` 可以是真实冻结 SAM 的闭包，也可以是 CPU mock
    （接收 ``1 x 1 x H x W`` 张量并返回 logits），因此单元测试无需权重。

    Args:
        prompt_fields: ``{"ridge": field, "binary": field, ...}``。
        decode: 单样本解码函数，返回 logits（``1x1xHxW`` / ``1xHxW`` /
            ``HxW``）。输出分辨率与 target 不同时会双线性对齐。
        target: 二值道路标注 ``H x W``。
        scales: 固定正尺度序列，映射 ``U = scale * clamp(H)``。先裁剪场，
            再施加尺度，因此 ``scale > 1`` 时可保留超过 1 的提示幅度。
        perturbations: ``none`` / ``zero`` / ``shuffled`` / ``local``。
        terminal_mapping: ``clamp`` 或 ``identity``。
        threshold: 概率阈值。
        seed: 扰动种子。
        valid_mask: 可选输出分辨率 bool 掩码；指标只在该有效区域内计算
            （例如裁掉右下 padding）。
        prompt_valid_mask: 可选提示分辨率 bool 掩码；零/打乱/局部扰动与提示
            统计只作用于该区域，padding 保持不变。

    Returns:
        ``{"rows": [...], "summary": {...}}``；每行键与
        :data:`ORACLE_ROW_KEYS` 一致。
    """
    if not isinstance(prompt_fields, Mapping) or not prompt_fields:
        raise ValueError("prompt_fields 必须是非空映射")
    target_binary = _oracle_mask(target)
    if not callable(decode):
        raise TypeError("decode 必须是可调用对象")

    scale_values: List[float] = []
    for value in scales:
        try:
            scale = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("scales 必须是正有限数序列") from exc
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError("scales 必须是正有限数序列")
        scale_values.append(scale)

    mode_values = [str(mode).strip().lower() for mode in perturbations]
    unknown = [mode for mode in mode_values if mode not in PERTURBATION_MODES]
    if unknown:
        raise ValueError(
            f"未知扰动模式：{unknown}；可选 {PERTURBATION_MODES}"
        )
    mapping = str(terminal_mapping).strip().lower()
    if mapping != "clamp":
        raise ValueError("第一版 terminal_mapping 只支持 clamp")
    threshold_value = float(threshold)
    if not 0.0 <= threshold_value <= 1.0:
        raise ValueError("threshold 必须在 [0, 1] 内")

    score_mask: Optional[np.ndarray] = None
    if valid_mask is not None:
        score_mask = np.asarray(valid_mask)
        if score_mask.shape != target_binary.shape:
            raise ValueError(
                f"valid_mask 形状 {score_mask.shape} 与 target "
                f"{target_binary.shape} 不一致"
            )
        score_mask = score_mask.astype(bool)
    prompt_mask: Optional[np.ndarray] = None
    if prompt_valid_mask is not None:
        prompt_mask = np.asarray(prompt_valid_mask)
        if prompt_mask.ndim == 3 and int(prompt_mask.shape[0]) == 1:
            prompt_mask = prompt_mask[0]
        if prompt_mask.ndim != 2:
            raise ValueError("prompt_valid_mask 必须是 HxW 或 1xHxW")
        prompt_mask = prompt_mask.astype(bool)

    rows: List[Dict[str, Any]] = []
    logits_cache: List[np.ndarray] = []
    for kind, raw_field in prompt_fields.items():
        base = _as_field_2d(raw_field, f"prompt_fields[{kind!r}]")
        if prompt_mask is not None and prompt_mask.shape != base.shape:
            raise ValueError(
                f"prompt_valid_mask 形状 {prompt_mask.shape} 与提示场 "
                f"{base.shape} 不一致"
            )
        # 先对未缩放的 clamp(H) 施加扰动，再乘以尺度；否则 local 扰动的
        # [-1, 1] 裁剪会抹掉 s > 1 的局部幅度。同一 kind 的各尺度/模式使用
        # 与模式绑定的种子，保证扰动在尺度之间可比。
        bounded = np.clip(base, -1.0, 1.0) if mapping == "clamp" else base
        perturbed_by_mode = {
            mode: perturb_prompt_field(
                bounded,
                mode,
                seed=seed + mode_index,
                valid_mask=prompt_mask,
            )
            for mode_index, mode in enumerate(mode_values)
        }
        for scale in scale_values:
            # 顺序固定为 U = scale * clamp(H)；不做 clamp(scale * H)。
            for mode in mode_values:
                prompt = perturbed_by_mode[mode] * scale
                tensor = torch.from_numpy(prompt).float()[None, None]
                logits = _oracle_logits_to_2d(decode(tensor))
                logits = _resize_oracle_output(logits, target_binary.shape)
                probability = 1.0 / (1.0 + np.exp(-logits))
                if score_mask is None:
                    prediction_region = probability >= threshold_value
                    target_region = target_binary
                else:
                    prediction_region = (probability >= threshold_value)[
                        score_mask
                    ]
                    target_region = target_binary[score_mask]
                metrics = _binary_metrics(prediction_region, target_region)
                prompt_values = (
                    prompt.reshape(-1)
                    if prompt_mask is None
                    else prompt[prompt_mask]
                )
                logits_values = (
                    logits.reshape(-1)
                    if score_mask is None
                    else logits[score_mask]
                )
                row: Dict[str, Any] = {
                    "prompt_kind": str(kind),
                    "scale": float(scale),
                    "perturbation": mode,
                    **metrics,
                    "prompt_mean": float(prompt_values.mean()),
                    "prompt_std": float(prompt_values.std(ddof=0)),
                    "prompt_positive_rate": float(
                        (prompt_values > 0.0).mean()
                    ),
                    "output_mean": float(logits_values.mean()),
                    "output_std": float(logits_values.std(ddof=0)),
                    "mean_abs_delta": float("nan"),
                }
                rows.append(row)
                logits_cache.append(logits)

    # 同 kind/scale 的 none 输出作为扰动响应基线。
    baseline: Dict[Tuple[str, float], np.ndarray] = {}
    for row, logits in zip(rows, logits_cache):
        if row["perturbation"] == "none":
            baseline[(row["prompt_kind"], row["scale"])] = logits
    for row, logits in zip(rows, logits_cache):
        reference = baseline.get((row["prompt_kind"], row["scale"]))
        if reference is not None and reference.shape == logits.shape:
            difference = np.abs(logits - reference)
            if score_mask is not None:
                # padding 不参与响应差异，避免大面积零区稀释指标。
                difference = difference[score_mask]
            row["mean_abs_delta"] = float(difference.mean())

    finite_iou = [row["iou"] for row in rows if row["iou"] == row["iou"]]
    best_index = int(np.argmax(finite_iou)) if finite_iou else -1

    def _mean_by(key: str) -> Dict[str, float]:
        groups: Dict[str, List[float]] = {}
        for row in rows:
            value = float(row["iou"])
            if math.isfinite(value):
                groups.setdefault(str(row[key]), []).append(value)
        return {
            name: float(np.mean(values)) for name, values in groups.items()
        }

    best = rows[best_index] if best_index >= 0 else None
    summary: Dict[str, Any] = {
        "n_rows": len(rows),
        "best_iou": float(best["iou"]) if best is not None else float("nan"),
        "best_kind": best["prompt_kind"] if best is not None else None,
        "best_scale": float(best["scale"]) if best is not None else float("nan"),
        "best_perturbation": (
            best["perturbation"] if best is not None else None
        ),
        "mean_iou_by_kind": _mean_by("prompt_kind"),
        "mean_iou_by_perturbation": _mean_by("perturbation"),
    }
    return {"rows": rows, "summary": summary}


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
