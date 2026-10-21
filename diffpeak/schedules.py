"""学习率预热与 cosine 退火。

训练引擎在每个阶段构造 ``LambdaLR`` 时使用这里的纯函数；把它们独立出来
是为了能在不加载模型、不创建优化器的情况下做单元测试。

调度形状
--------
1. 前 ``warmup_steps`` 次优化器更新：线性从 ``1/warmup_steps`` 升到 1；
2. 之后：从 1 按 cosine 曲线退火到 ``min_lr_ratio``；
3. 最后一步（``step >= total_steps``）保持在 ``min_lr_ratio``。

``warmup_steps`` 为 0 时退化为纯 cosine。所有边界都做了 clamp，避免
``total_steps <= warmup_steps`` 时出现除零或负 progress。
"""

from __future__ import annotations

import math

__all__ = ["resolve_warmup_steps", "warmup_cosine_factor"]


def _as_int(value: int, name: str) -> int:
    try:
        integer = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} 必须是整数") from exc
    if isinstance(value, bool) or integer != value:
        raise ValueError(f"{name} 必须是整数")
    return integer


def resolve_warmup_steps(
    total_steps: int,
    warmup_steps: int = 0,
    warmup_ratio: float = 0.0,
) -> int:
    """把绝对/比例预热配置解析成实际步数。

    显式 ``warmup_steps > 0`` 优先；否则使用 ``round(total_steps * ratio)``。
    结果被限制在 ``[0, total_steps]``。
    """
    total = _as_int(total_steps, "total_steps")
    steps = _as_int(warmup_steps, "warmup_steps")
    if total < 1:
        raise ValueError("total_steps 必须为正整数")
    if steps < 0:
        raise ValueError("warmup_steps 不能为负")
    try:
        ratio = float(warmup_ratio)
    except (TypeError, ValueError) as exc:
        raise ValueError("warmup_ratio 必须是 [0, 1] 内的有限数") from exc
    if not math.isfinite(ratio) or not (0.0 <= ratio <= 1.0):
        raise ValueError("warmup_ratio 必须是 [0, 1] 内的有限数")
    if steps <= 0 and ratio > 0.0:
        steps = int(round(total * ratio))
    return min(steps, total)


def warmup_cosine_factor(
    step: int,
    total_steps: int,
    warmup_steps: int = 0,
    min_lr_ratio: float = 0.0,
) -> float:
    """返回第 ``step`` 次更新后的学习率乘子。

    Args:
        step: 已完成的优化器更新次数（从 0 开始）。
        total_steps: 本阶段计划的总更新次数（含预热）。
        warmup_steps: 线性预热步数。
        min_lr_ratio: cosine 退火终点乘子，必须在 ``[0, 1]``。

    Returns:
        ``[min_lr_ratio, 1]`` 内的浮点乘子。
    """
    total = _as_int(total_steps, "total_steps")
    warmup = _as_int(warmup_steps, "warmup_steps")
    position = _as_int(step, "step")
    if total < 1:
        raise ValueError("total_steps 必须为正整数")
    if warmup < 0:
        raise ValueError("warmup_steps 不能为负")
    if position < 0:
        raise ValueError("step 不能为负")
    warmup = min(warmup, total)
    try:
        floor = float(min_lr_ratio)
    except (TypeError, ValueError) as exc:
        raise ValueError("min_lr_ratio 必须在 [0, 1] 内") from exc
    if not math.isfinite(floor) or not (0.0 <= floor <= 1.0):
        raise ValueError("min_lr_ratio 必须在 [0, 1] 内")

    if position >= total:
        return floor
    if warmup > 0 and position < warmup:
        # 第一步立即离开 0，避免优化器在首步完全不动。
        return max(1e-8, float(position + 1) / float(warmup))

    progress = (position - warmup) / float(max(1, total - warmup))
    progress = min(1.0, max(0.0, progress))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return floor + (1.0 - floor) * cosine
