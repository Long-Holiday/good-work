"""GRPO（Group Relative Policy Optimization）的纯函数工具。

本模块只实现与具体模型无关的数学步骤，便于单元测试：

* :func:`group_relative_advantages`：同一影像的 G 条轨迹奖励做组内标准化；
* :func:`grpo_policy_loss`：带裁剪的 GRPO/PPO 代理目标；
* :func:`gaussian_kl_divergence`：固定方差高斯策略之间的逐样本 KL。

约定
----
* 奖励张量形状为 ``[G, ...]``，**第 0 维必须是同一影像的候选组**；
  若为 ``[G, B]``，则每一列是一张影像的组，优势按列独立计算。
* 组内标准差使用 ``std.clamp_min(delta)`` 作为分母；只有奖励完全相同的组
  优势才为零，非零但方差很小的组按 ``delta`` 缩放，不会被跳过。
* 动作维联合对数概率在调用方按维求和；本模块不改变这一约定。
"""

from __future__ import annotations

from typing import Union

import numpy as np
import torch
from torch import Tensor

__all__ = [
    "group_relative_advantages",
    "grpo_policy_loss",
    "gaussian_kl_divergence",
]


def _as_float64_tensor(rewards: Union[Tensor, np.ndarray]) -> Tensor:
    if torch.is_tensor(rewards):
        return rewards.detach().to(torch.float64)
    return torch.as_tensor(np.asarray(rewards), dtype=torch.float64)


def group_relative_advantages(
    rewards: Union[Tensor, np.ndarray],
    delta: float = 1e-4,
    dim: int = 0,
) -> Tensor:
    """计算 GRPO 组内相对优势。

    ``A_i = (R_i - mean(R)) / max(std(R), delta)``，其中均值和标准差只使用
    ``dim`` 上的同一组候选。标准差使用总体标准差（``unbiased=False``），
    与“组内 G 个候选就是完整组”的语义一致。分母下限只限制缩放，不把
    低方差组整体置零；奖励完全相同时优势自然为零。

    Args:
        rewards: 组内奖励；默认第 0 维是候选维，每组至少两个候选。
        delta: 标准差下限，必须是严格正有限数。
        dim: 候选组所在维度。

    Returns:
        与输入同形状的 float64 优势张量，已 detach。

    Raises:
        ValueError: 奖励非有限、输入为空、组大小小于 2 或参数非法。
    """
    try:
        delta_value = float(delta)
    except (TypeError, ValueError) as exc:
        raise ValueError("delta 必须是严格正的有限数") from exc
    if not np.isfinite(delta_value) or delta_value <= 0.0:
        raise ValueError("delta 必须是严格正的有限数")

    values = _as_float64_tensor(rewards)
    if values.numel() == 0:
        raise ValueError("rewards 不能为空")
    if values.ndim == 0:
        raise ValueError("rewards 至少需要一维候选组")
    if not (-values.ndim <= int(dim) < values.ndim):
        raise ValueError(f"dim={dim} 超出 rewards 维度 {values.ndim}")
    if values.size(int(dim)) < 2:
        raise ValueError("每个 GRPO 组至少需要 2 个候选")
    if not bool(torch.isfinite(values).all()):
        raise ValueError("rewards 必须全部有限")

    mean = values.mean(dim=dim, keepdim=True)
    std = values.std(dim=dim, unbiased=False, keepdim=True)
    advantages = (values - mean) / std.clamp_min(delta_value)
    return advantages.detach()


def grpo_policy_loss(
    ratio: Tensor,
    advantage: Tensor,
    clip_epsilon: float,
) -> Tensor:
    """GRPO/PPO 裁剪代理损失（标量，取 batch 平均）。

    ``-mean(min(ratio * A, clip(ratio, 1-e, 1+e) * A))``。

    Args:
        ratio: 新策略与旧策略的概率比 ``exp(log p_new - log p_old)``。
        advantage: 同一候选的优势；应已 stopgrad。
        clip_epsilon: 裁剪半径，必须为非负有限数。

    Returns:
        标量损失张量。
    """
    if not np.isfinite(float(clip_epsilon)) or float(clip_epsilon) < 0.0:
        raise ValueError("clip_epsilon 必须是非负有限数")
    epsilon = float(clip_epsilon)
    clipped = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon)
    return -torch.minimum(ratio * advantage, clipped * advantage).mean()


def gaussian_kl_divergence(
    mean: Tensor,
    reference_mean: Tensor,
    std: Tensor,
) -> Tensor:
    """固定方差高斯分布之间的逐样本 KL，对动作维求和。

    ``KL(N(mu, sigma^2) || N(mu_ref, sigma^2))
      = 0.5 * sum((mu - mu_ref)^2 / sigma^2)``。

    Args:
        mean: 当前策略的高斯均值，形状 ``[B, C, H, W]``。
        reference_mean: 冻结参考策略的高斯均值，形状相同。
        std: 已知且相同的高斯标准差，形状可广播到 ``mean``。

    Returns:
        形状 ``[B]`` 的 float64 KL。
    """
    mean64 = mean.to(torch.float64)
    reference64 = reference_mean.to(torch.float64)
    std64 = std.to(torch.float64).clamp_min(1e-12)
    delta = (mean64 - reference64) / std64
    if delta.dim() <= 1:
        return 0.5 * delta.pow(2).sum()
    return 0.5 * delta.reshape(delta.shape[0], -1).pow(2).sum(dim=1)
