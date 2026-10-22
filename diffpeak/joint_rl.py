"""联合 ABC 端到端 GRPO：扩散轨迹与终端分割动作共享组内优势。

本模块只实现 ``training_protocol=joint_abc_v1`` 的 D 阶段更新，不修改引擎、
模型或数据代码。每个 GRPO 组由**同一影像的 G 条轨迹**组成，两条策略共享
同一组相对优势 ``A_gb``：

* **扩散转移策略**：``GaussianDiffusion.sample(record=True)`` 记录的每一步
  ``z_t -> z_{t-1}`` 是高斯动作；重放时条件特征由 ``condition_encoder``
  **带梯度**重新计算，梯度经冻结的 SAM 图像编码器之外的 condition_encoder
  与 DiT 回传。参考 KL 使用固定的 ``engine.ref_condition_encoder``
  （只编码一次，``no_grad``）与 ``engine.ref_dit``；
* **终端 Bernoulli 策略**：把最终热图作为 SAM 密集提示，经
  ``engine.sam.decode_heatmaps`` 得到逐像素道路 logits，并采样二值分割动作
  ``a ~ Bernoulli(sigmoid(logits))``。该动作只用于组奖励与策略重放；
  重放时终端热图被 ``detach``，因此梯度只训练 SAM mask decoder，不会把
  终端 Bernoulli 目标反传到扩散/条件策略。参考 KL 使用固定的完整参考
  mask decoder ``engine.ref_sam_decoder``（无需复制整个 SAM，提示编码器与
  图像编码器本来就冻结且共享）。

奖励使用已有的 numpy 指标（``compute_image_metrics``）在**采样得到的二值
掩码**上计算 IoU/F1/clDice/路径项，提示点计数固定为 0：密集热图没有稀疏
点，因此不施加任何点成本惩罚，只保留峰值/热图诊断语义。

目标函数（每个 ``update_epochs`` 轮）
------------------------------------
令 ``D = G * T * B`` 为扩散 sample-step 项数，终端项数为 ``G * B``::

    L = - 1/D * sum_{g,t,b} [ A_gb * min(rho^d, clip(rho^d))
                              - beta * KL(N(mu) || N(mu_ref)) ]
        - 1/(G*B) * sum_{g,b} [ A_gb * min(rho^s, clip(rho^s))
                                - beta * KL(Bern(p) || Bern(p_ref)) ]
        + alpha * L_anchor

其中 ``rho`` 是新旧策略概率比，``beta=kl_weight``，``alpha=anchor_weight``。
``L_anchor`` 是引擎统一监督锚定 ``_supervised_loss("ABC", ...)["loss"]``。
每个时间步/终端损失在计算后立即 ``backward``，每个更新轮只做一次
``optimizer.step()``。整组奖励完全常量且 ``kl_weight == 0``、
``anchor_weight == 0`` 时不构造计算图、不 backward、不 step。
"""

from __future__ import annotations

import math
from contextlib import nullcontext
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from diffpeak.grpo import (
    gaussian_kl_divergence,
    group_relative_advantages,
    grpo_policy_loss,
)
from diffpeak.inference import (
    compute_image_metrics,
    encode_images,
    make_generator,
    stable_seed,
)

__all__ = [
    "joint_grpo_group",
    "bernoulli_log_prob",
    "bernoulli_kl_divergence",
]

# 逐像素 Bernoulli 对数概率/KL 统一升到 float64 后求和，避免 1024² 动作维
# 在 float32 中累积出可观误差。
_FLOAT64 = torch.float64


# ---------------------------------------------------------------------------
# 有效区域与 Bernoulli 数学
# ---------------------------------------------------------------------------
def _valid_regions(
    valid_sizes: Optional[Union[Sequence[Optional[Tuple[int, int]]], Tensor]],
    batch_size: int,
    height: int,
    width: int,
) -> List[Tuple[int, int]]:
    """把 ``valid_sizes`` 规范为长度 B 的 ``(h, w)`` 列表。

    与 SAM 解码器一致：``None`` 元素表示整幅 ``(height, width)``。只接受
    输入画布 ``(height, width)`` 内的区域，越界直接报错，避免把 padding
    静默计入策略概率或 KL。
    """
    if valid_sizes is None:
        return [(int(height), int(width)) for _ in range(batch_size)]

    raw = (
        valid_sizes.detach().cpu().tolist()
        if torch.is_tensor(valid_sizes)
        else valid_sizes
    )
    if isinstance(raw, (tuple, list)) and len(raw) == 2 and all(
        isinstance(item, (int, float, np.integer, np.floating))
        and not isinstance(item, bool)
        for item in raw
    ):
        if batch_size != 1:
            raise ValueError("单个 (height, width) 只适用于 batch_size == 1")
        items: Sequence[Any] = [raw]
    else:
        try:
            items = list(raw)  # type: ignore[arg-type]
        except TypeError as exc:
            raise TypeError("valid_sizes 必须是序列或单个 (height, width)") from exc
        if len(items) != batch_size:
            raise ValueError(
                f"valid_sizes 长度 {len(items)} 与 batch_size {batch_size} 不一致"
            )

    regions: List[Tuple[int, int]] = []
    for item in items:
        if item is None:
            regions.append((int(height), int(width)))
            continue
        if torch.is_tensor(item):
            item = item.detach().cpu().tolist()
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            raise ValueError("valid_sizes 的每个元素必须是 (height, width) 或 None")
        valid_height, valid_width = int(item[0]), int(item[1])
        if not (
            0 < valid_height <= int(height)
            and 0 < valid_width <= int(width)
        ):
            raise ValueError(
                "valid_size 必须在 "
                f"(0, {int(height)}] x (0, {int(width)}] 内，"
                f"实际为 {(valid_height, valid_width)}"
            )
        regions.append((valid_height, valid_width))
    return regions


def _as_batched_pair(
    logits: Tensor,
    actions: Tensor,
    name: str,
) -> Tuple[Tensor, Tensor]:
    """把 ``[B, 1, H, W]`` 或 ``[B, H, W]`` 的 logits/actions 统一为四维。"""
    if not torch.is_tensor(logits):
        raise TypeError(f"logits 必须是张量，实际为 {type(logits)!r}")
    if not torch.is_tensor(actions):
        raise TypeError(f"{name} 必须是张量，实际为 {type(actions)!r}")
    if logits.ndim == 3:
        logits = logits.unsqueeze(1)
    if actions.ndim == 3:
        actions = actions.unsqueeze(1)
    if logits.ndim != 4 or logits.shape[1] != 1:
        raise ValueError(
            f"logits 应为 B x 1 x H x W 或 B x H x W，实际为 {tuple(logits.shape)}"
        )
    if actions.shape != logits.shape:
        raise ValueError(
            f"{name} 形状必须与 logits 一致："
            f"{tuple(actions.shape)} vs {tuple(logits.shape)}"
        )
    return logits, actions


def bernoulli_log_prob(
    logits: Tensor,
    actions: Tensor,
    valid_sizes: Optional[Union[Sequence[Optional[Tuple[int, int]]], Tensor]] = None,
) -> Tensor:
    """逐像素 Bernoulli 对数概率，对有效像素**求和**（不取平均）。

    ``log p(a|x) = a * log sigmoid(x) + (1-a) * log(1-sigmoid(x))``。
    使用 ``-softplus(-x)`` / ``-softplus(x)`` 的稳定形式，并在 float64 中
    按有效区域求和，保证高维终端动作的概率比不会被 padding 或精度污染。

    Args:
        logits: ``B x 1 x H x W``（或 ``B x H x W``）的 Bernoulli logits。
        actions: 与 logits 同形状的 0/1 或 bool 动作。
        valid_sizes: 可选每样本有效区域 ``(h, w)``；``None`` 元素表示整幅。

    Returns:
        ``[B]`` float64 张量，每个元素是该样本有效像素的联合对数概率。
    """
    logits, actions = _as_batched_pair(logits, actions, "actions")
    batch_size, _, height, width = (
        int(logits.shape[0]),
        int(logits.shape[1]),
        int(logits.shape[2]),
        int(logits.shape[3]),
    )
    regions = _valid_regions(valid_sizes, batch_size, height, width)
    totals: List[Tensor] = []
    for index, (valid_height, valid_width) in enumerate(regions):
        values = logits[index, 0, :valid_height, :valid_width].to(_FLOAT64)
        sampled = actions[index, 0, :valid_height, :valid_width].to(_FLOAT64)
        log_probability = sampled * (-F.softplus(-values)) + (
            1.0 - sampled
        ) * (-F.softplus(values))
        totals.append(log_probability.sum())
    return torch.stack(totals) if totals else logits.new_zeros((0,), dtype=_FLOAT64)


def bernoulli_kl_divergence(
    logits: Tensor,
    reference_logits: Tensor,
    valid_sizes: Optional[Union[Sequence[Optional[Tuple[int, int]]], Tensor]] = None,
) -> Tensor:
    """逐像素 Bernoulli KL ``KL(Bern(p) || Bern(q))``，对有效像素求和。

    解析式::

        p * (log p - log q) + (1-p) * (log(1-p) - log(1-q))

    其中 ``p=sigmoid(logits)``、``q=sigmoid(reference_logits)``。参考 logits
    在函数内 ``detach``，梯度只回到当前策略 logits。结果 clamp 到非负以
    消除浮点舍入造成的极小负值。
    """
    logits, reference_logits = _as_batched_pair(
        logits, reference_logits, "reference_logits"
    )
    batch_size, _, height, width = (
        int(logits.shape[0]),
        int(logits.shape[1]),
        int(logits.shape[2]),
        int(logits.shape[3]),
    )
    regions = _valid_regions(valid_sizes, batch_size, height, width)
    totals: List[Tensor] = []
    for index, (valid_height, valid_width) in enumerate(regions):
        values = logits[index, 0, :valid_height, :valid_width].to(_FLOAT64)
        reference = (
            reference_logits[index, 0, :valid_height, :valid_width]
            .to(_FLOAT64)
            .detach()
        )
        probability = torch.sigmoid(values)
        log_probability = -F.softplus(-values)
        log_one_minus_probability = -F.softplus(values)
        log_reference = -F.softplus(-reference)
        log_one_minus_reference = -F.softplus(reference)
        elementwise = probability * (log_probability - log_reference) + (
            1.0 - probability
        ) * (log_one_minus_probability - log_one_minus_reference)
        totals.append(elementwise.clamp_min(0.0).sum())
    return torch.stack(totals) if totals else logits.new_zeros((0,), dtype=_FLOAT64)


def _sample_terminal_actions(
    logits: Tensor,
    generator: Optional[torch.Generator],
) -> Tensor:
    """从 ``Bernoulli(sigmoid(logits))`` 采样二值动作。

    随机数使用调用方提供的生成器。``torch.Generator`` 可能位于 CPU 而
    logits 位于 CUDA；此时先在生成器设备上采样再搬到 logits 设备，避免
    "Expected a 'cuda' device type for generator" 之类的设备不匹配错误。
    """
    probabilities = torch.sigmoid(logits.float())
    if generator is None:
        uniform = torch.rand(
            probabilities.shape,
            device=probabilities.device,
            dtype=torch.float32,
        )
    else:
        uniform = torch.rand(
            probabilities.shape,
            device=generator.device,
            dtype=torch.float32,
            generator=generator,
        )
        if uniform.device != probabilities.device:
            uniform = uniform.to(probabilities.device)
    return uniform < probabilities


# ---------------------------------------------------------------------------
# 引擎与模型适配
# ---------------------------------------------------------------------------
def _autocast(engine: Any):
    """复用引擎的 AMP 上下文；轻量测试替身没有该方法时退回 nullcontext。"""
    method = getattr(engine, "_autocast", None)
    return method() if callable(method) else nullcontext()


def _current_lr(engine: Any) -> float:
    method = getattr(engine, "_current_lr", None)
    if callable(method):
        return float(method())
    optimizer = getattr(engine, "optimizer", None)
    if optimizer is not None and optimizer.param_groups:
        return float(optimizer.param_groups[0]["lr"])
    return float("nan")


def _prepare_batch(engine: Any, batch: Mapping[str, Any]) -> Dict[str, Any]:
    """引擎通常已传入 ``_prepare_batch`` 的结果；独立调用时按需补齐。"""
    if "valid_sizes" in batch and "ids" in batch:
        return dict(batch)
    method = getattr(engine, "_prepare_batch", None)
    if callable(method):
        return method(batch)
    return dict(batch)


def _condition_list(output: Any) -> List[Tensor]:
    """从 ``ConditionEncoder`` 的 ``(conditions, auxiliary)`` 或裸序列取条件。"""
    if isinstance(output, (tuple, list)):
        if len(output) == 2 and isinstance(output[0], (tuple, list)):
            return list(output[0])
        if output and all(torch.is_tensor(item) for item in output):
            return list(output)
        if output and isinstance(output[0], (tuple, list)):
            return list(output[0])
    if isinstance(output, (tuple, list)):
        return list(output)  # type: ignore[arg-type]
    raise TypeError(
        "条件编码器应返回 (conditions, auxiliary) 或条件张量序列，"
        f"实际为 {type(output)!r}"
    )


def _encode_conditions(
    module: torch.nn.Module,
    images: Tensor,
    embeddings: Any,
) -> List[Tensor]:
    return _condition_list(module(images, embeddings))


def _decode_heatmaps(
    engine: Any,
    embeddings: Any,
    heatmaps: Tensor,
    output_size: Tuple[int, int],
    valid_sizes: Any,
    decoder: Optional[torch.nn.Module] = None,
) -> Tensor:
    """调用 ``RoadSAM.decode_heatmaps``，可选覆盖 mask decoder。

    参考终端 KL 不需要复制整个 SAM：图像编码器与提示编码器始终冻结且与
    当前策略共享，只有 mask decoder 需要冻结副本 ``engine.ref_sam_decoder``。
    """
    method = getattr(getattr(engine, "sam", None), "decode_heatmaps", None)
    if not callable(method):
        raise AttributeError(
            "当前 SAM 模型缺少可微 decode_heatmaps；联合 ABC 协议需要 "
            "RoadSAM.decode_heatmaps(embeddings, heatmaps, output_size, "
            "valid_sizes=None, decoder=None)"
        )
    kwargs: Dict[str, Any] = {
        "output_size": (int(output_size[0]), int(output_size[1])),
        "valid_sizes": valid_sizes,
    }
    if decoder is not None:
        kwargs["decoder"] = decoder
    logits = method(embeddings, heatmaps, **kwargs)
    if not torch.is_tensor(logits) or logits.ndim != 4 or logits.shape[1] != 1:
        raise ValueError(
            "decode_heatmaps 必须返回 B x 1 x H x W 的 logits，"
            f"实际为 {type(logits)!r} / "
            f"{tuple(logits.shape) if torch.is_tensor(logits) else 'N/A'}"
        )
    return logits


def _eval_policy_modules(engine: Any) -> None:
    """rollout/replay 一律 eval；参考模块同时永久冻结。"""
    for name in ("sam", "condition_encoder", "dit"):
        module = getattr(engine, name, None)
        if isinstance(module, torch.nn.Module):
            module.eval()
    for name in ("ref_dit", "ref_condition_encoder", "ref_sam_decoder"):
        reference = getattr(engine, name, None)
        if reference is None:
            continue
        if isinstance(reference, torch.nn.Module):
            reference.eval()
            for parameter in reference.parameters():
                parameter.requires_grad_(False)


def _target_arrays(
    target_masks: Tensor,
    valid_sizes: Sequence[Tuple[int, int]],
) -> List[np.ndarray]:
    if target_masks.ndim == 3:
        target_masks = target_masks.unsqueeze(1)
    targets: List[np.ndarray] = []
    for index, (height, width) in enumerate(valid_sizes):
        targets.append(
            (target_masks[index, 0, :height, :width] > 0.5)
            .detach()
            .cpu()
            .numpy()
        )
    return targets


def _reference_conditions(
    engine: Any,
    images: Tensor,
    embeddings: Any,
) -> List[Tensor]:
    """参考条件只编码一次并固定；``no_grad`` 保证不会意外更新参考模块。"""
    encoder = getattr(engine, "ref_condition_encoder", None)
    if encoder is None:
        raise RuntimeError("joint GRPO 缺少 engine.ref_condition_encoder")
    with torch.no_grad(), _autocast(engine):
        conditions = _encode_conditions(encoder, images, embeddings)
    return [condition.detach() for condition in conditions]


def _require_references(engine: Any) -> None:
    if getattr(engine, "ref_dit", None) is None:
        raise RuntimeError("joint GRPO 缺少 engine.ref_dit")
    if getattr(engine, "ref_condition_encoder", None) is None:
        raise RuntimeError("joint GRPO 缺少 engine.ref_condition_encoder")
    if getattr(engine, "ref_sam_decoder", None) is None:
        raise RuntimeError(
            "joint GRPO 缺少 engine.ref_sam_decoder；终端 Bernoulli KL "
            "需要冻结的参考 mask decoder"
        )


# ---------------------------------------------------------------------------
# 组内 rollout / 更新
# ---------------------------------------------------------------------------
def _rollout_group(
    engine: Any,
    batch: Mapping[str, Any],
    epoch: int,
    reward_config: Any,
    use_reference_kl: bool,
) -> Tuple[
    List[List[Dict[str, Any]]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    Any,
    List[Tensor],
    List[np.ndarray],
    List[List[float]],
]:
    """执行 G 条完整 rollout 并计量终端二值动作的奖励。

    Returns:
        ``(trajectories, terminal_records, metric_rows, embeddings,
        reference_conditions, targets_np, rewards)``；``rewards`` 形状为
        ``[G][B]``，``metric_rows`` 按轨迹优先排列（``g * B + b``）。
    """
    images = batch["image"]
    target_masks = batch["mask"]
    valid_sizes = batch["valid_sizes"]
    ids = list(batch["ids"])
    batch_size = int(images.shape[0])
    output_size = (int(engine.image_size), int(engine.image_size))
    regions = _valid_regions(valid_sizes, batch_size, *output_size)
    targets_np = _target_arrays(target_masks, regions)
    reference_decoder = getattr(engine, "ref_sam_decoder", None)

    with torch.no_grad(), _autocast(engine):
        embeddings = encode_images(engine.sam, images, valid_sizes)
        # rollout 条件只用于采样，不保留计算图。
        rollout_conditions = [
            condition.detach()
            for condition in _encode_conditions(
                engine.condition_encoder, images, embeddings
            )
        ]
    reference_conditions = (
        _reference_conditions(engine, images, embeddings)
        if use_reference_kl
        else []
    )

    trajectories: List[List[Dict[str, Any]]] = []
    terminal_records: List[Dict[str, Any]] = []
    metric_rows: List[Dict[str, Any]] = []
    rewards: List[List[float]] = []

    for trajectory_index in range(int(engine.group_size)):
        generator = make_generator(
            stable_seed(
                int(engine.seed),
                "D",
                int(epoch),
                *ids,
                trajectory_index,
            )
        )
        with torch.no_grad(), _autocast(engine):
            heatmaps, trace = engine.diffusion.sample(
                engine.dit,
                rollout_conditions,
                generator=generator,
                record=True,
            )
            # 终端策略：密集热图 -> SAM logits -> Bernoulli 二值动作。
            terminal_logits = _decode_heatmaps(
                engine, embeddings, heatmaps, output_size, valid_sizes
            )
            actions = _sample_terminal_actions(terminal_logits, generator)
            old_terminal_log_prob = bernoulli_log_prob(
                terminal_logits, actions, valid_sizes
            )
            reference_logits: Optional[Tensor] = None
            if use_reference_kl:
                reference_logits = _decode_heatmaps(
                    engine,
                    embeddings,
                    heatmaps,
                    output_size,
                    valid_sizes,
                    decoder=reference_decoder,
                )

        sample_rewards: List[float] = []
        for image_index, (height, width) in enumerate(regions):
            prediction = (
                actions[image_index, 0, :height, :width]
                .detach()
                .cpu()
                .numpy()
                .astype(bool, copy=False)
            )
            # 密集提示没有离散点：point_count=0，点成本恒为 0。峰值/热图
            # 诊断可另由日志记录，但不进入奖励的稀疏点惩罚。
            metrics = compute_image_metrics(
                prediction, targets_np[image_index], 0, reward_config
            )
            sample_rewards.append(float(metrics["reward"]))
            metric_rows.append(
                {
                    **metrics,
                    "quality_reward": float(metrics["reward"]),
                    "point_penalty": 0.0,
                    "point_count": 0.0,
                    "positive_point_count": 0.0,
                    "negative_point_count": 0.0,
                    "terminal_foreground_rate": float(prediction.mean()),
                }
            )

        rewards.append(sample_rewards)
        trajectories.append(trace)
        terminal_records.append(
            {
                # 终端热图必须 detach：Bernoulli 重放只训练 SAM decoder，
                # 不能把终端策略梯度反传到条件编码器/DiT。
                "heatmaps": heatmaps.detach().to("cpu", dtype=torch.float32),
                "actions": actions.detach().to("cpu"),
                "old_log_prob": old_terminal_log_prob.detach().cpu().double(),
                "reference_logits": (
                    None
                    if reference_logits is None
                    else reference_logits.detach().to("cpu", dtype=torch.float32)
                ),
            }
        )
        del heatmaps, trace, terminal_logits, actions, old_terminal_log_prob, reference_logits

    return (
        trajectories,
        terminal_records,
        metric_rows,
        embeddings,
        reference_conditions,
        targets_np,
        rewards,
    )  # type: ignore[return-value]


def _empty_update_stats(engine: Any) -> Dict[str, float]:
    return {
        "grpo": float("nan"),
        "kl": float("nan"),
        "anchor": float("nan"),
        "clip_fraction": float("nan"),
        "logratio_abs_mean": float("nan"),
        "logratio_abs_max": float("nan"),
        "lr": _current_lr(engine),
        "updates": 0.0,
        "skipped_updates": float(max(1, int(getattr(engine, "update_epochs", 1)))),
        "policy_terms": 0.0,
        "diffusion_grpo": float("nan"),
        "terminal_grpo": float("nan"),
        "diffusion_kl": float("nan"),
        "terminal_kl": float("nan"),
    }


def _joint_update(
    engine: Any,
    batch: Mapping[str, Any],
    embeddings: Any,
    reference_conditions: Sequence[Tensor],
    trajectories: Sequence[Sequence[Mapping[str, Any]]],
    terminal_records: Sequence[Mapping[str, Any]],
    advantages: Tensor,
    active_groups: Tensor,
    epoch: int,
) -> Dict[str, float]:
    """固定整组轨迹后做联合 GRPO/KL/anchor 更新；每个更新轮 step 一次。"""
    optimizer = getattr(engine, "optimizer", None)
    scheduler = getattr(engine, "scheduler", None)
    scaler = getattr(engine, "scaler", None)
    if optimizer is None or scheduler is None or scaler is None:
        raise RuntimeError("joint GRPO 更新要求 optimizer/scheduler/scaler 已构建")

    if not trajectories:
        return _empty_update_stats(engine)

    batch_size = int(advantages.shape[1])
    group_size = int(advantages.shape[0])
    update_epochs = max(1, int(getattr(engine, "update_epochs", 1)))
    epsilon = float(getattr(engine, "ppo_clip", 0.0))
    kl_weight = float(getattr(engine, "kl_weight", 0.0))
    anchor_weight = float(getattr(engine, "anchor_weight", 0.0))
    has_kl = kl_weight > 0.0
    has_anchor = anchor_weight > 0.0

    active_mask = active_groups.detach().to(engine.device).reshape(-1).bool()
    if active_mask.numel() == 1 and batch_size > 1:
        active_mask = active_mask.expand(batch_size)
    if active_mask.numel() != batch_size:
        raise ValueError(
            "active_groups 数量与 batch size 不一致："
            f"{active_mask.numel()} vs {batch_size}"
        )
    has_active_policy = bool(active_mask.any())

    if not has_active_policy and not has_kl and not has_anchor:
        # 整组奖励完全相同且没有 KL/锚定：不构造计算图、不 backward、
        # 不调用 optimizer.step()，也不消耗 scheduler/global_step。
        stats = _empty_update_stats(engine)
        stats["skipped_updates"] = float(update_epochs)
        return stats

    # 扩散项分母 G*T*B；终端项分母 G*B。T 取实际记录步数，兼容不同
    # sampling_steps 的轨迹。
    diffusion_terms = sum(len(trajectory) for trajectory in trajectories) * batch_size
    terminal_terms = len(trajectories) * batch_size
    per_epoch_terms = diffusion_terms + terminal_terms
    total_terms = max(1, per_epoch_terms * update_epochs)
    if diffusion_terms <= 0:
        raise ValueError("扩散轨迹不能为空")

    images = batch["image"]
    valid_sizes = batch["valid_sizes"]
    output_size = (int(engine.image_size), int(engine.image_size))
    reference_decoder = getattr(engine, "ref_sam_decoder", None)

    diffusion_grpo_sum = 0.0
    terminal_grpo_sum = 0.0
    diffusion_kl_sum = 0.0
    terminal_kl_sum = 0.0
    anchor_values: List[float] = []
    clip_values: List[float] = []
    logratio_abs_sum = 0.0
    logratio_abs_max = 0.0
    logratio_count = 0
    policy_terms = 0
    update_steps = 0
    skipped_updates = 0

    def _record_ratio(
        raw_log_ratio: Tensor,
        clipped_mask: Tensor,
        active: Tensor,
        active_count: int,
    ) -> None:
        nonlocal logratio_abs_sum, logratio_abs_max, logratio_count
        if active_count <= 0:
            return
        clip_values.append(float(clipped_mask.double()[active].mean().cpu()))
        selected = raw_log_ratio.detach().abs()[active]
        logratio_abs_sum += float(selected.sum().cpu())
        logratio_abs_max = max(logratio_abs_max, float(selected.max().cpu()))
        logratio_count += active_count

    for _ in range(update_epochs):
        optimizer.zero_grad(set_to_none=True)

        # 统一监督锚定：ABC 单阶段总损失（扩散 + 热图 + 辅助 + 分割）。
        if has_anchor:
            supervised = getattr(engine, "_supervised_loss", None)
            if not callable(supervised):
                raise AttributeError(
                    "anchor_weight > 0 需要 engine._supervised_loss('ABC', ...)"
                )
            with _autocast(engine):
                anchor_result = supervised("ABC", batch, epoch)
            anchor_total = (
                anchor_result.get("loss")
                if isinstance(anchor_result, Mapping)
                else anchor_result
            )
            if (
                torch.is_tensor(anchor_total)
                and anchor_total.numel() > 0
                and bool(anchor_total.requires_grad)
            ):
                anchor_loss = anchor_total.float() * anchor_weight
                scaler.scale(anchor_loss).backward()
                anchor_values.append(float(anchor_loss.detach().cpu()))

        if has_active_policy or has_kl:
            for trajectory_index, trajectory in enumerate(trajectories):
                advantage64 = (
                    advantages[trajectory_index]
                    .detach()
                    .to(engine.device)
                    .reshape(-1)
                    .double()
                )
                if advantage64.numel() != batch_size:
                    raise ValueError(
                        "优势数量与 batch size 不一致："
                        f"{advantage64.numel()} vs {batch_size}"
                    )
                active_float = active_mask.to(torch.float64)
                active_count = int(active_mask.sum().item())
                # 常量组优势为 0；显式置零避免 ratio*0 产生浮点噪声。
                advantage64 = advantage64 * active_float
                if active_count > 0:
                    policy_terms += len(trajectory) * active_count

                for step in trajectory:
                    state = step["state"].to(
                        engine.device, dtype=torch.float32, non_blocking=True
                    )
                    action = step["action"].to(
                        engine.device, dtype=torch.float32, non_blocking=True
                    )
                    timestep = step["timestep"]
                    previous_timestep = step["previous_timestep"]
                    old_log_prob = (
                        step["old_log_prob"].to(engine.device).reshape(-1).double()
                    )

                    # 每个时间步都带梯度重算条件；这是联合训练 condition_encoder
                    # 的关键，不能复用 rollout 的 detached 条件。
                    with _autocast(engine):
                        conditions = _encode_conditions(
                            engine.condition_encoder, images, embeddings
                        )
                        mean, std = engine.diffusion.transition(
                            engine.dit,
                            state,
                            timestep,
                            previous_timestep,
                            conditions,
                        )
                        new_log_prob = engine.diffusion.log_prob(action, mean, std)

                    new_log_prob = new_log_prob.reshape(-1).double()
                    if old_log_prob.numel() == 1 and new_log_prob.numel() > 1:
                        old_log_prob = old_log_prob.expand_as(new_log_prob)
                    if old_log_prob.numel() != new_log_prob.numel():
                        raise ValueError(
                            "old/new 扩散 log-prob 形状不一致："
                            f"{old_log_prob.numel()} vs {new_log_prob.numel()}"
                        )
                    raw_log_ratio = new_log_prob - old_log_prob
                    log_ratio = torch.clamp(raw_log_ratio, min=-20.0, max=20.0)
                    ratio = torch.exp(log_ratio)
                    clipped_mask = (ratio < 1.0 - epsilon) | (
                        ratio > 1.0 + epsilon
                    )
                    _record_ratio(
                        raw_log_ratio, clipped_mask, active_mask, active_count
                    )

                    # grpo_policy_loss 对传入的 B 个样本取平均；乘回 B 得到
                    # 分子求和，再除以 G*T*B。
                    grpo_loss = grpo_policy_loss(ratio, advantage64, epsilon)
                    diffusion_grpo_sum += (
                        float(grpo_loss.detach().cpu()) * batch_size
                    )

                    kl_element = torch.zeros(
                        (), dtype=torch.float64, device=engine.device
                    )
                    if has_kl:
                        with torch.no_grad(), _autocast(engine):
                            reference_mean, _ = engine.diffusion.transition(
                                getattr(engine, "ref_dit"),
                                state,
                                timestep,
                                previous_timestep,
                                reference_conditions,
                            )
                        kl_element = gaussian_kl_divergence(
                            mean, reference_mean, std
                        )
                        diffusion_kl_sum += float(
                            kl_element.detach().sum().cpu()
                        )

                    loss = (
                        grpo_loss * batch_size
                        + kl_weight * kl_element.sum()
                    ) / float(diffusion_terms)
                    scaler.scale(loss.float()).backward()

                    del (
                        state,
                        action,
                        conditions,
                        mean,
                        std,
                        new_log_prob,
                        log_ratio,
                        ratio,
                        grpo_loss,
                        kl_element,
                    )

                # 终端 Bernoulli 策略：heatmaps 已 detach，只有 mask decoder
                # 接收梯度；同一 A_gb 同时驱动扩散与终端策略。
                terminal = terminal_records[trajectory_index]
                heatmaps = terminal["heatmaps"].to(
                    engine.device, dtype=torch.float32, non_blocking=True
                )
                actions = terminal["actions"].to(engine.device)
                old_terminal_log_prob = (
                    terminal["old_log_prob"].to(engine.device).reshape(-1).double()
                )
                with _autocast(engine):
                    terminal_logits = _decode_heatmaps(
                        engine,
                        embeddings,
                        heatmaps.detach(),
                        output_size,
                        valid_sizes,
                    )
                new_terminal_log_prob = bernoulli_log_prob(
                    terminal_logits, actions, valid_sizes
                ).reshape(-1).double()
                if (
                    old_terminal_log_prob.numel() == 1
                    and new_terminal_log_prob.numel() > 1
                ):
                    old_terminal_log_prob = old_terminal_log_prob.expand_as(
                        new_terminal_log_prob
                    )
                if old_terminal_log_prob.numel() != new_terminal_log_prob.numel():
                    raise ValueError(
                        "old/new 终端 log-prob 形状不一致："
                        f"{old_terminal_log_prob.numel()} vs "
                        f"{new_terminal_log_prob.numel()}"
                    )
                raw_terminal_log_ratio = (
                    new_terminal_log_prob - old_terminal_log_prob
                )
                terminal_log_ratio = torch.clamp(
                    raw_terminal_log_ratio, min=-20.0, max=20.0
                )
                terminal_ratio = torch.exp(terminal_log_ratio)
                terminal_clipped = (
                    terminal_ratio < 1.0 - epsilon
                ) | (terminal_ratio > 1.0 + epsilon)
                _record_ratio(
                    raw_terminal_log_ratio,
                    terminal_clipped,
                    active_mask,
                    active_count,
                )
                if active_count > 0:
                    policy_terms += active_count

                terminal_grpo_loss = grpo_policy_loss(
                    terminal_ratio, advantage64, epsilon
                )
                terminal_grpo_sum += (
                    float(terminal_grpo_loss.detach().cpu()) * batch_size
                )

                terminal_kl_element = torch.zeros(
                    (), dtype=torch.float64, device=engine.device
                )
                if has_kl:
                    reference_logits = terminal["reference_logits"]
                    if reference_logits is None:
                        raise RuntimeError(
                            "终端 KL 缺少 rollout 阶段保存的参考 logits"
                        )
                    terminal_kl_element = bernoulli_kl_divergence(
                        terminal_logits,
                        reference_logits.to(
                            engine.device, dtype=torch.float32, non_blocking=True
                        ),
                        valid_sizes,
                    )
                    terminal_kl_sum += float(
                        terminal_kl_element.detach().sum().cpu()
                    )

                terminal_loss = (
                    terminal_grpo_loss * batch_size
                    + kl_weight * terminal_kl_element.sum()
                ) / float(terminal_terms)
                scaler.scale(terminal_loss.float()).backward()

                del (
                    heatmaps,
                    actions,
                    terminal_logits,
                    new_terminal_log_prob,
                    terminal_log_ratio,
                    terminal_ratio,
                    terminal_grpo_loss,
                    terminal_kl_element,
                )

        if engine._optimizer_step():
            update_steps += 1
        else:
            skipped_updates += 1

    grpo_value = (
        (diffusion_grpo_sum + terminal_grpo_sum) / float(total_terms)
        if policy_terms > 0
        else float("nan")
    )
    kl_value = (
        (diffusion_kl_sum + terminal_kl_sum) / float(total_terms)
        if has_kl
        else float("nan")
    )
    return {
        "grpo": grpo_value,
        "kl": kl_value,
        "anchor": _mean(anchor_values),
        "clip_fraction": _mean(clip_values),
        "logratio_abs_mean": (
            logratio_abs_sum / float(logratio_count)
            if logratio_count > 0
            else float("nan")
        ),
        "logratio_abs_max": (
            logratio_abs_max if logratio_count > 0 else float("nan")
        ),
        "lr": _current_lr(engine),
        "updates": float(update_steps),
        "skipped_updates": float(skipped_updates),
        "policy_terms": float(policy_terms),
        "diffusion_grpo": (
            diffusion_grpo_sum / float(diffusion_terms * update_epochs)
            if policy_terms > 0
            else float("nan")
        ),
        "terminal_grpo": (
            terminal_grpo_sum / float(terminal_terms * update_epochs)
            if policy_terms > 0
            else float("nan")
        ),
        "diffusion_kl": (
            diffusion_kl_sum / float(diffusion_terms * update_epochs)
            if has_kl
            else float("nan")
        ),
        "terminal_kl": (
            terminal_kl_sum / float(terminal_terms * update_epochs)
            if has_kl
            else float("nan")
        ),
    }


def _mean(values: Sequence[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.mean(finite)) if finite else float("nan")


# ---------------------------------------------------------------------------
# 公共入口
# ---------------------------------------------------------------------------
def joint_grpo_group(
    engine: Any,
    batch: Mapping[str, Any],
    epoch: int,
) -> Dict[str, Any]:
    """执行一个联合 ABC GRPO 组并返回训练记录。

    调用方（引擎的 D 阶段循环）负责构建 optimizer/scheduler/scaler，并按
    ``training_protocol=joint_abc_v1`` 将 trainability 设置为 condition_encoder、
    dit 与 SAM mask decoder 可训练。本函数内部：

    1. 对整个策略与参考模块执行 ``eval()``，并在 ``no_grad`` 下 rollout；
    2. 用采样二值掩码计算奖励与组内优势；
    3. 多个 ``update_epochs`` 轮重放：扩散每步带梯度重算条件，终端热图
       detach 后只训练 SAM decoder；每个损失立即 backward，每轮一次 step。

    Returns:
        与引擎 D 阶段记录兼容的字典，包含 ``reward``/``iou``/``f1``/
        ``cldice``/``path``、组诊断、``grpo``/``kl``/``anchor``、
        ``logratio_*``、``clip_fraction``、``updates``/``skipped_updates``
        以及联合诊断 ``diffusion_grpo``/``terminal_grpo`` 等。
    """
    if int(getattr(engine, "group_size", 0)) < 2:
        raise ValueError("D 阶段要求 rl.group_size >= 2 才能做组内相对优势")
    if not math.isfinite(float(getattr(engine, "advantage_std_delta", 0.0))) or float(
        getattr(engine, "advantage_std_delta", 0.0)
    ) <= 0.0:
        raise ValueError("rl.advantage_std_delta 必须是严格正的有限数")

    prepared = _prepare_batch(engine, batch)
    _eval_policy_modules(engine)

    use_reference_kl = float(getattr(engine, "kl_weight", 0.0)) > 0.0
    if use_reference_kl:
        _require_references(engine)

    # 处理组计数按 D 组推进；即使本组常量且无 KL/anchor 而跳过 step，
    # 预热/诊断仍应看到该组已处理。
    if hasattr(engine, "grpo_groups_processed"):
        engine.grpo_groups_processed = int(engine.grpo_groups_processed) + 1

    reward_config_method = getattr(engine, "_effective_reward_config", None)
    reward_config = (
        reward_config_method()
        if callable(reward_config_method)
        else getattr(engine, "reward_config")
    )

    (
        trajectories,
        terminal_records,
        metric_rows,
        embeddings,
        reference_conditions,
        targets_np,
        rewards,
    ) = _rollout_group(  # type: ignore[misc]
        engine,
        prepared,
        int(epoch),
        reward_config,
        use_reference_kl,
    )
    del targets_np

    reward_tensor = torch.tensor(rewards, dtype=torch.float64)  # [G, B]
    if reward_tensor.ndim != 2 or reward_tensor.shape[1] != int(
        prepared["image"].shape[0]
    ):
        raise RuntimeError(
            "奖励矩阵形状应为 [G, B]，实际为 "
            f"{tuple(reward_tensor.shape)}"
        )
    advantages = group_relative_advantages(
        reward_tensor,
        delta=float(getattr(engine, "advantage_std_delta")),
    )
    per_image_std = reward_tensor.std(dim=0, unbiased=False)
    per_image_variance = per_image_std.pow(2)
    # 只有奖励完全相同的组没有策略信号；低方差组仍按 delta 缩放参与。
    active_groups = engine._active_group_mask(reward_tensor)

    update_stats = _joint_update(
        engine,
        prepared,
        embeddings,
        reference_conditions,
        trajectories,
        terminal_records,
        advantages,
        active_groups,
        int(epoch),
    )

    batch_size = int(prepared["image"].shape[0])
    merged: Dict[str, Any] = dict(update_stats)
    merged.update(
        {
            "reward": _mean([row["reward"] for row in metric_rows]),
            "iou": _mean([row["iou"] for row in metric_rows]),
            "f1": _mean([row["f1"] for row in metric_rows]),
            "cldice": _mean([row["cldice"] for row in metric_rows]),
            "path": _mean([row["path"] for row in metric_rows]),
            "reward_std": float(per_image_std.mean().cpu()),
            "reward_variance": float(per_image_variance.mean().cpu()),
            "active_group_rate": float(active_groups.double().mean().cpu()),
            "inactive_group_rate": 1.0
            - float(active_groups.double().mean().cpu()),
            "invalid_groups": float(
                batch_size - int(active_groups.sum().item())
            ),
            "advantage_abs_mean": (
                float(advantages.abs().mean().cpu())
                if advantages.numel()
                else float("nan")
            ),
            # 密集提示不使用离散点：点数与点成本严格为 0，避免把稀疏点
            # 惩罚误加到热图策略上。
            "point_count": 0.0,
            "positive_point_count": 0.0,
            "negative_point_count": 0.0,
            "point_penalty": 0.0,
            "quality_reward": _mean(
                [row["quality_reward"] for row in metric_rows]
            ),
            "point_cost": 0.0,
            "point_cost_scale": (
                float(engine._point_cost_scale())
                if callable(getattr(engine, "_point_cost_scale", None))
                else 1.0
            ),
            "terminal_foreground_rate": _mean(
                [
                    row["terminal_foreground_rate"]
                    for row in metric_rows
                    if "terminal_foreground_rate" in row
                ]
            ),
            "trajectory_diagnostics": [
                {
                    **row,
                    "image_id": prepared["ids"][position % batch_size],
                    "trajectory_index": position // batch_size,
                    "advantage": float(
                        advantages[position // batch_size, position % batch_size]
                    ),
                }
                for position, row in enumerate(metric_rows)
            ],
        }
    )

    # 复用已有的组诊断函数：quality_reward=reward、point_penalty=0 时，
    # 其 reward 与优势相关性与点数为常量的判断仍然成立。
    try:
        from diffpeak.diagnostics import reward_group_diagnostics

        merged.update(reward_group_diagnostics(metric_rows, advantages))
    except (ImportError, ValueError, KeyError):
        # 诊断不是训练正确性的必要条件；数值异常不应静默吞掉训练错误，
        # 但可选的日志辅助失败不应中断整个 D 阶段。
        pass

    return merged
