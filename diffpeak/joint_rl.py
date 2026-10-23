"""随机 Flow Matching 轨迹的 GRPO：冻结 SAM 的确定性评分 + 高斯轨迹策略更新。

本模块实现 ``training_protocol=signed_ridge_v1`` 新协议下的 D 阶段更新。与旧版
联合策略不同，**终端分割不再是一个随机 Bernoulli 策略**：

* **随机 Flow Matching 转移策略**：``ConditionalFlowMatching.sample(record=True)`` 记录的每一步
  ``z_t -> z_{t-1}`` 是高斯动作；重放时条件特征由 ``condition_encoder``
  **带梯度**重新计算，梯度只回传到 ``condition_encoder`` 与 UNet。参考 KL
  使用固定的 ``engine.ref_condition_encoder``（只编码一次，``no_grad``）与
  ``engine.ref_unet``；
* **终端评分**：最终的单通道有符号场直接作为密集提示交给冻结 SAM 的
  ``decode_heatmaps``，得到逐像素道路 logits，并用**确定性阈值**
  ``sigmoid(logits) >= threshold`` 生成二值掩码。该掩码只用于奖励，不参与
  策略重放；因此没有终端 Bernoulli 动作、终端对数概率、终端 KL，也不需要
  参考 SAM 解码器。

奖励使用已有的 numpy 指标（``compute_image_metrics``）在确定性二值掩码上
计算 IoU/F1/clDice/路径项；密集山脊场没有稀疏点，因此不产生任何提示点
计数或点成本日志。
路径项的随机采样通过 ``RewardConfig.seed`` 固定到**每张图像**的稳定种子
（``stable_seed(seed, "D-reward", image_id)``），同一图像的所有候选在不同
epoch 使用完全相同的采样规则，避免额外的非策略随机性并保证组内可比。

目标函数（每个 ``update_epochs`` 轮）
------------------------------------
令 ``D = G * T * B`` 为Flow Matching sample-step 项数::

    L = - 1/D * sum_{g,t,b} [ A_gb * min(rho, clip(rho))
                              - beta * KL(N(mu) || N(mu_ref)) ]
        + alpha * L_anchor

其中 ``rho = exp(log p_new - log p_old)``，``beta=kl_weight``，
``alpha=anchor_weight``。``L_anchor`` 是引擎统一监督锚定
``_supervised_loss("ABC", ...)["loss"]``。每个时间步损失在计算后立即
``backward``，每个更新轮只做一次 ``optimizer.step()``。

参数边界
--------
本函数在开始前把 SAM 整体设为 ``eval`` 并逐参数 ``requires_grad_(False)``，
同时冻结参考模块；更新过程中只重算条件编码器与 UNet，不调用任何 SAM 解码
路径。ABC 监督锚定依赖的 SAM 前向仍可传递输入梯度（冻结参数不等于禁止
输入梯度），但不会更新 SAM 参数。
"""

from __future__ import annotations

import dataclasses
import math
from contextlib import nullcontext
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import Tensor

from diffpeak.diagnostics import FIELD_METRIC_KEYS, field_quality
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

__all__ = ["joint_grpo_group"]

# 奖励路径采样的稳定种子域：与 Flow Matching 采样种子（"D"）分开，避免两者共享随机流。
_REWARD_SEED_TAG = "D-reward"
_DEFAULT_THRESHOLD = 0.5


# ---------------------------------------------------------------------------
# 有效区域与确定性阈值
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
    静默计入奖励。
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


def _segmentation_threshold(engine: Any) -> float:
    """返回确定性评分阈值，默认 ``0.5``。

    优先使用引擎已解析的 ``threshold``（``inference.threshold``）；轻量测试
    替身没有该属性时退回默认值。阈值必须位于 ``[0, 1]``。
    """
    value = getattr(engine, "threshold", None)
    if value is None:
        config = getattr(engine, "config", None)
        inference = config.get("inference", {}) if isinstance(config, Mapping) else {}
        value = inference.get("threshold", _DEFAULT_THRESHOLD)
    threshold = float(value)
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError(f"分割阈值必须是 [0, 1] 内的有限数，实际为 {threshold}")
    return threshold


def _threshold_predictions(
    logits: Tensor,
    threshold: float,
    regions: Sequence[Tuple[int, int]],
) -> List[np.ndarray]:
    """把 SAM logits 确定性二值化为逐图预测：``sigmoid(logits) >= threshold``。

    返回 ``H x W`` 的 bool numpy 数组（已裁掉右下 padding）。这里必须使用
    ``>=``：恰好在阈值上的像素视为道路，与方案约定一致。函数内不做任何
    采样，因此奖励不依赖终端随机动作。
    """
    if not torch.is_tensor(logits) or logits.ndim != 4 or logits.shape[1] != 1:
        raise ValueError(
            "decode_heatmaps 必须返回 B x 1 x H x W 的 logits，"
            f"实际为 {type(logits)!r} / "
            f"{tuple(logits.shape) if torch.is_tensor(logits) else 'N/A'}"
        )
    probabilities = torch.sigmoid(logits.detach().float())
    binary = probabilities >= float(threshold)
    predictions: List[np.ndarray] = []
    for index, (height, width) in enumerate(regions):
        predictions.append(
            binary[index, 0, :height, :width].detach().cpu().numpy()
        )
    return predictions


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
) -> Tensor:
    """调用冻结 SAM 的 ``RoadSAM.decode_heatmaps``。

    新协议只使用单通道有符号山脊场：提示张量原样传入解码器
    （``B x 1 x h x w``，范围 ``[-1, 1]``）。本模块不构造正负通道差，也不
    使用任何稀疏点或参考解码器。
    """
    method = getattr(getattr(engine, "sam", None), "decode_heatmaps", None)
    if not callable(method):
        raise AttributeError(
            "当前 SAM 模型缺少 decode_heatmaps；新协议需要 "
            "RoadSAM.decode_heatmaps(embeddings, heatmaps, output_size, "
            "valid_sizes=None)"
        )
    if (
        not torch.is_tensor(heatmaps)
        or heatmaps.ndim != 4
        or int(heatmaps.shape[1]) != 1
    ):
        raise ValueError(
            "heatmaps 必须是 B x 1 x H x W 的单通道有符号场，"
            f"实际为 {type(heatmaps)!r} / "
            f"{tuple(heatmaps.shape) if torch.is_tensor(heatmaps) else 'N/A'}"
        )
    logits = method(
        embeddings,
        heatmaps,
        output_size=(int(output_size[0]), int(output_size[1])),
        valid_sizes=valid_sizes,
    )
    if not torch.is_tensor(logits) or logits.ndim != 4 or logits.shape[1] != 1:
        raise ValueError(
            "decode_heatmaps 必须返回 B x 1 x H x W 的 logits，"
            f"实际为 {type(logits)!r} / "
            f"{tuple(logits.shape) if torch.is_tensor(logits) else 'N/A'}"
        )
    return logits


def _eval_policy_modules(engine: Any) -> None:
    """rollout/replay 一律 eval；参考模块同时永久冻结。"""
    for name in ("sam", "condition_encoder", "unet"):
        module = getattr(engine, name, None)
        if isinstance(module, torch.nn.Module):
            module.eval()
    for name in ("ref_unet", "ref_condition_encoder"):
        reference = getattr(engine, name, None)
        if reference is None:
            continue
        if isinstance(reference, torch.nn.Module):
            reference.eval()
            for parameter in reference.parameters():
                parameter.requires_grad_(False)


def _freeze_sam(engine: Any) -> None:
    """防御性地把 SAM 整体设为 eval 且所有参数冻结。

    新协议只允许 ``condition_encoder`` 与 UNet 接收梯度；即使引擎阶段设置
    残留了 SAM 解码器的可训练标记，本函数也会在 rollout/更新前清除。
    """
    sam = getattr(engine, "sam", None)
    if isinstance(sam, torch.nn.Module):
        sam.eval()
        for parameter in sam.parameters():
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
    """参考 KL 只需要参考条件编码器与参考 UNet；不需要参考 SAM。"""
    if getattr(engine, "ref_unet", None) is None:
        raise RuntimeError("joint GRPO 缺少 engine.ref_unet")
    if getattr(engine, "ref_condition_encoder", None) is None:
        raise RuntimeError("joint GRPO 缺少 engine.ref_condition_encoder")


def _image_reward_config(
    reward_config: Any,
    engine: Any,
    image_id: Any,
) -> Any:
    """为一张图像构造固定种子的奖励配置。

    ``compute_image_metrics`` 内部的路径采样使用 ``RewardConfig.seed``。这里
    把种子固定为 ``stable_seed(seed, "D-reward", image_id)``：同一图像的所有
    候选在任何 epoch 使用完全相同的路径采样规则，既不随全局 numpy 随机流
    漂移，也保证组内候选在同一采样规则下比较。配置不是 ``RewardConfig``
    数据类（例如测试用的轻量映射）时原样返回，保持
    ``compute_image_metrics`` 的鸭子类型契约。
    """
    if reward_config is None or not dataclasses.is_dataclass(reward_config):
        return reward_config
    if not hasattr(reward_config, "seed"):
        return reward_config
    seed = stable_seed(
        int(getattr(engine, "seed", 0)),
        _REWARD_SEED_TAG,
        image_id,
    ) % (1 << 32)
    try:
        return dataclasses.replace(reward_config, seed=int(seed))
    except (TypeError, ValueError):
        return reward_config


# ---------------------------------------------------------------------------
# 组内 rollout
# ---------------------------------------------------------------------------
def _rollout_group(
    engine: Any,
    batch: Mapping[str, Any],
    epoch: int,
    use_reference_kl: bool,
) -> Tuple[
    List[List[Dict[str, Any]]],
    List[Dict[str, Any]],
    Any,
    List[Tensor],
    List[List[float]],
]:
    """执行 G 条完整 Flow Matching rollout，并用确定性 SAM 掩码计算奖励。

    Returns:
        ``(trajectories, metric_rows, embeddings, reference_conditions,
        rewards)``；``rewards`` 形状为 ``[G][B]``，``metric_rows`` 按轨迹优先
        排列（``g * B + b``）。
    """
    images = batch["image"]
    target_masks = batch["mask"]
    valid_sizes = batch["valid_sizes"]
    ids = list(batch["ids"])
    batch_size = int(images.shape[0])
    output_size = (int(engine.image_size), int(engine.image_size))
    regions = _valid_regions(valid_sizes, batch_size, *output_size)
    targets_np = _target_arrays(target_masks, regions)
    threshold = _segmentation_threshold(engine)

    reward_config_method = getattr(engine, "_effective_reward_config", None)
    reward_config = (
        reward_config_method()
        if callable(reward_config_method)
        else getattr(engine, "reward_config", None)
    )

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
                engine.unet,
                rollout_conditions,
                generator=generator,
                record=True,
            )
            # 终端评分：单通道有符号场 -> 冻结 SAM logits -> 确定性阈值。
            terminal_logits = _decode_heatmaps(
                engine, embeddings, heatmaps, output_size, valid_sizes
            )
        predictions = _threshold_predictions(terminal_logits, threshold, regions)

        sample_rewards: List[float] = []
        for image_index, (height, width) in enumerate(regions):
            prediction = np.asarray(predictions[image_index], dtype=bool)
            if prediction.shape != (height, width):
                raise ValueError(
                    "确定性掩码形状与有效区域不一致："
                    f"{prediction.shape} vs {(height, width)}"
                )
            candidate_config = _image_reward_config(
                reward_config, engine, ids[image_index]
            )
            # 密集山脊场没有离散点；point_count 参数仅为兼容指标签名。
            metrics = compute_image_metrics(
                prediction, targets_np[image_index], 0, candidate_config
            )
            foreground_rate = float(prediction.mean())
            sample_rewards.append(float(metrics["reward"]))
            metric_rows.append(
                {
                    **metrics,
                    **field_quality(heatmaps[image_index], targets_np[image_index], image_size=engine.image_size),
                    "quality_reward": float(metrics["reward"]),
                    "prediction_foreground_rate": foreground_rate,
                }
            )

        rewards.append(sample_rewards)
        trajectories.append(trace)
        del heatmaps, trace, terminal_logits, predictions

    return trajectories, metric_rows, embeddings, reference_conditions, rewards


# ---------------------------------------------------------------------------
# 更新
# ---------------------------------------------------------------------------
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
        "diffusion_kl": float("nan"),
    }


def _joint_update(
    engine: Any,
    batch: Mapping[str, Any],
    embeddings: Any,
    reference_conditions: Sequence[Tensor],
    trajectories: Sequence[Sequence[Mapping[str, Any]]],
    advantages: Tensor,
    active_groups: Tensor,
    epoch: int,
) -> Dict[str, float]:
    """固定整组轨迹后做 Flow Matching GRPO/KL/anchor 更新；每轮 step 一次。

    只重放 Flow Matching 时间步，不调用 SAM 解码；因此梯度只可能流向
    ``condition_encoder`` 与 UNet。
    """
    optimizer = getattr(engine, "optimizer", None)
    scheduler = getattr(engine, "scheduler", None)
    scaler = getattr(engine, "scaler", None)
    if optimizer is None or scheduler is None or scaler is None:
        raise RuntimeError("joint GRPO 更新要求 optimizer/scheduler/scaler 已构建")

    if not trajectories:
        return _empty_update_stats(engine)

    batch_size = int(advantages.shape[1])
    group_size = int(advantages.shape[0])
    if group_size != len(trajectories):
        raise ValueError(
            "优势组数与轨迹数不一致：" f"{group_size} vs {len(trajectories)}"
        )
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

    # Flow Matching 项分母 G*T*B；T 取实际记录步数，兼容不同 sampling_steps 的轨迹。
    diffusion_terms = sum(len(trajectory) for trajectory in trajectories) * batch_size
    if diffusion_terms <= 0:
        raise ValueError("Flow Matching 轨迹不能为空")
    total_terms = max(1, diffusion_terms * update_epochs)

    images = batch["image"]

    diffusion_grpo_sum = 0.0
    diffusion_kl_sum = 0.0
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

        # 可选监督锚定：ABC 单阶段总损失（flow + 场 + 辅助 + 分割）。
        # SAM 已整体冻结，锚定只能更新条件编码器/UNet。
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

                    # 每个时间步都带梯度重算条件；这是训练 condition_encoder
                    # 的关键，不能复用 rollout 的 detached 条件。
                    with _autocast(engine):
                        conditions = _encode_conditions(
                            engine.condition_encoder, images, embeddings
                        )
                        mean, std = engine.diffusion.transition(
                            engine.unet,
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
                            "old/new Flow Matching log-prob 形状不一致："
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
                                getattr(engine, "ref_unet"),
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

        if engine._optimizer_step():
            update_steps += 1
        else:
            skipped_updates += 1

    grpo_value = (
        diffusion_grpo_sum / float(total_terms)
        if policy_terms > 0
        else float("nan")
    )
    kl_value = (
        diffusion_kl_sum / float(total_terms)
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
        "diffusion_kl": (
            diffusion_kl_sum / float(diffusion_terms * update_epochs)
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
    """执行一个随机 Flow Matching GRPO 组并返回训练记录（冻结 SAM 的确定性评分）。

    调用方（引擎的 D 阶段循环）负责构建 optimizer/scheduler/scaler，并把
    ``training_protocol=signed_ridge_v1`` 的 trainability 设置为 condition_encoder
    与 UNet。本函数内部：

    1. 防御性地冻结并 ``eval`` SAM，``eval`` 条件编码器/UNet，冻结参考模块；
    2. 在 ``no_grad`` 下采样 G 条随机 Flow Matching 轨迹，用冻结 SAM 与确定性阈值评分；
    3. 用每图组内优势计算 GRPO；多个 ``update_epochs`` 轮只重放 Flow Matching 步骤，
       每个时间步带梯度重算条件，每个损失立即 backward，每轮一次 step。

    Returns:
        与引擎 D 阶段记录兼容的字典，包含 ``reward``/``iou``/``f1``/
        ``cldice``/``path``、组诊断、``grpo``/``kl``/``anchor``、
        ``logratio_*``、``clip_fraction``、``updates``/``skipped_updates``
        以及 ``diffusion_grpo``/``diffusion_kl``。不再包含
        ``terminal_grpo``/``terminal_kl``。
    """
    if int(getattr(engine, "group_size", 0)) < 2:
        raise ValueError("D 阶段要求 rl.group_size >= 2 才能做组内相对优势")
    if not math.isfinite(float(getattr(engine, "advantage_std_delta", 0.0))) or float(
        getattr(engine, "advantage_std_delta", 0.0)
    ) <= 0.0:
        raise ValueError("rl.advantage_std_delta 必须是严格正的有限数")

    prepared = _prepare_batch(engine, batch)
    _eval_policy_modules(engine)
    _freeze_sam(engine)

    use_reference_kl = float(getattr(engine, "kl_weight", 0.0)) > 0.0
    if use_reference_kl:
        _require_references(engine)

    # 处理组计数按 D 组推进；即使本组常量且无 KL/anchor 而跳过 step，
    # 预热/诊断仍应看到该组已处理。
    if hasattr(engine, "grpo_groups_processed"):
        engine.grpo_groups_processed = int(engine.grpo_groups_processed) + 1

    (
        trajectories,
        metric_rows,
        embeddings,
        reference_conditions,
        rewards,
    ) = _rollout_group(  # type: ignore[misc]
        engine, prepared, int(epoch), use_reference_kl
    )

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
            # 密集山脊场没有离散点：只保留质量奖励诊断。
            "quality_reward": _mean(
                [row["quality_reward"] for row in metric_rows]
            ),
            "prediction_foreground_rate": _mean(
                [
                    row["prediction_foreground_rate"]
                    for row in metric_rows
                    if "prediction_foreground_rate" in row
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

    for key in FIELD_METRIC_KEYS:
        merged[key] = _mean([row[key] for row in metric_rows if key in row])
    return merged
