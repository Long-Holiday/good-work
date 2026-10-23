"""DiffPeak-SAM 的监督损失。

本模块只包含与网络结构解耦的损失函数，便于在阶段 A（道路适配）、
阶段 B（扩散预训练）和阶段 C（生成提示适配）中复用。

两个公开接口
------------
``segmentation_loss``
    道路二值分割损失：BCE + Dice + soft-clDice。
``auxiliary_loss``
    辅助头损失：道路区域头与中心线（骨架）头分别做 BCE + Dice。

形状约定
--------
* logits / targets 采用 channel-first：``(B, 1, H, W)``；
* 为调用方便，也接受 ``(B, H, W)``（内部会补一个通道维）；
* targets 可以是 bool、整型或浮点张量，内部会转换为 logits 的 dtype/device。

数值约定
--------
* Dice 与 clDice 只在“目标中存在前景”的样本上取平均。空标注图像仍由 BCE
  监督（让预测趋近于全零），这样既不会让空图抬高细长结构的 Dice 损失，
  也保留了背景假阳性的惩罚。
* soft-clDice 采用 Shit et al. (2020) 的可微骨架近似，迭代次数固定为 3。
  该迭代次数属于损失定义的一部分，实验中保持不变，便于跨阶段比较。
"""

from __future__ import annotations

from typing import Mapping, Sequence, Tuple, Union

import torch
import torch.nn.functional as F

__all__ = ["segmentation_loss", "auxiliary_loss"]


# ---------------------------------------------------------------------------
# 形状与输入检查
# ---------------------------------------------------------------------------


def _ensure_channel_first(x: torch.Tensor, name: str) -> torch.Tensor:
    """把 ``(B, H, W)`` 或 ``(B, 1, H, W)`` 统一为 ``(B, 1, H, W)``。"""
    if x.ndim == 3:
        return x.unsqueeze(1)
    if x.ndim == 4:
        if x.shape[1] != 1:
            raise ValueError(
                f"{name} 必须是单通道二值图，期望通道数为 1，实际为 {x.shape[1]}"
            )
        return x
    raise ValueError(
        f"{name} 必须是 (B, H, W) 或 (B, 1, H, W)，实际形状为 {tuple(x.shape)}"
    )


def _as_target_for_logits(
    target: torch.Tensor,
    logits: torch.Tensor,
    name: str,
) -> torch.Tensor:
    """检查 target 与 logits 空间尺寸一致，并转换到相同 dtype/device。"""
    target = _ensure_channel_first(target, name)
    if target.shape != logits.shape:
        raise ValueError(
            f"{name} 与 logits 的空间尺寸必须一致："
            f"logits={tuple(logits.shape)}，{name}={tuple(target.shape)}"
        )
    # 文档约定“非零值视为道路”。BCE/Dice/clDice 都要求 0/1 目标；
    # 若直接传入 0/255 的整型掩码，BCE 的目标会越界并产生错误的数值。
    return (target != 0).to(dtype=logits.dtype, device=logits.device)


# ---------------------------------------------------------------------------
# 基础损失
# ---------------------------------------------------------------------------


def _soft_dice_loss(
    probability: torch.Tensor,
    target: torch.Tensor,
    smooth: float = 1.0,
) -> torch.Tensor:
    """按样本计算 soft Dice 损失，并只对有前景目标的样本取平均。"""
    reduce_dims = tuple(range(1, probability.ndim))
    intersection = (probability * target).sum(dim=reduce_dims)
    denominator = probability.sum(dim=reduce_dims) + target.sum(dim=reduce_dims)
    per_sample = 1.0 - (2.0 * intersection + smooth) / (denominator + smooth)

    has_foreground = target.sum(dim=reduce_dims) > 0
    if bool(has_foreground.any()):
        return per_sample[has_foreground].mean()
    # 整个 batch 都为空图：Dice 不提供额外监督，返回一个可反向传播的 0。
    return probability.sum() * 0.0


def _soft_erode(x: torch.Tensor) -> torch.Tensor:
    """3x3 软腐蚀：对取负后的特征做 max pooling。"""
    vertical = -F.max_pool2d(-x, kernel_size=(3, 1), stride=1, padding=(1, 0))
    horizontal = -F.max_pool2d(-x, kernel_size=(1, 3), stride=1, padding=(0, 1))
    return torch.minimum(vertical, horizontal)


def _soft_dilate(x: torch.Tensor) -> torch.Tensor:
    """3x3 软膨胀：直接做 max pooling。"""
    return F.max_pool2d(x, kernel_size=3, stride=1, padding=1)


def _soft_skeleton(x: torch.Tensor, iterations: int = 3) -> torch.Tensor:
    """用软腐蚀/膨胀迭代近似可微骨架（soft-clDice 的核心）。"""
    # soft-open = dilate(erode(x))；骨架是原图减 opening，而不是 opening 减 erosion。
    opened = _soft_dilate(_soft_erode(x))
    skeleton = torch.relu(x - opened)
    for _ in range(iterations):
        x = _soft_erode(x)
        opened = _soft_dilate(_soft_erode(x))
        delta = torch.relu(x - opened)
        skeleton = skeleton + torch.relu(delta - skeleton * delta)
    return skeleton


def _soft_cldice_loss(
    probability: torch.Tensor,
    target: torch.Tensor,
    iterations: int = 3,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Soft-clDice 损失（1 - clDice），按样本计算后对有目标样本取平均。

    clDice 同时约束中心线覆盖（Tprec）与中心线灵敏度（Tsens），
    对道路这类细长结构比纯区域 Dice 更敏感。
    """
    reduce_dims = tuple(range(1, probability.ndim))

    pred_skeleton = _soft_skeleton(probability, iterations=iterations)
    target_skeleton = _soft_skeleton(target, iterations=iterations)

    # Tprec：预测骨架落在目标区域内的比例。
    tprec = (pred_skeleton * target).sum(dim=reduce_dims) / (
        pred_skeleton.sum(dim=reduce_dims) + eps
    )
    # Tsens：目标骨架被预测区域覆盖的比例。
    tsens = (target_skeleton * probability).sum(dim=reduce_dims) / (
        target_skeleton.sum(dim=reduce_dims) + eps
    )

    cldice = 2.0 * tprec * tsens / (tprec + tsens + eps)
    per_sample = 1.0 - cldice

    has_foreground = target.sum(dim=reduce_dims) > 0
    if bool(has_foreground.any()):
        return per_sample[has_foreground].mean()
    return probability.sum() * 0.0


def _bce_dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """BCE + Dice，供辅助头使用。"""
    logits, target = logits.float(), target.float()
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="mean")
    dice = _soft_dice_loss(torch.sigmoid(logits), target)
    return bce + dice


# ---------------------------------------------------------------------------
# 公开接口
# ---------------------------------------------------------------------------


def segmentation_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    cldice_weight: float = 0.2,
) -> torch.Tensor:
    """道路分割损失：``BCE + Dice + cldice_weight * soft-clDice``。

    Parameters
    ----------
    logits:
        未经过 sigmoid 的分割输出，形状 ``(B, 1, H, W)`` 或 ``(B, H, W)``。
    targets:
        二值道路标注，形状与 logits 相同（也接受 ``(B, H, W)``）。
        非零值视为道路。
    cldice_weight:
        soft-clDice 项的权重 ``lambda_s``。默认 0.2 对应设计文档中的起点配置。

    Returns
    -------
    torch.Tensor
        标量损失，可直接调用 ``backward()``。
    """
    if cldice_weight < 0:
        raise ValueError(f"cldice_weight 必须非负，实际为 {cldice_weight}")
    if not isinstance(logits, torch.Tensor):
        raise TypeError("logits 必须是 torch.Tensor")

    logits = _ensure_channel_first(logits, "logits")
    if not logits.is_floating_point():
        raise TypeError(f"logits 必须是浮点张量，实际 dtype 为 {logits.dtype}")
    # 1024² 上的求和超过 fp16 范围；在 AMP 中也必须用 float32 计算区域损失。
    logits = logits.float()
    targets = _as_target_for_logits(targets, logits, "targets")

    probability = torch.sigmoid(logits)
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="mean")
    dice = _soft_dice_loss(probability, targets)
    cldice = _soft_cldice_loss(probability, targets)

    return bce + dice + float(cldice_weight) * cldice


def _unpack_aux_logits(
    aux_logits: Union[
        torch.Tensor, Sequence[torch.Tensor], Mapping[str, torch.Tensor]
    ],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """把辅助头输出拆成 ``(道路 logits, 骨架 logits)``。

    支持三种写法：

    * 序列：``(road_logits, skeleton_logits)``；
    * 映射：``{"road": road_logits, "skeleton": skeleton_logits}``；
    * 单张量：``(B, 2, H, W)``，第 0 通道为道路、第 1 通道为骨架。
    """
    if isinstance(aux_logits, torch.Tensor):
        if aux_logits.ndim != 4 or aux_logits.shape[1] != 2:
            raise ValueError(
                "aux_logits 为张量时必须是 (B, 2, H, W)，"
                f"实际形状为 {tuple(aux_logits.shape)}"
            )
        return aux_logits[:, 0:1], aux_logits[:, 1:2]
    if isinstance(aux_logits, Mapping):
        try:
            return aux_logits["road"], aux_logits["skeleton"]
        except KeyError as exc:
            raise ValueError(
                "aux_logits 映射必须包含 'road' 与 'skeleton' 两个键"
            ) from exc
    if isinstance(aux_logits, (tuple, list)):
        if len(aux_logits) != 2:
            raise ValueError(
                f"aux_logits 序列长度必须为 2，实际为 {len(aux_logits)}"
            )
        return aux_logits[0], aux_logits[1]
    raise TypeError(
        "aux_logits 必须是 (road_logits, skeleton_logits) 序列或 "
        "{'road': ..., 'skeleton': ...} 映射"
    )


def _align_binary_target(
    target: torch.Tensor,
    reference: torch.Tensor,
    preserve_connectivity: bool,
) -> torch.Tensor:
    """把原始分辨率的二值 target 对齐到辅助头的输出分辨率。

    Parameters
    ----------
    target:
        原始分辨率标注，``(B, H, W)`` 或 ``(B, 1, H, W)``。
    reference:
        辅助头 logits，决定目标的空间尺寸。
    preserve_connectivity:
        当 target 分辨率更高时：

        * ``False``（道路区域）：使用 nearest 下采样，保持标签值不被插值混淆；
        * ``True``（骨架/中心线）：若为整数倍下采样，使用 max pooling。
          只要一个池化窗口内存在骨架像素，输出就为 1，避免 1 像素宽的
          中心线被 nearest 采样“跳过”而整体消失。

        若为任意比例（非整数倍），退化为 nearest；若 target 分辨率更低
        （logits 在更高分辨率上输出），同样使用 nearest 上采样。
    """
    target = _ensure_channel_first(target, "target")
    reference = _ensure_channel_first(reference, "logits")
    # 先按“非零即前景”二值化，再转 dtype：辅助头的 masks/skeletons 契约是二值图，
    # 非 0/1 的整型标签不应进入 BCE/Dice。
    target = (target != 0).to(dtype=reference.dtype, device=reference.device)

    if target.shape[-2:] == reference.shape[-2:]:
        return target

    if preserve_connectivity:
        src_h, src_w = target.shape[-2:]
        dst_h, dst_w = reference.shape[-2:]
        if src_h >= dst_h and src_w >= dst_w:
            scale_h = src_h // dst_h
            scale_w = src_w // dst_w
            is_integer_downsample = (
                scale_h >= 1
                and scale_w >= 1
                and src_h == dst_h * scale_h
                and src_w == dst_w * scale_w
            )
            if is_integer_downsample:
                return F.max_pool2d(
                    target,
                    kernel_size=(scale_h, scale_w),
                    stride=(scale_h, scale_w),
                )

    return F.interpolate(target, size=reference.shape[-2:], mode="nearest")


def auxiliary_loss(
    aux_logits: Union[
        torch.Tensor, Sequence[torch.Tensor], Mapping[str, torch.Tensor]
    ],
    masks: torch.Tensor,
    skeletons: torch.Tensor,
    valid_sizes: Sequence[Tuple[int, int]] | None = None,
) -> torch.Tensor:
    """辅助头损失：道路区域监督 + 中心线（骨架）监督。

    Parameters
    ----------
    aux_logits:
        辅助头输出，支持 ``(B, 2, H, W)`` 单张量、``(road_logits, skeleton_logits)``
        序列，或 ``{"road": ..., "skeleton": ...}`` 映射。空间分辨率通常低于原图
        （例如条件特征 ``C_256`` 上的 256x256）。
    masks:
        原始分辨率的道路掩码，与主分割标注一致。
    skeletons:
        原始分辨率的道路中心线/骨架二值图。

    Returns
    -------
    torch.Tensor
        ``0.5 * (L_road + L_skeleton)``，标量，可反向传播。

    Notes
    -----
    两个头都使用 BCE + Dice。骨架头的 target 在整数倍下采样时用 max pooling
    对齐到 logits 分辨率，避免细线在降采样中消失；道路掩码头使用 nearest。
    """
    road_logits, skeleton_logits = _unpack_aux_logits(aux_logits)

    if not isinstance(road_logits, torch.Tensor) or not isinstance(
        skeleton_logits, torch.Tensor
    ):
        raise TypeError("aux_logits 中的两个元素都必须是 torch.Tensor")
    if not road_logits.is_floating_point() or not skeleton_logits.is_floating_point():
        raise TypeError("aux_logits 必须是浮点张量")

    road_logits = _ensure_channel_first(road_logits, "road_logits")
    skeleton_logits = _ensure_channel_first(skeleton_logits, "skeleton_logits")

    road_target = _align_binary_target(
        masks, road_logits, preserve_connectivity=False
    )
    skeleton_target = _align_binary_target(
        skeletons, skeleton_logits, preserve_connectivity=True
    )

    if valid_sizes is None:
        road_loss = _bce_dice_loss(road_logits, road_target)
        skeleton_loss = _bce_dice_loss(skeleton_logits, skeleton_target)
        return 0.5 * (road_loss + skeleton_loss)
    if len(valid_sizes) != masks.shape[0]:
        raise ValueError("valid_sizes 数量必须与 batch 一致")
    # Targets are aligned on the full canvas before cropping so padding cannot
    # change the interpolation coordinates. Exclude partial cells at the edge.
    values = []
    for index, (height, width) in enumerate(valid_sizes):
        if not (0 < height <= masks.shape[-2] and 0 < width <= masks.shape[-1]):
            raise ValueError("valid_sizes 必须在标注画布内且非空")
        region_losses = []
        for logits, target in ((road_logits, road_target), (skeleton_logits, skeleton_target)):
            h = max(1, int(height * logits.shape[-2] / masks.shape[-2]))
            w = max(1, int(width * logits.shape[-1] / masks.shape[-1]))
            region_losses.append(_bce_dice_loss(
                logits[index:index + 1, :, :h, :w], target[index:index + 1, :, :h, :w]
            ))
        values.append(0.5 * (region_losses[0] + region_losses[1]))
    return torch.stack(values).mean()
