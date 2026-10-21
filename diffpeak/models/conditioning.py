"""多尺度条件特征编码器。

轻量细节分支与 SAM 图像特征融合，输出：
- 256/128/64/32 四个尺度的条件特征，通道数为 64/128/256/256；
- 256x256 的辅助道路/中心线 logits。
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _num_groups(channels: int, max_groups: int = 8) -> int:
    """选择能整除通道数的 GroupNorm 分组数。"""
    for groups in range(min(max_groups, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class ConvNormAct(nn.Module):
    """Conv2d + GroupNorm + SiLU，偏置由归一化吸收。"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        act: bool = True,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False
        )
        self.norm = nn.GroupNorm(_num_groups(out_channels), out_channels)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.norm(self.conv(x)))


class ResidualBlock(nn.Module):
    """两层 3x3 卷积的残差块，每层前使用 GroupNorm + SiLU。"""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(_num_groups(channels), channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(_num_groups(channels), channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        h = self.conv1(self.act(self.norm1(x)))
        h = self.conv2(self.act(self.norm2(h)))
        return x + h


class ConditionEncoder(nn.Module):
    """生成扩散生成器所需的多尺度条件特征。

    Args:
        condition_channels: 四个尺度的输出通道，默认 ``(64, 128, 256, 256)``。
        sam_channels: SAM 图像特征通道数，默认 256。
        auxiliary_classes: 辅助头输出通道，默认 2（道路区域、中心线）。

    ``forward`` 的输入为 RGB ``[0, 1]`` 图像与 SAM 图像特征，输出
    ``([C256, C128, C64, C32], auxiliary_logits)``。辅助头只在训练阶段使用。

    显存说明：SAM 特征在 64x64 低分辨率先做 1x1 投影到 ``C256`` 的通道数，
    再上采样到 256x256 与细节特征拼接。由于 1x1 卷积与双线性上采样都是
    线性操作，这不改变“拼接 + 1x1 卷积”融合的函数形式，但避免了直接在
    256² 分辨率上持有 256 通道 SAM 激活。
    """

    def __init__(
        self,
        condition_channels: Sequence[int] = (64, 128, 256, 256),
        sam_channels: int = 256,
        auxiliary_classes: int = 2,
    ) -> None:
        super().__init__()
        if len(condition_channels) != 4:
            raise ValueError("condition_channels 必须包含 256/128/64/32 四个尺度")
        self.condition_channels = tuple(int(c) for c in condition_channels)
        if any(channels <= 0 for channels in self.condition_channels):
            raise ValueError("condition_channels 的每个尺度都必须为正")
        if int(sam_channels) <= 0:
            raise ValueError("sam_channels 必须为正")
        c256, c128, c64, c32 = self.condition_channels

        # 细节分支：1024 -> 512 -> 256，保持 64 通道。
        self.detail_channels = 64
        self.detail_conv1 = ConvNormAct(3, 32, kernel_size=3, stride=2)
        self.detail_conv2 = ConvNormAct(
            32, self.detail_channels, kernel_size=3, stride=2
        )
        self.detail_blocks = nn.Sequential(
            ResidualBlock(self.detail_channels),
            ResidualBlock(self.detail_channels),
        )

        # SAM 特征在低分辨率投影到 C256，再上采样到 256² 后与细节特征拼接。
        # 这样避免 B x 256 x 256 x 256 的中间激活，同时保留线性融合形式。
        self.sam_projection = nn.Conv2d(
            int(sam_channels), c256, kernel_size=1, bias=False
        )
        self.fuse = ConvNormAct(
            self.detail_channels + c256, c256, kernel_size=1
        )

        # 步长卷积条件金字塔：256 -> 128 -> 64 -> 32。
        self.down_to_128 = ConvNormAct(c256, c128, kernel_size=3, stride=2)
        self.down_to_64 = ConvNormAct(c128, c64, kernel_size=3, stride=2)
        self.down_to_32 = ConvNormAct(c64, c32, kernel_size=3, stride=2)

        # 辅助头：道路区域与中心线 logits。
        self.auxiliary_head = nn.Sequential(
            ConvNormAct(c256, c256, kernel_size=3),
            nn.Conv2d(c256, auxiliary_classes, kernel_size=1),
        )

    def forward(self, images: Tensor, embeddings: Tensor) -> Tuple[List[Tensor], Tensor]:
        """编码条件特征。

        Args:
            images: ``B x 3 x 1024 x 1024``，取值范围 ``[0, 1]``。
            embeddings: SAM 图像特征 ``B x 256 x 64 x 64``。

        Returns:
            ``(conditions, auxiliary_logits)``，其中 conditions 依次为
            ``C256/Bx64x256x256``、``C128/Bx128x128x128``、
            ``C64/Bx256x64x64``、``C32/Bx256x32x32``，
            auxiliary_logits 为 ``B x 2 x 256 x 256``。

        Note:
            SAM 特征先投影再上采样，因此 256² 尺度上只有 ``C256`` 通道的
            投影结果参与拼接，而不是原始 256 通道特征。
        """
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"images 应为 B x 3 x H x W，实际为 {tuple(images.shape)}")
        if embeddings.ndim != 4:
            raise ValueError(f"embeddings 应为 B x C x H x W，实际为 {tuple(embeddings.shape)}")
        if images.shape[0] != embeddings.shape[0]:
            raise ValueError("images 与 embeddings 的 batch size 不一致")

        detail = self.detail_conv1(images)
        detail = self.detail_conv2(detail)
        detail = self.detail_blocks(detail)

        sam_projected = self.sam_projection(embeddings)
        sam_up = F.interpolate(
            sam_projected, size=detail.shape[-2:], mode="bilinear", align_corners=False
        )
        # SAM 编码器可能在 autocast 之外运行；显式对齐到细节分支的 dtype，
        # 避免混合精度下拼接失败。
        sam_up = sam_up.to(dtype=detail.dtype)
        c256 = self.fuse(torch.cat([detail, sam_up], dim=1))

        c128 = self.down_to_128(c256)
        c64 = self.down_to_64(c128)
        c32 = self.down_to_32(c64)

        auxiliary_logits = self.auxiliary_head(c256)
        return [c256, c128, c64, c32], auxiliary_logits
