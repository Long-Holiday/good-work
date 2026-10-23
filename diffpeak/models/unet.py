"""四尺度条件扩散 U-Net：逐尺度图像条件、时间调制与低分辨率注意力。

直接在单通道场上预测 Flow Matching 速度，使用连续时间[0,1]。
默认空间尺度为 256/128/64/32，仅在最小尺度使用全局自注意力；
编码器和解码器均逐层融合对应条件，并通过拼接跳连保留局部细节。
"""

from __future__ import annotations

import math
from typing import Sequence, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils import checkpoint as grad_checkpoint

from .conditioning import _num_groups


def _positive_int(value: object, name: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} 必须为正整数") from error
    if result != value or result <= 0:
        raise ValueError(f"{name} 必须为正整数")
    return result


def timestep_embedding(timesteps: Tensor, dim: int) -> Tensor:
    """正弦时间编码，返回 B x dim。"""
    half = dim // 2
    frequencies = torch.exp(
        -math.log(10000.0)
        * torch.arange(half, device=timesteps.device, dtype=torch.float32)
        / max(half, 1)
    )
    angles = timesteps.float().reshape(-1, 1) * frequencies
    embedding = torch.cat((angles.cos(), angles.sin()), dim=1)
    return F.pad(embedding, (0, dim % 2))


class TimeResidualBlock(nn.Module):
    """GroupNorm 残差卷积；时间向量产生逐通道 scale/shift。"""

    def __init__(self, in_channels: int, out_channels: int, time_channels: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(_num_groups(in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(time_channels, 2 * out_channels))
        self.norm2 = nn.GroupNorm(_num_groups(out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip = (
            nn.Identity() if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, 1)
        )

    def forward(self, x: Tensor, time: Tensor) -> Tensor:
        hidden = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.time_projection(time).to(hidden.dtype).chunk(2, dim=1)
        hidden = self.norm2(hidden) * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        return self.skip(x) + self.conv2(F.silu(hidden))


class SpatialAttention(nn.Module):
    """只在配置的低分辨率层使用 SDPA，dropout 为 0。"""

    def __init__(self, channels: int, num_heads: int) -> None:
        super().__init__()
        if channels % num_heads:
            raise ValueError("注意力通道数必须能被 num_heads 整除")
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(_num_groups(channels), channels)
        self.qkv = nn.Conv2d(channels, 3 * channels, 1)
        self.projection = nn.Conv2d(channels, channels, 1)
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, x: Tensor) -> Tensor:
        batch, channels, height, width = x.shape
        qkv = self.qkv(self.norm(x)).reshape(
            batch, 3, self.num_heads, channels // self.num_heads, height * width
        )
        q, k, v = qkv.permute(1, 0, 2, 4, 3).unbind(0)
        hidden = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        hidden = hidden.transpose(-1, -2).reshape(batch, channels, height, width)
        return x + self.projection(hidden)


class ConditionalLevel(nn.Module):
    """每个残差块重新融合本尺度图像条件，避免仅入口注入。"""

    def __init__(
        self, in_channels: int, channels: int, condition_channels: int,
        time_channels: int, num_res_blocks: int, attention: bool, num_heads: int,
    ) -> None:
        super().__init__()
        self.condition_projection = nn.Conv2d(condition_channels, channels, 1)
        self.blocks = nn.ModuleList(
            TimeResidualBlock(
                in_channels + channels if index == 0 else channels * 2,
                channels, time_channels,
            )
            for index in range(num_res_blocks)
        )
        self.attention = SpatialAttention(channels, num_heads) if attention else nn.Identity()

    def forward(self, x: Tensor, time: Tensor, condition: Tensor) -> Tensor:
        condition = self.condition_projection(condition)
        if condition.shape[-2:] != x.shape[-2:]:
            condition = F.interpolate(condition, size=x.shape[-2:], mode="bilinear", align_corners=False)
        for block in self.blocks:
            x = block(torch.cat((x, condition), dim=1), time)
        return self.attention(x)


class ConditionalUNet(nn.Module):
    """四尺度条件速度预测器，支持奇数及矩形尺寸。

    ``attention_levels`` 是从高到低的尺度索引，默认仅 3（256 输入时为32²）。
    为避免误配置造成高分辨率全局注意力，只允许索引2和3。
    梯度检查点使用非重入模式，冻结输入时仍可训练网络参数。
    """

    def __init__(
        self, in_channels: int = 1, out_channels: int = 1,
        base_channels: int = 64, channel_mults: Sequence[int] = (1, 2, 4, 4),
        num_res_blocks: int = 2, attention_levels: Sequence[int] = (3,),
        num_heads: int = 8, condition_channels: Sequence[int] = (64, 128, 256, 256),
        gradient_checkpointing: bool = True, time_scale: float = 1000.0,
    ) -> None:
        super().__init__()
        if not math.isfinite(time_scale) or time_scale <= 0:
            raise ValueError("time_scale 必须为正有限数")
        self.time_scale = float(time_scale)
        self.in_channels = _positive_int(in_channels, "in_channels")
        self.out_channels = _positive_int(out_channels, "out_channels")
        self.base_channels = _positive_int(base_channels, "base_channels")
        self.num_res_blocks = _positive_int(num_res_blocks, "num_res_blocks")
        self.num_heads = _positive_int(num_heads, "num_heads")
        self.channel_mults = tuple(_positive_int(c, "channel_mults") for c in channel_mults)
        self.condition_channels = tuple(_positive_int(c, "condition_channels") for c in condition_channels)
        if len(self.channel_mults) != 4 or len(self.condition_channels) != 4:
            raise ValueError("U-Net 要求四个尺度的 channel_mults 和 condition_channels")
        levels = tuple(attention_levels)
        if any(isinstance(level, bool) or not isinstance(level, int) or level not in (2, 3) for level in levels):
            raise ValueError("attention_levels 只允许低分辨率整数尺度索引 2、3")
        self.attention_levels = tuple(sorted(set(int(level) for level in levels)))
        self.gradient_checkpointing = bool(gradient_checkpointing)
        channels = tuple(self.base_channels * mult for mult in self.channel_mults)
        for level in self.attention_levels:
            if channels[level] % self.num_heads:
                raise ValueError("注意力层通道数必须能被 num_heads 整除")
        time_channels = self.base_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(self.base_channels, time_channels), nn.SiLU(),
            nn.Linear(time_channels, time_channels),
        )
        self.input_conv = nn.Conv2d(self.in_channels, channels[0], 3, padding=1)
        self.encoders = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for level, channel in enumerate(channels):
            self.encoders.append(ConditionalLevel(
                channel, channel, self.condition_channels[level], time_channels,
                self.num_res_blocks, level in self.attention_levels, self.num_heads,
            ))
            if level < 3:
                self.downsamples.append(nn.Conv2d(channel, channels[level + 1], 3, stride=2, padding=1))
        self.middle = TimeResidualBlock(channels[-1], channels[-1], time_channels)
        for level in range(3, -1, -1):
            channel = channels[level]
            self.decoders.append(ConditionalLevel(
                channel * 2, channel, self.condition_channels[level], time_channels,
                self.num_res_blocks, level in self.attention_levels, self.num_heads,
            ))
            if level > 0:
                self.upsamples.append(nn.Conv2d(channel, channels[level - 1], 3, padding=1))
        self.output_norm = nn.GroupNorm(_num_groups(channels[0]), channels[0])
        self.output_conv = nn.Conv2d(channels[0], self.out_channels, 3, padding=1)
        nn.init.zeros_(self.output_conv.weight)
        nn.init.zeros_(self.output_conv.bias)

    def _run(self, module: nn.Module, *inputs: Tensor) -> Tensor:
        if self.gradient_checkpointing and torch.is_grad_enabled():
            return grad_checkpoint.checkpoint(module, *inputs, use_reentrant=False)
        return module(*inputs)

    def forward(
        self, x: Tensor, t: Union[float, Tensor], conditions: Sequence[Tensor],
    ) -> Tensor:
        if x.ndim != 4 or x.shape[1] != self.in_channels or min(x.shape[-2:]) <= 0:
            raise ValueError("x 必须为非空 B x in_channels x H x W")
        if isinstance(conditions, Tensor) or len(conditions) != 4:
            raise ValueError("conditions 必须包含四个尺度")
        prepared = []
        for index, (condition, channel) in enumerate(zip(conditions, self.condition_channels)):
            if not isinstance(condition, Tensor) or condition.ndim != 4:
                raise ValueError(f"conditions[{index}] 必须为四维张量")
            if condition.shape[:2] != (x.shape[0], channel) or min(condition.shape[-2:]) <= 0:
                raise ValueError(f"conditions[{index}] 的批次、通道或空间尺寸不匹配")
            prepared.append(condition.to(device=x.device, dtype=x.dtype))
        time = torch.as_tensor(t, device=x.device, dtype=torch.float32).reshape(-1)
        if time.numel() == 1:
            time = time.expand(x.shape[0])
        if time.numel() != x.shape[0]:
            raise ValueError("时间步数量与 batch size 不一致")
        if not bool(torch.isfinite(time).all()) or bool(((time < 0) | (time > 1)).any()):
            raise ValueError("Flow Matching 时间必须为[0,1]内有限浮点数")
        time = self.time_embed(timestep_embedding(time * self.time_scale, self.base_channels).to(x.dtype))
        hidden = self.input_conv(x)
        skips = []
        for level, encoder in enumerate(self.encoders):
            hidden = self._run(encoder, hidden, time, prepared[level])
            skips.append(hidden)
            if level < 3:
                hidden = self.downsamples[level](hidden)
        hidden = self._run(self.middle, hidden, time)
        for index, decoder in enumerate(self.decoders):
            level = 3 - index
            hidden = self._run(decoder, torch.cat((hidden, skips[level]), dim=1), time, prepared[level])
            if level > 0:
                hidden = F.interpolate(hidden, size=skips[level - 1].shape[-2:], mode="nearest")
                hidden = self.upsamples[index](hidden)
        return self.output_conv(F.silu(self.output_norm(hidden)))


__all__ = ["ConditionalUNet", "SpatialAttention", "TimeResidualBlock"]
