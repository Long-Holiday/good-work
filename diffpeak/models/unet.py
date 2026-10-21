"""条件扩散生成器使用的四尺度 U-Net。

输入为双通道热图潜变量 ``z_t``、时间步 ``t`` 与四个尺度的影像条件特征，
输出同形状的噪声预测。每个尺度包含两个 GroupNorm + SiLU 残差块，时间步通过
正弦编码与 MLP 注入，条件特征通过拼接 + 1x1 卷积融合。
"""

from __future__ import annotations

import math
from typing import List, Sequence, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils import checkpoint as grad_checkpoint


def _num_groups(channels: int, max_groups: int = 8) -> int:
    """选择能整除通道数的 GroupNorm 分组数。"""
    for groups in range(min(max_groups, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


def timestep_embedding(timesteps: Tensor, dim: int, max_period: float = 10000.0) -> Tensor:
    """标准正弦时间步编码，返回 ``B x dim``。"""
    if dim <= 0:
        raise ValueError("时间步编码维度必须为正")
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(half, device=timesteps.device, dtype=torch.float32)
        / max(half, 1)
    )
    args = timesteps.to(torch.float32).reshape(-1, 1) * freqs.reshape(1, -1)
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=1)
    if dim % 2 == 1:
        embedding = F.pad(embedding, (0, 1))
    return embedding


class ResidualBlock(nn.Module):
    """带时间步注入的两层残差块。"""

    def __init__(self, in_channels: int, out_channels: int, time_embed_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(_num_groups(in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        # 注意：这里的 SiLU 不能原地修改 time_embedding。该张量由
        # ConditionalUNet.forward 共享给所有残差块，原地 SiLU 会让后续
        # 块看到被重复激活的时间嵌入。
        self.time_mlp = nn.Sequential(
            nn.SiLU(inplace=False),
            nn.Linear(time_embed_dim, out_channels),
        )
        self.norm2 = nn.GroupNorm(_num_groups(out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.act = nn.SiLU(inplace=True)
        self.skip = (
            nn.Conv2d(in_channels, out_channels, 1, bias=False)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: Tensor, time_embedding: Tensor) -> Tensor:
        h = self.conv1(self.act(self.norm1(x)))
        h = h + self.time_mlp(time_embedding).reshape(
            time_embedding.shape[0], -1, 1, 1
        )
        h = self.conv2(self.act(self.norm2(h)))
        return self.skip(x) + h


class ConditionFusion(nn.Module):
    """把对应尺度的条件特征与主干特征拼接后卷积融合。"""

    def __init__(self, channels: int, condition_channels: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(channels + condition_channels, channels, 1)

    def forward(self, x: Tensor, condition: Tensor) -> Tensor:
        if condition.shape[-2:] != x.shape[-2:]:
            condition = F.interpolate(
                condition, size=x.shape[-2:], mode="bilinear", align_corners=False
            )
        # 条件编码器可能在不同 autocast 区域运行；对齐 dtype/device 以保证
        # 拼接稳定。
        condition = condition.to(device=x.device, dtype=x.dtype)
        return x + self.proj(torch.cat([x, condition], dim=1))


class ConditionalLevel(nn.Module):
    """一个 U-Net 尺度：两个残差块 + 条件融合。"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        condition_channels: int,
        time_embed_dim: int,
        num_res_blocks: int = 2,
    ) -> None:
        super().__init__()
        blocks: List[nn.Module] = []
        for index in range(num_res_blocks):
            blocks.append(
                ResidualBlock(
                    in_channels if index == 0 else out_channels,
                    out_channels,
                    time_embed_dim,
                )
            )
        self.blocks = nn.ModuleList(blocks)
        self.fusion = ConditionFusion(out_channels, condition_channels)

    def forward(self, x: Tensor, time_embedding: Tensor, condition: Tensor) -> Tensor:
        for block in self.blocks:
            x = block(x, time_embedding)
        return self.fusion(x, condition)


class Downsample(nn.Module):
    """3x3 步长卷积下采样。"""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    """最近邻上采样 + 3x3 卷积。"""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class ConditionalUNet(nn.Module):
    """四尺度条件 U-Net，预测双通道热图噪声。

    Args:
        in_channels: 热图潜变量通道数，默认 2。
        out_channels: 输出通道数，默认 2。
        base_channels: 第一尺度通道数，默认 64。
        channel_mult: 四个尺度的通道倍率，默认 ``(1, 2, 4, 4)``，即
            64/128/256/256。
        condition_channels: 条件特征通道数，默认 ``(64, 128, 256, 256)``。
        num_res_blocks: 每个尺度的残差块数，默认 2。
        time_embed_mult: 时间步嵌入维度相对 ``base_channels`` 的倍率。
        gradient_checkpointing: 是否在梯度开启的前向中启用梯度检查点，
            默认 True。使用非重入实现，因此 ``z_t`` 或冻结条件特征不
            ``requires_grad`` 时，U-Net 参数仍能获得梯度；``no_grad``
            采样会自动跳过检查点。
    """

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        base_channels: int = 64,
        channel_mult: Sequence[int] = (1, 2, 4, 4),
        condition_channels: Sequence[int] = (64, 128, 256, 256),
        num_res_blocks: int = 2,
        time_embed_mult: int = 4,
        gradient_checkpointing: bool = True,
    ) -> None:
        super().__init__()
        if len(channel_mult) != 4:
            raise ValueError("channel_mult 必须包含四个尺度")
        if len(condition_channels) != 4:
            raise ValueError("condition_channels 必须包含四个尺度")

        self.base_channels = int(base_channels)
        self.channel_mult = tuple(int(m) for m in channel_mult)
        self.condition_channels = tuple(int(c) for c in condition_channels)
        self.gradient_checkpointing = bool(gradient_checkpointing)

        channels = [self.base_channels * mult for mult in self.channel_mult]
        time_embed_dim = self.base_channels * int(time_embed_mult)
        self.time_embed_dim = time_embed_dim

        self.time_embed = nn.Sequential(
            nn.Linear(self.base_channels, time_embed_dim),
            nn.SiLU(inplace=True),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        c0, c1, c2, c3 = channels
        cond0, cond1, cond2, cond3 = self.condition_channels

        self.enc0 = ConditionalLevel(in_channels, c0, cond0, time_embed_dim, num_res_blocks)
        self.down0 = Downsample(c0)
        self.enc1 = ConditionalLevel(c0, c1, cond1, time_embed_dim, num_res_blocks)
        self.down1 = Downsample(c1)
        self.enc2 = ConditionalLevel(c1, c2, cond2, time_embed_dim, num_res_blocks)
        self.down2 = Downsample(c2)
        self.mid = ConditionalLevel(c2, c3, cond3, time_embed_dim, num_res_blocks)

        self.up2 = Upsample(c3)
        self.dec2 = ConditionalLevel(c3 + c2, c2, cond2, time_embed_dim, num_res_blocks)
        self.up1 = Upsample(c2)
        self.dec1 = ConditionalLevel(c2 + c1, c1, cond1, time_embed_dim, num_res_blocks)
        self.up0 = Upsample(c1)
        self.dec0 = ConditionalLevel(c1 + c0, c0, cond0, time_embed_dim, num_res_blocks)

        self.out_norm = nn.GroupNorm(_num_groups(c0), c0)
        self.out_act = nn.SiLU(inplace=True)
        self.out_conv = nn.Conv2d(c0, out_channels, 3, padding=1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    @staticmethod
    def _as_batch_timesteps(
        timesteps: Union[Tensor, int], batch_size: int, device: torch.device
    ) -> Tensor:
        """把标量/张量时间步整理为 ``B`` 的 long 张量。"""
        if isinstance(timesteps, Tensor):
            t = timesteps.to(device=device, dtype=torch.long).reshape(-1)
            if t.numel() == 1:
                return t.expand(batch_size)
            if t.numel() != batch_size:
                raise ValueError("时间步数量与 batch size 不一致")
            return t
        return torch.full((batch_size,), int(timesteps), device=device, dtype=torch.long)

    def _run(
        self,
        module: nn.Module,
        x: Tensor,
        time_embedding: Tensor,
        condition: Tensor,
    ) -> Tensor:
        """按需使用梯度检查点执行一个尺度模块。

        ``use_reentrant=False`` 支持“输入不需要梯度、仅模块参数需要梯度”的
        场景（例如阶段 D 中条件特征已冻结，``z_t`` 也不带梯度）。若误用
        重入实现，这类调用不会产生参数梯度。采样在 ``torch.no_grad()``
        下进行，此时不启用检查点。
        """
        if (
            self.gradient_checkpointing
            and torch.is_grad_enabled()
        ):
            return grad_checkpoint.checkpoint(
                module, x, time_embedding, condition, use_reentrant=False
            )
        return module(x, time_embedding, condition)

    def forward(
        self, z: Tensor, t: Union[Tensor, int], conditions: Sequence[Tensor]
    ) -> Tensor:
        """预测噪声。

        Args:
            z: 带噪热图 ``B x 2 x 256 x 256``。
            t: 时间步，标量或长度 B 的张量。
            conditions: 四个尺度的条件特征，通道数与分辨率需与 U-Net 对齐。

        Returns:
            与 ``z`` 同形状的噪声预测。
        """
        if z.ndim != 4:
            raise ValueError(f"z 应为 B x C x H x W，实际为 {tuple(z.shape)}")
        if len(conditions) != 4:
            raise ValueError("conditions 必须包含四个尺度")
        batch_size = z.shape[0]

        t_batch = self._as_batch_timesteps(t, batch_size, z.device)
        time_embedding = self.time_embed(
            timestep_embedding(t_batch, self.base_channels).to(dtype=z.dtype)
        )

        s0 = self._run(self.enc0, z, time_embedding, conditions[0])
        s1 = self._run(self.enc1, self.down0(s0), time_embedding, conditions[1])
        s2 = self._run(self.enc2, self.down1(s1), time_embedding, conditions[2])
        mid = self._run(self.mid, self.down2(s2), time_embedding, conditions[3])

        u2 = self.up2(mid)
        u2 = self._run(self.dec2, torch.cat([u2, s2], dim=1), time_embedding, conditions[2])
        u1 = self.up1(u2)
        u1 = self._run(self.dec1, torch.cat([u1, s1], dim=1), time_embedding, conditions[1])
        u0 = self.up0(u1)
        u0 = self._run(self.dec0, torch.cat([u0, s0], dim=1), time_embedding, conditions[0])

        return self.out_conv(self.out_act(self.out_norm(u0)))
