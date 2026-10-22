"""条件 DiT（Diffusion Transformer）去噪器。

``forward(x, t, conditions)`` 接收双通道热图潜变量、
时间步与四个尺度的条件特征，返回同形状的噪声预测。

架构要点：

* 图像按 ``patch_size`` 切分为 token，使用可配置的 ``Conv2d`` patch embed；
* 固定 2D 正弦位置编码，支持任意（含矩形）patch 网格，不引入尺寸相关参数；
* 每个 Transformer block 通过 adaLN-Zero 注入时间步调制（shift/scale/gate）；
* 四个尺度的条件特征分别经 1x1 投影、双线性对齐到热图分辨率后求和，
  再 patchify 并加到 token 上（空间条件融合）；
* 注意力使用 ``F.scaled_dot_product_attention``，dropout 固定为 0；
* 最终层为 adaLN 调制 + 线性头，线性头与调制层零初始化，因此未训练时
  噪声预测恒为零；
* 当高/宽不能被 ``patch_size`` 整除时，在右侧/下侧补零到整数倍，前向后
  再裁剪回原始尺寸。

梯度检查点使用 ``use_reentrant=False``：即使 ``x`` 与条件特征都被 detach
（GRPO 中冻结条件编码器的场景），模块参数仍能获得梯度；``torch.no_grad()``
采样时会自动跳过检查点。
"""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils import checkpoint as grad_checkpoint

Timestep = Union[int, Tensor]


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


def _positive_int(value: object, name: str) -> int:
    """校验并返回正整数配置项。"""
    try:
        result = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} 必须为正整数，实际为 {value!r}") from error
    if result != value or result <= 0:
        raise ValueError(f"{name} 必须为正整数，实际为 {value!r}")
    return result


def _sincos_1d(positions: Tensor, dim: int, max_period: float = 10000.0) -> Tensor:
    """一维正弦编码，返回 ``len(positions) x dim``。"""
    if dim <= 0:
        return positions.new_zeros((positions.shape[0], 0))
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(half, device=positions.device, dtype=torch.float32)
        / max(half, 1)
    )
    args = positions.to(torch.float32).reshape(-1, 1) * freqs.reshape(1, -1)
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=1)
    if dim % 2 == 1:
        embedding = F.pad(embedding, (0, 1))
    return embedding


def build_2d_sincos_pos_embed(
    height: int,
    width: int,
    dim: int,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """构造 ``(height * width) x dim`` 的 2D 正弦位置编码。

    ``dim`` 的前一半编码行坐标、后一半编码列坐标；不使用可学习参数，因此
    可以直接用于动态的矩形 patch 网格。``dim`` 为奇数时仍能返回精确的
    ``dim`` 通道（末尾补零）。
    """
    if height <= 0 or width <= 0:
        raise ValueError("位置编码的 height/width 必须为正")
    if dim <= 0:
        raise ValueError("位置编码维度必须为正")
    if device is None:
        device = torch.device("cpu")

    dim_h = dim // 2
    dim_w = dim - dim_h
    rows = torch.arange(height, device=device)
    cols = torch.arange(width, device=device)
    emb_h = _sincos_1d(rows, dim_h).reshape(height, 1, dim_h)
    emb_w = _sincos_1d(cols, dim_w).reshape(1, width, dim_w)
    embedding = torch.cat(
        [
            emb_h.expand(height, width, dim_h),
            emb_w.expand(height, width, dim_w),
        ],
        dim=-1,
    )
    return embedding.reshape(height * width, dim).to(dtype=dtype)


def _modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    """adaLN 调制：``x * (1 + scale) + shift``，广播到 token 维。"""
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTAttention(nn.Module):
    """多头自注意力，使用 SDPA（``dropout_p=0``）。"""

    def __init__(self, hidden_size: int, num_heads: int) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size={hidden_size} 必须能被 num_heads={num_heads} 整除"
            )
        self.num_heads = int(num_heads)
        self.head_dim = int(hidden_size) // self.num_heads
        self.qkv = nn.Linear(hidden_size, hidden_size * 3)
        self.proj = nn.Linear(hidden_size, hidden_size)
        # 固定为 0，确保训练/评估都无注意力随机性，也便于扩散的确定性重算。
        self.attention_dropout = 0.0

    def forward(self, x: Tensor) -> Tensor:
        batch, num_tokens, hidden = x.shape
        qkv = self.qkv(x).reshape(
            batch, num_tokens, 3, self.num_heads, self.head_dim
        )
        # B x N x 3 x H x D -> 3 x B x H x N x D
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        attended = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.attention_dropout,
            is_causal=False,
        )
        attended = attended.transpose(1, 2).reshape(batch, num_tokens, hidden)
        return self.proj(attended)


class DiTBlock(nn.Module):
    """adaLN-Zero 调制的 Transformer block。"""

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float) -> None:
        super().__init__()
        mlp_hidden = max(1, int(hidden_size * mlp_ratio))
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = DiTAttention(hidden_size, num_heads)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden, hidden_size),
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size),
        )
        # adaLN-Zero：初始 gate 为 0，block 退化为恒等映射。
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x: Tensor, condition: Tensor) -> Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(condition).chunk(6, dim=1)
        )
        x = x + gate_msa.unsqueeze(1) * self.attn(
            _modulate(self.norm1(x), shift_msa, scale_msa)
        )
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            _modulate(self.norm2(x), shift_mlp, scale_mlp)
        )
        return x


class DiTFinalLayer(nn.Module):
    """最终 adaLN 调制 + 零初始化的 patch 线性头。"""

    def __init__(
        self, hidden_size: int, patch_size: int, out_channels: int
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size),
        )
        self.linear = nn.Linear(
            hidden_size, patch_size * patch_size * out_channels
        )
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: Tensor, condition: Tensor) -> Tensor:
        shift, scale = self.adaLN_modulation(condition).chunk(2, dim=1)
        return self.linear(_modulate(self.norm(x), shift, scale))


class ConditionalDiT(nn.Module):
    """四尺度条件 Transformer 去噪器。

    Args:
        in_channels: 热图潜变量通道数，默认 2。
        out_channels: 噪声预测通道数，默认 2。
        hidden_size: Transformer 宽度，默认 256。
        depth: Transformer block 数量，默认 6。
        num_heads: 注意力头数，默认 8（须整除 ``hidden_size``）。
        patch_size: patch 边长，默认 8（256 热图得到 32x32 token）。
        mlp_ratio: FFN 隐层相对 ``hidden_size`` 的倍率，默认 4.0。
        condition_channels: 四个尺度的条件通道数，默认
            ``(64, 128, 256, 256)``，与
            :class:`diffpeak.models.conditioning.ConditionEncoder` 一致。
        gradient_checkpointing: 是否为每个 block 启用非重入梯度检查点，
            默认 True。仅在梯度开启时启用。

    ``forward`` 的条件特征可以是任意空间尺寸（会双线性对齐到热图大小）；
    批量、通道数与数量会被严格校验。
    """

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        hidden_size: int = 256,
        depth: int = 6,
        num_heads: int = 8,
        patch_size: int = 8,
        mlp_ratio: float = 4.0,
        condition_channels: Sequence[int] = (64, 128, 256, 256),
        gradient_checkpointing: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = _positive_int(in_channels, "in_channels")
        self.out_channels = _positive_int(out_channels, "out_channels")
        self.hidden_size = _positive_int(hidden_size, "hidden_size")
        self.depth = _positive_int(depth, "depth")
        self.num_heads = _positive_int(num_heads, "num_heads")
        self.patch_size = _positive_int(patch_size, "patch_size")
        if self.hidden_size % self.num_heads != 0:
            raise ValueError(
                f"hidden_size={self.hidden_size} 必须能被 num_heads="
                f"{self.num_heads} 整除"
            )
        try:
            self.mlp_ratio = float(mlp_ratio)
        except (TypeError, ValueError) as error:
            raise ValueError(f"mlp_ratio 必须为正有限数，实际为 {mlp_ratio!r}") from error
        if not math.isfinite(self.mlp_ratio) or self.mlp_ratio <= 0:
            raise ValueError(f"mlp_ratio 必须为正有限数，实际为 {mlp_ratio!r}")

        channels = tuple(condition_channels)
        if len(channels) != 4:
            raise ValueError("condition_channels 必须包含四个尺度")
        self.condition_channels = tuple(
            _positive_int(channel, f"condition_channels[{index}]")
            for index, channel in enumerate(channels)
        )
        self.gradient_checkpointing = bool(gradient_checkpointing)

        self.patch_embed = nn.Conv2d(
            self.in_channels,
            self.hidden_size,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        # 四个尺度独立投影到 hidden_size 后逐像素相加；条件分辨率可不同。
        self.cond_proj = nn.ModuleList(
            nn.Conv2d(channel, self.hidden_size, kernel_size=1)
            for channel in self.condition_channels
        )
        self.cond_patch_embed = nn.Conv2d(
            self.hidden_size,
            self.hidden_size,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

        time_hidden = self.hidden_size * 4
        self.time_embed = nn.Sequential(
            nn.Linear(self.hidden_size, time_hidden),
            nn.SiLU(),
            nn.Linear(time_hidden, self.hidden_size),
        )

        self.blocks = nn.ModuleList(
            DiTBlock(self.hidden_size, self.num_heads, self.mlp_ratio)
            for _ in range(self.depth)
        )
        self.final_layer = DiTFinalLayer(
            self.hidden_size, self.patch_size, self.out_channels
        )

    # ------------------------------------------------------------------
    # 输入整理
    # ------------------------------------------------------------------
    @staticmethod
    def _as_batch_timesteps(
        timesteps: Timestep, batch_size: int, device: torch.device
    ) -> Tensor:
        """把标量/张量时间步整理为 ``B`` 的 long 张量。"""
        if isinstance(timesteps, Tensor):
            t = timesteps.to(device=device, dtype=torch.long).reshape(-1)
            if t.numel() == 1:
                return t.expand(batch_size)
            if t.numel() != batch_size:
                raise ValueError("时间步数量与 batch size 不一致")
            return t
        return torch.full(
            (batch_size,), int(timesteps), device=device, dtype=torch.long
        )

    def _prepare_conditions(
        self,
        conditions: Sequence[Tensor],
        batch_size: int,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        """校验并融合四个尺度，返回 ``B x hidden x H x W`` 条件图。"""
        if isinstance(conditions, Tensor) or len(conditions) != 4:
            raise ValueError("conditions 必须包含四个尺度")
        fused: Optional[Tensor] = None
        for index, (condition, projection, expected_channels) in enumerate(
            zip(conditions, self.cond_proj, self.condition_channels)
        ):
            if not torch.is_tensor(condition):
                raise TypeError(
                    f"conditions[{index}] 必须是 Tensor，实际为 {type(condition)!r}"
                )
            if condition.ndim != 4:
                raise ValueError(
                    f"conditions[{index}] 应为 B x C x H x W，"
                    f"实际为 {tuple(condition.shape)}"
                )
            if int(condition.shape[0]) != batch_size:
                raise ValueError(
                    f"conditions[{index}] 的 batch size {int(condition.shape[0])} "
                    f"与输入的 {batch_size} 不一致"
                )
            if int(condition.shape[1]) != expected_channels:
                raise ValueError(
                    f"conditions[{index}] 的通道数 {int(condition.shape[1])} 与 "
                    f"condition_channels[{index}]={expected_channels} 不一致"
                )
            if condition.shape[-2] <= 0 or condition.shape[-1] <= 0:
                raise ValueError(f"conditions[{index}] 的空间尺寸必须为正")
            condition = condition.to(device=device, dtype=dtype)
            projected = projection(condition)
            if projected.shape[-2:] != (height, width):
                projected = F.interpolate(
                    projected,
                    size=(height, width),
                    mode="bilinear",
                    align_corners=False,
                )
            fused = projected if fused is None else fused + projected
        assert fused is not None  # len(conditions) == 4 已保证。
        return fused

    # ------------------------------------------------------------------
    # 前向
    # ------------------------------------------------------------------
    def _run_block(self, block: DiTBlock, x: Tensor, condition: Tensor) -> Tensor:
        """按需使用非重入梯度检查点。"""
        if self.gradient_checkpointing and torch.is_grad_enabled():
            return grad_checkpoint.checkpoint(
                block, x, condition, use_reentrant=False
            )
        return block(x, condition)

    def _unpatchify(self, tokens: Tensor, height: int, width: int) -> Tensor:
        """把 ``B x (h*w) x (p*p*C)`` token 还原为 ``B x C x H x W``。"""
        batch = tokens.shape[0]
        patch = self.patch_size
        grid_h = height // patch
        grid_w = width // patch
        tokens = tokens.reshape(
            batch, grid_h, grid_w, patch, patch, self.out_channels
        )
        # B x h x w x p x p x C -> B x C x h x p x w x p
        tokens = tokens.permute(0, 5, 1, 3, 2, 4)
        return tokens.reshape(batch, self.out_channels, height, width)

    def forward(
        self, x: Tensor, t: Timestep, conditions: Sequence[Tensor]
    ) -> Tensor:
        """预测噪声。

        Args:
            x: 带噪热图 ``B x in_channels x H x W``，支持任意矩形尺寸。
            t: 时间步，标量或长度 B 的张量。
            conditions: 四个尺度的条件特征；空间尺寸任意，会双线性对齐。

        Returns:
            与 ``x`` 完全同形状的噪声预测。
        """
        if x.ndim != 4:
            raise ValueError(f"x 应为 B x C x H x W，实际为 {tuple(x.shape)}")
        if int(x.shape[1]) != self.in_channels:
            raise ValueError(
                f"x 的通道数 {int(x.shape[1])} 与 in_channels="
                f"{self.in_channels} 不一致"
            )
        if int(x.shape[2]) <= 0 or int(x.shape[3]) <= 0:
            raise ValueError("x 的空间尺寸必须为正")

        batch_size, _, height, width = x.shape
        device = x.device
        dtype = x.dtype
        timesteps = self._as_batch_timesteps(t, batch_size, device)

        time_embedding = timestep_embedding(
            timesteps, self.hidden_size
        ).to(device=device, dtype=dtype)
        time_embedding = self.time_embed(time_embedding)

        condition_map = self._prepare_conditions(
            conditions, batch_size, height, width, device, dtype
        )

        patch = self.patch_size
        pad_h = (-int(height)) % patch
        pad_w = (-int(width)) % patch
        if pad_h or pad_w:
            x_padded = F.pad(x, (0, pad_w, 0, pad_h))
            condition_map = F.pad(condition_map, (0, pad_w, 0, pad_h))
        else:
            x_padded = x
        padded_h = int(height) + pad_h
        padded_w = int(width) + pad_w

        tokens = self.patch_embed(x_padded).flatten(2).transpose(1, 2)
        condition_tokens = (
            self.cond_patch_embed(condition_map).flatten(2).transpose(1, 2)
        )
        # 位置编码与 patch token 对齐 dtype（autocast 下 token 可能是 fp16）。
        position = build_2d_sincos_pos_embed(
            padded_h // patch,
            padded_w // patch,
            self.hidden_size,
            device=device,
            dtype=tokens.dtype,
        )
        tokens = tokens + condition_tokens + position.unsqueeze(0)

        for block in self.blocks:
            tokens = self._run_block(block, tokens, time_embedding)

        output = self.final_layer(tokens, time_embedding)
        output = self._unpatchify(output, padded_h, padded_w)
        if pad_h or pad_w:
            output = output[:, :, : int(height), : int(width)]
        return output


# 便捷别名：调用方可用 ``from diffpeak.models.dit import DiT``。
DiT = ConditionalDiT

__all__ = [
    "ConditionalDiT",
    "DiT",
    "DiTAttention",
    "DiTBlock",
    "DiTFinalLayer",
    "build_2d_sincos_pos_embed",
    "timestep_embedding",
]
