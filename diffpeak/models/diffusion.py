"""条件高斯扩散与 DDIM 随机转移。

该模块同时服务于：
- 扩散预训练：``training_losses`` 返回扩散损失与热图重建损失；
- DDPO 采样与重算：``sample`` 记录每步状态、动作和旧策略对数概率，
  ``transition`` / ``log_prob`` 用于更新时重新计算概率。

热图映射只使用固定仿射变换 ``z0 = 2H - 1`` 与 ``H = clip((z0 + 1) / 2)``，
不做任何逐图 min-max 归一化或通道间 Softmax。
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn

Timestep = Union[int, Tensor]


class GaussianDiffusion(nn.Module):
    """双通道热图上的条件扩散过程。

    Args:
        timesteps: 训练时使用的扩散步数，默认 1000。
        sampling_steps: 推理/采样时的 DDIM 步数，默认 20。
        min_std: 每个转移（含末步）的高斯标准差下限，默认 0.02。
            该下限保证 DDPO 中对任意一步都能计算非退化的高斯对数概率；
            它是对策略分布的独立扰动，不改变 DDIM 均值中的系数。
        beta_start: 线性 beta 调度起点。
        beta_end: 线性 beta 调度终点。
        eta: DDIM 随机性系数；0 表示均值采用确定性 DDIM，仅由 ``min_std``
            提供策略方差。
        latent_channels: 热图潜变量通道数，默认 2。
        heatmap_size: 期望的热图边长元数据；``sample`` 的实际尺寸由
            ``conditions[0]`` 推断。
        heat_loss_max_timestep: 热图重建损失参与的最大时间步（不含）。
            默认使用前一半时间步。
        peak_threshold: 构造峰值邻域掩码的热图阈值。
        peak_loss_weight: 峰值邻域损失在热图重建损失中的权重。
    """

    def __init__(
        self,
        timesteps: int = 1000,
        sampling_steps: int = 20,
        min_std: float = 0.02,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        eta: float = 0.0,
        latent_channels: int = 2,
        heatmap_size: int = 256,
        heat_loss_max_timestep: Optional[int] = None,
        peak_threshold: float = 0.1,
        peak_loss_weight: float = 0.5,
    ) -> None:
        super().__init__()
        if timesteps <= 0:
            raise ValueError("timesteps 必须为正")
        if sampling_steps <= 0 or sampling_steps > timesteps:
            raise ValueError("sampling_steps 必须在 [1, timesteps] 内")
        if min_std <= 0:
            raise ValueError("min_std 必须为正，以保证转移方差非零")

        self.timesteps = int(timesteps)
        self.sampling_steps = int(sampling_steps)
        self.min_std = float(min_std)
        self.eta = float(eta)
        self.latent_channels = int(latent_channels)
        self.heatmap_size = int(heatmap_size)
        self.peak_threshold = float(peak_threshold)
        self.peak_loss_weight = float(peak_loss_weight)
        self.heat_loss_max_timestep = (
            self.timesteps // 2
            if heat_loss_max_timestep is None
            else int(heat_loss_max_timestep)
        )

        betas = torch.linspace(beta_start, beta_end, self.timesteps, dtype=torch.float32)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer(
            "sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod), persistent=False
        )
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod), persistent=False
        )

        # DDIM 的 current 端点从 timesteps-1 递减到 0，保留 sampling_steps
        # 个；末尾追加 -1 表示 alpha_bar=1 的干净端点。若仍以 0 结尾，最后
        # 一步只走到 t=0，采样结果不会成为干净热图。
        #
        # sampling_steps=timesteps 时直接使用完整序列，避免 linspace 中出现
        # 重复时间步；其余情况取 sampling_steps+1 等分点的前
        # sampling_steps 个，最后一步由 current -> -1 完成。
        if self.sampling_steps == self.timesteps:
            current_timesteps = torch.arange(
                self.timesteps - 1, -1, -1, dtype=torch.long
            )
        else:
            current_timesteps = torch.linspace(
                0,
                self.timesteps - 1,
                self.sampling_steps + 1,
                dtype=torch.float32,
            ).round().long().flip(0)[:-1]
        schedule = torch.cat(
            [current_timesteps, torch.tensor([-1], dtype=torch.long)]
        )
        self.register_buffer("sampling_schedule", schedule, persistent=False)

    # ------------------------------------------------------------------
    # 工具函数
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
        return torch.full((batch_size,), int(timesteps), device=device, dtype=torch.long)

    @staticmethod
    def _extract(buffer: Tensor, timesteps: Tensor, shape: torch.Size) -> Tensor:
        """按 batch 时间步取出调度系数并广播到 ``shape``。"""
        index = timesteps.clamp(0, buffer.numel() - 1)
        values = buffer.to(device=timesteps.device).gather(0, index)
        return values.reshape(values.shape[0], *([1] * (len(shape) - 1)))

    def _predict_eps_and_x0(
        self,
        unet: nn.Module,
        state: Tensor,
        timestep: Tensor,
        conditions: Sequence[Tensor],
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """由去噪器预测噪声，并恢复 ``[-1, 1]`` 范围的干净热图潜变量。"""
        # 网络卷积可由 autocast 加速；高维随机动作和转移系数保留 float32。
        state = state.float()
        eps = unet(state, timestep, conditions).float()
        alpha_bar = self._extract(self.alphas_cumprod, timestep, state.shape).to(state.dtype)
        sqrt_alpha = self._extract(
            self.sqrt_alphas_cumprod, timestep, state.shape
        ).to(state.dtype)
        x0 = (state - (1.0 - alpha_bar).clamp_min(0).sqrt() * eps) / sqrt_alpha.clamp_min(1e-12)
        return eps, x0.clamp(-1.0, 1.0), alpha_bar

    def _randn(
        self,
        shape: torch.Size,
        device: torch.device,
        dtype: torch.dtype,
        generator: Optional[torch.Generator],
    ) -> Tensor:
        """生成随机噪声；支持 CPU 生成器配合 GPU 张量。"""
        if generator is None:
            return torch.randn(shape, device=device, dtype=dtype)
        noise = torch.randn(shape, device=generator.device, dtype=dtype, generator=generator)
        return noise.to(device)

    # ------------------------------------------------------------------
    # 训练损失
    # ------------------------------------------------------------------
    def training_losses(
        self,
        unet: nn.Module,
        conditions: Sequence[Tensor],
        target_heatmaps: Tensor,
    ) -> Dict[str, Tensor]:
        """计算扩散损失与热图重建损失。

        Args:
            unet: 条件 U-Net。
            conditions: 四个尺度的条件特征。
            target_heatmaps: 目标双通道热图 ``B x 2 x 256 x 256``，范围 ``[0, 1]``。

        Returns:
            ``{"diff": 扩散 MSE, "heat": 热图重建损失}``。
        """
        if target_heatmaps.ndim != 4 or target_heatmaps.shape[1] != 2:
            raise ValueError(
                f"target_heatmaps 应为 B x 2 x H x W，实际为 {tuple(target_heatmaps.shape)}"
            )
        batch_size = target_heatmaps.shape[0]
        device = target_heatmaps.device
        dtype = target_heatmaps.dtype

        t = torch.randint(0, self.timesteps, (batch_size,), device=device)
        noise = torch.randn_like(target_heatmaps)
        z0 = target_heatmaps * 2.0 - 1.0
        alpha_bar = self._extract(self.alphas_cumprod, t, z0.shape).to(dtype)
        sqrt_alpha_bar = self._extract(self.sqrt_alphas_cumprod, t, z0.shape).to(dtype)
        sqrt_one_minus = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, z0.shape
        ).to(dtype)
        z_t = sqrt_alpha_bar * z0 + sqrt_one_minus * noise

        eps_pred = unet(z_t, t, conditions)
        diff_loss = F.mse_loss(eps_pred, noise)

        # 固定仿射恢复干净热图，不做 min-max。
        x0_pred = (z_t - (1.0 - alpha_bar).clamp_min(0).sqrt() * eps_pred) / alpha_bar.sqrt().clamp_min(
            1e-12
        )
        heat_pred = ((x0_pred + 1.0) * 0.5).clamp(0.0, 1.0)

        error = (heat_pred - target_heatmaps) ** 2
        peak_mask = (target_heatmaps > self.peak_threshold).to(dtype)
        background_mask = 1.0 - peak_mask

        reduce_dims = (1, 2, 3)
        peak_count = peak_mask.sum(dim=reduce_dims)
        background_count = background_mask.sum(dim=reduce_dims)
        peak_loss = (error * peak_mask).sum(dim=reduce_dims) / peak_count.clamp_min(1.0)
        background_loss = (error * background_mask).sum(dim=reduce_dims) / background_count.clamp_min(
            1.0
        )
        weighted = (
            self.peak_loss_weight * peak_loss
            + (1.0 - self.peak_loss_weight) * background_loss
        )
        # 无峰样本（例如无道路影像）只使用背景损失，避免峰值项稀释。
        per_sample_heat = torch.where(peak_count > 0, weighted, background_loss)

        # 高噪声时间步的重建不稳定，只在前一半时间步监督热图。
        time_weight = (t < self.heat_loss_max_timestep).to(per_sample_heat.dtype)
        heat_loss = (per_sample_heat * time_weight).sum() / time_weight.sum().clamp_min(1.0)

        return {"diff": diff_loss, "heat": heat_loss}

    # ------------------------------------------------------------------
    # 转移与对数概率
    # ------------------------------------------------------------------
    def transition(
        self,
        unet: nn.Module,
        state: Tensor,
        timestep: Timestep,
        previous_timestep: Timestep,
        conditions: Sequence[Tensor],
    ) -> Tuple[Tensor, Tensor]:
        """一次 DDIM 随机转移的高斯均值与标准差。

        Args:
            unet: 条件 U-Net。
            state: 当前带噪热图 ``z_t``。
            timestep: 当前时间步 ``t``（标量或长度 B 的张量）。
            previous_timestep: 上一时间步；允许 -1，此时视为 ``alpha_bar=1``。
            conditions: 四个尺度的条件特征。

        Returns:
            ``(mean, std)``，形状均与 ``state`` 相同；``std >= min_std``。

        Notes:
            采用标准 DDIM 随机转移

            ``mean = sqrt(alpha_prev) * x0 + sqrt(1 - alpha_prev - sigma^2) * eps``

            其中 ``sigma`` 是 eta 给出的**名义** DDIM 标准差，而不是被
            ``min_std`` 抬高后的值。末步 ``previous_timestep=-1`` 时
            ``alpha_prev=1``、名义 ``sigma=0``，均值退化为 ``x0``；此时
            ``min_std`` 提供的额外噪声是 DDPO 的策略探索扰动，不属于
            DDIM 边缘分布。其余步若名义 ``sigma < min_std`` 同理。
        """
        if state.ndim != 4:
            raise ValueError(f"state 应为 B x C x H x W，实际为 {tuple(state.shape)}")
        batch_size = state.shape[0]
        device = state.device
        t = self._as_batch_timesteps(timestep, batch_size, device)
        previous = self._as_batch_timesteps(previous_timestep, batch_size, device)

        eps, x0, alpha_t = self._predict_eps_and_x0(unet, state, t, conditions)

        alpha_prev = self._extract(self.alphas_cumprod, previous, state.shape).to(state.dtype)
        alpha_prev = torch.where(
            (previous < 0).reshape(batch_size, 1, 1, 1),
            torch.ones_like(alpha_prev),
            alpha_prev,
        )

        if self.eta > 0:
            variance = ((1.0 - alpha_prev) / (1.0 - alpha_t).clamp_min(1e-12)) * (
                1.0 - alpha_t / alpha_prev.clamp_min(1e-12)
            )
            sigma_nominal = self.eta * variance.clamp_min(0.0).sqrt()
        else:
            sigma_nominal = torch.zeros_like(alpha_t)

        # 名义 sigma 只进入 DDIM 均值系数；采样标准差再取 min_std 下限。
        mean_coef = (
            1.0 - alpha_prev - sigma_nominal * sigma_nominal
        ).clamp_min(0.0).sqrt()
        mean = alpha_prev.sqrt() * x0 + mean_coef * eps
        std = sigma_nominal.clamp_min(self.min_std).expand_as(state)
        return mean, std

    def log_prob(self, action: Tensor, mean: Tensor, std: Tensor) -> Tensor:
        """计算逐样本高斯对数概率，并对所有动作维度求和（不取平均）。

        使用 float64 计算，避免 2x256x256 维动作求和时的数值误差。返回值形状
        为 ``B``；若输入无 batch 维则返回标量。
        """
        action64 = action.to(torch.float64)
        mean64 = mean.to(torch.float64)
        # 方差下限由 ``transition`` 保证；这里只用极小值避免除零。
        std64 = std.to(torch.float64).clamp_min(1e-12)
        standardized = (action64 - mean64) / std64
        elementwise = (
            -0.5 * standardized * standardized
            - torch.log(std64)
            - 0.5 * math.log(2.0 * math.pi)
        )
        if elementwise.dim() == 0:
            return elementwise
        if elementwise.dim() < 4:
            return elementwise.sum()
        return elementwise.reshape(elementwise.shape[0], -1).sum(dim=1)

    # ------------------------------------------------------------------
    # 采样
    # ------------------------------------------------------------------
    @torch.no_grad()
    def sample(
        self,
        unet: nn.Module,
        conditions: Sequence[Tensor],
        generator: Optional[torch.Generator] = None,
        record: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, List[Dict[str, Any]]]]:
        """DDIM 随机采样双通道热图。

        Args:
            unet: 条件 U-Net。
            conditions: 四个尺度的条件特征。
            generator: 可选随机数生成器；允许 CPU 生成器配合 GPU 条件。
            record: 是否记录每步的 DDPO 轨迹信息。

        Returns:
            ``record=False`` 时返回 ``B x 2 x 256 x 256`` 热图；
            ``record=True`` 时返回 ``(heatmaps, records)``。records 中每项包含
            ``state``、``action``、``timestep``、``previous_timestep``、
            ``old_log_prob``，其中张量均已移到 CPU 以节省显存。

            采样从 ``timesteps-1`` 开始，最后一步的 ``previous_timestep`` 为
            ``-1``，因此返回的热图对应 ``alpha_bar=1`` 的干净端点；末步仍会
            加入 ``min_std`` 的策略探索噪声，符合 DDPO 的非零方差要求。
        """
        if len(conditions) != 4:
            raise ValueError("conditions 必须包含四个尺度")
        reference = conditions[0]
        batch_size, _, height, width = reference.shape
        device, dtype = reference.device, torch.float32

        state = self._randn(
            (batch_size, self.latent_channels, height, width), device, dtype, generator
        )
        records: List[Dict[str, Any]] = []
        # 调度表是静态的，转成 Python 列表可避免每步 GPU 同步。
        schedule = self.sampling_schedule.tolist()

        for index in range(self.sampling_steps):
            current = int(schedule[index])
            previous = int(schedule[index + 1])
            t = torch.full((batch_size,), current, device=device, dtype=torch.long)
            prev = torch.full((batch_size,), previous, device=device, dtype=torch.long)

            mean, std = self.transition(unet, state, t, prev, conditions)
            noise = self._randn(state.shape, device, dtype, generator)
            action = mean + std * noise

            if record:
                old_log_prob = self.log_prob(action, mean, std)
                records.append(
                    {
                        "state": state.detach().to("cpu"),
                        "action": action.detach().to("cpu"),
                        "timestep": current,
                        "previous_timestep": previous,
                        "old_log_prob": old_log_prob.detach().to("cpu"),
                    }
                )
            state = action

        heatmaps = ((state + 1.0) * 0.5).clamp(0.0, 1.0)
        if record:
            return heatmaps, records
        return heatmaps
