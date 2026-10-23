"""单通道有符号道路山脊场的条件高斯扩散与 DDIM 随机转移。

该模块同时服务于：

- 扩散预训练：``training_losses`` 返回扩散损失与山脊场重建损失；
- GRPO 采样与重算：``sample`` 记录每步状态、**原始高斯动作**和旧策略对数
  概率，``transition`` / ``log_prob`` 用于更新时重新计算概率。

新协议直接在有符号目标场 ``[-1, 1]`` 上扩散：

- 不使用旧的 ``[0, 1] -> [-1, 1]`` 仿射目标变换，也不在采样后做逆变换；
- 带噪状态 ``z_t`` 不限制在 ``[-1, 1]``，最终采样场在输出时裁剪到
  ``[-1, 1]``；裁剪发生在记录动作之后，因此 GRPO 始终以原始高斯样本计算
  轨迹概率；
- 不再使用依赖旧背景语义的采样均值补偿（背景先验）。
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor, nn

Timestep = Union[int, Tensor]


class GaussianDiffusion(nn.Module):
    """单通道有符号山脊场上的条件扩散过程。

    Args:
        timesteps: 训练时使用的扩散步数，默认 1000。
        sampling_steps: 推理/采样时的 DDIM 步数，默认 20。
        min_std: 每个转移（含末步）的高斯标准差下限，默认 0.02。
            该下限保证 GRPO 中对任意一步都能计算非退化的高斯对数概率；
            它是对策略分布的独立扰动，不改变 DDIM 均值中的系数。
        beta_start: 线性 beta 调度起点。
        beta_end: 线性 beta 调度终点。
        eta: DDIM 随机性系数；0 表示均值采用确定性 DDIM，仅由 ``min_std``
            提供策略方差。
        latent_channels: 潜变量通道数；单通道协议固定为 1。
        heatmap_size: 期望的热图边长元数据；``sample`` 的实际尺寸由
            ``conditions[0]`` 推断。
        heat_loss_max_timestep: 山脊场重建损失参与的最大时间步（不含）。
            默认使用前一半时间步；高噪声时间步的重建不稳定。
    """

    def __init__(
        self,
        timesteps: int = 1000,
        sampling_steps: int = 20,
        min_std: float = 0.02,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        eta: float = 0.0,
        latent_channels: int = 1,
        heatmap_size: int = 256,
        heat_loss_max_timestep: Optional[int] = None,
    ) -> None:
        super().__init__()
        if timesteps <= 0:
            raise ValueError("timesteps 必须为正")
        if sampling_steps <= 0 or sampling_steps > timesteps:
            raise ValueError("sampling_steps 必须在 [1, timesteps] 内")
        if not math.isfinite(min_std) or min_std <= 0:
            raise ValueError("min_std 必须为正有限数，以保证转移方差非零")
        if not math.isfinite(eta) or not 0.0 <= eta <= 1.0:
            raise ValueError("eta 必须是 [0, 1] 内的有限数")
        if not (0.0 < beta_start <= beta_end < 1.0):
            raise ValueError("beta 必须满足 0 < beta_start <= beta_end < 1")
        if int(latent_channels) != 1:
            raise ValueError(
                "单通道山脊场协议要求 latent_channels=1，"
                f"实际为 {latent_channels!r}"
            )

        self.timesteps = int(timesteps)
        self.sampling_steps = int(sampling_steps)
        self.min_std = float(min_std)
        self.eta = float(eta)
        self.latent_channels = 1
        self.heatmap_size = int(heatmap_size)
        self.heat_loss_max_timestep = (
            self.timesteps // 2
            if heat_loss_max_timestep is None
            else int(heat_loss_max_timestep)
        )
        if not 0 <= self.heat_loss_max_timestep <= self.timesteps:
            raise ValueError(
                "heat_loss_max_timestep 必须在 [0, timesteps] 内，"
                f"实际为 {self.heat_loss_max_timestep}"
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
        # 一步只走到 t=0，采样结果不会成为干净场。
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

    @staticmethod
    def _prepare_valid_mask(
        valid_mask: Optional[Tensor], reference: Tensor
    ) -> Tensor:
        """把有效区域掩码校验并整理为 ``B x 1 x H x W`` 的 bool 张量。

        接受 ``B x H x W`` 或 ``B x 1 x H x W``；``None`` 表示全部有效。
        非零值视为有效，padding 为零。
        """
        batch, _, height, width = reference.shape
        if valid_mask is None:
            return torch.ones(
                (batch, 1, height, width), dtype=torch.bool, device=reference.device
            )
        if not torch.is_tensor(valid_mask):
            raise TypeError(f"valid_mask 必须是张量，实际为 {type(valid_mask)!r}")
        mask = valid_mask
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
        if mask.ndim != 4:
            raise ValueError(
                "valid_mask 应为 B x 1 x H x W 或 B x H x W，"
                f"实际为 {tuple(mask.shape)}"
            )
        if int(mask.shape[0]) != batch or tuple(mask.shape[-2:]) != (height, width):
            raise ValueError(
                "valid_mask 的批次/空间尺寸必须与目标场一致："
                f"{tuple(mask.shape)} vs {(batch, height, width)}"
            )
        if int(mask.shape[1]) != 1:
            raise ValueError(
                f"valid_mask 通道数必须为 1，实际为 {int(mask.shape[1])}"
            )
        return (mask > 0).to(device=reference.device)

    @staticmethod
    def _prepare_field_regions(
        field_regions: Optional[Tensor], target: Tensor
    ) -> Tensor:
        """把四个互斥区域整理为 ``B x R x H x W`` 的 bool 张量。

        调用方缺失区域时使用 ``target > 0`` / ``target <= 0`` 两个区域作为
        兜底，避免大面积 ``-1`` 背景主导场重建损失。
        """
        batch, _, height, width = target.shape
        if field_regions is None:
            positive = target > 0
            return torch.cat((positive, ~positive), dim=1)
        if not torch.is_tensor(field_regions):
            raise TypeError(
                f"field_regions 必须是张量，实际为 {type(field_regions)!r}"
            )
        if field_regions.ndim != 4:
            raise ValueError(
                "field_regions 应为 B x 4 x H x W，"
                f"实际为 {tuple(field_regions.shape)}"
            )
        if (
            int(field_regions.shape[0]) != batch
            or int(field_regions.shape[1]) != 4
            or tuple(field_regions.shape[-2:]) != (height, width)
        ):
            raise ValueError(
                "field_regions 必须为 B x 4 x H x W 且与目标场对齐，"
                f"实际为 {tuple(field_regions.shape)}"
            )
        return (field_regions > 0).to(device=target.device)

    def _predict_eps_and_x0(
        self,
        dit: nn.Module,
        state: Tensor,
        timestep: Tensor,
        conditions: Sequence[Tensor],
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """由去噪器预测噪声，并恢复 ``[-1, 1]`` 范围的有符号干净场。"""
        # 去噪网络可由 autocast 加速；高维随机动作和转移系数保留 float32。
        state = state.float()
        eps = dit(state, timestep, conditions).float()
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
    def _field_loss(
        self,
        prediction: Tensor,
        target: Tensor,
        valid_mask: Tensor,
        field_regions: Optional[Tensor] = None,
        sample_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """按四个互斥区域分别归一化的有符号场重建损失。

        每个区域先在该区域有效像素上取均方误差，再对存在（非空）的区域取
        平均；缺失区域跳过，避免除零。有效掩码为零的样本不参与损失。

        Args:
            prediction: ``B x 1 x H x W`` 的预测场。
            target: 同形状的目标场，范围 ``[-1, 1]``。
            valid_mask: ``B x 1 x H x W`` 的 bool 有效区域掩码。
            field_regions: ``B x 4 x H x W`` 的互斥区域；``None`` 时回退到
                目标符号的两区域平衡。
            sample_mask: 可选的 ``B`` 样本权重（例如时间步筛选）；``None``
                表示全部样本参与。
        """
        batch = int(prediction.shape[0])
        regions = self._prepare_field_regions(field_regions, target)
        valid = valid_mask.to(device=prediction.device, dtype=prediction.dtype)
        error = (prediction - target) ** 2
        region_masks = regions.to(prediction.dtype) * valid
        counts = region_masks.sum(dim=(2, 3))
        # error 为 B x 1 x H x W，region_masks 为 B x R x H x W，
        # 广播后按区域求和。
        sums = (error * region_masks).sum(dim=(2, 3))
        per_region = sums / counts.clamp_min(1.0)
        present = counts > 0
        per_sample = (per_region * present.to(prediction.dtype)).sum(dim=1) / (
            present.sum(dim=1).clamp_min(1)
        )

        sample_valid = present.any(dim=1) & valid.reshape(batch, -1).any(dim=1)
        if sample_mask is not None:
            if not torch.is_tensor(sample_mask):
                raise TypeError(
                    f"sample_mask 必须是张量，实际为 {type(sample_mask)!r}"
                )
            weights = sample_mask.to(device=prediction.device).reshape(-1)
            if weights.numel() != batch:
                raise ValueError(
                    f"sample_mask 长度 {weights.numel()} 与 batch {batch} 不一致"
                )
            sample_valid = sample_valid & weights.bool()

        weight = sample_valid.to(prediction.dtype)
        return (per_sample * weight).sum() / weight.sum().clamp_min(1.0)

    def training_losses(
        self,
        dit: nn.Module,
        conditions: Sequence[Tensor],
        target_heatmaps: Tensor,
        return_prediction: bool = False,
        valid_mask: Optional[Tensor] = None,
        field_regions: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """计算扩散损失与单通道山脊场重建损失。

        Args:
            dit: 条件去噪器（默认多尺度 U-Net；保留参数名以兼容调用）。
            conditions: 四个尺度的条件特征。
            target_heatmaps: 目标有符号场 ``B x 1 x H x W``，范围 ``[-1, 1]``；
                直接作为扩散目标，不做仿射变换。
            return_prediction: 为 ``True`` 时在返回值中加入 ``"heat_pred"``，
                即由当前去噪器恢复的 ``[-1, 1]`` 有符号场。该张量保留计算图，
                可直接作为可微密集提示送入冻结的 SAM 提示编码器。
            valid_mask: 可选的有效区域掩码 ``B x 1 x H x W`` 或 ``B x H x W``；
                扩散 MSE 只在有效像素上平均，padding 不参与。
            field_regions: 可选的 ``B x 4 x H x W`` 互斥区域，用于平衡场重建
                损失；缺失区域跳过。未提供时按目标符号做两区域兜底。

        Returns:
            ``{"diff": 扩散 MSE, "heat": 场重建损失}``；当
            ``return_prediction=True`` 时额外包含 ``"heat_pred"``。
        """
        if target_heatmaps.ndim != 4 or target_heatmaps.shape[1] != self.latent_channels:
            raise ValueError(
                "target_heatmaps 应为 B x 1 x H x W 的有符号场，"
                f"实际为 {tuple(target_heatmaps.shape)}"
            )
        batch_size = target_heatmaps.shape[0]
        device = target_heatmaps.device
        dtype = target_heatmaps.dtype
        valid = self._prepare_valid_mask(valid_mask, target_heatmaps)
        valid_float = valid.to(dtype)

        t = torch.randint(0, self.timesteps, (batch_size,), device=device)
        noise = torch.randn_like(target_heatmaps)
        # 目标已经是有符号场：直接扩散，不做 [0, 1] -> [-1, 1] 变换。
        z0 = target_heatmaps
        alpha_bar = self._extract(self.alphas_cumprod, t, z0.shape).to(dtype)
        sqrt_alpha_bar = self._extract(self.sqrt_alphas_cumprod, t, z0.shape).to(dtype)
        sqrt_one_minus = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, z0.shape
        ).to(dtype)
        z_t = sqrt_alpha_bar * z0 + sqrt_one_minus * noise

        eps_pred = dit(z_t, t, conditions)
        squared_error = (eps_pred - noise) ** 2
        valid_broadcast = valid_float.expand_as(squared_error)
        diff_loss = (squared_error * valid_broadcast).sum() / valid_broadcast.sum().clamp_min(
            1.0
        )

        x0_pred = (z_t - sqrt_one_minus * eps_pred) / sqrt_alpha_bar.clamp_min(1e-12)
        heat_pred = x0_pred.clamp(-1.0, 1.0)

        # 高噪声时间步的重建不稳定，只在前一半时间步监督山脊场。
        time_mask = t < self.heat_loss_max_timestep
        heat_loss = self._field_loss(
            heat_pred,
            target_heatmaps,
            valid,
            field_regions=field_regions,
            sample_mask=time_mask,
        )

        losses = {"diff": diff_loss, "heat": heat_loss}
        if return_prediction:
            losses["heat_pred"] = heat_pred
        return losses

    # ------------------------------------------------------------------
    # 转移与对数概率
    # ------------------------------------------------------------------
    def transition(
        self,
        dit: nn.Module,
        state: Tensor,
        timestep: Timestep,
        previous_timestep: Timestep,
        conditions: Sequence[Tensor],
    ) -> Tuple[Tensor, Tensor]:
        """一次 DDIM 随机转移的高斯均值与标准差。

        Args:
            dit: 条件去噪器（默认多尺度 U-Net；保留参数名以兼容调用）。
            state: 当前带噪场 ``z_t``，不限制在 ``[-1, 1]``。
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
            ``min_std`` 提供的额外噪声是 GRPO 的策略探索扰动，不属于
            DDIM 边缘分布。其余步若名义 ``sigma < min_std`` 同理。
        """
        if state.ndim != 4:
            raise ValueError(f"state 应为 B x C x H x W，实际为 {tuple(state.shape)}")
        batch_size = state.shape[0]
        device = state.device
        t = self._as_batch_timesteps(timestep, batch_size, device)
        previous = self._as_batch_timesteps(previous_timestep, batch_size, device)

        eps, x0, alpha_t = self._predict_eps_and_x0(dit, state, t, conditions)

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

        使用 float64 计算，避免单通道 256x256 维动作求和时的数值误差。返回值
        形状为 ``B``；若输入无 batch 维则返回标量。
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
    def _sample_impl(
        self,
        dit: nn.Module,
        conditions: Sequence[Tensor],
        generator: Optional[torch.Generator],
        record: bool,
    ) -> Union[Tensor, Tuple[Tensor, List[Dict[str, Any]]]]:
        """采样实现；梯度开关由调用方（``sample`` / ``sample_differentiable``）控制。"""
        if len(conditions) != 4:
            raise ValueError("conditions 必须包含四个尺度")
        reference = conditions[0]
        batch_size, _, height, width = reference.shape
        device, dtype = reference.device, torch.float32

        # 从纯噪声开始；新协议没有背景均值补偿。
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

            mean, std = self.transition(dit, state, t, prev, conditions)
            noise = self._randn(state.shape, device, dtype, generator)
            action = mean + std * noise

            if record:
                old_log_prob = self.log_prob(action, mean, std)
                # 记录**裁剪前**的原始高斯动作；终端裁剪只影响返回场。
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

        # 单通道有符号场：只做范围裁剪，不做 [−1, 1] -> [0, 1] 逆变换。
        heatmaps = state.clamp(-1.0, 1.0)
        if record:
            return heatmaps, records
        return heatmaps

    def sample(
        self,
        dit: nn.Module,
        conditions: Sequence[Tensor],
        generator: Optional[torch.Generator] = None,
        record: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, List[Dict[str, Any]]]]:
        """DDIM 随机采样单通道有符号场（不构建计算图）。

        Args:
            dit: 条件去噪器（默认多尺度 U-Net；保留参数名以兼容调用）。
            conditions: 四个尺度的条件特征。
            generator: 可选随机数生成器；允许 CPU 生成器配合 GPU 条件。
            record: 是否记录每步的 GRPO 轨迹信息。

        Returns:
            ``record=False`` 时返回 ``B x 1 x H x W`` 的有符号场，范围
            ``[-1, 1]``；``record=True`` 时返回 ``(heatmaps, records)``。records
            中每项包含 ``state``、``action``、``timestep``、
            ``previous_timestep``、``old_log_prob``，其中张量均已移到 CPU 以
            节省显存。``action`` 是终端裁剪前的原始高斯样本。

            采样从 ``timesteps-1`` 开始，最后一步的 ``previous_timestep`` 为
            ``-1``，因此返回的场对应 ``alpha_bar=1`` 的干净端点；末步仍会
            加入 ``min_std`` 的策略探索噪声，符合 GRPO 的非零方差要求。
        """
        with torch.no_grad():
            return self._sample_impl(dit, conditions, generator, record)

    def sample_differentiable(
        self,
        dit: nn.Module,
        conditions: Sequence[Tensor],
        generator: Optional[torch.Generator] = None,
        record: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, List[Dict[str, Any]]]]:
        """与 :meth:`sample` 完全相同的 DDIM 采样，但保留计算图。

        用于联合 ABC 端到端训练：采样场作为冻结 SAM 提示编码器的可微密集
        输入，梯度可以经冻结的提示编码器回传到 DiT 与条件编码器。除不包裹
        ``torch.no_grad()`` 外，随机数消耗、调度和返回值与 :meth:`sample`
        逐位一致；固定 ``generator`` 时两者输出相同。
        """
        return self._sample_impl(dit, conditions, generator, record)
