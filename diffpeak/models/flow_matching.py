"""条件线性 Flow Matching 与可重放的随机积分策略。

时间从噪声端0走向数据端1，监督速度 H* - epsilon。生成采用 Euler/Heun
积分均值及 policy_std * sqrt(dt) 的独立高斯探索，包含最后一步。
这是为 GRPO 显式定义的随机策略，不声称保持确定性 ODE 的边缘分布。
ABC 可微生成、D rollout/replay 以及推理使用同一转移；只在终端裁剪。
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor, nn

from .diffusion import GaussianDiffusion


class ConditionalFlowMatching(nn.Module):
    """图像条件速度回归及带高斯探索的连续时间积分。

    ``time_scale`` 是网络时间编码的契约元数据，网络接收的仍是[0,1]浮点
    时间，由 ConditionalUNet 内部缩放；积分器不将时间量化或重复缩放。
    保留 transition 的 previous_timestep 参数名以兼容轨迹消费者，但该参数
    在此表示更大的下一端点，而非 DDPM 的前一时间步。
    """

    # 复用与扩散公式无关的掩码、区域归一化、随机数和高斯密度工具。
    _prepare_valid_mask = staticmethod(GaussianDiffusion._prepare_valid_mask)
    _prepare_field_regions = staticmethod(GaussianDiffusion._prepare_field_regions)
    _field_loss = GaussianDiffusion._field_loss
    _randn = GaussianDiffusion._randn
    log_prob = GaussianDiffusion.log_prob

    def __init__(
        self, sampling_steps: int = 20, policy_std: float = 0.02,
        latent_channels: int = 1, heatmap_size: int = 256,
        time_scale: float = 1000.0, field_loss_min_time: float = 0.5,
        solver: str = "heun",
    ) -> None:
        super().__init__()
        if isinstance(sampling_steps, bool) or not isinstance(sampling_steps, int) or sampling_steps <= 0:
            raise ValueError("sampling_steps 必须为正整数")
        if latent_channels != 1:
            raise ValueError("单通道山脊场要求 latent_channels=1")
        if not math.isfinite(policy_std) or policy_std <= 0:
            raise ValueError("policy_std 必须为正有限数，保证 GRPO 的高斯转移非退化")
        if not math.isfinite(time_scale) or time_scale <= 0:
            raise ValueError("time_scale 必须为正有限数")
        if not math.isfinite(field_loss_min_time) or not 0 <= field_loss_min_time <= 1:
            raise ValueError("field_loss_min_time 必须在[0,1]内")
        if solver not in ("euler", "heun"):
            raise ValueError("solver 只支持 euler 或 heun")
        self.sampling_steps = int(sampling_steps)
        self.policy_std = float(policy_std)
        self.latent_channels = 1
        self.heatmap_size = int(heatmap_size)
        self.time_scale = float(time_scale)
        self.field_loss_min_time = float(field_loss_min_time)
        self.solver = solver
        self.register_buffer("flow_signature", torch.tensor(1, dtype=torch.long))
        self.register_buffer("policy_std_buffer", torch.tensor(policy_std, dtype=torch.float64))
        self.register_buffer("time_scale_buffer", torch.tensor(time_scale, dtype=torch.float64))
        self.register_buffer("field_loss_min_time_buffer", torch.tensor(field_loss_min_time, dtype=torch.float64))
        self.register_buffer("solver_id", torch.tensor(int(solver == "heun"), dtype=torch.long))
        self.register_buffer("sampling_steps_buffer", torch.tensor(self.sampling_steps, dtype=torch.long))
        self.register_buffer("sampling_schedule", torch.linspace(0, 1, self.sampling_steps + 1), persistent=False)

    @staticmethod
    def _batch_time(time: Union[float, Tensor], state: Tensor) -> Tensor:
        result = torch.as_tensor(time, device=state.device, dtype=torch.float32).reshape(-1)
        if result.numel() == 1:
            result = result.expand(state.shape[0])
        if result.numel() != state.shape[0]:
            raise ValueError("时间数量与 batch size 不一致")
        if not bool(torch.isfinite(result).all()) or bool(((result < 0) | (result > 1)).any()):
            raise ValueError("Flow Matching 时间必须为[0,1]内有限浮点数")
        return result

    def training_losses(
        self, unet: nn.Module, conditions: Sequence[Tensor], target_heatmaps: Tensor,
        return_prediction: bool = False, valid_mask: Optional[Tensor] = None,
        field_regions: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        if target_heatmaps.ndim != 4 or target_heatmaps.shape[1] != 1:
            raise ValueError("target_heatmaps 必须为 B x 1 x H x W")
        target = target_heatmaps.float()
        valid = self._prepare_valid_mask(valid_mask, target)
        time = torch.rand(target.shape[0], device=target.device, dtype=torch.float32)
        blend = time[:, None, None, None]
        noise = torch.randn_like(target)
        state = (1 - blend) * noise + blend * target
        velocity = unet(state, time, conditions).float()
        error = (velocity - (target - noise)).square()
        flow_loss = (error * valid).sum() / valid.sum().clamp_min(1)
        # 线性路径的一步终端估计；场监督在未裁剪估计上计算，避免饱和死梯度。
        estimate = state + (1 - blend) * velocity
        field_loss = self._field_loss(
            estimate, target, valid, field_regions=field_regions,
            sample_mask=time >= self.field_loss_min_time,
        )
        losses = {"flow": flow_loss, "heat": field_loss}
        if return_prediction:
            losses["heat_pred"] = estimate.clamp(-1, 1)
        return losses

    def transition(
        self, unet: nn.Module, state: Tensor,
        timestep: Union[float, Tensor], previous_timestep: Union[float, Tensor],
        conditions: Sequence[Tensor],
    ) -> Tuple[Tensor, Tensor]:
        if state.ndim != 4 or state.shape[1] != 1:
            raise ValueError("state 必须为 B x 1 x H x W")
        state = state.float()
        time = self._batch_time(timestep, state)
        next_time = self._batch_time(previous_timestep, state)
        if bool((next_time <= time).any()):
            raise ValueError("Flow Matching 转移要求 0 <= t < next_t <= 1")
        dt = (next_time - time)[:, None, None, None]
        velocity = unet(state, time, conditions).float()
        mean = state + dt * velocity
        if self.solver == "heun":
            next_velocity = unet(mean, next_time, conditions).float()
            mean = state + 0.5 * dt * (velocity + next_velocity)
        std = (self.policy_std * dt.sqrt()).expand_as(state)
        return mean, std

    def _sample_impl(
        self, unet: nn.Module, conditions: Sequence[Tensor],
        generator: Optional[torch.Generator], record: bool,
    ) -> Union[Tensor, Tuple[Tensor, List[Dict[str, Any]]]]:
        if len(conditions) != 4:
            raise ValueError("conditions 必须包含四个尺度")
        reference = conditions[0]
        batch, _, height, width = reference.shape
        state = self._randn(
            torch.Size((batch, 1, height, width)), reference.device,
            torch.float32, generator,
        )
        records: List[Dict[str, Any]] = []
        schedule = self.sampling_schedule.tolist()
        for start, end in zip(schedule[:-1], schedule[1:]):
            mean, std = self.transition(unet, state, start, end, conditions)
            action = mean + std * self._randn(state.shape, state.device, state.dtype, generator)
            if record:
                records.append({
                    "state": state.detach().cpu(), "action": action.detach().cpu(),
                    "timestep": float(start), "previous_timestep": float(end),
                    "old_log_prob": self.log_prob(action, mean, std).detach().cpu(),
                })
            state = action
        result = state.clamp(-1, 1)
        return (result, records) if record else result

    def sample(
        self, unet: nn.Module, conditions: Sequence[Tensor],
        generator: Optional[torch.Generator] = None, record: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, List[Dict[str, Any]]]]:
        with torch.no_grad():
            return self._sample_impl(unet, conditions, generator, record)

    def sample_differentiable(
        self, unet: nn.Module, conditions: Sequence[Tensor],
        generator: Optional[torch.Generator] = None, record: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, List[Dict[str, Any]]]]:
        return self._sample_impl(unet, conditions, generator, record)


__all__ = ["ConditionalFlowMatching"]
