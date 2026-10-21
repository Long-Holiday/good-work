"""RTX 3050 小型 GPU smoke test（阶段 B 与阶段 D）。

只依赖 ``torch`` 与 ``diffpeak`` 的 models/losses 模块；不导入 inference/engine，
不加载 SAM 权重（用随机张量模拟 SAM 图像嵌入），不安装或下载任何东西。

运行::

    conda run -n pytorch python scripts/gpu_smoke.py

阶段 B：128x128 输入 -> ConditionEncoder 多尺度条件 -> ConditionalUNet +
GaussianDiffusion.training_losses + auxiliary_loss，AMP float16 + GradScaler
前向/反向，检查损失、梯度有限且参数确实更新。

阶段 D：U-Net eval、条件 detach，固定 CUDA generator 采样 G=2 轨迹
（timesteps=32, steps=4），检查记录 dtype、逐步重算 log-prob 的 ratio≈1，
再用固定优势 ±1 做 PPO + KL + 扩散 anchor 组合反向，检查梯度有限、参数更新。

显存只报告当前设备（4 GiB RTX 3050 Laptop）上的实测峰值，不涉及其他显卡。
"""

from __future__ import annotations

import copy
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diffpeak.losses import auxiliary_loss  # noqa: E402
from diffpeak.models.conditioning import ConditionEncoder  # noqa: E402
from diffpeak.models.diffusion import GaussianDiffusion  # noqa: E402
from diffpeak.models.unet import ConditionalUNet  # noqa: E402

DEVICE = torch.device("cuda")
IMAGE_SIZE = 128
EMBED_SIZE = 8
BATCH = 1
GROUP = 2
TIMESTEPS = 32
SAMPLING_STEPS = 4
SEED = 1234


def autocast():
    """统一的 CUDA float16 autocast 上下文。"""
    return torch.autocast(device_type="cuda", dtype=torch.float16)


def mib(value: int) -> float:
    return round(int(value) / (1024.0 * 1024.0), 2)


def all_finite(tensors: Sequence[torch.Tensor]) -> bool:
    return all(bool(torch.isfinite(t).all()) for t in tensors)


def snapshot(params: Sequence[torch.nn.Parameter]) -> List[torch.Tensor]:
    return [p.detach().clone() for p in params]


def changed_elements(before: Sequence[torch.Tensor], params: Sequence[torch.nn.Parameter]) -> int:
    return sum(int((old != new.detach()).sum()) for old, new in zip(before, params))


def grad_stats(params: Sequence[torch.nn.Parameter]) -> Dict[str, Any]:
    """统计缺失/非有限梯度数量与全局梯度范数（不修改 .grad）。"""
    missing = 0
    nonfinite = 0
    squared = 0.0
    for param in params:
        if param.grad is None:
            missing += 1
            continue
        grad = param.grad.detach()
        if not bool(torch.isfinite(grad).all()):
            nonfinite += 1
        squared += float(grad.double().pow(2).sum().item())
    return {
        "missing": missing,
        "nonfinite": nonfinite,
        "global_norm": round(math.sqrt(squared), 6),
    }


def build_targets() -> Dict[str, torch.Tensor]:
    """构造含细线的原始 128x128 掩码；热图池化到 32x32，道路/骨架保留原分辨率。"""
    yy, xx = torch.meshgrid(
        torch.arange(IMAGE_SIZE, device=DEVICE),
        torch.arange(IMAGE_SIZE, device=DEVICE),
        indexing="ij",
    )
    horizontal = (yy == 64) & (xx >= 8) & (xx < 120)
    diagonal = (yy == xx) & (xx >= 16) & (xx < 112)
    skeleton = (horizontal | diagonal).to(torch.float32).view(1, 1, IMAGE_SIZE, IMAGE_SIZE)
    # 3x3 max pooling 得到约 3 像素宽的道路区域，细线骨架单独保留。
    road = F.max_pool2d(skeleton, kernel_size=3, stride=1, padding=1).clamp(0.0, 1.0)

    heatmaps = torch.cat([skeleton, road * 0.5], dim=1)
    # 与 diffusion.sample 的潜变量空间（conditions[0] 的 32x32）保持一致；
    # 128x128 仅用于辅助头的原始分辨率监督。
    heatmaps = F.adaptive_max_pool2d(
        heatmaps, (IMAGE_SIZE // 4, IMAGE_SIZE // 4)
    )
    assert list(heatmaps.shape) == [BATCH, 2, IMAGE_SIZE // 4, IMAGE_SIZE // 4], tuple(
        heatmaps.shape
    )
    return {
        "heatmaps": heatmaps,
        "road": road,
        "skeleton": skeleton,
    }


def phase_b(
    encoder: ConditionEncoder,
    unet: ConditionalUNet,
    diffusion: GaussianDiffusion,
    images: torch.Tensor,
    embedding: torch.Tensor,
    targets: Dict[str, torch.Tensor],
) -> Dict[str, Any]:
    """阶段 B：条件编码 + 扩散/热图/辅助损失 + AMP 反向与参数更新。"""
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()

    encoder.train()
    unet.train()
    params = list(encoder.parameters()) + list(unet.parameters())
    optimizer = torch.optim.Adam(params, lr=2e-3)
    scaler = torch.amp.GradScaler("cuda")

    with autocast():
        conditions, aux_logits = encoder(images, embedding)
        losses = diffusion.training_losses(unet, conditions, targets["heatmaps"])
        aux = auxiliary_loss(aux_logits, targets["road"], targets["skeleton"])
        total = losses["diff"] + losses["heat"] + 0.5 * aux

    condition_shapes = [list(c.shape) for c in conditions]
    assert condition_shapes == [
        [BATCH, 64, 32, 32],
        [BATCH, 128, 16, 16],
        [BATCH, 256, 8, 8],
        [BATCH, 256, 4, 4],
    ], f"条件形状不符合预期: {condition_shapes}"
    assert list(aux_logits.shape) == [BATCH, 2, 32, 32], tuple(aux_logits.shape)
    assert all_finite([*conditions, aux_logits]), "条件特征或辅助 logits 出现非有限值"
    assert all_finite([losses["diff"], losses["heat"], aux, total]), "阶段 B 损失非有限"
    assert total.requires_grad, "阶段 B 总损失未连接计算图"

    before = snapshot(params)
    optimizer.zero_grad(set_to_none=True)
    scaler.scale(total.float()).backward()
    scaler.unscale_(optimizer)
    stats = grad_stats(params)
    assert stats["missing"] == 0, f"阶段 B 有 {stats['missing']} 个参数没有梯度"
    assert stats["nonfinite"] == 0, f"阶段 B 有 {stats['nonfinite']} 个参数梯度非有限"
    assert stats["global_norm"] > 0.0, "阶段 B 梯度范数为 0"

    scaler.step(optimizer)
    scaler.update()
    changed = changed_elements(before, params)
    finite_after = all_finite(params)
    assert finite_after, "阶段 B 更新后参数出现非有限值"
    assert changed > 0, "阶段 B 参数没有更新"

    elapsed = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated()
    return {
        "condition_shapes": condition_shapes,
        "target_heatmap_shape": list(targets["heatmaps"].shape),
        "latent_shape": list(conditions[0].shape),
        "aux_logits_shape": list(aux_logits.shape),
        "losses": {
            "diff": round(float(losses["diff"].detach()), 6),
            "heat": round(float(losses["heat"].detach()), 6),
            "aux": round(float(aux.detach()), 6),
            "total": round(float(total.detach()), 6),
        },
        "all_losses_finite": True,
        "gradients": stats,
        "changed_elements": changed,
        "params_finite_after_step": finite_after,
        "generator_scale": round(float(scaler.get_scale()), 4),
        "seconds": round(elapsed, 3),
        "peak_allocated_mib": mib(peak),
    }


def phase_d(
    encoder: ConditionEncoder,
    unet: ConditionalUNet,
    diffusion: GaussianDiffusion,
    images: torch.Tensor,
    embedding: torch.Tensor,
    targets: Dict[str, torch.Tensor],
) -> Dict[str, Any]:
    """阶段 D：eval U-Net、detach 条件、记录式采样、重算与策略梯度反向。"""
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()

    # 条件只编码一次并冻结；G=2 由 batch 维 expand 模拟同一组的两个动作。
    encoder.eval()
    with torch.no_grad(), autocast():
        conditions, _ = encoder(images, embedding)
    conditions = [c.detach().expand(GROUP, *c.shape[1:]) for c in conditions]
    # diffusion.sample 从 conditions[0] 推断潜变量边长；128 输入经 4x 下采样为 32。
    latent_size = int(conditions[0].shape[-1])
    assert latent_size == IMAGE_SIZE // 4, latent_size

    unet.eval()
    assert unet.gradient_checkpointing, "D 阶段要求 gradient_checkpointing=True"

    # 与 engine/inference.make_generator 一致：CPU generator，噪声生成后搬到 GPU。
    generator = torch.Generator(device="cpu")
    generator.manual_seed(SEED)

    with autocast():
        heatmaps, records = diffusion.sample(
            unet, conditions, generator=generator, record=True
        )

    assert list(heatmaps.shape) == [GROUP, 2, latent_size, latent_size], tuple(
        heatmaps.shape
    )
    assert torch.isfinite(heatmaps).all(), "采样热图非有限"
    assert float(heatmaps.min()) >= 0.0 and float(heatmaps.max()) <= 1.0
    assert len(records) == SAMPLING_STEPS, f"记录步数 {len(records)} != {SAMPLING_STEPS}"

    record_shapes = []
    for record in records:
        state, action = record["state"], record["action"]
        old_log_prob = record["old_log_prob"]
        assert state.device.type == "cpu" and action.device.type == "cpu"
        assert old_log_prob.device.type == "cpu"
        assert state.dtype == torch.float32, state.dtype
        assert action.dtype == torch.float32, action.dtype
        assert old_log_prob.dtype == torch.float64, old_log_prob.dtype
        assert list(state.shape) == [GROUP, 2, latent_size, latent_size]
        assert list(old_log_prob.shape) == [GROUP]
        assert bool(torch.isfinite(state).all()) and bool(torch.isfinite(action).all())
        assert bool(torch.isfinite(old_log_prob).all())
        record_shapes.append(
            {
                "state": str(state.dtype),
                "action": str(action.dtype),
                "old_log_prob": str(old_log_prob.dtype),
                "shape": list(state.shape),
            }
        )

    # 固定 generator 的采样可复现性（eval + no_grad）。
    generator.manual_seed(SEED)
    with autocast():
        heatmaps_repeat, _ = diffusion.sample(
            unet, conditions, generator=generator, record=True
        )
    repeat_diff = float((heatmaps - heatmaps_repeat).abs().max())
    assert repeat_diff < 1e-6, f"固定 generator 采样不可复现: {repeat_diff}"

    advantages = torch.tensor([1.0, -1.0], device=DEVICE, dtype=torch.float64)
    ppo_clip = 1e-4  # 与 engine rl.ppo_clip 默认配置一致

    # 合成参考 U-Net：微小扰动权重，专门让 KL 项有非零梯度可走。
    ref_unet = copy.deepcopy(unet)
    ref_unet.eval()
    ref_generator = torch.Generator(device="cuda")
    ref_generator.manual_seed(SEED + 1)
    with torch.no_grad():
        for param in ref_unet.parameters():
            param.add_(torch.randn_like(param, generator=ref_generator) * 1e-4)
            param.requires_grad_(False)

    ref_mean_stats: List[float] = []
    ratio_errors: List[float] = []
    ppo_values: List[float] = []
    kl_values: List[float] = []
    policy_mean_grads: List[float] = []
    kl_mean_grads: List[float] = []

    optimizer = torch.optim.Adam(unet.parameters(), lr=1e-4)
    # 高维高斯 log-prob 对均值的梯度约为 noise/std（~50/元素），默认 65536
    # 的初始 scale 会把 fp16 中间梯度推到 inf。这里用保守 scale 且不增长。
    scaler = torch.amp.GradScaler(
        "cuda", init_scale=64.0, growth_interval=1000000
    )
    before = snapshot(list(unet.parameters()))

    total_terms = float(len(records))
    policy_kl_loss = torch.zeros((), device=DEVICE, dtype=torch.float64)
    for record in records:
        state = record["state"].to(DEVICE, dtype=torch.float32)
        action = record["action"].to(DEVICE, dtype=torch.float32)
        old_log_prob = record["old_log_prob"].to(DEVICE).reshape(-1).double()

        with torch.no_grad(), autocast():
            ref_mean, _ = diffusion.transition(
                ref_unet,
                state,
                record["timestep"],
                record["previous_timestep"],
                conditions,
            )
        ref_mean_stats.append(float(ref_mean.double().abs().mean()))

        # rollout 与 replay 使用同一 AMP 上下文和同一 eval U-Net。
        with autocast():
            mean, std = diffusion.transition(
                unet,
                state,
                record["timestep"],
                record["previous_timestep"],
                conditions,
            )
            new_log_prob = diffusion.log_prob(action, mean, std).reshape(-1).double()

        assert bool(torch.isfinite(new_log_prob).all()), "重算 log-prob 非有限"
        log_ratio = torch.clamp(new_log_prob - old_log_prob, min=-20.0, max=20.0)
        ratio = torch.exp(log_ratio)
        ratio_errors.append(float((ratio - 1.0).abs().max().detach()))

        clipped = torch.clamp(ratio, 1.0 - ppo_clip, 1.0 + ppo_clip)
        ppo_loss = -torch.minimum(ratio * advantages, clipped * advantages).mean()

        delta = (mean.double() - ref_mean.double()) / std.double().clamp_min(1e-12)
        kl = 0.5 * delta.reshape(delta.shape[0], -1).pow(2).sum(dim=1).mean()

        # 单独验证 log-prob 的 PPO 梯度与 KL 梯度都能到达动作均值。
        ppo_mean_grad = torch.autograd.grad(ppo_loss, mean, retain_graph=True)[0]
        kl_mean_grad = torch.autograd.grad(kl, mean, retain_graph=True)[0]
        assert torch.isfinite(ppo_mean_grad).all(), "PPO 对 mean 的梯度非有限"
        assert torch.isfinite(kl_mean_grad).all(), "KL 对 mean 的梯度非有限"
        policy_mean_grads.append(float(ppo_mean_grad.abs().mean().detach()))
        kl_mean_grads.append(float(kl_mean_grad.abs().mean().detach()))

        ppo_values.append(float(ppo_loss.detach()))
        kl_values.append(float(kl.detach()))
        policy_kl_loss = policy_kl_loss + (ppo_loss + 0.01 * kl) / total_terms

    # 扩散 anchor：与 engine 的 _ddpo_update 一致，只用噪声 MSE。
    with autocast():
        anchor_losses = diffusion.training_losses(
            unet, conditions, targets["heatmaps"].expand(GROUP, -1, -1, -1)
        )
    anchor = 0.1 * anchor_losses["diff"]
    combined = (policy_kl_loss + anchor.double()).float()

    assert all_finite([combined, anchor, anchor_losses["diff"]]), "D 阶段组合损失非有限"
    assert combined.requires_grad, "D 阶段组合损失未连接计算图"
    assert len(ratio_errors) == SAMPLING_STEPS
    max_ratio_error = max(ratio_errors)
    assert max_ratio_error < 5e-3, f"重算 ratio 偏离 1 过大: {max_ratio_error}"
    assert all(math.isfinite(v) for v in ppo_values + kl_values), "PPO/KL 出现非有限值"
    assert max(policy_mean_grads) > 0.0, "PPO 对 mean 的梯度为 0"
    assert max(kl_mean_grads) > 0.0, "KL 对 mean 的梯度为 0"
    assert max(ref_mean_stats) > 0.0

    optimizer.zero_grad(set_to_none=True)
    scaler.scale(combined).backward()
    scaler.unscale_(optimizer)
    stats = grad_stats(list(unet.parameters()))
    assert stats["missing"] == 0, f"D 阶段有 {stats['missing']} 个参数没有梯度"
    assert stats["nonfinite"] == 0, f"D 阶段有 {stats['nonfinite']} 个参数梯度非有限"
    assert stats["global_norm"] > 0.0, "D 阶段梯度范数为 0"

    scaler.step(optimizer)
    scaler.update()
    changed = changed_elements(before, list(unet.parameters()))
    finite_after = all_finite(unet.parameters())
    assert finite_after, "D 阶段更新后参数出现非有限值"
    assert changed > 0, "D 阶段参数没有更新"

    elapsed = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated()
    return {
        "sampling_steps": len(records),
        "sample_latent_shape": list(heatmaps.shape),
        "anchor_target_shape": list(targets["heatmaps"].shape),
        "record_dtypes": record_shapes[0],
        "sample_generator": "cpu manual_seed(%d)" % SEED,
        "sample_reproducible_max_abs_diff": repeat_diff,
        "heatmap_range": [round(float(heatmaps.min()), 4), round(float(heatmaps.max()), 4)],
        "logprob_ratio_max_abs_error": max_ratio_error,
        "advantages": advantages.tolist(),
        "ppo_clip": ppo_clip,
        "ppo_loss_mean": round(sum(ppo_values) / len(ppo_values), 6),
        "ppo_mean_gradient_abs_mean": round(
            sum(policy_mean_grads) / len(policy_mean_grads), 6
        ),
        "kl_mean": round(sum(kl_values) / len(kl_values), 6),
        "kl_mean_gradient_abs_mean": round(
            sum(kl_mean_grads) / len(kl_mean_grads), 6
        ),
        "kl_reference": "deepcopy(unet) + 1e-4 合成扰动，仅用于打通 KL 梯度路径",
        "anchor_diff": round(float(anchor_losses["diff"].detach()), 6),
        "combined_loss": round(float(combined.detach()), 6),
        "gradients": stats,
        "changed_elements": changed,
        "params_finite_after_step": finite_after,
        "generator_scale": round(float(scaler.get_scale()), 4),
        "seconds": round(elapsed, 3),
        "peak_allocated_mib": mib(peak),
    }


def main() -> None:
    assert torch.cuda.is_available(), "CUDA 不可用，无法执行 GPU smoke test"
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    properties = torch.cuda.get_device_properties(0)
    device_total_mib = mib(properties.total_memory)
    started = time.perf_counter()

    generator = torch.Generator(device="cuda")
    generator.manual_seed(SEED)
    images = torch.rand(
        BATCH, 3, IMAGE_SIZE, IMAGE_SIZE, device=DEVICE, generator=generator
    )
    embedding = torch.randn(
        BATCH, 256, EMBED_SIZE, EMBED_SIZE, device=DEVICE, generator=generator
    )
    targets = build_targets()

    encoder = ConditionEncoder().to(DEVICE)
    # 默认通道 (64, 128, 256, 256)，gradient_checkpointing=True。
    unet = ConditionalUNet().to(DEVICE)
    diffusion = GaussianDiffusion(
        timesteps=TIMESTEPS,
        sampling_steps=SAMPLING_STEPS,
        heatmap_size=IMAGE_SIZE,
    ).to(DEVICE)

    phase_b_result = phase_b(encoder, unet, diffusion, images, embedding, targets)
    phase_d_result = phase_d(encoder, unet, diffusion, images, embedding, targets)

    elapsed = time.perf_counter() - started
    peak_mib = max(phase_b_result["peak_allocated_mib"], phase_d_result["peak_allocated_mib"])
    result = {
        "passed": True,
        "command": "conda run -n pytorch python scripts/gpu_smoke.py",
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": properties.name,
            "device_total_memory_mib": device_total_mib,
        },
        "config": {
            "images": list(images.shape),
            "fake_sam_embedding": list(embedding.shape),
            "heatmap_target_shape": list(targets["heatmaps"].shape),
            "aux_target_shape": list(targets["road"].shape),
            "unet_base_channels": unet.base_channels,
            "unet_channels": [unet.base_channels * m for m in unet.channel_mult],
            "condition_channels": list(unet.condition_channels),
            "gradient_checkpointing": unet.gradient_checkpointing,
            "amp": "cuda autocast float16 + GradScaler",
            "diffusion": {"timesteps": TIMESTEPS, "sampling_steps": SAMPLING_STEPS},
            "phase_d_group_size": GROUP,
        },
        "phase_b": phase_b_result,
        "phase_d": phase_d_result,
        "memory": {
            "phase_b_peak_allocated_mib": phase_b_result["peak_allocated_mib"],
            "phase_d_peak_allocated_mib": phase_d_result["peak_allocated_mib"],
            "max_memory_allocated_mib": peak_mib,
            "max_memory_reserved_mib": mib(torch.cuda.max_memory_reserved()),
            "device_total_memory_mib": device_total_mib,
            "peak_fraction_of_device": round(peak_mib / device_total_mib, 4),
        },
        "timings_sec": {
            "phase_b": phase_b_result["seconds"],
            "phase_d": phase_d_result["seconds"],
            "total": round(elapsed, 3),
        },
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
