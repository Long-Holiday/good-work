"""RTX 3050 小型 GPU smoke test（阶段 B 与阶段 D）。

只依赖 ``torch`` 与 ``diffpeak`` 的 models/losses 模块；不导入 inference/engine，
不加载 SAM 权重（用随机低分辨率与两级高分辨率张量模拟 SAM2 图像特征），
不安装或下载任何东西。

运行::

    conda run -n pytorch python scripts/gpu_smoke.py

阶段 B：128x128 输入 -> ConditionEncoder 多尺度条件 -> ConditionalDiT +
GaussianDiffusion.training_losses + auxiliary_loss，AMP float16 + GradScaler
前向/反向，检查损失、梯度有限且参数确实更新。

阶段 D：DiT eval、条件 detach，固定 CUDA generator 采样 G=2 轨迹
（timesteps=32, steps=4），检查记录 dtype、逐步重算 log-prob 的 ratio≈1，
再用组内奖励 [0.4, 0.6] 走真实的 ``diffpeak.grpo`` 优势/裁剪/KL 工具，
做 GRPO + KL + 扩散 anchor 组合反向，检查梯度有限、参数更新。

另有一组纯张量检查覆盖 GRPO 工具的多图组（``[G, B]`` 按列独立标准化）、
常量组（零优势、策略项零梯度）与裁剪边界；不绘制任何图表。

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

from diffpeak.grpo import (  # noqa: E402
    gaussian_kl_divergence,
    group_relative_advantages,
    grpo_policy_loss,
)
from diffpeak.losses import auxiliary_loss  # noqa: E402
from diffpeak.models.conditioning import ConditionEncoder  # noqa: E402
from diffpeak.models.diffusion import GaussianDiffusion  # noqa: E402
from diffpeak.models.dit import ConditionalDiT  # noqa: E402
from diffpeak.models.sam import SAM2ImageFeatures  # noqa: E402

DEVICE = torch.device("cuda")
IMAGE_SIZE = 128
EMBED_SIZE = 8
BATCH = 1
GROUP = 2
TIMESTEPS = 32
SAMPLING_STEPS = 4
SEED = 1234
# 冒烟测试用的小 DiT：32x32 潜变量按 patch_size=4 得到 8x8 token。
DIT_PATCH_SIZE = 4
DIT_HIDDEN_SIZE = 128
DIT_DEPTH = 2
DIT_NUM_HEADS = 4
DIT_MLP_RATIO = 4.0
DIT_CONDITION_CHANNELS = (64, 128, 256, 256)


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


def grpo_utility_checks() -> Dict[str, Any]:
    """用真实 ``diffpeak.grpo`` 工具覆盖多图组、常量组与裁剪/KL 边界。"""
    # 多图：[G=2, B=2]。两列奖励尺度不同，但每列都应按各自组标准化为 ∓1，
    # 不能把整张 [G, B] 当成一个组。
    multi_rewards = torch.tensor(
        [[0.40, 0.90], [0.60, 0.91]], dtype=torch.float64
    )
    multi_advantages = group_relative_advantages(multi_rewards, delta=1e-9)
    torch.testing.assert_close(
        multi_advantages,
        torch.tensor([[-1.0, -1.0], [1.0, 1.0]], dtype=torch.float64),
    )

    # 常量组：组内奖励完全相同 -> 零优势；策略项及其对 ratio 的梯度都必须为零。
    constant_rewards = torch.full((GROUP, 2), 0.75, dtype=torch.float64)
    constant_advantages = group_relative_advantages(
        constant_rewards, delta=1e-4
    )
    assert torch.equal(constant_advantages, torch.zeros_like(constant_rewards))
    constant_ratio = torch.tensor(
        [0.5, 1.5, 2.0, 0.1], dtype=torch.float64, requires_grad=True
    )
    constant_loss = grpo_policy_loss(
        constant_ratio, constant_advantages.reshape(-1), clip_epsilon=1e-4
    )
    constant_grad = torch.autograd.grad(constant_loss, constant_ratio)[0]
    assert float(constant_loss.detach()) == 0.0, f"常量组策略损失非零: {constant_loss}"
    assert torch.equal(
        constant_grad, torch.zeros_like(constant_grad)
    ), "常量组策略项仍产生非零梯度"

    # 裁剪边界：A>0 的超上界与 A<0 的超下界都必须走 clamp 分支。
    clip_ratio = torch.tensor([2.0, 0.1], dtype=torch.float64)
    clip_advantage = torch.tensor([1.0, -1.0], dtype=torch.float64)
    utility_clip = 0.2
    clipped_loss = grpo_policy_loss(clip_ratio, clip_advantage, utility_clip)
    # min(2, 1.2) * 1 + min(0.1, 0.8) * -1 = 1.2 - 0.8 = 0.4 -> loss = -0.2。
    torch.testing.assert_close(
        clipped_loss, torch.tensor(-0.2, dtype=torch.float64)
    )

    # KL 工具：相同均值为 0；逐元素差 1 时为 0.5 * 动作维元素数，且梯度非零。
    kl_mean = torch.zeros(2, 2, 4, 4, dtype=torch.float64, requires_grad=True)
    kl_std = torch.ones_like(kl_mean)
    identical_kl = gaussian_kl_divergence(
        kl_mean, kl_mean.detach().clone(), kl_std
    )
    assert torch.equal(identical_kl, torch.zeros(2, dtype=torch.float64))
    kl_reference = torch.ones_like(kl_mean)
    kl = gaussian_kl_divergence(kl_mean, kl_reference, kl_std)
    torch.testing.assert_close(kl, torch.full((2,), 16.0, dtype=torch.float64))
    kl_grad = torch.autograd.grad(kl.mean(), kl_mean)[0]
    assert bool(torch.isfinite(kl_grad).all()) and float(kl_grad.abs().max()) > 0.0

    return {
        "multi_image_rewards": multi_rewards.tolist(),
        "multi_image_advantages": multi_advantages.tolist(),
        "constant_group_rewards": constant_rewards[:, 0].tolist(),
        "constant_group_advantages": constant_advantages[:, 0].tolist(),
        "constant_group_loss": float(constant_loss.detach()),
        "constant_group_ratio_grad_abs_max": float(constant_grad.abs().max()),
        "clip_epsilon": utility_clip,
        "clip_ratio": clip_ratio.tolist(),
        "clip_advantages": clip_advantage.tolist(),
        "clip_loss": float(clipped_loss),
        "kl_identical": identical_kl.tolist(),
        "kl_unit_mean_shift": kl.tolist(),
        "kl_mean_gradient_abs_max": float(kl_grad.abs().max()),
    }


def phase_b(
    encoder: ConditionEncoder,
    dit: ConditionalDiT,
    diffusion: GaussianDiffusion,
    images: torch.Tensor,
    embedding: SAM2ImageFeatures,
    targets: Dict[str, torch.Tensor],
) -> Dict[str, Any]:
    """阶段 B：条件编码 + 扩散/热图/辅助损失 + AMP 反向与参数更新。"""
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()

    encoder.train()
    dit.train()
    params = list(encoder.parameters()) + list(dit.parameters())
    optimizer = torch.optim.Adam(params, lr=2e-3)
    scaler = torch.amp.GradScaler("cuda")

    with autocast():
        conditions, aux_logits = encoder(images, embedding)
        losses = diffusion.training_losses(dit, conditions, targets["heatmaps"])
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
    dit: ConditionalDiT,
    diffusion: GaussianDiffusion,
    images: torch.Tensor,
    embedding: SAM2ImageFeatures,
    targets: Dict[str, torch.Tensor],
) -> Dict[str, Any]:
    """阶段 D：eval DiT、detach 条件、记录式采样、重算与策略梯度反向。"""
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

    dit.eval()
    assert dit.gradient_checkpointing, "D 阶段要求 gradient_checkpointing=True"

    # 与 engine/inference.make_generator 一致：CPU generator，噪声生成后搬到 GPU。
    generator = torch.Generator(device="cpu")
    generator.manual_seed(SEED)

    with autocast():
        heatmaps, records = diffusion.sample(
            dit, conditions, generator=generator, record=True
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
            dit, conditions, generator=generator, record=True
        )
    repeat_diff = float((heatmaps - heatmaps_repeat).abs().max())
    assert repeat_diff < 1e-6, f"固定 generator 采样不可复现: {repeat_diff}"

    group_rewards = torch.tensor(
        [[0.4], [0.6]], dtype=torch.float64
    )  # [G=2, B=1]
    advantages = group_relative_advantages(
        group_rewards, delta=1e-4
    ).reshape(-1).to(DEVICE)
    torch.testing.assert_close(
        advantages, torch.tensor([-1.0, 1.0], dtype=torch.float64, device=DEVICE)
    )
    clip_epsilon = 1e-4  # 与 engine rl.ppo_clip（GRPO 裁剪半径）默认配置一致

    # 合成参考 DiT：微小扰动权重，专门让 KL 项有非零梯度可走。
    ref_dit = copy.deepcopy(dit)
    ref_dit.eval()
    ref_generator = torch.Generator(device="cuda")
    ref_generator.manual_seed(SEED + 1)
    with torch.no_grad():
        for param in ref_dit.parameters():
            param.add_(torch.randn_like(param, generator=ref_generator) * 1e-4)
            param.requires_grad_(False)

    ref_mean_stats: List[float] = []
    ratio_errors: List[float] = []
    grpo_values: List[float] = []
    kl_values: List[float] = []
    policy_mean_grads: List[float] = []
    kl_mean_grads: List[float] = []

    optimizer = torch.optim.Adam(dit.parameters(), lr=1e-4)
    # 高维高斯 log-prob 对均值的梯度约为 noise/std（~50/元素），默认 65536
    # 的初始 scale 会把 fp16 中间梯度推到 inf。这里用保守 scale 且不增长。
    scaler = torch.amp.GradScaler(
        "cuda", init_scale=64.0, growth_interval=1000000
    )
    before = snapshot(list(dit.parameters()))

    total_terms = float(len(records))
    policy_kl_loss = torch.zeros((), device=DEVICE, dtype=torch.float64)
    for record in records:
        state = record["state"].to(DEVICE, dtype=torch.float32)
        action = record["action"].to(DEVICE, dtype=torch.float32)
        old_log_prob = record["old_log_prob"].to(DEVICE).reshape(-1).double()

        with torch.no_grad(), autocast():
            ref_mean, _ = diffusion.transition(
                ref_dit,
                state,
                record["timestep"],
                record["previous_timestep"],
                conditions,
            )
        ref_mean_stats.append(float(ref_mean.double().abs().mean()))

        # rollout 与 replay 使用同一 AMP 上下文和同一 eval DiT。
        with autocast():
            mean, std = diffusion.transition(
                dit,
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

        grpo_loss = grpo_policy_loss(ratio, advantages, clip_epsilon)

        # 与 engine._grpo_update 相同的同方差高斯 KL 工具：动作维求和后取组均值。
        kl = gaussian_kl_divergence(mean, ref_mean, std).mean()

        # 单独验证 GRPO 梯度与 KL 梯度都能到达动作均值。
        grpo_mean_grad = torch.autograd.grad(grpo_loss, mean, retain_graph=True)[0]
        kl_mean_grad = torch.autograd.grad(kl, mean, retain_graph=True)[0]
        assert torch.isfinite(grpo_mean_grad).all(), "GRPO 对 mean 的梯度非有限"
        assert torch.isfinite(kl_mean_grad).all(), "KL 对 mean 的梯度非有限"
        policy_mean_grads.append(float(grpo_mean_grad.abs().mean().detach()))
        kl_mean_grads.append(float(kl_mean_grad.abs().mean().detach()))

        grpo_values.append(float(grpo_loss.detach()))
        kl_values.append(float(kl.detach()))
        policy_kl_loss = policy_kl_loss + (grpo_loss + 0.01 * kl) / total_terms

    # 扩散 anchor：与 engine 的 _grpo_update 一致，只用噪声 MSE。
    with autocast():
        anchor_losses = diffusion.training_losses(
            dit, conditions, targets["heatmaps"].expand(GROUP, -1, -1, -1)
        )
    anchor = 0.1 * anchor_losses["diff"]
    combined = (policy_kl_loss + anchor.double()).float()

    assert all_finite([combined, anchor, anchor_losses["diff"]]), "D 阶段组合损失非有限"
    assert combined.requires_grad, "D 阶段组合损失未连接计算图"
    assert len(ratio_errors) == SAMPLING_STEPS
    max_ratio_error = max(ratio_errors)
    assert max_ratio_error < 5e-3, f"重算 ratio 偏离 1 过大: {max_ratio_error}"
    assert all(math.isfinite(v) for v in grpo_values + kl_values), "GRPO/KL 出现非有限值"
    assert max(policy_mean_grads) > 0.0, "GRPO 对 mean 的梯度为 0"
    assert max(kl_mean_grads) > 0.0, "KL 对 mean 的梯度为 0"
    assert max(ref_mean_stats) > 0.0

    optimizer.zero_grad(set_to_none=True)
    scaler.scale(combined).backward()
    scaler.unscale_(optimizer)
    stats = grad_stats(list(dit.parameters()))
    assert stats["missing"] == 0, f"D 阶段有 {stats['missing']} 个参数没有梯度"
    assert stats["nonfinite"] == 0, f"D 阶段有 {stats['nonfinite']} 个参数梯度非有限"
    assert stats["global_norm"] > 0.0, "D 阶段梯度范数为 0"

    scaler.step(optimizer)
    scaler.update()
    changed = changed_elements(before, list(dit.parameters()))
    finite_after = all_finite(dit.parameters())
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
        "group_rewards": group_rewards.reshape(-1).tolist(),
        "grpo_advantages": advantages.tolist(),
        "grpo_clip_epsilon": clip_epsilon,
        "grpo_loss_mean": round(sum(grpo_values) / len(grpo_values), 6),
        "grpo_mean_gradient_abs_mean": round(
            sum(policy_mean_grads) / len(policy_mean_grads), 6
        ),
        "kl_mean": round(sum(kl_values) / len(kl_values), 6),
        "kl_mean_gradient_abs_mean": round(
            sum(kl_mean_grads) / len(kl_mean_grads), 6
        ),
        "kl_tool": "diffpeak.grpo.gaussian_kl_divergence",
        "kl_reference": "deepcopy(dit) + 1e-4 合成扰动，仅用于打通 KL 梯度路径",
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
    # 128 输入对应 SAM2 的 1/16 低分辨率与 1/4、1/8 两级高分辨率特征。
    embedding = SAM2ImageFeatures(
        torch.randn(
            BATCH, 256, EMBED_SIZE, EMBED_SIZE, device=DEVICE, generator=generator
        ),
        (
            torch.randn(
                BATCH, 256, IMAGE_SIZE // 4, IMAGE_SIZE // 4,
                device=DEVICE, generator=generator,
            ),
            torch.randn(
                BATCH, 256, IMAGE_SIZE // 8, IMAGE_SIZE // 8,
                device=DEVICE, generator=generator,
            ),
        ),
    )
    targets = build_targets()
    utility_checks = grpo_utility_checks()

    encoder = ConditionEncoder().to(DEVICE)
    # 小型 DiT（patch 4，hidden 128，depth 2），保持 CPU/GPU 冒烟可控。
    dit = ConditionalDiT(
        patch_size=DIT_PATCH_SIZE,
        hidden_size=DIT_HIDDEN_SIZE,
        depth=DIT_DEPTH,
        num_heads=DIT_NUM_HEADS,
        mlp_ratio=DIT_MLP_RATIO,
        condition_channels=DIT_CONDITION_CHANNELS,
        gradient_checkpointing=True,
    ).to(DEVICE)
    diffusion = GaussianDiffusion(
        timesteps=TIMESTEPS,
        sampling_steps=SAMPLING_STEPS,
        heatmap_size=IMAGE_SIZE,
    ).to(DEVICE)

    phase_b_result = phase_b(encoder, dit, diffusion, images, embedding, targets)
    phase_d_result = phase_d(encoder, dit, diffusion, images, embedding, targets)

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
            "fake_sam_features": {
                "low_res": list(embedding.low_res.shape),
                "high_res": [list(level.shape) for level in embedding.high_res],
            },
            "heatmap_target_shape": list(targets["heatmaps"].shape),
            "aux_target_shape": list(targets["road"].shape),
            "denoiser": "ConditionalDiT",
            "dit": {
                "patch_size": DIT_PATCH_SIZE,
                "hidden_size": DIT_HIDDEN_SIZE,
                "depth": DIT_DEPTH,
                "num_heads": DIT_NUM_HEADS,
                "mlp_ratio": DIT_MLP_RATIO,
                "condition_channels": list(DIT_CONDITION_CHANNELS),
                "gradient_checkpointing": dit.gradient_checkpointing,
            },
            "amp": "cuda autocast float16 + GradScaler",
            "diffusion": {"timesteps": TIMESTEPS, "sampling_steps": SAMPLING_STEPS},
            "phase_d_group_size": GROUP,
        },
        "grpo_utility_checks": utility_checks,
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
