"""ABC 联合预训练与 D 整体 GRPO 后训练。

ABC 用一个总损失和优化器更新条件编码器与 DiT，SAM 全部冻结但保留输入梯度。
SAM 图像/提示编码器权重冻结，可微密集提示保留输入梯度。
D 使用扩散转移及终端 Bernoulli 分割动作的组相对策略梯度；整个策略
采用 eval 模式重放，并保存条件编码器、DiT、SAM 解码器的固定参考。
旧 A/B/C 损失辅助实现仅供历史测试，不再作为训练入口。
"""

from __future__ import annotations

import copy
import json
import math
import os
import random
import shutil
import time
import warnings
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader

from diffpeak.data import DeepGlobeDataset
from diffpeak.diagnostics import (
    PROMPT_METRIC_KEYS,
    REWARD_GROUP_METRIC_KEYS,
    prompt_quality,
    reward_group_diagnostics,
)
from diffpeak.grpo import (
    gaussian_kl_divergence,
    group_relative_advantages,
    grpo_policy_loss,
)
from diffpeak.inference import (
    aggregate_metrics,
    build_models,
    compute_image_metrics,
    compute_resized_size,
    decode_with_prompts,
    encode_images,
    evaluation_seed,
    get_points,
    load_checkpoint_models,
    make_generator,
    make_reward_config,
    normalize_valid_sizes,
    points_to_prompt,
    read_checkpoint,
    resolve_data_root,
    resolve_output_dir,
    resolve_sam_checkpoint,
    resolve_split_file,
    seed_everything,
    stable_seed,
)
from diffpeak.losses import auxiliary_loss, segmentation_loss
from diffpeak.schedules import resolve_warmup_steps, warmup_cosine_factor

__all__ = ["DiffPeakEngine", "STAGE_ORDER"]

STAGE_ORDER: Tuple[str, ...] = ("ABC", "D")


def _decoder_module(sam: torch.nn.Module) -> Optional[torch.nn.Module]:
    """定位 RoadSAM 的掩码解码器，兼容 SAM2 与旧测试替身。"""
    decoder = getattr(sam, "mask_decoder", None)
    if decoder is not None:
        return decoder
    inner = getattr(sam, "sam", None)
    for name in ("sam_mask_decoder", "mask_decoder"):
        decoder = getattr(inner, name, None)
        if decoder is not None:
            return decoder
    return None


# ---------------------------------------------------------------------------
# DataLoader 辅助
# ---------------------------------------------------------------------------
def _seed_worker(worker_id: int) -> None:
    """让不同 worker 的 Python/NumPy 随机流彼此独立。"""
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def _collate_samples(samples: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """自定义 collate：张量堆叠，元数据（id/尺寸）保留逐样本列表。"""
    result: Dict[str, Any] = {}
    keys = set()
    for sample in samples:
        keys.update(sample.keys())
    for key in keys:
        values = [sample.get(key) for sample in samples]
        if all(torch.is_tensor(value) for value in values):
            result[key] = torch.stack(values, dim=0)
        else:
            result[key] = values
    return result


def _micro_groups(loader: Iterable[Any], size: int) -> Iterable[List[Any]]:
    """把批次切成梯度累积组；最后一组可能不足 ``size``。"""
    group: List[Any] = []
    for batch in loader:
        group.append(batch)
        if len(group) >= size:
            yield group
            group = []
    if group:
        yield group


def _mean_of(values: Sequence[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.mean(finite)) if finite else float("nan")


def _scalar(value: Any) -> Optional[float]:
    """把标量张量/数字转成 float；None 原样返回。"""
    if value is None:
        return None
    if torch.is_tensor(value):
        if value.numel() == 0:
            return None
        return float(value.detach().float().mean().cpu())
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class DiffPeakEngine:
    """负责阶段切换、训练循环、验证、checkpoint 与 GRPO 的引擎。"""

    def __init__(
        self,
        config: Mapping[str, Any],
        device: Optional[Union[str, torch.device]] = None,
        config_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        self.config = copy.deepcopy(dict(config))
        self.config_dir = (
            Path(config_dir).expanduser().resolve() if config_dir is not None else None
        )
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.seed = int(self.config.get("seed", 42))

        self.data_cfg = dict(self.config.get("data", {}))
        self.train_cfg = dict(self.config.get("train", {}))
        self.stage_cfgs = dict(self.config.get("stages", {}))
        self.rl_cfg = dict(self.config.get("rl", {}))
        self.model_cfg = dict(self.config.get("model", {}))
        self.inference_cfg = dict(self.config.get("inference", {}))

        seed_everything(
            self.seed, deterministic=bool(self.train_cfg.get("deterministic", False))
        )
        torch.backends.cudnn.benchmark = False

        self.image_size = 1024
        self.heatmap_size = 256

        self.data_root = Path(
            resolve_data_root(self.config, self.config_dir)
        ).expanduser().resolve()
        if not self.data_root.is_dir():
            raise FileNotFoundError(f"数据根目录不存在：{self.data_root}")
        self.sam_checkpoint = Path(
            resolve_sam_checkpoint(self.config, self.config_dir)
        ).expanduser().resolve()

        self.models, self.peak_extractor = build_models(
            self.config, self.device, sam_checkpoint=self.sam_checkpoint
        )
        self.sam = self.models["sam"]
        self.condition_encoder = self.models["condition_encoder"]
        self.dit = self.models["dit"]
        self.diffusion = self.models["diffusion"]
        self.reward_config = make_reward_config(self.config)

        self.output_dir = Path(
            resolve_output_dir(self.config, self.config_dir)
        ).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        requested_amp = bool(self.train_cfg.get("amp", True))
        self.amp = bool(requested_amp and self.device.type == "cuda")
        if requested_amp and self.device.type != "cuda":
            warnings.warn("非 CUDA 设备上已关闭 AMP。", RuntimeWarning)

        # 吞吐参数的全局默认值；run_stage 会按 stages.<stage> 覆盖后写入
        # 同名运行时属性。未配置覆盖时行为与旧版本完全一致。
        self.batch_size = max(1, int(self.train_cfg.get("batch_size", 1)))
        self.accumulation_steps = max(1, int(self.train_cfg.get("accumulation_steps", 1)))
        self.num_workers = max(0, int(self.data_cfg.get("num_workers", 0)))
        self.heat_weight = float(self.train_cfg.get("heat_weight", 0.1))
        self.aux_weight = float(self.train_cfg.get("aux_weight", 0.2))
        self.cldice_weight = float(self.train_cfg.get("cldice_weight", 0.2))
        self.weight_decay = float(self.train_cfg.get("weight_decay", 0.01))
        self.grad_clip = float(self.train_cfg.get("grad_clip", 1.0))
        self.save_every = max(1, int(self.train_cfg.get("save_every", 1)))
        self.log_every = max(1, int(self.train_cfg.get("log_every", 10)))
        self.val_every = max(1, int(self.train_cfg.get("val_every", 1)))
        self.val_limit = self.train_cfg.get("val_limit")
        self.threshold = float(self.inference_cfg.get("threshold", 0.5))

        self.rl_algorithm = str(self.rl_cfg.get("algorithm", "grpo")).lower()
        if self.rl_algorithm != "grpo":
            raise ValueError(
                "当前实现只支持 GRPO；rl.algorithm 必须为 'grpo'，"
                f"收到 {self.rl_algorithm!r}。旧 DDPO 配置请迁移后再训练。"
            )
        self.group_size = int(self.rl_cfg.get("group_size", 4))
        # 键名保留 ppo_clip 以兼容已有配置；它在这里是 GRPO 的裁剪半径。
        self.ppo_clip = float(self.rl_cfg.get("ppo_clip", 1e-4))
        self.kl_weight = float(self.rl_cfg.get("kl_weight", 0.01))
        self.anchor_weight = float(self.rl_cfg.get("anchor_weight", 0.0))
        self.update_epochs = max(1, int(self.rl_cfg.get("update_epochs", 1)))
        self.advantage_std_delta = float(
            self.rl_cfg.get("advantage_std_delta", 1e-4)
        )
        self.point_cost_warmup_steps = max(
            0, int(self.rl_cfg.get("point_cost_warmup_steps", 0))
        )
        if self.ppo_clip < 0.0 or not math.isfinite(self.ppo_clip):
            raise ValueError("rl.ppo_clip（GRPO 裁剪半径）必须是非负有限数")
        if self.kl_weight < 0.0 or not math.isfinite(self.kl_weight):
            raise ValueError("rl.kl_weight 必须是非负有限数")
        if self.anchor_weight < 0.0 or not math.isfinite(self.anchor_weight):
            raise ValueError("rl.anchor_weight 必须是非负有限数")
        if (
            self.advantage_std_delta <= 0.0
            or not math.isfinite(self.advantage_std_delta)
        ):
            raise ValueError("rl.advantage_std_delta 必须是严格正的有限数")

        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
        self.scheduler_total_steps = 0
        self.scheduler_warmup_steps = 0
        self.scheduler_min_lr_ratio = 0.0
        self.scheduler_stage_lr = 0.0
        self.scheduler_steps_per_epoch = 0
        self.scheduler_epochs = 0
        self.scheduler_lr_schedule = "cosine"
        self.scaler: Optional[Any] = None
        self.ref_dit: Optional[torch.nn.Module] = None
        self.ref_condition_encoder: Optional[torch.nn.Module] = None
        self.ref_sam_decoder: Optional[torch.nn.Module] = None
        self.global_step = 0
        # 点成本预热按“已处理的 D 组数”推进，而不是按成功的 optimizer step；
        # 否则整组常量奖励且 KL/anchor 关闭时预热会永远停在 0。
        self.grpo_groups_processed = 0
        self.best_metric: Optional[float] = None
        self.history: List[Dict[str, Any]] = []
        self._stage: Optional[str] = None
        self._epoch = 0
        self._train_split_cache: Optional[Path] = None
        self._val_split_cache: Optional[Path] = None
        self._val_split_cached = False
        self.experiment_logger = None

        self._validate_peak_target_config()
        # 提前解析划分路径：train 缺失或显式配置的 val 缺失时立即失败；
        # val_split=None 才会禁用验证。
        self._resolve_train_split()
        self._resolve_val_split()

    # ------------------------------------------------------------------
    # 配置与数据
    # ------------------------------------------------------------------
    def _validate_peak_target_config(self) -> None:
        """显式检查目标采样间距与最终 NMS 间距一致。"""
        peaks_min = float(self.config.get("peaks", {}).get("min_distance", 4.0))
        target_min = float(self.data_cfg.get("target_min_distance", 4.0))
        if abs(peaks_min - target_min) > 1e-9:
            raise ValueError(
                "peaks.min_distance "
                f"({peaks_min}) 与 data.target_min_distance ({target_min}) 不一致。"
                "两者必须显式同步，否则目标热图在构造阶段就可能被合并。"
            )

    def _dataset_kwargs(self, split_file: Path, augment: bool) -> Dict[str, Any]:
        return {
            "root": str(self.data_root),
            "split_file": str(split_file),
            "augment": bool(augment),
            "image_size": self.image_size,
            "heatmap_size": self.heatmap_size,
            "seed": self.seed,
            "sampling_config": self.data_cfg.get("sampling_config"),
            "heatmap_sigma": float(self.data_cfg.get("heatmap_sigma", 1.0)),
            "target_min_distance": float(
                self.data_cfg.get("target_min_distance", 4.0)
            ),
        }

    def _resolve_train_split(self) -> Path:
        if self._train_split_cache is not None:
            return self._train_split_cache
        raw = self.data_cfg.get("train_split")
        if not raw:
            raise ValueError("配置缺少 data.train_split")
        path = resolve_split_file(
            raw, self.data_root, self.config_dir, must_exist=True
        )
        self._train_split_cache = Path(path).expanduser().resolve()
        return self._train_split_cache

    def _resolve_val_split(self) -> Optional[Path]:
        """解析验证划分；``val_split: null`` 表示显式禁用验证。

        配置了路径但文件不存在时直接报错，避免静默跳过验证或用错数据。
        """
        if self._val_split_cached:
            return self._val_split_cache
        raw = self.data_cfg.get("val_split")
        if raw is None or (isinstance(raw, str) and not raw.strip()):
            path = None
        else:
            path = Path(
                resolve_split_file(
                    raw, self.data_root, self.config_dir, must_exist=True
                )
            ).expanduser().resolve()
        self._val_split_cache = path
        self._val_split_cached = True
        return path

    def _apply_stage_config(self, stage: str) -> None:
        """解析当前阶段的吞吐覆盖，写回运行时属性。

        必须在构建 DataLoader 与优化器/调度器**之前**调用。每个阶段开始时
        都从该阶段的配置重新计算：显式键覆盖全局值，缺失键回退到
        ``train``/``data``，因此上一阶段的覆盖不会泄漏到下一阶段。
        """
        stage_cfg = self.stage_cfgs.get(stage, {})

        def effective(key: str, default: Any) -> Any:
            value = stage_cfg.get(key)
            return default if value is None else value

        self.batch_size = max(
            1, int(effective("batch_size", self.train_cfg.get("batch_size", 1)))
        )
        self.accumulation_steps = max(
            1,
            int(
                effective(
                    "accumulation_steps",
                    self.train_cfg.get("accumulation_steps", 1),
                )
            ),
        )
        self.num_workers = max(
            0, int(effective("num_workers", self.data_cfg.get("num_workers", 0)))
        )

    def _build_train_loader(self, epoch: int) -> DataLoader:
        dataset = DeepGlobeDataset(
            **self._dataset_kwargs(self._resolve_train_split(), augment=bool(
                self.data_cfg.get("augment", False)
            ))
        )
        # 每轮重建数据集，清空 occurrence；epoch 显式写入，配合 loader 的
        # seed+epoch 生成器，使同一 epoch 在 resume 后可以复现。
        dataset.set_epoch(int(epoch))
        generator = torch.Generator()
        generator.manual_seed(
            (self.seed + int(epoch) * 1_000_003) % (1 << 63)
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=_collate_samples,
            pin_memory=self.device.type == "cuda",
            drop_last=False,
            worker_init_fn=_seed_worker,
            generator=generator,
            persistent_workers=False,
        )

    def _build_val_loader(self) -> Optional[DataLoader]:
        split = self._resolve_val_split()
        if split is None:
            return None
        # 每轮验证重建数据集：A 阶段的目标布局固定为 occurrence=0。
        dataset = DeepGlobeDataset(**self._dataset_kwargs(split, augment=False))
        dataset.set_epoch(0)
        return DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            collate_fn=_collate_samples,
            pin_memory=False,
            drop_last=False,
        )

    # ------------------------------------------------------------------
    # 冻结、模式与优化器
    # ------------------------------------------------------------------
    def _set_requires_grad(self, stage: str) -> None:
        for parameter in self.sam.parameters():
            parameter.requires_grad_(False)
        if stage in ("D", "A", "C"):
            decoder = _decoder_module(self.sam)
            if decoder is None:
                raise AttributeError("RoadSAM 缺少 mask decoder")
            for parameter in decoder.parameters():
                parameter.requires_grad_(True)

        for parameter in self.condition_encoder.parameters():
            parameter.requires_grad_(stage in ("ABC", "B", "D"))
        for parameter in self.dit.parameters():
            parameter.requires_grad_(stage in ("ABC", "B", "D"))

    def _set_train_modes(self, stage: str) -> None:
        """设置训练模式；冻结模块始终 eval。"""
        self.sam.eval()
        if stage in ("D", "A", "C"):
            decoder = _decoder_module(self.sam)
            if decoder is not None:
                decoder.train(stage != "D")
        self.condition_encoder.train(stage in ("ABC", "B"))
        # D 的 rollout 与 replay 对整个策略使用 eval，梯度仍然开启。
        self.dit.train(stage in ("ABC", "B"))
        if self.ref_dit is not None:
            self.ref_dit.eval()

    def _set_eval_all(self) -> None:
        for module in self.models.values():
            module.eval()
        if self.ref_dit is not None:
            self.ref_dit.eval()

    def _trainable_parameters(self) -> List[torch.nn.Parameter]:
        parameters: List[torch.nn.Parameter] = []
        seen = set()
        for module in (self.sam, self.condition_encoder, self.dit):
            for parameter in module.parameters():
                if id(parameter) in seen:
                    continue
                seen.add(id(parameter))
                if parameter.requires_grad:
                    parameters.append(parameter)
        return parameters

    def _autocast(self, enabled: Optional[bool] = None):
        use_amp = self.amp if enabled is None else bool(enabled and self.amp)
        if not use_amp:
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=torch.float16)

    def _build_optimizer(
        self, stage: str, epochs: int, steps_per_epoch: int
    ) -> None:
        parameters = self._trainable_parameters()
        if not parameters:
            raise RuntimeError(f"阶段 {stage} 没有可训练参数")
        stage_cfg = self.stage_cfgs.get(stage, {})
        if "lr" not in stage_cfg:
            raise ValueError(f"stages.{stage} 缺少 lr")
        learning_rate = float(stage_cfg["lr"])
        self.optimizer = torch.optim.AdamW(
            parameters, lr=learning_rate, weight_decay=self.weight_decay
        )

        total_steps = max(1, int(epochs) * max(1, int(steps_per_epoch)))
        warmup_steps = resolve_warmup_steps(
            total_steps,
            warmup_steps=int(
                stage_cfg.get(
                    "warmup_steps", self.train_cfg.get("warmup_steps", 0)
                )
                or 0
            ),
            warmup_ratio=float(
                stage_cfg.get(
                    "warmup_ratio", self.train_cfg.get("warmup_ratio", 0.0)
                )
                or 0.0
            ),
        )
        min_lr_ratio = float(
            stage_cfg.get(
                "min_lr_ratio", self.train_cfg.get("min_lr_ratio", 0.0)
            )
            or 0.0
        )
        if not (0.0 <= min_lr_ratio <= 1.0):
            raise ValueError("min_lr_ratio 必须在 [0, 1] 内")

        def lr_lambda(step: int) -> float:
            # Read restored metadata rather than closing over the new CLI config.
            return warmup_cosine_factor(
                step,
                total_steps=self.scheduler_total_steps,
                warmup_steps=self.scheduler_warmup_steps,
                min_lr_ratio=self.scheduler_min_lr_ratio,
            )

        self.scheduler_total_steps = total_steps
        self.scheduler_warmup_steps = warmup_steps
        self.scheduler_min_lr_ratio = min_lr_ratio

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=lr_lambda
        )
        self.scheduler_stage_lr = learning_rate
        self.scheduler_steps_per_epoch = int(steps_per_epoch)
        self.scheduler_epochs = int(epochs)
        self.scheduler_lr_schedule = "cosine"
        print(
            f"[{stage}] lr={learning_rate:.3e} total_steps={total_steps} "
            f"warmup_steps={warmup_steps} min_lr_ratio={min_lr_ratio}",
            flush=True,
        )
        # 联合动作 log-prob 的梯度较大；D 使用更保守的初始缩放。
        scale_config = self.rl_cfg if stage == "D" else self.train_cfg
        initial_scale = float(scale_config.get("amp_init_scale", 64.0 if stage == "D" else 65536.0))
        try:
            self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp, init_scale=initial_scale)
        except (AttributeError, TypeError):  # pragma: no cover - 旧版 PyTorch
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp, init_scale=initial_scale)

    def _optimizer_step(self) -> bool:
        """执行一次优化器更新；返回是否真正推进了参数与调度。"""
        assert self.optimizer is not None and self.scheduler is not None
        assert self.scaler is not None
        parameters = [p for p in self._trainable_parameters() if p.grad is not None]
        if self.scaler.is_enabled():
            self.scaler.unscale_(self.optimizer)
        if parameters and self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(parameters, self.grad_clip)
        old_scale = self.scaler.get_scale()
        if self.scaler.is_enabled():
            self.scaler.step(self.optimizer)
        else:
            self.optimizer.step()
        self.scaler.update()
        skipped = self.scaler.is_enabled() and self.scaler.get_scale() < old_scale
        self.optimizer.zero_grad(set_to_none=True)
        # 溢出跳过参数更新时，不能提前消耗学习率调度或成功更新计数。
        if skipped:
            warnings.warn("AMP 检测到非有限梯度，本次更新已跳过并降低 scale。", RuntimeWarning)
        else:
            self.scheduler.step()
            self.global_step += 1
            # 注意：点成本预热使用独立的 grpo_groups_processed，不能在这里
            # 同步到 global_step；常量组安全跳过时 optimizer 不 step，但 D
            # 组仍然处理过，预热需要按组推进。
        return not skipped

    def _current_lr(self) -> float:
        """返回当前优化器学习率；尚未构造优化器时返回 NaN。"""
        if self.optimizer is not None and self.optimizer.param_groups:
            return float(self.optimizer.param_groups[0]["lr"])
        return float("nan")

    # ------------------------------------------------------------------
    # 批次整理与损失
    # ------------------------------------------------------------------
    def _prepare_batch(self, batch: Mapping[str, Any]) -> Dict[str, Any]:
        prepared = dict(batch)
        image = prepared.get("image")
        if image is None:
            raise KeyError("数据集样本缺少 image")
        if image.ndim == 3:
            image = image.unsqueeze(0)
        if image.dtype == torch.uint8:
            image = image.float().div_(255.0)
        elif image.is_floating_point() and image.numel() and float(image.max()) > 1.5:
            image = image.float().div_(255.0)
        else:
            image = image.float()
        prepared["image"] = image

        for key in ("mask", "skeleton"):
            value = prepared.get(key)
            if value is None:
                continue
            if value.ndim == 3:
                value = value.unsqueeze(1)
            if value.is_floating_point() and value.numel() and float(value.max()) > 1.5:
                value = value.float().div_(255.0)
            else:
                value = value.float()
            prepared[key] = value
        if "skeleton" not in prepared or prepared["skeleton"] is None:
            prepared["skeleton"] = torch.zeros_like(prepared["mask"])

        heatmaps = prepared.get("heatmaps")
        if heatmaps is None:
            raise KeyError("数据集样本缺少 heatmaps")
        if heatmaps.ndim == 3:
            heatmaps = heatmaps.unsqueeze(0)
        if heatmaps.is_floating_point() and heatmaps.numel() and float(heatmaps.max()) > 1.5:
            heatmaps = heatmaps.float().div_(255.0)
        else:
            heatmaps = heatmaps.float()
        prepared["heatmaps"] = heatmaps

        batch_size = int(image.shape[0])
        valid_sizes = normalize_valid_sizes(
            prepared.get("resized_size"), batch_size, self.image_size
        )
        if all(item is None for item in valid_sizes):
            original_sizes = normalize_valid_sizes(
                prepared.get("original_size"), batch_size, self.image_size
            )
            if any(item is not None for item in original_sizes):
                valid_sizes = [
                    compute_resized_size(item[0], item[1], self.image_size)
                    if item is not None
                    else (self.image_size, self.image_size)
                    for item in original_sizes
                ]
            else:
                valid_sizes = [
                    (self.image_size, self.image_size) for _ in range(batch_size)
                ]
        prepared["valid_sizes"] = valid_sizes

        for key in ("image", "mask", "skeleton", "heatmaps"):
            prepared[key] = prepared[key].to(self.device, non_blocking=True)

        prepared["ids"] = self._batch_ids(prepared, batch_size)
        return prepared

    def _batch_ids(self, batch: Mapping[str, Any], batch_size: int) -> List[str]:
        ids = batch.get("id")
        if ids is None:
            return [str(index) for index in range(batch_size)]
        if torch.is_tensor(ids):
            ids = ids.detach().cpu().tolist()
        if isinstance(ids, (list, tuple)):
            return [str(item) for item in ids]
        return [str(ids) for _ in range(batch_size)]

    def _supervised_loss(
        self, stage: str, batch: Mapping[str, Any], epoch: int
    ) -> Dict[str, Any]:
        images = batch["image"]
        masks = batch["mask"]
        skeletons = batch["skeleton"]
        target_heatmaps = batch["heatmaps"]
        valid_sizes = batch["valid_sizes"]
        ids = batch["ids"]

        # 图像编码器冻结；即使误开梯度也不会回传。
        with torch.no_grad():
            embeddings = encode_images(self.sam, images, valid_sizes)

        if stage == "ABC":
            conditions, auxiliary_logits = self.condition_encoder(images, embeddings)
            losses = self.diffusion.training_losses(
                self.dit, conditions, target_heatmaps, return_prediction=True
            )
            # 单步重建监督与纯噪声采样监督同时训练，避免只依赖带真值的 z_t。
            reconstructed = self.sam.decode_heatmaps(
                embeddings, losses["heat_pred"],
                output_size=tuple(masks.shape[-2:]), valid_sizes=valid_sizes,
            )
            generated_heatmaps = self.diffusion.sample_differentiable(
                self.dit, conditions,
                generator=make_generator(stable_seed(self.seed, "ABC", epoch, *ids)),
            )
            generated = self.sam.decode_heatmaps(
                embeddings, generated_heatmaps,
                output_size=tuple(masks.shape[-2:]), valid_sizes=valid_sizes,
            )
            # 有效区域逐图归一，右下 padding 不参与分割损失。
            def seg_loss(logits):
                return torch.stack([
                    segmentation_loss(
                        logits[i:i + 1, :, :h, :w], masks[i:i + 1, :, :h, :w],
                        cldice_weight=self.cldice_weight,
                    )
                    for i, (h, w) in enumerate(valid_sizes)
                ]).mean()

            segmentation = seg_loss(reconstructed)
            sampled_segmentation = seg_loss(generated)
            auxiliary = auxiliary_loss(auxiliary_logits, masks, skeletons)
            total = (
                losses["diff"] + self.heat_weight * losses["heat"]
                + self.aux_weight * auxiliary
                + float(self.train_cfg.get("seg_weight", 1.0)) * segmentation
                + float(self.train_cfg.get("sampled_seg_weight", 1.0)) * sampled_segmentation
            )
            return {
                "loss": total, "diff": losses["diff"], "heat": losses["heat"],
                "aux": auxiliary, "seg": segmentation, "sampled_seg": sampled_segmentation,
            }

        if stage == "A":
            # 按设计 A 阶段只训练 SAM mask decoder；条件/细节编码器和辅助头
            # 属于 B 阶段，这里不做无用 forward，也不把 aux 损失加进来。
            points = get_points(
                self.peak_extractor,
                target_heatmaps,
                image_size=self.image_size,
                valid_sizes=valid_sizes,
            )
            prompts = [points_to_prompt(item, self.device) for item in points]
            logits, active = decode_with_prompts(
                self.sam,
                embeddings,
                prompts,
                output_size=(self.image_size, self.image_size),
            )
            indices = [index for index, flag in enumerate(active) if flag]
            segmentation = None
            if indices:
                index_tensor = torch.as_tensor(
                    indices, dtype=torch.long, device=self.device
                )
                segmentation = segmentation_loss(
                    logits.index_select(0, index_tensor),
                    masks.index_select(0, index_tensor),
                    cldice_weight=self.cldice_weight,
                )
            return {
                "loss": segmentation,
                "seg": segmentation,
                "aux": None,
                "points": float(sum(int(item[1].numel()) for item in points)),
            }

        if stage == "B":
            conditions, auxiliary_logits = self.condition_encoder(images, embeddings)
            diffusion_losses = self.diffusion.training_losses(
                self.dit, conditions, target_heatmaps
            )
            diffusion_loss = diffusion_losses["diff"]
            heat_loss = diffusion_losses["heat"]
            auxiliary = auxiliary_loss(auxiliary_logits, masks, skeletons)
            total = (
                diffusion_loss
                + self.heat_weight * heat_loss
                + self.aux_weight * auxiliary
            )
            return {
                "loss": total,
                "diff": diffusion_loss,
                "heat": heat_loss,
                "aux": auxiliary,
            }

        if stage == "C":
            with torch.no_grad():
                conditions, _ = self.condition_encoder(images, embeddings)
                conditions = [item.detach() for item in conditions]
                generator = make_generator(
                    stable_seed(self.seed, "C", int(epoch), *ids)
                )
                heatmaps = self.diffusion.sample(
                    self.dit,
                    conditions,
                    generator=generator,
                    record=False,
                )
            points = get_points(
                self.peak_extractor,
                heatmaps,
                image_size=self.image_size,
                valid_sizes=valid_sizes,
            )
            prompts = [points_to_prompt(item, self.device) for item in points]
            logits, active = decode_with_prompts(
                self.sam,
                embeddings,
                prompts,
                output_size=(self.image_size, self.image_size),
            )
            indices = [index for index, flag in enumerate(active) if flag]
            if not indices:
                # 没有正峰：SAM 输出常量负 logits，分割损失跳过；
                # stage C 没有辅助头可训练，因此本批次不产生梯度。
                return {
                    "loss": None,
                    "seg": None,
                    "points": float(sum(int(item[1].numel()) for item in points)),
                }
            index_tensor = torch.as_tensor(
                indices, dtype=torch.long, device=self.device
            )
            segmentation = segmentation_loss(
                logits.index_select(0, index_tensor),
                masks.index_select(0, index_tensor),
                cldice_weight=self.cldice_weight,
            )
            return {
                "loss": segmentation,
                "seg": segmentation,
                "points": float(sum(int(item[1].numel()) for item in points)),
            }

        raise ValueError(f"未知监督阶段：{stage}")

    # ------------------------------------------------------------------
    # 训练循环
    # ------------------------------------------------------------------
    def _log_training_record(
        self, stage: str, scope: str, record: Mapping[str, Any],
        epoch: int, index: int, total: int,
    ) -> None:
        logger = getattr(self, "experiment_logger", None)
        if logger is not None and (
            index == 1 or index == total or index % logger.log_every == 0
        ):
            logger.log_scalars(
                stage, scope, record, step=epoch * total + index, epoch=epoch + 1
            )

    def _train_epoch_supervised(
        self, stage: str, epoch: int, loader: DataLoader
    ) -> Dict[str, float]:
        self._set_train_modes(stage)
        records: List[Dict[str, Any]] = []
        optimizer_steps = 0
        skipped_groups = 0
        batch_index = 0
        for group in _micro_groups(loader, self.accumulation_steps):
            assert self.optimizer is not None and self.scheduler is not None
            assert self.scaler is not None
            self.optimizer.zero_grad(set_to_none=True)
            group_size = len(group)
            had_gradient = False
            for batch in group:
                prepared = self._prepare_batch(batch)
                with self._autocast():
                    loss_dict = self._supervised_loss(stage, prepared, epoch)
                loss = loss_dict.get("loss")
                if loss is not None and torch.is_tensor(loss) and loss.requires_grad:
                    # 尾组按实际组大小归一。
                    self.scaler.scale(loss / float(group_size)).backward()
                    had_gradient = True
                record = {
                    "loss": _scalar(loss),
                    "seg": _scalar(loss_dict.get("seg")),
                    "sampled_seg": _scalar(loss_dict.get("sampled_seg")),
                    "aux": _scalar(loss_dict.get("aux")),
                    "diff": _scalar(loss_dict.get("diff")),
                    "heat": _scalar(loss_dict.get("heat")),
                    "points": _scalar(loss_dict.get("points")),
                }
                records.append(record)
                batch_index += 1
                self._log_training_record(
                    stage, "train_batch", {**record, "lr": self._current_lr()},
                    epoch, batch_index, len(loader),
                )

            if had_gradient:
                # AMP overflow 时 _optimizer_step 返回 False，不推 scheduling。
                if self._optimizer_step():
                    optimizer_steps += 1
                else:
                    skipped_groups += 1
            else:
                # 例如 stage C 整组都没有正峰：完全跳过优化器/调度/计数。
                self.optimizer.zero_grad(set_to_none=True)
                skipped_groups += 1

            if optimizer_steps > 0 and optimizer_steps % self.log_every == 0:
                latest = records[-1] if records else {}
                learning_rate = (
                    self.optimizer.param_groups[0]["lr"]
                    if self.optimizer.param_groups
                    else float("nan")
                )
                print(
                    f"[{stage}] epoch {epoch + 1} step {optimizer_steps} "
                    f"loss={latest.get('loss')} lr={learning_rate:.3e}",
                    flush=True,
                )

        stats: Dict[str, float] = {}
        for key in ("loss", "seg", "sampled_seg", "aux", "diff", "heat", "points"):
            values = [
                record[key]
                for record in records
                if record.get(key) is not None
            ]
            stats[key] = _mean_of(values) if values else float("nan")
        stats["optimizer_steps"] = float(optimizer_steps)
        stats["skipped_groups"] = float(skipped_groups)
        return stats

    def _train_epoch_grpo(
        self, stage: str, epoch: int, loader: DataLoader
    ) -> Dict[str, float]:
        self._set_train_modes(stage)
        records: List[Dict[str, Any]] = []
        for group_index, batch in enumerate(loader, start=1):
            prepared = self._prepare_batch(batch)
            record = self._grpo_group(prepared, epoch)
            self._log_training_record(
                stage, "train_group", record, epoch, group_index, len(loader)
            )
            # 逐轨迹详情只写 JSONL，不在整轮内保留所有样本记录。
            records.append({k: v for k, v in record.items() if k != "trajectory_diagnostics"})
            if group_index % self.log_every == 0:
                print(
                    f"[D] epoch {epoch + 1} group {group_index} "
                    f"reward={record.get('reward')} "
                    f"iou={record.get('iou')} "
                    f"points={record.get('point_count')} "
                    f"pts+/-={record.get('positive_point_count')}/"
                    f"{record.get('negative_point_count')} "
                    f"div={record.get('point_set_diversity')} "
                    f"active={record.get('active_group_rate')} "
                    f"inactive={record.get('inactive_group_rate')} "
                    f"var={record.get('reward_variance')} "
                    f"cost_scale={record.get('point_cost_scale')} "
                    f"lr={record.get('lr')} "
                    f"clip_frac={record.get('clip_fraction')}",
                    flush=True,
                )

        stats: Dict[str, float] = {}
        keys = (
            "reward",
            "reward_std",
            "reward_variance",
            "advantage_abs_mean",
            "active_group_rate",
            "inactive_group_rate",
            "invalid_groups",
            "iou",
            "f1",
            "cldice",
            "path",
            "point_count",
            "positive_point_count",
            "negative_point_count",
            "point_set_diversity",
            "unique_point_sets",
            "point_set_identical_ratio",
            "point_cost_scale",
            "point_cost",
            "lr",
            "grpo",
            "kl",
            "diffusion_grpo",
            "terminal_grpo",
            "diffusion_kl",
            "terminal_kl",
            "terminal_foreground_rate",
            "anchor",
            "logratio_abs_mean",
            "logratio_abs_max",
            "clip_fraction",
            "policy_terms",
            "quality_reward",
            "point_penalty",
            *PROMPT_METRIC_KEYS,
            *REWARD_GROUP_METRIC_KEYS,
        )
        for key in keys:
            values = [
                record[key] for record in records if record.get(key) is not None
            ]
            stats[key] = _mean_of(values) if values else float("nan")
        stats["optimizer_steps"] = float(
            sum(float(record.get("updates", 0.0)) for record in records)
        )
        stats["skipped_update_steps"] = float(
            sum(float(record.get("skipped_updates", 0.0)) for record in records)
        )
        stats["skipped_groups"] = float(
            sum(float(record.get("updates", 0.0)) == 0.0 for record in records)
        )
        return stats

    def _point_cost_scale(self, groups_processed: Optional[int] = None) -> float:
        """返回提示成本预热乘子；``1`` 表示已到配置值。

        定义：按**已处理的 D 组数**线性推进。第 1 组使用 ``1/N``，
        第 ``N`` 组到 ``1`` 后保持（``N = rl.point_cost_warmup_steps``，
        ``0`` 禁用）。
        使用独立的组计数器而不是 ``global_step``：整组奖励常量且
        ``kl_weight=0``、``anchor_weight=0`` 时更新会被安全跳过，
        ``global_step`` 永不增加；若按它预热，点成本会永远停在 0。
        AMP 溢出不会提前推进组计数以外的任何状态。
        """
        warmup_steps = int(getattr(self, "point_cost_warmup_steps", 0) or 0)
        if warmup_steps <= 0:
            return 1.0
        if groups_processed is None:
            # 唯一事实来源是独立组计数器；不退回 global_step，避免重新引入
            # “整组常量时预热永远停在 0”的问题。
            step = int(getattr(self, "grpo_groups_processed", 0))
        else:
            step = int(groups_processed)
        step = max(0, min(step, warmup_steps))
        return float(step) / float(warmup_steps)

    def _effective_reward_config(self):
        """D 阶段的提示成本从 0 线性升到配置值。

        验证始终使用 ``self.reward_config`` 的基准值，避免同一检查点在
        不同 epoch 的验证协议漂移。
        """
        if self._stage != "D" or self.point_cost_warmup_steps <= 0:
            return self.reward_config
        return replace(
            self.reward_config,
            point_cost=float(self.reward_config.point_cost)
            * self._point_cost_scale(),
        )

    @staticmethod
    def _point_set_signature(points: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[Any, ...]:
        """返回与点顺序无关的点集签名，用于组内 diversity 诊断。

        坐标取整到 SAM 输入像素；只用于日志，不参与损失。
        """
        coords, labels = points
        coords = coords.detach().cpu().reshape(-1, 2)
        labels = labels.detach().cpu().reshape(-1)
        if coords.shape[0] == 0:
            return ()
        if labels.numel() != coords.shape[0]:
            raise ValueError("点集坐标与标签数量不一致")
        rounded = torch.round(coords).to(torch.long)
        rows = [
            (int(labels[index]), int(rounded[index, 0]), int(rounded[index, 1]))
            for index in range(rounded.shape[0])
        ]
        return tuple(sorted(rows))

    @staticmethod
    def _payload_algorithm_name(payload: Mapping[str, Any]) -> str:
        algorithm = payload.get("algorithm")
        if isinstance(algorithm, Mapping):
            return str(algorithm.get("name", "")).strip().lower()
        return str(algorithm or "").strip().lower()

    def _validate_resume_algorithm(self, payload: Mapping[str, Any]) -> None:
        """拒绝把旧 DDPO 的 optimizer/scheduler 状态直接接入 GRPO。

        ``--init`` 只加载权重，不经过这里，因此旧 DDPO 快照仍可作为初始化。
        """
        stage = str(payload.get("stage", "")).upper()
        if getattr(self, "config", {}).get("training_protocol") == "joint_abc_v1":
            if payload.get("training_protocol") != "joint_abc_v1":
                raise ValueError("旧分阶段快照不能 --resume 联合训练；请用 --init 迁移权重")
        if stage == "ABC" and payload.get("abc_sam_frozen") is not True:
            raise ValueError("此 ABC 快照曾训练 SAM，优化器参数集合已改变；请用 --init 加载权重")
        if stage != "D":
            return
        version = int(payload.get("format_version", 1) or 1)
        name = self._payload_algorithm_name(payload)
        if name != "grpo" or version < 2:
            raise ValueError(
                "检测到旧 DDPO 阶段的 checkpoint：不能直接 --resume 到 GRPO "
                "（优化器、调度器和旧策略状态语义不同）。请改用 --init 只加载"
                "模型权重，然后从阶段 D 的第 0 个 epoch 重新开始。"
            )

    @staticmethod
    def _active_group_mask(reward_tensor: torch.Tensor) -> torch.Tensor:
        """标记有策略信号的组：只有奖励**完全常量**的组才跳过。

        不能把 ``std <= delta`` 的组判为无效；``delta`` 只限制优势分母，
        非零低方差组仍应以缩放后的优势参与更新。
        """
        if reward_tensor.ndim != 2:
            raise ValueError("reward_tensor 必须是 [G, B]")
        return reward_tensor.amax(dim=0) > reward_tensor.amin(dim=0)

    def _grpo_group(self, batch: Mapping[str, Any], epoch: int) -> Dict[str, Any]:
        """采样一组轨迹、计算 GRPO 优势，并做一次（或多次）更新。"""
        if getattr(self, "config", {}).get("training_protocol") == "joint_abc_v1":
            from diffpeak.joint_rl import joint_grpo_group
            return joint_grpo_group(self, batch, epoch)
        if self.group_size < 2:
            raise ValueError("D 阶段要求 rl.group_size >= 2 才能做组内相对优势")
        if self.ref_dit is None:
            raise RuntimeError("D 阶段缺少冻结参考 DiT")

        images = batch["image"]
        target_heatmaps = batch["heatmaps"]
        target_masks = batch["mask"]
        valid_sizes = batch["valid_sizes"]
        ids = batch["ids"]
        batch_size = int(images.shape[0])

        with torch.no_grad(), self._autocast():
            embeddings = encode_images(self.sam, images, valid_sizes)
            conditions_raw, _ = self.condition_encoder(images, embeddings)
            conditions = [item.detach() for item in conditions_raw]

        # 先推进独立组计数，再计算成本乘子：第 1 组为 1/N，第 N 组到
        # 配置值。即使本组最终没有 optimizer step（常量组且 KL/anchor
        # 关闭），预热仍会随“处理过的 D 组”前进，不会永远停在 0。
        self.grpo_groups_processed = int(
            getattr(self, "grpo_groups_processed", 0)
        ) + 1
        point_cost_scale = self._point_cost_scale()
        reward_config = self._effective_reward_config()
        effective_point_cost = float(reward_config.point_cost)

        targets_np: List[np.ndarray] = []
        for index in range(batch_size):
            valid = valid_sizes[index] or (self.image_size, self.image_size)
            height, width = int(valid[0]), int(valid[1])
            targets_np.append(
                (target_masks[index, 0, :height, :width] > 0.5)
                .detach()
                .cpu()
                .numpy()
            )

        rewards: List[List[float]] = []
        trajectories: List[List[Dict[str, Any]]] = []
        metric_rows: List[Dict[str, float]] = []
        point_signatures: List[List[Tuple[Any, ...]]] = []

        for trajectory_index in range(self.group_size):
            generator = make_generator(
                stable_seed(
                    self.seed, "D", int(epoch), *ids, trajectory_index
                )
            )
            with torch.no_grad(), self._autocast():
                heatmaps, trace = self.diffusion.sample(
                    self.dit,
                    conditions,
                    generator=generator,
                    record=True,
                )
                points = get_points(
                    self.peak_extractor,
                    heatmaps,
                    image_size=self.image_size,
                    valid_sizes=valid_sizes,
                )
                prompts = [points_to_prompt(item, self.device) for item in points]
                logits, _ = decode_with_prompts(
                    self.sam,
                    embeddings,
                    prompts,
                    output_size=(self.image_size, self.image_size),
                )
                probabilities = torch.sigmoid(logits)

                sample_rewards: List[float] = []
                trajectory_signatures: List[Tuple[Any, ...]] = []
                for index in range(batch_size):
                    valid = valid_sizes[index] or (self.image_size, self.image_size)
                    height, width = int(valid[0]), int(valid[1])
                    prediction = (
                        probabilities[index, 0, :height, :width] > self.threshold
                    ).detach().cpu().numpy()
                    coords, labels = points[index]
                    point_count = int(labels.numel())
                    positive_points = int((labels == 1).sum().item())
                    negative_points = int((labels == 0).sum().item())
                    metrics = compute_image_metrics(
                        prediction,
                        targets_np[index],
                        point_count,
                        reward_config,
                    )
                    sample_rewards.append(float(metrics["reward"]))
                    penalty = effective_point_cost * point_count / float(reward_config.point_reference)
                    quality = prompt_quality(
                        points[index], targets_np[index], heatmaps[index],
                        image_size=self.image_size,
                        threshold=tuple(
                            getattr(self.peak_extractor, f"{channel}_config", {}).get("threshold", 0.3)
                            for channel in ("positive", "negative")
                        ),
                    )
                    metric_rows.append(
                        {
                            **metrics,
                            **quality,
                            "quality_reward": float(metrics["reward"]) + penalty,
                            "point_penalty": penalty,
                            "point_count": float(point_count),
                            "positive_point_count": float(positive_points),
                            "negative_point_count": float(negative_points),
                        }
                    )
                    trajectory_signatures.append(
                        self._point_set_signature(points[index])
                    )
                rewards.append(sample_rewards)
                point_signatures.append(trajectory_signatures)
                # trace 已由扩散模块保存在 CPU；这里只保留轨迹。
                trajectories.append(trace)
            # 显式释放采样图，避免 G 条轨迹的完整计算图/激活驻留。
            del heatmaps, logits, probabilities

        reward_tensor = torch.tensor(rewards, dtype=torch.float64)  # [G, B]
        # GRPO：同一影像的 G 个候选构成一个组，组内标准化后才作为优势。
        # 组内奖励完全相同时优势为零，策略项不会用数值噪声放大出虚假信号。
        advantages = group_relative_advantages(
            reward_tensor, delta=self.advantage_std_delta
        )
        per_image_std = reward_tensor.std(dim=0, unbiased=False)
        per_image_variance = per_image_std.pow(2)
        # 只有奖励完全相同的组才真正没有策略信号；低方差组由 delta 限制
        # 优势幅度，而不是直接跳过。
        active_groups = self._active_group_mask(reward_tensor)
        reward_std = float(per_image_std.mean().cpu())
        reward_variance = float(per_image_variance.mean().cpu())
        active_group_rate = float(active_groups.double().mean().cpu())
        inactive_group_rate = 1.0 - active_group_rate
        invalid_groups = int(batch_size - int(active_groups.sum().item()))

        # 组内点集 diversity：每张图 G 个候选中有多少个不同的点集。
        # identical_ratio 是“该图所有候选点集完全相同（unique == 1）”
        # 的图所占比例，不能与 1 - diversity 混为一谈。
        diversity_values: List[float] = []
        unique_point_sets: List[float] = []
        identical_values: List[float] = []
        for index in range(batch_size):
            signatures = [
                point_signatures[trajectory_index][index]
                for trajectory_index in range(len(point_signatures))
            ]
            if signatures:
                unique_count = len(set(signatures))
                unique_point_sets.append(float(unique_count))
                diversity_values.append(
                    float(unique_count) / float(len(signatures))
                )
                identical_values.append(
                    1.0 if unique_count == 1 else 0.0
                )
            else:
                diversity_values.append(float("nan"))
                unique_point_sets.append(float("nan"))
                identical_values.append(float("nan"))
        point_set_diversity = (
            _mean_of(diversity_values) if diversity_values else float("nan")
        )
        point_set_identical_ratio = (
            _mean_of(identical_values) if identical_values else float("nan")
        )

        update_stats = self._grpo_update(
            conditions,
            target_heatmaps,
            trajectories,
            advantages,
            active_groups,
        )

        merged: Dict[str, Any] = dict(update_stats)
        merged.update(reward_group_diagnostics(metric_rows, advantages))
        merged["trajectory_diagnostics"] = [
            {
                **row,
                "image_id": ids[position % batch_size],
                "trajectory_index": position // batch_size,
                "advantage": float(advantages[position // batch_size, position % batch_size]),
            }
            for position, row in enumerate(metric_rows)
        ]
        merged["reward_std"] = reward_std
        merged["reward_variance"] = reward_variance
        merged["active_group_rate"] = active_group_rate
        merged["inactive_group_rate"] = inactive_group_rate
        merged["invalid_groups"] = float(invalid_groups)
        merged["point_set_diversity"] = point_set_diversity
        merged["unique_point_sets"] = _mean_of(unique_point_sets)
        merged["point_set_identical_ratio"] = point_set_identical_ratio
        merged["advantage_abs_mean"] = float(
            advantages.abs().mean().cpu()
        ) if advantages.numel() else float("nan")
        merged["point_cost_scale"] = point_cost_scale
        merged["point_cost"] = effective_point_cost
        for key in (
            "reward",
            "iou",
            "f1",
            "cldice",
            "path",
            "point_count",
            "positive_point_count",
            "negative_point_count",
            "quality_reward",
            "point_penalty",
            *PROMPT_METRIC_KEYS,
        ):
            values = [
                float(row[key])
                for row in metric_rows
                if key in row and math.isfinite(float(row[key]))
            ]
            merged[key] = _mean_of(values) if values else float("nan")
        return merged

    def _grpo_update(
        self,
        conditions: Sequence[torch.Tensor],
        target_heatmaps: torch.Tensor,
        trajectories: Sequence[Sequence[Mapping[str, Any]]],
        advantages: torch.Tensor,
        active_groups: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """固定整组轨迹后做 GRPO/KL/anchor 更新；每组 step 一次。"""
        assert self.optimizer is not None and self.scheduler is not None
        assert self.scaler is not None

        def _empty_stats() -> Dict[str, float]:
            return {
                "grpo": float("nan"),
                "kl": float("nan"),
                "anchor": float("nan"),
                "clip_fraction": float("nan"),
                "logratio_abs_mean": float("nan"),
                "logratio_abs_max": float("nan"),
                "lr": self._current_lr(),
                "updates": 0.0,
                "skipped_updates": 0.0,
                "policy_terms": 0.0,
            }

        if not trajectories:
            return _empty_stats()

        batch_size = int(conditions[0].shape[0])
        steps_per_trajectory = sum(len(trajectory) for trajectory in trajectories)
        # 单轮更新分母是 G*T*B；无效组在分子中贡献 0，但仍占用分母。
        per_epoch_terms = steps_per_trajectory * batch_size
        # 日志中的 grpo_sum/kl_sum 会累计 update_epochs 轮，最终分母同步乘上
        # update_epochs，避免同一组多轮更新时诊断值随轮数线性放大。
        total_terms = per_epoch_terms * max(1, int(self.update_epochs))
        if per_epoch_terms <= 0:
            return _empty_stats()

        active_mask: Optional[torch.Tensor] = None
        if active_groups is not None:
            active_mask = active_groups.detach().to(self.device).reshape(-1).bool()
            if active_mask.numel() == 1 and batch_size > 1:
                active_mask = active_mask.expand(batch_size)
            if active_mask.numel() != batch_size:
                raise ValueError(
                    "active_groups 数量与 batch size 不一致："
                    f"{active_mask.numel()} vs {batch_size}"
                )
        has_active_policy = (
            bool(active_mask.any()) if active_mask is not None else True
        )
        has_kl = self.kl_weight > 0
        has_anchor = self.anchor_weight > 0
        if not has_active_policy and not has_kl and not has_anchor:
            # 整组奖励完全相同且没有 KL/锚定项：不构造计算图、不 backward、
            # 不调用 optimizer.step()，也不消耗 scheduler/global_step。
            skipped = _empty_stats()
            skipped["skipped_updates"] = float(self.update_epochs)
            return skipped

        grpo_sum = 0.0
        kl_sum = 0.0
        anchor_values: List[float] = []
        clip_values: List[float] = []
        logratio_abs_sum = 0.0
        logratio_abs_max = 0.0
        logratio_count = 0
        update_steps = 0
        skipped_updates = 0
        policy_terms = 0
        epsilon = float(self.ppo_clip)

        for _ in range(self.update_epochs):
            self.optimizer.zero_grad(set_to_none=True)

            # 扩散锚定：只使用噪声 MSE，避免后训练把热图结构完全带偏。
            if self.anchor_weight > 0:
                with self._autocast():
                    anchor_losses = self.diffusion.training_losses(
                        self.dit, conditions, target_heatmaps
                    )
                    anchor = anchor_losses["diff"] * self.anchor_weight
                self.scaler.scale(anchor.float()).backward()
                anchor_values.append(float(anchor.detach().float().cpu()))

            # 策略项与 KL 项都依赖 rollout 轨迹；只有锚定时可以完全跳过
            # 轨迹重放，避免无意义的 backward。
            if has_active_policy or has_kl:
                for trajectory_index, trajectory in enumerate(trajectories):
                    advantage64 = (
                        advantages[trajectory_index]
                        .detach()
                        .to(self.device)
                        .reshape(-1)
                        .double()
                    )
                    if advantage64.numel() != batch_size:
                        raise ValueError(
                            "优势数量与 batch size 不一致："
                            f"{advantage64.numel()} vs {batch_size}"
                        )
                    sample_active = (
                        active_mask
                        if active_mask is not None
                        else torch.ones(batch_size, dtype=torch.bool, device=self.device)
                    )
                    active_float = sample_active.to(torch.float64)
                    active_count = int(sample_active.sum().item())
                    if active_count > 0:
                        # 记录真正参与策略项的 sample-step 数；分母仍然是
                        # G*T*B，不能用这个数替换。
                        policy_terms += len(trajectory) * active_count
                    # 常量组优势为 0；显式置零，避免 ratio*0 仍产生浮点噪声。
                    advantage64 = advantage64 * active_float

                    for step in trajectory:
                        # 高维动作统一保持 float32；log_prob 内部再升到 float64。
                        state = step["state"].to(
                            self.device, dtype=torch.float32, non_blocking=True
                        )
                        action = step["action"].to(
                            self.device, dtype=torch.float32, non_blocking=True
                        )
                        timestep = step["timestep"]
                        previous_timestep = step["previous_timestep"]
                        old_log_prob = (
                            step["old_log_prob"].to(self.device).reshape(-1).double()
                        )

                        # rollout 与 replay 使用同一 AMP 上下文和同一 eval DiT。
                        with self._autocast():
                            mean, std = self.diffusion.transition(
                                self.dit,
                                state,
                                timestep,
                                previous_timestep,
                                conditions,
                            )
                            new_log_prob = self.diffusion.log_prob(action, mean, std)

                        # 全维 SUM log-prob 的差值在 float64 中计算。
                        new_log_prob = new_log_prob.reshape(-1).double()
                        if old_log_prob.numel() == 1 and new_log_prob.numel() > 1:
                            old_log_prob = old_log_prob.expand_as(new_log_prob)
                        if old_log_prob.numel() != new_log_prob.numel():
                            raise ValueError(
                                "old/new log-prob 形状不一致："
                                f"{old_log_prob.numel()} vs {new_log_prob.numel()}"
                            )
                        raw_log_ratio = new_log_prob - old_log_prob
                        log_ratio = torch.clamp(raw_log_ratio, min=-20.0, max=20.0)
                        ratio = torch.exp(log_ratio)
                        clipped_mask = (
                            (ratio < 1.0 - epsilon)
                            | (ratio > 1.0 + epsilon)
                        )

                        if active_count > 0:
                            selected_clip = clipped_mask[sample_active]
                            clip_values.append(
                                float(selected_clip.double().mean().cpu())
                            )
                            selected_abs = raw_log_ratio.detach().abs()[sample_active]
                            logratio_abs_sum += float(selected_abs.sum().cpu())
                            logratio_abs_max = max(
                                logratio_abs_max,
                                float(selected_abs.max().cpu()),
                            )
                            logratio_count += active_count

                        # grpo_policy_loss 对传入的 B 个样本取平均；乘回
                        # batch_size 得到分子求和，再统一除以 G*T*B。
                        grpo_loss = grpo_policy_loss(
                            ratio, advantage64, epsilon
                        )
                        grpo_sum += float(grpo_loss.detach().cpu()) * batch_size

                        kl_element = torch.zeros(
                            (), dtype=torch.float64, device=self.device
                        )
                        if has_kl:
                            with torch.no_grad(), self._autocast():
                                reference_mean, _ = self.diffusion.transition(
                                    self.ref_dit,
                                    state,
                                    timestep,
                                    previous_timestep,
                                    conditions,
                                )
                            # 同方差高斯 KL = 0.5 * sum((mu - mu_ref)^2 / sigma^2)。
                            kl_element = gaussian_kl_divergence(
                                mean, reference_mean, std
                            )
                            kl_sum += float(kl_element.detach().sum().cpu())

                        loss = (
                            grpo_loss * batch_size
                            + self.kl_weight * kl_element.sum()
                        ) / float(per_epoch_terms)
                        self.scaler.scale(loss.float()).backward()

                        del state, action, mean, std, new_log_prob, log_ratio, ratio

            if self._optimizer_step():
                update_steps += 1
            else:
                skipped_updates += 1

        grpo_value = (
            grpo_sum / float(total_terms)
            if policy_terms > 0
            else float("nan")
        )
        kl_value = (
            kl_sum / float(total_terms) if has_kl else float("nan")
        )
        return {
            "grpo": grpo_value,
            "kl": kl_value,
            "anchor": _mean_of(anchor_values),
            "clip_fraction": _mean_of(clip_values),
            "logratio_abs_mean": (
                logratio_abs_sum / float(logratio_count)
                if logratio_count > 0
                else float("nan")
            ),
            "logratio_abs_max": (
                logratio_abs_max if logratio_count > 0 else float("nan")
            ),
            "lr": self._current_lr(),
            "updates": float(update_steps),
            "skipped_updates": float(skipped_updates),
            "policy_terms": float(policy_terms),
        }

    # ------------------------------------------------------------------
    # 验证
    # ------------------------------------------------------------------
    def validate(self, stage: str) -> Optional[Dict[str, float]]:
        """验证集评估。

        A 使用真实目标热图提示；B/C/D 使用逐图生成的提示。验证逐图进行，
        扩散种子只由 ``(seed, canonical_image_id)`` 决定，与阶段、顺序无关。
        """
        loader = self._build_val_loader()
        if loader is None:
            return None

        self._set_eval_all()
        rows: List[Dict[str, float]] = []
        point_counts: List[float] = []
        empty_predictions = 0
        empty_targets = 0
        count = 0
        try:
            with torch.no_grad(), self._autocast():
                for batch in loader:
                    prepared = self._prepare_batch(batch)
                    images = prepared["image"]
                    valid_sizes = prepared["valid_sizes"]
                    ids = prepared["ids"]
                    embeddings = encode_images(self.sam, images, valid_sizes)

                    if stage == "A":
                        heatmaps = prepared["heatmaps"]
                        points = get_points(
                            self.peak_extractor,
                            heatmaps,
                            image_size=self.image_size,
                            valid_sizes=valid_sizes,
                        )
                    else:
                        conditions, _ = self.condition_encoder(images, embeddings)
                        conditions = [item.detach() for item in conditions]
                        identifier = ids[0] if ids else ""
                        generator = make_generator(
                            evaluation_seed(self.seed, identifier)
                        )
                        heatmaps = self.diffusion.sample(
                            self.dit,
                            conditions,
                            generator=generator,
                            record=False,
                        )
                        points = get_points(
                            self.peak_extractor,
                            heatmaps,
                            image_size=self.image_size,
                            valid_sizes=valid_sizes,
                        )

                    prompts = [
                        points_to_prompt(item, self.device) for item in points
                    ]
                    if self.config.get("training_protocol") == "joint_abc_v1":
                        logits = self.sam.decode_heatmaps(
                            embeddings, heatmaps,
                            output_size=(self.image_size, self.image_size), valid_sizes=valid_sizes,
                        )
                    else:
                        logits, _ = decode_with_prompts(
                            self.sam, embeddings, prompts,
                            output_size=(self.image_size, self.image_size),
                        )
                    probabilities = torch.sigmoid(logits)

                    for index in range(int(images.shape[0])):
                        valid = valid_sizes[index] or (
                            self.image_size,
                            self.image_size,
                        )
                        height, width = int(valid[0]), int(valid[1])
                        prediction = (
                            probabilities[index, 0, :height, :width]
                            > self.threshold
                        ).detach().cpu().numpy()
                        target = (
                            prepared["mask"][index, 0, :height, :width] > 0.5
                        ).detach().cpu().numpy()
                        point_count = (0 if self.config.get("training_protocol") == "joint_abc_v1"
                                       else int(points[index][1].numel()))
                        metrics = compute_image_metrics(
                            prediction,
                            target,
                            point_count,
                            self.reward_config,
                        )
                        metrics.update(prompt_quality(
                            points[index], target, heatmaps[index],
                            image_size=self.image_size,
                            threshold=tuple(
                                getattr(self.peak_extractor, f"{channel}_config", {}).get("threshold", 0.3)
                                for channel in ("positive", "negative")
                            ),
                        ))
                        penalty = self.reward_config.point_cost * point_count / self.reward_config.point_reference
                        metrics["point_penalty"] = float(penalty)
                        metrics["quality_reward"] = float(metrics["reward"]) + float(penalty)
                        rows.append(metrics)
                        point_counts.append(float(point_count))
                        if not prediction.any():
                            empty_predictions += 1
                        if not target.any():
                            empty_targets += 1
                        count += 1
                        if (
                            self.val_limit is not None
                            and count >= int(self.val_limit)
                        ):
                            break
                    if self.val_limit is not None and count >= int(self.val_limit):
                        break
        finally:
            self._set_train_modes(stage)

        if not rows:
            return None
        summary = aggregate_metrics(rows)
        for key in (*PROMPT_METRIC_KEYS, "quality_reward", "point_penalty"):
            values = [float(row[key]) for row in rows if math.isfinite(float(row[key]))]
            summary[key] = _mean_of(values) if values else float("nan")
        summary["mean_point_count"] = _mean_of(point_counts)
        summary["empty_prediction_rate"] = float(empty_predictions) / float(len(rows))
        summary["empty_target_rate"] = float(empty_targets) / float(len(rows))
        return summary

    # ------------------------------------------------------------------
    # 阶段与 epoch 调度
    # ------------------------------------------------------------------
    def _prepare_stage(
        self, stage: str, epochs: int, steps_per_epoch: int
    ) -> None:
        self._stage = stage
        self._set_requires_grad(stage)
        if stage == "D":
            if self.group_size < 2:
                raise ValueError("D 阶段要求 rl.group_size >= 2")
            # rollout 与 replay 必须给出相同的高斯参数；固定 cuDNN 算法选择，
            # 避免同一输入在两次前向中产生非确定性差异。
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            if self.ref_dit is None:
                self.ref_dit = copy.deepcopy(self.dit)
            self.ref_dit = self.ref_dit.to(self.device)
            self.ref_dit.eval()
            for parameter in self.ref_dit.parameters():
                parameter.requires_grad_(False)
            for name, module in (
                ("ref_condition_encoder", self.condition_encoder),
                ("ref_sam_decoder", _decoder_module(self.sam)),
            ):
                reference = getattr(self, name, None)
                if reference is None:
                    reference = copy.deepcopy(module)
                reference.to(self.device).eval().requires_grad_(False)
                setattr(self, name, reference)
        else:
            self.ref_dit = None
            self.ref_condition_encoder = None
            self.ref_sam_decoder = None

        self._set_train_modes(stage)
        self._build_optimizer(stage, epochs, steps_per_epoch)
        self.best_metric = None
        self.global_step = 0
        self.grpo_groups_processed = 0

    def run_stage(
        self,
        stage: str,
        start_epoch: int = 0,
        resume_payload: Optional[Mapping[str, Any]] = None,
    ) -> Optional[Dict[str, float]]:
        """运行一个阶段；返回最后一个验证 epoch 的指标。"""
        stage = str(stage).upper()
        if stage not in STAGE_ORDER:
            raise ValueError(f"未知阶段：{stage}；请使用 ABC 联合预训练或 D 后训练")
        if self.config.get("training_protocol") != "joint_abc_v1":
            raise ValueError("训练配置需设置 training_protocol: joint_abc_v1，并定义 stages.ABC/D")
        if stage not in self.stage_cfgs:
            raise ValueError(f"配置缺少 stages.{stage}")
        # 必须在 probe_loader（用于计算调度步数）和 _prepare_stage（构建
        # 优化器/调度器）之前解析阶段吞吐覆盖。
        self._apply_stage_config(stage)
        epochs = int(self.stage_cfgs[stage].get("epochs", 0))
        start_epoch = int(start_epoch)
        if epochs <= 0 or start_epoch >= epochs:
            print(f"[{stage}] 无需运行（epochs={epochs}, start={start_epoch}）", flush=True)
            return None

        probe_loader = self._build_train_loader(start_epoch)
        if stage == "D":
            # 每个轨迹组按 update_epochs 做优化器 step。
            steps_per_epoch = len(probe_loader) * self.update_epochs
        else:
            steps_per_epoch = int(
                math.ceil(len(probe_loader) / float(self.accumulation_steps))
            )
        del probe_loader
        # run_stage 自身负责恢复模型权重，不能依赖 CLI 预加载；
        # D 阶段的 ref_dit 也在这里严格恢复。
        if resume_payload is not None:
            # 先检查算法归属：旧 DDPO 的 D 阶段快照即使带有 ref_dit，也不能
            # 把其 optimizer/scheduler 状态直接接到 GRPO。
            self._validate_resume_algorithm(resume_payload)
            self.load_models_from_payload(
                resume_payload,
                include_ref=True,
                allow_missing_ref=stage != "D",
            )
            # 权重已装入模块；释放快照中的大块模型 state_dict，避免整个
            # 阶段训练期间多保留一份 CPU 权重。优化器/RNG 等仍在 payload 中。
            if isinstance(resume_payload, dict):
                resume_payload.pop("models", None)
        self._prepare_stage(stage, epochs, steps_per_epoch)

        if resume_payload is not None:
            self._restore_resume(resume_payload)
            start_epoch = int(resume_payload.get("epoch", start_epoch))
            if start_epoch >= epochs:
                print(f"[{stage}] checkpoint 已在该阶段结束", flush=True)
                return None

        logger = getattr(self, "experiment_logger", None)
        if logger is not None:
            logger.event(
                "stage_start", stage=stage, start_epoch=start_epoch, epochs=epochs,
                resumed=resume_payload is not None, global_step=self.global_step,
                schedule={"total_steps": self.scheduler_total_steps,
                          "warmup_steps": self.scheduler_warmup_steps,
                          "min_lr_ratio": self.scheduler_min_lr_ratio,
                          "learning_rate": self.scheduler_stage_lr},
                algorithm={"name": self.rl_algorithm, "group_size": self.group_size,
                           "update_epochs": self.update_epochs, "ppo_clip": self.ppo_clip,
                           "kl_weight": self.kl_weight, "anchor_weight": self.anchor_weight,
                           "advantage_std_delta": self.advantage_std_delta,
                           "point_cost_warmup_steps": self.point_cost_warmup_steps},
            )
        last_validation: Optional[Dict[str, float]] = None
        for epoch in range(start_epoch, epochs):
            epoch_started = time.monotonic()
            if logger is not None and self.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(self.device)
            self._epoch = epoch
            train_loader = self._build_train_loader(epoch)
            if stage == "D":
                train_stats = self._train_epoch_grpo(stage, epoch, train_loader)
            else:
                train_stats = self._train_epoch_supervised(stage, epoch, train_loader)
            del train_loader

            validation: Optional[Dict[str, float]] = None
            if (epoch + 1) % self.val_every == 0 or epoch + 1 == epochs:
                validation = self.validate(stage)
            last_validation = validation if validation is not None else last_validation
            score = (
                float(validation["score"])
                if validation is not None and validation.get("score") is not None
                else None
            )

            self.history.append(
                {
                    "stage": stage,
                    "epoch": epoch + 1,
                    "train": train_stats,
                    "val": validation,
                }
            )
            self._write_metrics()
            if logger is not None:
                logger.log_scalars(stage, "train_epoch", train_stats, epoch + 1, epoch + 1)
                if validation is not None:
                    logger.log_scalars(stage, "val_epoch", validation, epoch + 1, epoch + 1)
                system_stats = {
                    "epoch_seconds": time.monotonic() - epoch_started,
                    "lr_next": self._current_lr(),
                    "optimizer_step": self.global_step,
                    "amp_scale": float(self.scaler.get_scale()),
                }
                if self.device.type == "cuda":
                    system_stats["peak_cuda_memory_mb"] = (
                        torch.cuda.max_memory_allocated(self.device) / 1024**2
                    )
                logger.log_scalars(stage, "system", system_stats, epoch + 1, epoch + 1)
                logger.flush()

            # 先更新 best，再写 last；这样 last.pt 中的 best_metric 不会落后，
            # 中断后从 last.pt 恢复不会回退最佳指标跟踪。
            if score is not None and (
                self.best_metric is None or score > float(self.best_metric)
            ):
                self.best_metric = float(score)
                self.save_best(stage, epoch + 1)
                if logger is not None:
                    logger.event("checkpoint", kind="best", stage=stage, epoch=epoch + 1,
                                 path=str(self.output_dir / "best.pt"), score=self.best_metric)

            if (epoch + 1) % self.save_every == 0 or epoch + 1 == epochs:
                self.save_last(stage, epoch + 1, self.best_metric)
                if logger is not None:
                    logger.event("checkpoint", kind="last", stage=stage, epoch=epoch + 1,
                                 path=str(self.output_dir / "last.pt"))

            print(
                f"[{stage}] epoch {epoch + 1}/{epochs} "
                f"train_loss={train_stats.get('loss', train_stats.get('reward'))} "
                f"val_iou={None if validation is None else validation.get('iou')} "
                f"best={self.best_metric}"
                + (
                    f" clip_frac={train_stats.get('clip_fraction')}"
                    if stage == "D"
                    else ""
                )
                + (
                    f" skipped={train_stats.get('skipped_groups')}"
                    if train_stats.get("skipped_groups", 0.0) > 0
                    else ""
                ),
                flush=True,
            )

        # 阶段结束：确保 last/best 存在，并保留一个仅用于 --init 的阶段文件。
        self.save_last(stage, epochs, self.best_metric)
        if self.best_metric is None:
            self._copy_last_to_best()
        self._link_stage_checkpoint(stage)
        if logger is not None:
            logger.event("stage_end", stage=stage, epochs=epochs,
                         checkpoint=str(self.output_dir / f"stage_{stage}.pth"),
                         best_metric=self.best_metric)
            logger.flush()
        print(
            f"[{stage}] 完成；last={self.output_dir / 'last.pt'} "
            f"best_metric={self.best_metric}",
            flush=True,
        )
        return last_validation

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------
    def _capture_rng(self) -> Dict[str, Any]:
        state: Dict[str, Any] = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            try:
                state["cuda"] = torch.cuda.get_rng_state_all()
            except Exception:
                pass
        return state

    def _restore_rng(self, state: Optional[Mapping[str, Any]]) -> None:
        if not state:
            return
        try:
            if "python" in state:
                random.setstate(state["python"])
            if "numpy" in state:
                np.random.set_state(state["numpy"])
            if "torch" in state:
                torch.set_rng_state(state["torch"].cpu().to(torch.uint8))
            if "cuda" in state and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(
                    [item.cpu() for item in state["cuda"]]
                )
        except Exception as exc:  # pragma: no cover - 跨设备恢复的防御性警告
            warnings.warn(f"恢复 RNG 状态失败：{exc}", RuntimeWarning)

    @staticmethod
    def _to_cpu_tree(value: Any) -> Any:
        if torch.is_tensor(value):
            return value.detach().cpu()
        if isinstance(value, Mapping):
            return {key: DiffPeakEngine._to_cpu_tree(item) for key, item in value.items()}
        if isinstance(value, list):
            return [DiffPeakEngine._to_cpu_tree(item) for item in value]
        if isinstance(value, tuple):
            return tuple(DiffPeakEngine._to_cpu_tree(item) for item in value)
        return value

    def _model_states(self) -> Dict[str, Any]:
        states: Dict[str, Any] = {
            "sam": self._to_cpu_tree(self.sam.state_dict()),
            "condition_encoder": self._to_cpu_tree(
                self.condition_encoder.state_dict()
            ),
            "dit": self._to_cpu_tree(self.dit.state_dict()),
        }
        # 始终写入 diffusion 字段（包含 beta/alpha 调度 buffer），
        # 严格加载器要求四个必需模块齐全。
        states["diffusion"] = self._to_cpu_tree(self.diffusion.state_dict())
        if self.ref_dit is not None:
            states["ref_dit"] = self._to_cpu_tree(self.ref_dit.state_dict())
        for name in ("ref_condition_encoder", "ref_sam_decoder"):
            reference = getattr(self, name, None)
            if reference is not None:
                states[name] = self._to_cpu_tree(reference.state_dict())
        return states

    def _resolved_config(self) -> Dict[str, Any]:
        """返回写入 checkpoint 的绝对路径配置，换工作目录后仍可使用。"""
        resolved = copy.deepcopy(self.config)
        data = dict(resolved.get("data", {}))
        data["root"] = str(Path(self.data_root).expanduser().resolve())
        data["train_split"] = str(
            Path(self._resolve_train_split()).expanduser().resolve()
        )
        val_split = self._resolve_val_split()
        data["val_split"] = (
            None
            if val_split is None
            else str(Path(val_split).expanduser().resolve())
        )
        resolved["data"] = data
        model = dict(resolved.get("model", {}))
        model["sam_checkpoint"] = str(
            Path(self.sam_checkpoint).expanduser().resolve()
        )
        # 快照必须能独立说明使用的是哪一代 SAM 与哪个 Hiera 配置；
        # 加载时会据此拒绝 SAM1 快照或错误骨干。
        model["sam_config"] = str(
            model.get(
                "sam_config", getattr(self.sam, "config_name", "")
            )
        )
        model["sam_architecture"] = str(
            model.get(
                "sam_architecture", getattr(self.sam, "architecture", "")
            )
        )
        resolved["model"] = model
        resolved["output_dir"] = str(
            Path(self.output_dir).expanduser().resolve()
        )
        return resolved

    def _checkpoint_payload(
        self, stage: str, epoch: int, best_metric: Optional[float]
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            # v3：SAM2.1 迁移。SAM 权重键与 SAM1 完全不兼容，v2 快照不能
            # 继续 --resume/--init；payload 同时记录骨干标识。
            "format_version": 4,
            "training_protocol": self.config.get("training_protocol"),
            "abc_sam_frozen": True,
            "sam": {
                "architecture": str(getattr(self.sam, "architecture", "")),
                "config": str(getattr(self.sam, "config_name", "")),
                "apply_postprocessing": bool(
                    getattr(self.sam, "apply_postprocessing", False)
                ),
            },
            "algorithm": {
                "name": self.rl_algorithm,
                "version": 2,
                "group_size": int(self.group_size),
                "advantage_std_delta": float(self.advantage_std_delta),
                "ppo_clip": float(self.ppo_clip),
                "kl_weight": float(self.kl_weight),
                "anchor_weight": float(self.anchor_weight),
                "update_epochs": int(self.update_epochs),
                "point_cost_warmup_steps": int(self.point_cost_warmup_steps),
                "grpo_groups_processed": int(
                    getattr(self, "grpo_groups_processed", 0)
                ),
                "reward_point_cost": float(
                    getattr(self.reward_config, "point_cost", 0.0)
                ),
            },
            "schedule": {
                "name": "warmup_cosine",
                "stage": stage,
                "learning_rate": float(self.scheduler_stage_lr),
                "total_steps": int(self.scheduler_total_steps),
                "warmup_steps": int(self.scheduler_warmup_steps),
                "min_lr_ratio": float(self.scheduler_min_lr_ratio),
                "steps_per_epoch": int(self.scheduler_steps_per_epoch),
                "epochs": int(self.scheduler_epochs),
            },
            "config": self._resolved_config(),
            "stage": stage,
            "epoch": int(epoch),
            "global_step": int(self.global_step),
            "best_metric": None if best_metric is None else float(best_metric),
            "history": self._to_cpu_tree(self.history),
            "models": self._model_states(),
            "rng": self._capture_rng(),
        }
        if self.optimizer is not None:
            payload["optimizer"] = self._to_cpu_tree(self.optimizer.state_dict())
        if self.scheduler is not None:
            payload["scheduler"] = self._to_cpu_tree(self.scheduler.state_dict())
        if self.scaler is not None:
            payload["scaler"] = self._to_cpu_tree(self.scaler.state_dict())
        return payload

    def _save_payload(self, path: Path, payload: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        torch.save(dict(payload), temporary)
        os.replace(temporary, path)

    def save_last(self, stage: str, epoch: int, best_metric: Optional[float]) -> Path:
        path = self.output_dir / "last.pt"
        self._save_payload(
            path, self._checkpoint_payload(stage, epoch, best_metric)
        )
        return path

    def save_best(self, stage: str, epoch: int) -> Path:
        path = self.output_dir / "best.pt"
        self._save_payload(
            path,
            self._checkpoint_payload(stage, epoch, self.best_metric),
        )
        return path

    def _copy_last_to_best(self) -> None:
        last = self.output_dir / "last.pt"
        best = self.output_dir / "best.pt"
        if last.is_file():
            shutil.copy2(last, best)

    def _link_stage_checkpoint(self, stage: str) -> None:
        """用硬链接保留阶段结束权重，避免额外复制整份 SAM 权重。"""
        last = self.output_dir / "last.pt"
        target = self.output_dir / f"stage_{stage}.pth"
        if not last.is_file():
            return
        try:
            if target.exists():
                target.unlink()
            try:
                os.link(last, target)
            except OSError:
                shutil.copy2(last, target)
        except OSError as exc:  # pragma: no cover - 文件系统限制
            warnings.warn(f"创建阶段权重文件失败：{exc}", RuntimeWarning)

    def _write_metrics(self) -> None:
        path = self.output_dir / "metrics.json"
        temporary = path.with_name(path.name + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(self.history, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary, path)

    def _restore_resume(self, payload: Mapping[str, Any]) -> None:
        self._validate_resume_algorithm(payload)
        payload_stage = str(payload.get("stage", self._stage)).upper()
        if self._stage is not None and payload_stage != self._stage:
            raise ValueError(
                f"checkpoint 属于阶段 {payload_stage}，不能恢复阶段 {self._stage}"
            )
        for required in ("optimizer", "scheduler", "scaler"):
            if required not in payload:
                raise ValueError(
                    f"用于 --resume 的 checkpoint 缺少 {required} 字段，"
                    "请使用 last.pt/best.pt/stage_X.pth 完整快照"
                )
        assert self.optimizer is not None and self.scheduler is not None
        assert self.scaler is not None
        self.optimizer.load_state_dict(payload["optimizer"])
        self.scheduler.load_state_dict(payload["scheduler"])
        self.scaler.load_state_dict(payload["scaler"])

        # 算法超参与 warmup/cosine 配置随 checkpoint 恢复；旧快照没有这些
        # 字段时保持当前配置。匹配完整 config 时值相同，这里主要是让日志
        # 与后续 checkpoint 明确反映训练时的实际协议。
        algorithm = payload.get("algorithm")
        if isinstance(algorithm, Mapping):
            self.group_size = int(algorithm.get("group_size", self.group_size))
            self.advantage_std_delta = float(
                algorithm.get("advantage_std_delta", self.advantage_std_delta)
            )
            self.ppo_clip = float(algorithm.get("ppo_clip", self.ppo_clip))
            self.kl_weight = float(algorithm.get("kl_weight", self.kl_weight))
            self.anchor_weight = float(
                algorithm.get("anchor_weight", self.anchor_weight)
            )
            self.update_epochs = max(
                1, int(algorithm.get("update_epochs", self.update_epochs))
            )
            self.point_cost_warmup_steps = max(
                0,
                int(
                    algorithm.get(
                        "point_cost_warmup_steps",
                        self.point_cost_warmup_steps,
                    )
                ),
            )
            self.grpo_groups_processed = max(
                0,
                int(
                    algorithm.get(
                        "grpo_groups_processed",
                        self.grpo_groups_processed,
                    )
                ),
            )
        schedule = payload.get("schedule")
        if isinstance(schedule, Mapping):
            self.scheduler_stage_lr = float(
                schedule.get("learning_rate", self.scheduler_stage_lr)
            )
            self.scheduler_total_steps = int(
                schedule.get("total_steps", self.scheduler_total_steps)
            )
            self.scheduler_warmup_steps = int(
                schedule.get("warmup_steps", self.scheduler_warmup_steps)
            )
            self.scheduler_min_lr_ratio = float(
                schedule.get("min_lr_ratio", self.scheduler_min_lr_ratio)
            )
            self.scheduler_steps_per_epoch = int(
                schedule.get("steps_per_epoch", self.scheduler_steps_per_epoch)
            )
            self.scheduler_epochs = int(
                schedule.get("epochs", self.scheduler_epochs)
            )
            self.scheduler_lr_schedule = str(
                schedule.get("name", self.scheduler_lr_schedule)
            )

        self.global_step = int(payload.get("global_step", 0))
        self.best_metric = payload.get("best_metric")
        if self.best_metric is not None:
            self.best_metric = float(self.best_metric)
        self.history = list(payload.get("history", []))
        self._restore_rng(payload.get("rng"))
        self._set_train_modes(self._stage or payload_stage)
        print(
            f"已恢复阶段 {payload_stage}：epoch={payload.get('epoch')} "
            f"global_step={self.global_step}",
            flush=True,
        )

    def load_models_from_payload(
        self,
        payload: Mapping[str, Any],
        *,
        include_ref: bool = True,
        allow_missing_ref: bool = True,
    ) -> Dict[str, Any]:
        """严格加载模型权重。

        ``allow_missing_ref=True`` 时，D 快照缺少 ``ref_dit`` 也可只做
        ``--init``；真正的 ``--resume`` 由 :meth:`run_stage` 传 ``False``，
        以保持参考策略的严格恢复。
        """
        report = load_checkpoint_models(
            self.models, payload, include_ref=False
        )
        source = payload["models"]
        payload_stage = str(payload.get("stage", "")).upper()
        if include_ref and "ref_dit" in source:
            if self.ref_dit is None:
                self.ref_dit = copy.deepcopy(self.dit)
            self.ref_dit.load_state_dict(source["ref_dit"], strict=True)
            self.ref_dit = self.ref_dit.to(self.device).eval()
            for parameter in self.ref_dit.parameters():
                parameter.requires_grad_(False)
        elif include_ref and payload_stage == "D" and not allow_missing_ref:
            raise ValueError("D 阶段 checkpoint 缺少 ref_dit，不能恢复 KL 参考策略")
        if include_ref:
            for name, module in (
                ("ref_condition_encoder", self.condition_encoder),
                ("ref_sam_decoder", _decoder_module(self.sam)),
            ):
                if name in source:
                    reference = copy.deepcopy(module)
                    reference.load_state_dict(source[name], strict=True)
                    reference.to(self.device).eval().requires_grad_(False)
                    setattr(self, name, reference)
                elif payload_stage == "D" and not allow_missing_ref and self.config.get("training_protocol") == "joint_abc_v1":
                    raise ValueError(f"联合 D 快照缺少 {name}，不能恢复固定参考策略")
        return report

    def load_init_weights(self, path: Union[str, Path]) -> Mapping[str, Any]:
        """``--init``：只加载模型权重，不恢复优化器/epoch/RNG。

        快照含 ref_dit 时保留该冻结参考；缺失时允许只初始化当前策略，
        随后由 ``_prepare_stage`` 从当前 DiT 建立新的冻结参考模型。
        """
        payload = read_checkpoint(path)
        report = self.load_models_from_payload(
            payload, include_ref=True, allow_missing_ref=True
        )
        print(f"已从 {path} 初始化模型权重（不含优化器状态）", flush=True)
        return report

    def read_checkpoint(self, path: Union[str, Path]) -> Any:
        """读取 checkpoint 的便捷入口。"""
        return read_checkpoint(path)
