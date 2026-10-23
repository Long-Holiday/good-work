"""Signed ridge diffusion training: ABC supervision and diffusion-only GRPO.

SAM stays frozen and in eval mode while retaining derivatives with respect to fields.
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
    FIELD_METRIC_KEYS,
    field_quality,
)
from diffpeak.inference import (
    aggregate_metrics,
    build_models,
    compute_image_metrics,
    compute_resized_size,
    encode_images,
    evaluation_seed,
    load_checkpoint_models,
    validate_checkpoint_protocol,
    validate_signed_ridge_config,
    SIGNED_RIDGE_PROTOCOL,
    make_generator,
    make_reward_config,
    normalize_valid_sizes,
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
        validate_signed_ridge_config(self.config)
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

        self.models, _ = build_models(
            self.config, self.device, sam_checkpoint=self.sam_checkpoint
        )
        self.sam = self.models["sam"]
        self.condition_encoder = self.models["condition_encoder"]
        self.dit = self.models["dit"]
        self.diffusion = self.models["diffusion"]
        self.reward_config = make_reward_config(self.config)
        if self.reward_config.point_cost != 0:
            raise ValueError("signed_ridge_v1 不使用离散点，reward.point_cost 必须为0")

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
        self.global_step = 0
        # 处理组数独立于优化器步数，常量奖励组也计入诊断。
        self.grpo_groups_processed = 0
        self.best_metric: Optional[float] = None
        self.history: List[Dict[str, Any]] = []
        self._stage: Optional[str] = None
        self._epoch = 0
        self._train_split_cache: Optional[Path] = None
        self._val_split_cache: Optional[Path] = None
        self._val_split_cached = False
        self.experiment_logger = None

        # 提前解析划分路径：train 缺失或显式配置的 val 缺失时立即失败；
        # val_split=None 才会禁用验证。
        self._resolve_train_split()
        self._resolve_val_split()

    # ------------------------------------------------------------------
    # 配置与数据
    # ------------------------------------------------------------------

    def _dataset_kwargs(self, split_file: Path, augment: bool) -> Dict[str, Any]:
        return {
            "root": str(self.data_root),
            "split_file": str(split_file),
            "augment": bool(augment),
            "image_size": self.image_size,
            "heatmap_size": self.heatmap_size,
            "seed": self.seed,
            "ridge_config": self.data_cfg.get("ridge_config"),
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
        if stage not in STAGE_ORDER:
            raise ValueError(f"未知阶段：{stage}")
        self.sam.requires_grad_(False).eval()
        self.condition_encoder.requires_grad_(True)
        self.dit.requires_grad_(True)

    def _set_train_modes(self, stage: str) -> None:
        self.sam.requires_grad_(False).eval()
        self.condition_encoder.train(stage == "ABC")
        self.dit.train(stage == "ABC")
        for reference in (self.ref_dit, self.ref_condition_encoder):
            if reference is not None:
                reference.requires_grad_(False).eval()

    def _set_eval_all(self) -> None:
        for module in self.models.values():
            module.eval()
        if self.ref_dit is not None:
            self.ref_dit.eval()

    def _trainable_parameters(self) -> List[torch.nn.Parameter]:
        parameters: List[torch.nn.Parameter] = []
        seen = set()
        for module in (self.condition_encoder, self.dit):
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
        heatmaps = heatmaps.float()
        if heatmaps.ndim != 4 or heatmaps.shape[1] != 1:
            raise ValueError("heatmaps 必须是 B x 1 x H x W 有符号山脊场")
        if not torch.isfinite(heatmaps).all() or bool((heatmaps.abs() > 1).any()):
            raise ValueError("目标山脊场必须有限且在 [-1,1] 内")
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

        for key in ("image", "mask", "skeleton", "heatmaps", "field_valid_mask", "field_regions"):
            if key in prepared:
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
        if stage != "ABC":
            raise ValueError(f"未知监督阶段：{stage}")
        images, masks = batch["image"], batch["mask"]
        valid_sizes = batch["valid_sizes"]
        with torch.no_grad():
            embeddings = encode_images(self.sam, images, valid_sizes)
        conditions, auxiliary_logits = self.condition_encoder(images, embeddings)
        losses = self.diffusion.training_losses(
            self.dit, conditions, batch["heatmaps"], return_prediction=True,
            valid_mask=batch.get("field_valid_mask"), field_regions=batch.get("field_regions"),
        )
        reconstructed = self.sam.decode_heatmaps(
            embeddings, losses["heat_pred"], output_size=tuple(masks.shape[-2:]),
            valid_sizes=valid_sizes,
        )
        generated_heatmaps = self.diffusion.sample_differentiable(
            self.dit, conditions,
            generator=make_generator(stable_seed(self.seed, "ABC", epoch, *batch["ids"])),
        )
        generated = self.sam.decode_heatmaps(
            embeddings, generated_heatmaps, output_size=tuple(masks.shape[-2:]),
            valid_sizes=valid_sizes,
        )
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
        auxiliary = auxiliary_loss(auxiliary_logits, masks, batch["skeleton"], valid_sizes=valid_sizes)
        total = (
            losses["diff"] + self.heat_weight * losses["heat"]
            + self.aux_weight * auxiliary
            + float(self.train_cfg.get("seg_weight", 1.0)) * segmentation
            + float(self.train_cfg.get("sampled_seg_weight", 1.0)) * sampled_segmentation
        )
        return {"loss": total, "diff": losses["diff"], "heat": losses["heat"],
                "aux": auxiliary, "seg": segmentation, "sampled_seg": sampled_segmentation}

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
                # 没有可反传损失时不推进优化器和调度器。
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
        for key in ("loss", "seg", "sampled_seg", "aux", "diff", "heat"):
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
                    f"active={record.get('active_group_rate')} "
                    f"var={record.get('reward_variance')} "
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
            "lr",
            "grpo",
            "kl",
            "diffusion_grpo",
            "diffusion_kl",
            "anchor",
            "logratio_abs_mean",
            "logratio_abs_max",
            "clip_fraction",
            "policy_terms",
            "quality_reward",
            "prediction_foreground_rate",
            *FIELD_METRIC_KEYS,
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


    def _effective_reward_config(self):
        return self.reward_config


    @staticmethod
    def _payload_algorithm_name(payload: Mapping[str, Any]) -> str:
        algorithm = payload.get("algorithm")
        if isinstance(algorithm, Mapping):
            return str(algorithm.get("name", "")).strip().lower()
        return str(algorithm or "").strip().lower()

    def _validate_resume_algorithm(self, payload: Mapping[str, Any]) -> None:
        validate_checkpoint_protocol(payload, self.config)
        if int(payload.get("format_version", 0)) < 5:
            raise ValueError("旧快照不能 --resume；请显式 --init --migrate-legacy 迁移兼容权重")
        saved_train = payload.get("config", {}).get("train", {})
        for key in (("seg_weight", "sampled_seg_weight", "cldice_weight") if "config" in payload else ()):
            default = 0.2 if key == "cldice_weight" else 1.0
            # Missing segmentation weights identify the field-only objective.
            saved = float(saved_train.get(key, 0.0 if key != "cldice_weight" else default))
            current = float(self.config.get("train", {}).get(key, default))
            if saved != current:
                raise ValueError(
                    "checkpoint 的 ABC 分割监督目标与当前配置不同，不能 --resume；"
                    "请用 --config 和 --init 复用权重并开始新实验"
                )
        if str(payload.get("stage", "")).upper() == "D":
            algorithm = payload.get("algorithm", {})
            if self._payload_algorithm_name(payload) != "grpo" or algorithm.get("version") != 3:
                raise ValueError("D 快照必须使用仅扩散轨迹的 Gaussian GRPO v3")

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
        from diffpeak.joint_rl import joint_grpo_group
        return joint_grpo_group(self, batch, epoch)


    # ------------------------------------------------------------------
    # 验证
    # ------------------------------------------------------------------
    def validate(self, stage: str) -> Optional[Dict[str, float]]:
        """Evaluate one generated field per image under the inference sampling protocol."""
        loader = self._build_val_loader()
        if loader is None:
            return None
        self._set_eval_all()
        rows: List[Dict[str, float]] = []
        try:
            with torch.no_grad(), self._autocast():
                for batch in loader:
                    prepared = self._prepare_batch(batch)
                    images, sizes = prepared["image"], prepared["valid_sizes"]
                    embeddings = encode_images(self.sam, images, sizes)
                    conditions, _ = self.condition_encoder(images, embeddings)
                    fields = self.diffusion.sample(
                        self.dit, conditions,
                        generator=make_generator(evaluation_seed(self.seed, prepared["ids"][0])),
                    )
                    logits = self.sam.decode_heatmaps(
                        embeddings, fields, output_size=(self.image_size, self.image_size),
                        valid_sizes=sizes,
                    )
                    for i, (height, width) in enumerate(sizes):
                        prediction = (logits[i, 0, :height, :width].sigmoid() >= self.threshold).cpu().numpy()
                        target = (prepared["mask"][i, 0, :height, :width] > 0.5).cpu().numpy()
                        metrics = compute_image_metrics(
                            prediction, target, 0,
                            replace(self.reward_config, seed=evaluation_seed(self.seed, prepared["ids"][i]) % (2**32)),
                        )
                        metrics.update(field_quality(fields[i], target, image_size=self.image_size))
                        metrics["empty_prediction_rate"] = float(not prediction.any())
                        metrics["empty_target_rate"] = float(not target.any())
                        rows.append(metrics)
                        if self.val_limit is not None and len(rows) >= int(self.val_limit):
                            break
                    if self.val_limit is not None and len(rows) >= int(self.val_limit):
                        break
        finally:
            self._set_train_modes(stage)
        if not rows:
            return None
        summary = aggregate_metrics(rows)
        for key in (*FIELD_METRIC_KEYS, "empty_prediction_rate", "empty_target_rate"):
            summary[key] = _mean_of([row[key] for row in rows if key in row])
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
            ):
                reference = getattr(self, name, None)
                if reference is None:
                    reference = copy.deepcopy(module)
                reference.to(self.device).eval().requires_grad_(False)
                setattr(self, name, reference)
        else:
            self.ref_dit = None
            self.ref_condition_encoder = None

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
        if self.config.get("training_protocol") != SIGNED_RIDGE_PROTOCOL:
            raise ValueError("训练配置需设置 training_protocol: signed_ridge_v1，并定义 stages.ABC/D")
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
                           "advantage_std_delta": self.advantage_std_delta},
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
        for name in ("ref_condition_encoder",):
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
            # v5：单通道有符号目标、SAM 全冻结及仅扩散轨迹 GRPO。
            "format_version": 5,
            "training_protocol": self.config.get("training_protocol"),
            "target_representation": "signed_ridge_v1",
            "latent_channels": 1,
            "sam_frozen": True,
            "prompt": copy.deepcopy(self.config.get("prompt", {"scale": 1.0, "terminal_mapping": "clamp"})),
            "sam": {
                "architecture": str(getattr(self.sam, "architecture", "")),
                "config": str(getattr(self.sam, "config_name", "")),
                "apply_postprocessing": bool(
                    getattr(self.sam, "apply_postprocessing", False)
                ),
            },
            "algorithm": {
                "name": self.rl_algorithm,
                "version": 3,
                "group_size": int(self.group_size),
                "advantage_std_delta": float(self.advantage_std_delta),
                "ppo_clip": float(self.ppo_clip),
                "kl_weight": float(self.kl_weight),
                "anchor_weight": float(self.anchor_weight),
                "update_epochs": int(self.update_epochs),
                "grpo_groups_processed": int(
                    getattr(self, "grpo_groups_processed", 0)
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
        validate_checkpoint_protocol(payload, self.config)
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
            ):
                if name in source:
                    reference = copy.deepcopy(module)
                    reference.load_state_dict(source[name], strict=True)
                    reference.to(self.device).eval().requires_grad_(False)
                    setattr(self, name, reference)
                elif payload_stage == "D" and not allow_missing_ref:
                    raise ValueError(f"联合 D 快照缺少 {name}，不能恢复固定参考策略")
        return report

    def load_init_weights(self, path: Union[str, Path]) -> Mapping[str, Any]:
        """``--init``：只加载模型权重，不恢复优化器/epoch/RNG。

        只初始化当前策略。新 D 实验从已加载的条件编码器和 DiT
        建立新的固定参考；恢复旧参考必须使用 --resume。
        """
        payload = read_checkpoint(path)
        report = self.load_models_from_payload(
            payload, include_ref=False, allow_missing_ref=True
        )
        print(f"已从 {path} 初始化模型权重（不含优化器状态）", flush=True)
        return report

    def read_checkpoint(self, path: Union[str, Path]) -> Any:
        """读取 checkpoint 的便捷入口。"""
        return read_checkpoint(path)
