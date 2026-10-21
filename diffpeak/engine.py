"""DiffPeak-SAM 四阶段训练引擎。

阶段职责
--------
* **A**：训练 SAM mask decoder、细节/条件分支与辅助头；提示来自真实目标热图。
* **B**：训练扩散 U-Net、条件分支与辅助头；扩散 MSE + 热图重建 + 辅助损失。
* **C**：只训练 SAM mask decoder；提示来自冻结扩散模型生成的提示。
* **D**：只训练扩散 U-Net；DDPO + 参考策略 KL + 扩散锚定，奖励来自冻结
  SAM 的解码结果。

关键实现约定
------------
* SAM 图像编码器与提示编码器始终冻结；冻结模块始终保持 ``eval``。
* A/B/C 使用 AMP + ``GradScaler``。D 的 rollout 与 replay 使用**相同的**
  autocast 上下文和 ``eval`` U-Net，保证更新前联合 log-prob 比值为 1。
* 梯度累积按实际尾组大小归一：``loss / len(group)``。
* D 以“一张影像的 G 条轨迹”为一个更新组：先整组 rollout（``no_grad``），
  再逐轨迹、逐步重算并立即 ``backward``，最后每组只做一次优化器更新。
  因此 ``train.accumulation_steps`` 只作用于 A/B/C；D 的组语义不同。
* 轨迹中的状态/动作/旧 log-prob 由 ``GaussianDiffusion.sample(record=True)``
  保存在 CPU。重放时一次只把一个时间步的张量搬到 GPU。
* PPO 使用 ``log_prob`` 返回的动作维**求和**，不做 mean；新旧概率差在
  float64 中计算，避免 2x256x256 维动作的精度损失。
* 参考 U-Net 只在 D 开始时 ``deepcopy``，永久冻结；KL 使用同方差高斯解析式。
"""

from __future__ import annotations

import copy
import json
import math
import os
import random
import shutil
import warnings
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from diffpeak.data import DeepGlobeDataset
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

__all__ = ["DiffPeakEngine", "STAGE_ORDER"]

STAGE_ORDER: Tuple[str, ...] = ("A", "B", "C", "D")


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
    """负责阶段切换、训练循环、验证、checkpoint 与 DDPO 的引擎。"""

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
        self.unet = self.models["unet"]
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

        self.accumulation_steps = max(1, int(self.train_cfg.get("accumulation_steps", 1)))
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

        self.group_size = int(self.rl_cfg.get("group_size", 4))
        self.ppo_clip = float(self.rl_cfg.get("ppo_clip", 1e-4))
        self.kl_weight = float(self.rl_cfg.get("kl_weight", 0.01))
        self.anchor_weight = float(self.rl_cfg.get("anchor_weight", 0.1))
        self.update_epochs = max(1, int(self.rl_cfg.get("update_epochs", 1)))

        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
        self.scaler: Optional[Any] = None
        self.ref_unet: Optional[torch.nn.Module] = None
        self.global_step = 0
        self.best_metric: Optional[float] = None
        self.history: List[Dict[str, Any]] = []
        self._stage: Optional[str] = None
        self._epoch = 0
        self._train_split_cache: Optional[Path] = None
        self._val_split_cache: Optional[Path] = None
        self._val_split_cached = False

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
        num_workers = max(0, int(self.data_cfg.get("num_workers", 0)))
        return DataLoader(
            dataset,
            batch_size=max(1, int(self.train_cfg.get("batch_size", 1))),
            shuffle=True,
            num_workers=num_workers,
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
        if stage in ("A", "C"):
            decoder = getattr(getattr(self.sam, "sam", None), "mask_decoder", None)
            if decoder is None:
                raise AttributeError("RoadSAM 缺少 sam.mask_decoder")
            for parameter in decoder.parameters():
                parameter.requires_grad_(True)

        for parameter in self.condition_encoder.parameters():
            parameter.requires_grad_(stage in ("A", "B"))
        for parameter in self.unet.parameters():
            parameter.requires_grad_(stage in ("B", "D"))

    def _set_train_modes(self, stage: str) -> None:
        """设置训练模式；冻结模块始终 eval。"""
        self.sam.eval()
        if stage in ("A", "C"):
            decoder = getattr(getattr(self.sam, "sam", None), "mask_decoder", None)
            if decoder is not None:
                decoder.train()
        self.condition_encoder.train(stage in ("A", "B"))
        # D 的 rollout 与 replay 必须使用同一个 eval U-Net 分布。
        self.unet.train(stage == "B")
        if self.ref_unet is not None:
            self.ref_unet.eval()

    def _set_eval_all(self) -> None:
        for module in self.models.values():
            module.eval()
        if self.ref_unet is not None:
            self.ref_unet.eval()

    def _trainable_parameters(self) -> List[torch.nn.Parameter]:
        parameters: List[torch.nn.Parameter] = []
        seen = set()
        for module in (self.sam, self.condition_encoder, self.unet):
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
        warmup_steps = int(
            stage_cfg.get("warmup_steps", self.train_cfg.get("warmup_steps", 0)) or 0
        )
        warmup_ratio = float(
            stage_cfg.get("warmup_ratio", self.train_cfg.get("warmup_ratio", 0.0)) or 0.0
        )
        if warmup_steps <= 0 and warmup_ratio > 0:
            warmup_steps = int(round(total_steps * warmup_ratio))

        def lr_lambda(step: int) -> float:
            step = int(step)
            if warmup_steps > 0 and step < warmup_steps:
                return max(1e-8, float(step + 1) / float(warmup_steps))
            progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            progress = min(1.0, max(0.0, progress))
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=lr_lambda
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

        if stage == "A":
            _, auxiliary_logits = self.condition_encoder(images, embeddings)
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
            auxiliary = auxiliary_loss(auxiliary_logits, masks, skeletons)
            total = self.aux_weight * auxiliary
            if segmentation is not None:
                total = total + segmentation
            return {
                "loss": total,
                "seg": segmentation,
                "aux": auxiliary,
                "points": float(sum(int(item[1].numel()) for item in points)),
            }

        if stage == "B":
            conditions, auxiliary_logits = self.condition_encoder(images, embeddings)
            diffusion_losses = self.diffusion.training_losses(
                self.unet, conditions, target_heatmaps
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
                    self.unet,
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
    def _train_epoch_supervised(
        self, stage: str, epoch: int, loader: DataLoader
    ) -> Dict[str, float]:
        self._set_train_modes(stage)
        records: List[Dict[str, Any]] = []
        optimizer_steps = 0
        skipped_groups = 0
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
                    "aux": _scalar(loss_dict.get("aux")),
                    "diff": _scalar(loss_dict.get("diff")),
                    "heat": _scalar(loss_dict.get("heat")),
                    "points": _scalar(loss_dict.get("points")),
                }
                records.append(record)

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
        for key in ("loss", "seg", "aux", "diff", "heat", "points"):
            values = [
                record[key]
                for record in records
                if record.get(key) is not None
            ]
            stats[key] = _mean_of(values) if values else float("nan")
        stats["optimizer_steps"] = float(optimizer_steps)
        stats["skipped_groups"] = float(skipped_groups)
        return stats

    def _train_epoch_ddpo(
        self, stage: str, epoch: int, loader: DataLoader
    ) -> Dict[str, float]:
        self._set_train_modes(stage)
        records: List[Dict[str, Any]] = []
        for group_index, batch in enumerate(loader, start=1):
            prepared = self._prepare_batch(batch)
            record = self._ddpo_group(prepared, epoch)
            records.append(record)
            if group_index % self.log_every == 0:
                print(
                    f"[D] epoch {epoch + 1} group {group_index} "
                    f"reward={record.get('reward')} "
                    f"iou={record.get('iou')} "
                    f"points={record.get('point_count')} "
                    f"clip_frac={record.get('clip_fraction')}",
                    flush=True,
                )

        stats: Dict[str, float] = {}
        keys = (
            "reward",
            "reward_std",
            "iou",
            "f1",
            "cldice",
            "path",
            "point_count",
            "ppo",
            "kl",
            "anchor",
            "clip_fraction",
        )
        for key in keys:
            values = [
                record[key] for record in records if record.get(key) is not None
            ]
            stats[key] = _mean_of(values) if values else float("nan")
        stats["optimizer_steps"] = float(
            sum(float(record.get("updates", 0.0)) for record in records)
        )
        stats["skipped_groups"] = float(
            sum(float(record.get("skipped_updates", 0.0)) for record in records)
        )
        return stats

    def _ddpo_group(self, batch: Mapping[str, Any], epoch: int) -> Dict[str, float]:
        """采样一组轨迹、计算奖励，并做一次（或 ``update_epochs`` 次）更新。"""
        if self.group_size < 2:
            raise ValueError("D 阶段要求 rl.group_size >= 2 才能做 leave-one-out")
        if self.ref_unet is None:
            raise RuntimeError("D 阶段缺少冻结参考 U-Net")

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

        for trajectory_index in range(self.group_size):
            generator = make_generator(
                stable_seed(
                    self.seed, "D", int(epoch), *ids, trajectory_index
                )
            )
            with torch.no_grad(), self._autocast():
                heatmaps, trace = self.diffusion.sample(
                    self.unet,
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
                for index in range(batch_size):
                    valid = valid_sizes[index] or (self.image_size, self.image_size)
                    height, width = int(valid[0]), int(valid[1])
                    prediction = (
                        probabilities[index, 0, :height, :width] > self.threshold
                    ).detach().cpu().numpy()
                    point_count = int(points[index][1].numel())
                    metrics = compute_image_metrics(
                        prediction,
                        targets_np[index],
                        point_count,
                        self.reward_config,
                    )
                    sample_rewards.append(float(metrics["reward"]))
                    metric_rows.append(
                        {**metrics, "point_count": float(point_count)}
                    )
                rewards.append(sample_rewards)
                # trace 已由扩散模块保存在 CPU；这里只保留轨迹。
                trajectories.append(trace)
            # 显式释放采样图，避免 G 条轨迹的完整计算图/激活驻留。
            del heatmaps, logits, probabilities

        reward_tensor = torch.tensor(rewards, dtype=torch.float64)  # [G, B]
        leave_one_out = (
            reward_tensor.sum(dim=0, keepdim=True) - reward_tensor
        ) / float(self.group_size - 1)
        advantages = reward_tensor - leave_one_out
        reward_std = float(
            reward_tensor.std(dim=0, unbiased=False).mean().cpu()
        )

        update_stats = self._ddpo_update(
            conditions, target_heatmaps, trajectories, advantages
        )

        merged: Dict[str, float] = dict(update_stats)
        merged["reward_std"] = reward_std
        for key in ("reward", "iou", "f1", "cldice", "path", "point_count"):
            values = [
                float(row[key])
                for row in metric_rows
                if key in row and math.isfinite(float(row[key]))
            ]
            merged[key] = _mean_of(values) if values else float("nan")
        return merged

    def _ddpo_update(
        self,
        conditions: Sequence[torch.Tensor],
        target_heatmaps: torch.Tensor,
        trajectories: Sequence[Sequence[Mapping[str, Any]]],
        advantages: torch.Tensor,
    ) -> Dict[str, float]:
        """固定整组轨迹后做 PPO/KL/anchor 更新；每组 step 一次。"""
        assert self.optimizer is not None and self.scheduler is not None
        assert self.scaler is not None
        if not trajectories:
            return {
                "ppo": float("nan"),
                "kl": float("nan"),
                "anchor": float("nan"),
                "clip_fraction": float("nan"),
                "updates": 0.0,
                "skipped_updates": 0.0,
            }

        total_terms = sum(len(trajectory) for trajectory in trajectories)
        if total_terms <= 0:
            return {
                "ppo": float("nan"),
                "kl": float("nan"),
                "anchor": float("nan"),
                "clip_fraction": float("nan"),
                "updates": 0.0,
                "skipped_updates": 0.0,
            }

        ppo_values: List[float] = []
        kl_values: List[float] = []
        anchor_values: List[float] = []
        clip_values: List[float] = []
        update_steps = 0
        skipped_updates = 0
        epsilon = float(self.ppo_clip)

        for _ in range(self.update_epochs):
            self.optimizer.zero_grad(set_to_none=True)

            # 扩散锚定：只使用噪声 MSE，避免后训练把热图结构完全带偏。
            if self.anchor_weight > 0:
                with self._autocast():
                    anchor_losses = self.diffusion.training_losses(
                        self.unet, conditions, target_heatmaps
                    )
                    anchor = anchor_losses["diff"] * self.anchor_weight
                self.scaler.scale(anchor.float()).backward()
                anchor_values.append(float(anchor.detach().float().cpu()))

            for trajectory_index, trajectory in enumerate(trajectories):
                advantage = advantages[trajectory_index].to(self.device)
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

                    # rollout 与 replay 使用同一 AMP 上下文和同一 eval U-Net。
                    with self._autocast():
                        mean, std = self.diffusion.transition(
                            self.unet,
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
                    log_ratio = torch.clamp(
                        new_log_prob - old_log_prob, min=-20.0, max=20.0
                    )
                    ratio = torch.exp(log_ratio)
                    clipped = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon)
                    clipped_mask = (ratio < 1.0 - epsilon) | (ratio > 1.0 + epsilon)
                    clip_values.append(float(clipped_mask.double().mean().cpu()))
                    advantage64 = advantage.reshape(-1).double()
                    ppo_loss = -torch.minimum(
                        ratio * advantage64, clipped * advantage64
                    ).mean()

                    kl_value = torch.zeros((), dtype=torch.float64, device=self.device)
                    if self.kl_weight > 0:
                        with torch.no_grad(), self._autocast():
                            reference_mean, _ = self.diffusion.transition(
                                self.ref_unet,
                                state,
                                timestep,
                                previous_timestep,
                                conditions,
                            )
                        delta = (
                            mean.double() - reference_mean.double()
                        ) / std.double().clamp_min(1e-12)
                        # 同方差高斯 KL = 0.5 * sum((mu - mu_ref)^2 / sigma^2)。
                        kl_value = 0.5 * delta.reshape(
                            delta.shape[0], -1
                        ).pow(2).sum(dim=1).mean()
                        kl_values.append(float(kl_value.detach().cpu()))

                    loss = (ppo_loss + self.kl_weight * kl_value) / float(total_terms)
                    self.scaler.scale(loss.float()).backward()
                    ppo_values.append(float(ppo_loss.detach().cpu()))

                    del state, action, mean, std, new_log_prob, log_ratio, ratio

            if self._optimizer_step():
                update_steps += 1
            else:
                skipped_updates += 1

        return {
            "ppo": _mean_of(ppo_values),
            "kl": _mean_of(kl_values),
            "anchor": _mean_of(anchor_values),
            "clip_fraction": _mean_of(clip_values),
            "updates": float(update_steps),
            "skipped_updates": float(skipped_updates),
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
                        points = get_points(
                            self.peak_extractor,
                            prepared["heatmaps"],
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
                            self.unet,
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
                    logits, _ = decode_with_prompts(
                        self.sam,
                        embeddings,
                        prompts,
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
                        point_count = int(points[index][1].numel())
                        metrics = compute_image_metrics(
                            prediction,
                            target,
                            point_count,
                            self.reward_config,
                        )
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
            if self.ref_unet is None:
                self.ref_unet = copy.deepcopy(self.unet)
            self.ref_unet = self.ref_unet.to(self.device)
            self.ref_unet.eval()
            for parameter in self.ref_unet.parameters():
                parameter.requires_grad_(False)
        else:
            self.ref_unet = None

        self._set_train_modes(stage)
        self._build_optimizer(stage, epochs, steps_per_epoch)
        self.best_metric = None
        self.global_step = 0

    def run_stage(
        self,
        stage: str,
        start_epoch: int = 0,
        resume_payload: Optional[Mapping[str, Any]] = None,
    ) -> Optional[Dict[str, float]]:
        """运行一个阶段；返回最后一个验证 epoch 的指标。"""
        stage = str(stage).upper()
        if stage not in STAGE_ORDER:
            raise ValueError(f"未知阶段：{stage}")
        if stage not in self.stage_cfgs:
            raise ValueError(f"配置缺少 stages.{stage}")
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
        # D 阶段的 ref_unet 也在这里严格恢复。
        if resume_payload is not None:
            self.load_models_from_payload(resume_payload, include_ref=True)
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

        last_validation: Optional[Dict[str, float]] = None
        for epoch in range(start_epoch, epochs):
            self._epoch = epoch
            train_loader = self._build_train_loader(epoch)
            if stage == "D":
                train_stats = self._train_epoch_ddpo(stage, epoch, train_loader)
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

            # 先更新 best，再写 last；这样 last.pt 中的 best_metric 不会落后，
            # 中断后从 last.pt 恢复不会回退最佳指标跟踪。
            if score is not None and (
                self.best_metric is None or score > float(self.best_metric)
            ):
                self.best_metric = float(score)
                self.save_best(stage, epoch + 1)

            if (epoch + 1) % self.save_every == 0 or epoch + 1 == epochs:
                self.save_last(stage, epoch + 1, self.best_metric)

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
            "unet": self._to_cpu_tree(self.unet.state_dict()),
        }
        # 始终写入 diffusion 字段（包含 beta/alpha 调度 buffer），
        # 严格加载器要求四个必需模块齐全。
        states["diffusion"] = self._to_cpu_tree(self.diffusion.state_dict())
        if self.ref_unet is not None:
            states["ref_unet"] = self._to_cpu_tree(self.ref_unet.state_dict())
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
        resolved["model"] = model
        resolved["output_dir"] = str(
            Path(self.output_dir).expanduser().resolve()
        )
        return resolved

    def _checkpoint_payload(
        self, stage: str, epoch: int, best_metric: Optional[float]
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "format_version": 1,
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
    ) -> Dict[str, Any]:
        """严格加载模型权重；D 快照必须同时包含参考 U-Net。"""
        report = load_checkpoint_models(
            self.models, payload, include_ref=False
        )
        source = payload["models"]
        payload_stage = str(payload.get("stage", "")).upper()
        if include_ref and "ref_unet" in source:
            if self.ref_unet is None:
                self.ref_unet = copy.deepcopy(self.unet)
            self.ref_unet.load_state_dict(source["ref_unet"], strict=True)
            self.ref_unet = self.ref_unet.to(self.device).eval()
            for parameter in self.ref_unet.parameters():
                parameter.requires_grad_(False)
        elif include_ref and payload_stage == "D":
            raise ValueError("D 阶段 checkpoint 缺少 ref_unet，不能恢复 KL 参考策略")
        return report

    def load_init_weights(self, path: Union[str, Path]) -> Mapping[str, Any]:
        """``--init``：只加载模型权重，不恢复优化器/epoch/RNG。"""
        payload = read_checkpoint(path)
        report = self.load_models_from_payload(payload, include_ref=True)
        print(f"已从 {path} 初始化模型权重（不含优化器状态）", flush=True)
        return report

    def read_checkpoint(self, path: Union[str, Path]) -> Any:
        """读取 checkpoint 的便捷入口。"""
        return read_checkpoint(path)
