"""DeepGlobe Road 监督数据集。

一张样本的构造流程
------------------
1. 读取相对划分项对应的 ``*_sat.jpg`` 与同目录 ``*_mask.png``；
2. 可选地做翻转 / 90° 旋转增广（标签与影像使用同一变换）；
3. 在**原始标注分辨率**下由道路掩码计算骨架，并沿骨架/背景采样提示点；
4. 按保持纵横比、右下填充的规则把影像、掩码、骨架缩放到 ``image_size``；
5. 把原始分辨率下的点映射到 ``heatmap_size`` 热图并渲染高斯目标；
6. 返回训练所需张量及几何信息。

随机性
------
每个样本使用独立的 ``numpy.random.Generator``。种子由
``(seed, worker_id, epoch, index, occurrence)`` 组成：

* ``worker_id`` 来自 DataLoader worker，保证不同 worker 不共享随机流；
* 训练模式（``augment=True``）下，``occurrence`` 是同一 worker 内同索引
  被访问的次数，即使使用 persistent workers，多次访问也不会得到完全
  相同的布局；评估模式（``augment=False``）不推进 ``occurrence``，同一
  epoch 内重复读取会得到相同布局；
* ``epoch`` 由 :meth:`set_epoch` 显式写入。训练脚本每轮重建数据集并调用
  ``set_epoch(epoch)`` 时，同一张图在不同 epoch 会得到新的提示布局。

验证集若需要跨 epoch 固定布局，应重建数据集并调用 ``set_epoch(0)``
（``diffpeak/engine.py`` 即如此）；由于 ``augment=False`` 下同一 epoch 内
布局固定，重复读取或离线复现都会得到同一组提示。
该设计保证：相同的 seed / worker 划分 / epoch / 访问顺序可复现；它不承诺
在不同 ``num_workers`` 或不同样本访问顺序下得到逐样本完全相同的结果。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from skimage.morphology import skeletonize
from torch.utils.data import Dataset, get_worker_info

from .geometry import ImageGeometry, build_geometry, resize_and_pad
from .sampling import (
    HardNegativeProvider,
    RoadTargetSampler,
    render_gaussian_heatmap,
    select_spread_points,
)
from .splits import (
    load_split_entries,
    mask_path_for_image,
    resolve_under_root,
    sample_id_from_image_path,
)

__all__ = ["DeepGlobeDataset"]

_ROAD_THRESHOLD = 127


@dataclass(frozen=True)
class _SampleEntry:
    relative_path: str
    image_path: Path
    mask_path: Path
    sample_id: str


def _filter_heatmap_points(
    points_yx: np.ndarray,
    heatmap_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """校验并夹紧热图坐标，返回点与 keep 掩码。

    标注内容不会落在右下填充区；这里只处理极边缘处因缩放舍入产生的
    微小越界，避免丢失图像最外圈的道路点。
    """
    points = np.asarray(points_yx, dtype=np.float64)
    if points.size == 0:
        return points.reshape(0, 2), np.zeros(0, dtype=bool)
    keep = (
        np.isfinite(points).all(axis=1)
        & (points[:, 0] >= -0.5)
        & (points[:, 0] <= heatmap_size - 0.5)
        & (points[:, 1] >= -0.5)
        & (points[:, 1] <= heatmap_size - 0.5)
    )
    points = points[keep]
    if len(points):
        points = np.clip(points, 0.0, float(heatmap_size - 1))
    return points, keep


class DeepGlobeDataset(Dataset):
    """DeepGlobe 道路监督数据集。

    Args:
        root: 数据根目录，例如 ``"DeepGlobe_Road"``。
        split_file: JSON 字符串数组路径，或字符串列表。项是相对 ``root``
            的影像路径，例如 ``"train/100034_sat.jpg"``。
        augment: 是否做翻转 / 90° 旋转增广。默认关闭。该开关同时决定
            提示布局是否随重复访问变化：``True`` 用于训练（每次访问重新
            采样），``False`` 用于评估（同一 epoch 内布局固定、可复现）。
        image_size: 网络输入边长，默认 1024。
        heatmap_size: 提示热图边长，默认 256。
        seed: 样本随机种子，默认 42。
        sampling_config: 可选的点采样配置，形如
            ``{"positive": {...}, "negative": {...}}``。
        heatmap_sigma: 目标高斯宽度（热图像素），默认 1.0。
        target_min_distance: 目标点在热图空间的最小间距，默认 4.0，
            应与 :class:`diffpeak.prompts.PeakExtractor` 的 ``min_distance``
            保持一致，以减少“热图能拟合但回提合并”的风险。
        hard_negative_provider: 可选的困难负点回调
            ``(image, road_mask, rng) -> (N, 2) (y, x)``。本类不自动挖掘
            困难负点，只负责把调用方提供且合法的背景点并入负点集合。

    Returns (``__getitem__``):
        dict:

        * ``image``: ``float32``，``3 x image_size x image_size``，范围 ``[0, 1]``；
        * ``mask``: ``float32``，``1 x image_size x image_size``，二值道路；
        * ``skeleton``: ``float32``，``1 x image_size x image_size``，二值骨架；
        * ``heatmaps``: ``float32``，``2 x heatmap_size x heatmap_size``，
          通道 0 为正提示，通道 1 为负提示；
        * ``id``: 字符串，例如 ``"100034"``；
        * ``original_size``: 当前标注分辨率 ``(H, W)``（几何增广后）；
        * ``resized_size``: 填充前的缩放尺寸 ``(H, W)``，可直接用于
          ``PeakExtractor(..., valid_sizes=[resized_size])``。
    """

    def __init__(
        self,
        root: str,
        split_file: str,
        augment: bool = False,
        image_size: int = 1024,
        heatmap_size: int = 256,
        seed: int = 42,
        *,
        sampling_config: Optional[Mapping[str, Any]] = None,
        heatmap_sigma: float = 1.0,
        target_min_distance: float = 4.0,
        hard_negative_provider: Optional[HardNegativeProvider] = None,
    ) -> None:
        super().__init__()
        self.root = Path(root).expanduser().resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(f"数据根目录不存在：{self.root}")
        if image_size <= 0 or heatmap_size <= 0:
            raise ValueError("image_size 与 heatmap_size 必须为正")
        if heatmap_sigma < 0:
            raise ValueError("heatmap_sigma 不能为负")
        if target_min_distance < 0:
            raise ValueError("target_min_distance 不能为负")
        if hard_negative_provider is not None and not callable(hard_negative_provider):
            raise TypeError("hard_negative_provider 必须可调用或为 None")

        self.split_file = split_file
        self.augment = bool(augment)
        self.image_size = int(image_size)
        self.heatmap_size = int(heatmap_size)
        self.seed = int(seed)
        self.heatmap_sigma = float(heatmap_sigma)
        self.target_min_distance = float(target_min_distance)
        self.hard_negative_provider = hard_negative_provider

        if sampling_config is None:
            sampling_config = {}
        if not isinstance(sampling_config, Mapping):
            raise TypeError("sampling_config 必须是 dict 或 None")
        unknown = set(sampling_config) - {"positive", "negative"}
        if unknown:
            raise ValueError(f"sampling_config 包含未知字段: {sorted(unknown)}")
        self.sampler = RoadTargetSampler(
            positive=sampling_config.get("positive"),
            negative=sampling_config.get("negative"),
        )

        entries = load_split_entries(split_file)
        self._samples: List[_SampleEntry] = []
        missing: List[str] = []
        for relative_path in entries:
            image_path = resolve_under_root(self.root, relative_path)
            mask_path = mask_path_for_image(image_path)
            if not image_path.is_file():
                missing.append(f"影像不存在: {relative_path}")
                continue
            if not mask_path.is_file():
                missing.append(
                    f"掩码不存在: {mask_path.relative_to(self.root).as_posix()}"
                )
                continue
            self._samples.append(
                _SampleEntry(
                    relative_path=relative_path,
                    image_path=image_path,
                    mask_path=mask_path,
                    sample_id=sample_id_from_image_path(image_path),
                )
            )
        if missing:
            preview = "\n  ".join(missing[:8])
            raise FileNotFoundError(
                "划分中缺少影像或掩码：\n  "
                + preview
                + (
                    "\n  ……" if len(missing) > 8 else ""
                )
                + "\n注意：官方 valid/test 只有影像、没有 mask，不能当作监督集。"
            )
        if not self._samples:
            raise ValueError("split_file 中没有可用样本")

        self._epoch = 0
        # occurrence 只增不减；仅在训练模式（augment=True）下使用。
        self._occurrence: Dict[int, int] = {}

    # ------------------------------------------------------------------
    # 基本属性
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._samples)

    @property
    def epoch(self) -> int:
        return self._epoch

    def set_epoch(self, epoch: int) -> None:
        """显式设置 epoch。

        训练时每个 epoch 调用一次即可让同一张图获得新的提示布局。验证集
        保持 ``set_epoch(0)`` 即可跨 epoch 复现；``augment=False`` 下同一
        epoch 内重复读取也不会改变布局。
        """
        self._epoch = int(epoch)

    @property
    def image_paths(self) -> List[Path]:
        return [entry.image_path for entry in self._samples]

    @property
    def mask_paths(self) -> List[Path]:
        return [entry.mask_path for entry in self._samples]

    # ------------------------------------------------------------------
    # 随机状态
    # ------------------------------------------------------------------
    def _worker_id(self) -> int:
        info = get_worker_info()
        return 0 if info is None else int(info.id)

    def _rng_for(self, index: int) -> np.random.Generator:
        """返回样本专属随机流。

        ``occurrence`` 只在训练模式（``augment=True``）下推进；评估/固定
        布局模式（``augment=False``）不推进它，因此同一 epoch 内重复访问
        同一索引会得到完全相同的布局。``epoch`` 始终参与种子：验证集固定
        ``set_epoch(0)`` 即可跨 epoch 复现（``diffpeak/engine.py`` 如此），
        训练集仍可通过逐 epoch 的 ``set_epoch`` 获得新布局。
        """
        if self.augment:
            occurrence = self._occurrence.get(index, 0)
            self._occurrence[index] = occurrence + 1
        else:
            occurrence = 0
        seed_sequence = np.random.SeedSequence(
            [
                self.seed & 0xFFFFFFFF,
                self._worker_id() & 0xFFFFFFFF,
                self._epoch & 0xFFFFFFFF,
                index & 0xFFFFFFFF,
                occurrence & 0xFFFFFFFF,
            ]
        )
        return np.random.default_rng(seed_sequence)

    # ------------------------------------------------------------------
    # 读取与增广
    # ------------------------------------------------------------------
    @staticmethod
    def _read_image(path: Path) -> np.ndarray:
        with Image.open(path) as image:
            return np.array(image.convert("RGB"), dtype=np.uint8, copy=True)

    @staticmethod
    def _read_road_mask(path: Path) -> np.ndarray:
        with Image.open(path) as image:
            mask = np.array(image.convert("RGB"), dtype=np.uint8, copy=True)
        # DeepGlobe 的道路为白色 (255,255,255)，背景为黑色 (0,0,0)。
        road = np.any(mask > _ROAD_THRESHOLD, axis=2)
        if not road.any() and mask.max() > 0:
            # 兼容测试/派生数据中 0/1 二值掩码；正式数据仍走 255 阈值。
            road = np.any(mask > 0, axis=2)
        return road

    def _augment(
        self,
        image: np.ndarray,
        road_mask: np.ndarray,
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if not self.augment:
            return image, road_mask
        if rng.random() < 0.5:
            image = image[:, ::-1]
            road_mask = road_mask[:, ::-1]
        if rng.random() < 0.5:
            image = image[::-1]
            road_mask = road_mask[::-1]
        rotations = int(rng.integers(0, 4))
        if rotations:
            image = np.rot90(image, rotations)
            road_mask = np.rot90(road_mask, rotations)
        return np.ascontiguousarray(image), np.ascontiguousarray(road_mask)

    # ------------------------------------------------------------------
    # 目标构造
    # ------------------------------------------------------------------
    def _build_sample(
        self,
        index: int,
        return_meta: bool = False,
    ):
        entry = self._samples[index]
        rng = self._rng_for(index)

        image = self._read_image(entry.image_path)
        road_mask = self._read_road_mask(entry.mask_path)
        if image.shape[:2] != road_mask.shape:
            raise ValueError(
                f"{entry.relative_path} 的影像与掩码尺寸不一致："
                f"{image.shape[:2]} vs {road_mask.shape}"
            )
        image, road_mask = self._augment(image, road_mask, rng)

        source_height, source_width = road_mask.shape
        geometry: ImageGeometry = build_geometry(
            source_height,
            source_width,
            image_size=self.image_size,
            heatmap_size=self.heatmap_size,
        )

        # 骨架必须在原始标注分辨率下计算；细道路在缩放到 256 后会断开。
        skeleton = skeletonize(road_mask) if road_mask.any() else np.zeros_like(road_mask)

        targets = self.sampler.sample(
            skeleton=skeleton,
            road_mask=road_mask,
            rng=rng,
            image=image,
            hard_negative_provider=self.hard_negative_provider,
        )

        positive_heatmap = geometry.original_yx_to_heatmap_yx(targets.positive_yx)
        negative_heatmap = geometry.original_yx_to_heatmap_yx(targets.negative_yx)
        positive_heatmap, positive_keep = _filter_heatmap_points(
            positive_heatmap, self.heatmap_size
        )
        negative_heatmap, negative_keep = _filter_heatmap_points(
            negative_heatmap, self.heatmap_size
        )
        priorities = np.asarray(targets.positive_priorities, dtype=np.int64)
        if len(priorities):
            priorities = priorities[positive_keep]

        # 让目标点自身满足最终 NMS 的间距；结构点优先保留。
        positive_heatmap = select_spread_points(
            positive_heatmap,
            min_distance=self.target_min_distance,
            rng=rng,
            priorities=priorities if len(priorities) else None,
        )
        negative_heatmap = select_spread_points(
            negative_heatmap,
            min_distance=self.target_min_distance,
            rng=rng,
        )

        heatmaps = np.zeros(
            (2, self.heatmap_size, self.heatmap_size), dtype=np.float32
        )
        if road_mask.any():
            # 无道路影像保持双通道全零，而不是用背景负点填满负通道。
            heatmaps[0] = render_gaussian_heatmap(
                positive_heatmap, self.heatmap_size, self.heatmap_sigma
            )
            heatmaps[1] = render_gaussian_heatmap(
                negative_heatmap, self.heatmap_size, self.heatmap_sigma
            )

        image_padded = resize_and_pad(image, geometry, resample="bilinear")
        mask_padded = resize_and_pad(road_mask, geometry, resample="nearest")
        skeleton_padded = resize_and_pad(skeleton, geometry, resample="nearest")

        image_tensor = torch.from_numpy(
            np.ascontiguousarray(image_padded.transpose(2, 0, 1))
        ).float()
        image_tensor.div_(255.0)
        mask_tensor = torch.from_numpy(
            np.ascontiguousarray(mask_padded[None].astype(np.float32))
        )
        skeleton_tensor = torch.from_numpy(
            np.ascontiguousarray(skeleton_padded[None].astype(np.float32))
        )
        heatmap_tensor = torch.from_numpy(
            np.ascontiguousarray(heatmaps)
        )

        sample = {
            "image": image_tensor,
            "mask": mask_tensor,
            "skeleton": skeleton_tensor,
            "heatmaps": heatmap_tensor,
            "id": entry.sample_id,
            "original_size": (int(source_height), int(source_width)),
            "resized_size": tuple(int(v) for v in geometry.resized_size),
        }
        if not return_meta:
            return sample

        positive_image = geometry.heatmap_yx_to_image_xy(positive_heatmap)
        negative_image = geometry.heatmap_yx_to_image_xy(negative_heatmap)
        labels = np.concatenate(
            [
                np.ones(len(positive_image), dtype=np.int64),
                np.zeros(len(negative_image), dtype=np.int64),
            ]
        )
        image_xy = np.concatenate([positive_image, negative_image], axis=0)
        meta = {
            "positive_heatmap_yx": positive_heatmap,
            "negative_heatmap_yx": negative_heatmap,
            "positive_image_xy": positive_image,
            "negative_image_xy": negative_image,
            "image_xy": image_xy,
            "labels": labels,
            "geometry": geometry,
        }
        return sample, meta

    def __getitem__(self, index: int) -> Dict[str, Any]:
        index = int(index)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        return self._build_sample(index, return_meta=False)

    # ------------------------------------------------------------------
    # 诊断接口
    # ------------------------------------------------------------------
    def get_target_points(self, index: int) -> Dict[str, Any]:
        """返回某样本本次随机布局的点目标，便于可视化与回提诊断。

        ``augment=True`` 时每次调用都会推进该索引的 occurrence，因此连续
        调用会得到不同但同样可复现（给定 seed/worker/epoch）的布局；
        ``augment=False``（评估/阶段 A 验证）时不推进 occurrence，同一
        epoch 内连续调用结果完全一致。
        """
        index = int(index)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        sample, meta = self._build_sample(index, return_meta=True)
        meta = dict(meta)
        meta["id"] = sample["id"]
        meta["original_size"] = sample["original_size"]
        meta["resized_size"] = sample["resized_size"]
        return meta

    def recovery_report(
        self,
        index: int,
        extractor=None,
        **report_kwargs: Any,
    ) -> Dict[str, Any]:
        """用最终峰值提取器回提目标热图，并返回诊断报告。

        该接口用于检查“目标峰是否在构造阶段就被合并/过滤”。它只做诊断，
        不参与训练损失，也不改变峰值提取器的固定参数。

        Args:
            index: 样本索引。
            extractor: 可选的 :class:`diffpeak.prompts.PeakExtractor`；
                为 ``None`` 时使用默认配置。
            **report_kwargs: 透传给
                :func:`diffpeak.prompts.target_recovery_report`，例如
                ``match_radius``。
        """
        index = int(index)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        sample, meta = self._build_sample(index, return_meta=True)
        # 延迟导入，保持数据模块与提示提取模块的解耦。
        from diffpeak.prompts import target_recovery_report

        report = target_recovery_report(
            sample["heatmaps"].unsqueeze(0),
            [meta["image_xy"]],
            [meta["labels"]],
            extractor=extractor,
            image_size=self.image_size,
            valid_sizes=[sample["resized_size"]],
            **report_kwargs,
        )
        report["id"] = sample["id"]
        return report
