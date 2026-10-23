"""DeepGlobe Road 监督数据集（单通道有符号山脊场协议）。

一张样本的构造流程
------------------
1. 读取相对划分项对应的 ``*_sat.jpg`` 与同目录 ``*_mask.png``；
2. 可选地做翻转 / 90° 旋转增广（影像与掩码使用同一变换）；
3. 在**原始标注分辨率**下由道路掩码构造骨架与单通道有符号山脊场
   （见 :mod:`diffpeak.data.ridge`）；
4. 按保持纵横比、右下填充的规则把影像、掩码、骨架缩放到 ``image_size``；
5. 按同一像素中心映射把山脊场降到 ``heatmap_size``，并生成独立的有效区域
   掩码与四区域标签。

不再沿中心线采样离散点，也不构造双通道高斯热图。旧点采样器不在本数据集
的任何执行路径上。

随机性
------
目标场由道路几何确定性构造，随机性只影响几何增广。每个样本使用独立的
``numpy.random.Generator``，种子由 ``(seed, worker_id, epoch, index,
occurrence)`` 组成：

* ``worker_id`` 来自 DataLoader worker，保证不同 worker 不共享随机流；
* 训练模式（``augment=True``）下，``occurrence`` 是同一 worker 内同索引
  被访问的次数，即使使用 persistent workers，多次访问也会得到不同的增广；
* 评估模式（``augment=False``）不推进 ``occurrence``，同一 epoch 内重复
  读取会得到相同的增广与目标；
* ``epoch`` 由 :meth:`set_epoch` 显式写入。

验证集若需要跨 epoch 固定布局，应重建数据集并调用 ``set_epoch(0)``
（``diffpeak/engine.py`` 即如此）。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, get_worker_info

from .geometry import ImageGeometry, build_geometry, resize_and_pad
from .ridge import (
    PADDING_VALUE,
    RidgeFieldConfig,
    build_ridge_field,
    coerce_ridge_config,
    field_statistics,
    regions_from_one_hot,
    regions_to_one_hot,
    source_retention_statistics,
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


class DeepGlobeDataset(Dataset):
    """DeepGlobe 道路监督数据集。

    Args:
        root: 数据根目录，例如 ``"DeepGlobe_Road"``。
        split_file: JSON 字符串数组路径，或字符串列表。项是相对 ``root``
            的影像路径，例如 ``"train/100034_sat.jpg"``。
        augment: 是否做翻转 / 90° 旋转增广。默认关闭。该开关只控制几何
            增广的随机性；目标场始终由增广后的道路几何确定。
        image_size: 网络输入边长，默认 1024。
        heatmap_size: 提示热图边长，默认 256。
        seed: 样本随机种子，默认 42。
        ridge_config: 可选的山脊场参数，:class:`RidgeFieldConfig` 或
            ``{"center_radius": ..., "boundary_band": ..., "epsilon": ...}``。
            半径与带宽的单位均为热图像素。

    Returns (``__getitem__``):
        dict:

        * ``image``: ``float32``，``3 x image_size x image_size``，范围 ``[0, 1]``；
        * ``mask``: ``float32``，``1 x image_size x image_size``，二值道路；
        * ``skeleton``: ``float32``，``1 x image_size x image_size``，二值骨架；
        * ``heatmaps``: ``float32``，``1 x heatmap_size x heatmap_size``，
          单通道有符号山脊场：道路中心线 ``+1``、有效背景 ``-1``、
          padding ``0``；
        * ``field_valid_mask``: ``float32``，``1 x heatmap_size x heatmap_size``，
          ``1`` 表示有效图像内容、``0`` 表示右下 padding；
        * ``field_regions``: ``float32``，``4 x heatmap_size x heatmap_size``
          的互斥 one-hot 区域掩码，通道顺序为 ``0`` 中心线邻域、``1`` 其余
          道路、``2`` 边界外窄带、``3`` 远端背景；padding 在四个通道中
          全为 ``0``。引擎/扩散损失直接消费该形状；内部标签表示见
          :mod:`diffpeak.data.ridge`；
        * ``id``: 字符串，例如 ``"100034"``；
        * ``original_size``: 当前标注分辨率 ``(H, W)``（几何增广后）；
        * ``resized_size``: 填充前的缩放尺寸 ``(H, W)``，可直接作为 SAM
          的 ``valid_sizes``。
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
        ridge_config: Optional[Any] = None,
    ) -> None:
        super().__init__()
        self.root = Path(root).expanduser().resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(f"数据根目录不存在：{self.root}")
        if image_size <= 0 or heatmap_size <= 0:
            raise ValueError("image_size 与 heatmap_size 必须为正")

        self.split_file = split_file
        self.augment = bool(augment)
        self.image_size = int(image_size)
        self.heatmap_size = int(heatmap_size)
        self.seed = int(seed)
        self.ridge_config: RidgeFieldConfig = coerce_ridge_config(ridge_config)

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
        """显式设置 epoch；它参与几何增广的随机种子。"""
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
        """返回样本专属随机流，仅用于几何增广。

        ``occurrence`` 只在训练模式（``augment=True``）下推进；评估/固定
        布局模式（``augment=False``）不推进它，因此同一 epoch 内重复访问
        同一索引会得到完全相同的增广与目标。
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
    def _build_sample(self, index: int, *, return_ridge: bool = False):
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

        # 山脊场在原始标注分辨率下构造，再按同一几何映射降到热图分辨率。
        ridge = build_ridge_field(road_mask, geometry, self.ridge_config)

        image_padded = resize_and_pad(image, geometry, resample="bilinear")
        mask_padded = resize_and_pad(road_mask, geometry, resample="nearest")
        skeleton_padded = resize_and_pad(
            ridge.source_skeleton, geometry, resample="nearest"
        )

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
            np.ascontiguousarray(ridge.field[None].astype(np.float32))
        )
        valid_tensor = torch.from_numpy(
            np.ascontiguousarray(ridge.valid_mask[None].astype(np.float32))
        )
        regions_tensor = torch.from_numpy(
            np.ascontiguousarray(regions_to_one_hot(ridge.regions))
        )

        sample = {
            "image": image_tensor,
            "mask": mask_tensor,
            "skeleton": skeleton_tensor,
            "heatmaps": heatmap_tensor,
            "field_valid_mask": valid_tensor,
            "field_regions": regions_tensor,
            "id": entry.sample_id,
            "original_size": (int(source_height), int(source_width)),
            "resized_size": tuple(int(v) for v in geometry.resized_size),
        }
        if return_ridge:
            return sample, ridge
        return sample

    def __getitem__(self, index: int) -> Dict[str, Any]:
        index = int(index)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        return self._build_sample(index)

    # ------------------------------------------------------------------
    # 诊断接口
    # ------------------------------------------------------------------
    def field_report(self, index: int) -> Dict[str, Any]:
        """返回某样本山脊场目标的标量统计，便于离线检查。

        统计只包含 JSON 可序列化的值，不返回整张场。``augment=True`` 时
        每次调用会先推进该索引的 occurrence，因此连续调用对应不同增广；
        ``augment=False`` 时同一 epoch 内结果固定。
        """
        index = int(index)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        sample, ridge = self._build_sample(index, return_ridge=True)
        valid_mask = sample["field_valid_mask"][0].numpy() > 0
        stats = field_statistics(
            sample["heatmaps"][0].numpy(),
            valid_mask,
            regions_from_one_hot(
                sample["field_regions"].numpy(), valid_mask=valid_mask
            ),
        )
        stats.update(source_retention_statistics(ridge))
        stats.update(
            {
                "id": sample["id"],
                "original_size": list(sample["original_size"]),
                "resized_size": list(sample["resized_size"]),
                "heatmap_size": self.heatmap_size,
                "image_size": self.image_size,
                "padding_value": float(PADDING_VALUE),
                "ridge_config": {
                    "center_radius": self.ridge_config.center_radius,
                    "boundary_band": self.ridge_config.boundary_band,
                    "epsilon": self.ridge_config.epsilon,
                },
            }
        )
        return stats
