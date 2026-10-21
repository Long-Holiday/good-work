"""双通道提示热图的固定峰值提取与回提诊断。

提取流程严格对应 ``docs/model.md``：

1. 对每个通道独立做轻度高斯平滑；
2. 用 h-maxima 得到显著峰区域，并用绝对阈值删除低响应峰；
3. 对峰区域做连通域标记，每个区域取一个确定性代表点
   （平顶时选离区域中心最近、再按 ``(y, x)`` 字典序的像素）；
4. 按响应降序执行**同通道**距离 NMS，不设固定点数上限；
5. 先完成峰值检测，再按 ``valid_sizes`` 过滤右下填充区域，避免填充
   边界参与 NMS 后压掉有效峰；
6. 正负通道坐标完全重合时，保留分数高者；分数相同时正点优先。

坐标约定
--------
* 输入热图为 ``B x 2 x h x w``，通道 0 为正提示、通道 1 为负提示；
* 默认热图 256、SAM 输入 1024；热图坐标与 SAM 坐标都使用像素索引空间，
  映射采用像素中心对齐：``x_image = (x_heat + 0.5) * 1024 / 256 - 0.5``，
  与数据集的 ``original_yx_to_heatmap_yx`` 互为逆变换；
* ``valid_sizes`` 中每项是填充前的 ``(height, width)``，单位为 SAM 输入
  像素，可直接使用 ``DeepGlobeDataset`` 返回的 ``resized_size``；
* 返回坐标为 ``(x, y)``，标签 1 为正、0 为负。

随机性
------
本模块的峰值提取是确定性的，不含任何随机数。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from scipy import ndimage
from skimage.morphology import h_maxima, local_maxima

__all__ = [
    "DEFAULT_PEAK_CONFIG",
    "PeakExtractor",
    "extract_numpy",
    "target_recovery_report",
]


DEFAULT_PEAK_CONFIG = {
    "sigma": 0.5,
    "h": 0.05,
    "threshold": 0.3,
    "min_distance": 4.0,
}

_CFG_KEYS = ("sigma", "h", "threshold", "min_distance")
_CHANNEL_ALIASES = {
    "positive": "positive",
    "pos": "positive",
    "channel_0": "positive",
    "0": "positive",
    "negative": "negative",
    "neg": "negative",
    "channel_1": "negative",
    "1": "negative",
}


@dataclass(frozen=True)
class _ChannelConfig:
    sigma: float
    h: float
    threshold: float
    min_distance: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.sigma) or self.sigma < 0:
            raise ValueError("sigma 必须是非负有限数")
        if not np.isfinite(self.h) or self.h < 0:
            raise ValueError("h 必须是非负有限数")
        if np.isnan(self.threshold):
            raise ValueError("threshold 不能是 NaN")
        if not np.isfinite(self.min_distance) or self.min_distance < 0:
            raise ValueError("min_distance 必须是非负有限数")


@dataclass(frozen=True)
class _Peak:
    score: float
    y_heat: float
    x_heat: float


@dataclass(frozen=True)
class _MappedPeak:
    score: float
    y_heat: float
    x_heat: float
    x_image: float
    y_image: float
    label: int


def _build_channel_config(mapping: Mapping[str, Any], name: str) -> _ChannelConfig:
    unknown = set(mapping) - set(_CFG_KEYS)
    if unknown:
        raise ValueError(f"通道 {name} 包含未知峰值参数: {sorted(unknown)}")
    values = {key: float(mapping.get(key, DEFAULT_PEAK_CONFIG[key])) for key in _CFG_KEYS}
    return _ChannelConfig(**values)


def _merge_channel_block(
    target: dict,
    block: Any,
    name: str,
) -> None:
    """把一个通道的覆盖项合并到 ``target``。"""
    if block is None:
        return
    if not isinstance(block, Mapping):
        raise TypeError(f"{name} 的通道配置必须是 dict")
    for key, value in block.items():
        alias = _CHANNEL_ALIASES.get(str(key).lower())
        if alias is None:
            raise ValueError(f"{name} 中存在未知通道名: {key}")
        if value is not None and not isinstance(value, Mapping):
            raise TypeError(f"{name}.{key} 必须是 dict")
        target[alias].update(dict(value or {}))


def _split_channel_value(value: Any, key: str) -> Optional[dict]:
    """把一个顶层峰值参数拆成逐通道覆盖。

    * 标量：返回 ``None``，表示正负两个通道共用该值；
    * ``{"positive": x, "negative": y}``（键也接受 ``pos`` / ``neg`` /
      ``0`` / ``1`` 等别名）：返回逐通道覆盖；
    * 长度为 2 的标量序列：按 ``(positive, negative)`` 解释。
    """
    if isinstance(value, Mapping):
        overrides: dict = {}
        for channel, item in value.items():
            alias = _CHANNEL_ALIASES.get(str(channel).lower())
            if alias is None:
                raise ValueError(f"参数 {key} 中存在未知通道名: {channel}")
            if isinstance(item, (str, bytes, Mapping)) or not np.isscalar(item):
                raise TypeError(f"参数 {key}.{channel} 必须是标量")
            overrides[alias] = item
        return overrides
    if _is_scalar_pair(value):
        return {"positive": value[0], "negative": value[1]}
    return None


def _normalise_config(
    config: Optional[Mapping[str, Any]],
) -> dict:
    """把用户 dict 归一化为 ``{"positive": cfg, "negative": cfg}``。

    顶层 ``sigma`` / ``h`` / ``threshold`` / ``min_distance`` 既可以是标量
    （两通道共用），也可以是逐通道映射或 ``(positive, negative)`` 二元组。
    显式的 ``channels`` / ``positive`` / ``negative`` 块优先级最高。
    """
    if config is None:
        config = {}
    if not isinstance(config, Mapping):
        raise TypeError("PeakExtractor 配置必须是 dict 或 None")

    global_values = dict(DEFAULT_PEAK_CONFIG)
    parameter_overrides = {"positive": {}, "negative": {}}
    block_overrides = {"positive": {}, "negative": {}}
    for key, value in config.items():
        lowered = str(key).lower()
        if lowered in _CFG_KEYS:
            per_channel = _split_channel_value(value, str(key))
            if per_channel is None:
                global_values[lowered] = value
            else:
                for name, item in per_channel.items():
                    parameter_overrides[name][lowered] = item
        elif lowered in {"channels", "channel_configs"}:
            _merge_channel_block(block_overrides, value, str(key))
        elif lowered in _CHANNEL_ALIASES:
            _merge_channel_block(
                block_overrides, {lowered: value}, str(key)
            )
        else:
            raise ValueError(f"PeakExtractor 配置包含未知字段: {key}")

    channels = {}
    for name in ("positive", "negative"):
        merged = dict(global_values)
        merged.update(parameter_overrides[name])
        merged.update(block_overrides[name])
        channels[name] = _build_channel_config(merged, name)
    return channels


def _as_numpy_heatmaps(heatmaps: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    if torch.is_tensor(heatmaps):
        array = heatmaps.detach().cpu().numpy()
    else:
        array = np.asarray(heatmaps)
    if array.ndim == 3:
        if array.shape[0] != 2:
            raise ValueError(
                f"未分批热图必须是 2 x h x w，实际为 {array.shape}"
            )
        array = array[None]
    if array.ndim != 4 or array.shape[1] != 2:
        raise ValueError(f"heatmaps 必须是 B x 2 x h x w，实际为 {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("heatmaps 含 NaN 或 Inf，无法做确定性峰值提取")
    return array.astype(np.float64, copy=False)


def _is_scalar_pair(value: Any) -> bool:
    try:
        return (
            not isinstance(value, (str, bytes, Mapping))
            and len(value) == 2
            and np.isscalar(value[0])
            and np.isscalar(value[1])
        )
    except TypeError:
        return False


def _coerce_valid_size(value: Any, image_size: int) -> Optional[Tuple[int, int]]:
    if value is None:
        # 逐样本 None 表示该图没有任何 padding；与 valid_sizes=None 等价。
        return None
    if isinstance(value, Mapping):
        if "height" in value and "width" in value:
            height, width = value["height"], value["width"]
        elif "h" in value and "w" in value:
            height, width = value["h"], value["w"]
        else:
            raise ValueError("valid_sizes 的 dict 项必须包含 height/width")
    elif torch.is_tensor(value):
        value = value.detach().cpu().tolist()
        height, width = value
    else:
        if not _is_scalar_pair(value):
            raise ValueError("valid_sizes 的每个元素必须是 (height, width)")
        height, width = value
    height, width = int(height), int(width)
    if not (0 < height <= image_size and 0 < width <= image_size):
        raise ValueError(
            f"valid_size 必须在 (0, image_size] 内，实际为 {(height, width)}"
        )
    return height, width


def _normalise_valid_sizes(
    valid_sizes: Any,
    batch_size: int,
    image_size: int,
) -> List[Optional[Tuple[int, int]]]:
    if valid_sizes is None:
        return [None] * batch_size
    if isinstance(valid_sizes, Mapping):
        if batch_size != 1:
            raise ValueError("单个 valid_size dict 只适用于 batch_size=1")
        return [_coerce_valid_size(valid_sizes, image_size)]
    if torch.is_tensor(valid_sizes):
        valid_sizes = valid_sizes.detach().cpu().tolist()
    if _is_scalar_pair(valid_sizes):
        if batch_size != 1:
            raise ValueError("单个 (height, width) 只适用于 batch_size=1")
        return [_coerce_valid_size(valid_sizes, image_size)]
    try:
        items = list(valid_sizes)
    except TypeError as exc:
        raise TypeError("valid_sizes 必须是序列、单个 (h, w) 或 None") from exc
    if len(items) != batch_size:
        raise ValueError(
            f"valid_sizes 长度 {len(items)} 与 batch_size {batch_size} 不一致"
        )
    return [_coerce_valid_size(item, image_size) for item in items]


def _detect_peaks(heatmap: np.ndarray, config: _ChannelConfig) -> List[_Peak]:
    """单通道 h-maxima + 连通域代表点。"""
    if config.sigma > 0:
        # mode='nearest' 避免把热图边缘当作零背景而制造/抑制边界峰。
        smooth = ndimage.gaussian_filter(
            heatmap, sigma=float(config.sigma), mode="nearest"
        )
    else:
        smooth = heatmap
    # 全零/全负热图没有可提取的峰。显式处理，避免 threshold=0、h=0 时
    # local_maxima 把整张常量图当成一个平台。
    if not np.any(smooth > 0):
        return []

    if config.h > 0:
        peak_mask = h_maxima(smooth, float(config.h))
    else:
        # h=0 等价于区域极大值；local_maxima 保留整个平顶区域。
        # 显式使用 8 连通（connectivity=ndim），与 h_maxima 默认的
        # fully-connected footprint 保持一致，平顶标记不会因连通性不同
        # 而在不同 skimage 版本/调用之间变化。
        peak_mask = local_maxima(smooth, connectivity=smooth.ndim)
    peak_mask = np.asarray(peak_mask, dtype=bool)
    if config.threshold > -np.inf:
        peak_mask &= smooth >= float(config.threshold)
    if not peak_mask.any():
        return []

    components, num_components = ndimage.label(
        peak_mask, structure=np.ones((3, 3), dtype=bool)
    )
    peaks: List[_Peak] = []
    for component_id in range(1, int(num_components) + 1):
        ys, xs = np.nonzero(components == component_id)
        if len(ys) == 0:
            continue
        values = smooth[ys, xs]
        max_value = float(values.max())
        # 平顶：在最大值并列的像素中，选择离区域中心最近者；
        # 再以 (y, x) 字典序固定最终平局，保证完全确定性。
        tied = values >= max_value - 1e-7
        tied_y = ys[tied].astype(np.float64)
        tied_x = xs[tied].astype(np.float64)
        center_y = float(ys.mean())
        center_x = float(xs.mean())
        distance_sq = (tied_y - center_y) ** 2 + (tied_x - center_x) ** 2
        order = np.lexsort((tied_x, tied_y, distance_sq))
        chosen = int(order[0])
        peaks.append(
            _Peak(
                score=max_value,
                y_heat=float(tied_y[chosen]),
                x_heat=float(tied_x[chosen]),
            )
        )
    return peaks


def _nms_peaks(peaks: Sequence[Any], min_distance: float) -> List[Any]:
    """按分数降序做同通道距离 NMS；分数相同用 (y, x) 稳定排序。"""
    ordered = sorted(peaks, key=lambda peak: (-peak.score, peak.y_heat, peak.x_heat))
    if min_distance <= 0:
        return ordered
    minimum_sq = float(min_distance) ** 2
    kept: List[Any] = []
    for peak in ordered:
        if all(
            (peak.y_heat - other.y_heat) ** 2
            + (peak.x_heat - other.x_heat) ** 2
            >= minimum_sq
            for other in kept
        ):
            kept.append(peak)
    return kept


def _map_and_filter_peaks(
    peaks: Iterable[_Peak],
    scale: float,
    valid_size: Optional[Tuple[int, int]],
    image_size: int,
    label: int,
) -> List[_MappedPeak]:
    """映射到 SAM 坐标并过滤填充区域；随后才做 NMS。"""
    if valid_size is None:
        valid_height = valid_width = image_size
    else:
        valid_height, valid_width = valid_size

    mapped: List[_MappedPeak] = []
    for peak in peaks:
        x_image = (peak.x_heat + 0.5) * scale - 0.5
        y_image = (peak.y_heat + 0.5) * scale - 0.5
        # 有效内容覆盖连续区间 [-0.5, valid-0.5]；第一个填充像素中心
        # 位于 valid，因此这里不会把填充峰当作有效峰，也不会让 padding
        # 参与 NMS 后压掉有效峰。
        if not (
            -0.5 <= x_image <= valid_width - 0.5
            and -0.5 <= y_image <= valid_height - 0.5
        ):
            continue
        # SAM 提示坐标落在有效内容的像素索引范围内。
        x_image = float(np.clip(x_image, 0.0, max(valid_width - 1.0, 0.0)))
        y_image = float(np.clip(y_image, 0.0, max(valid_height - 1.0, 0.0)))
        mapped.append(
            _MappedPeak(
                score=peak.score,
                y_heat=peak.y_heat,
                x_heat=peak.x_heat,
                x_image=x_image,
                y_image=y_image,
                label=label,
            )
        )
    return mapped


def _resolve_cross_channel(peaks: Sequence[_MappedPeak]) -> List[_MappedPeak]:
    """坐标重合的异类峰只保留分数高者，分数相同时正点优先。

    重合判定使用热图上的离散像素坐标：检测得到的峰值本来就是整数热图
    坐标，两个通道共享同一网格，因此这是精确的“同坐标”判定；若改用
    SAM 坐标再取整，在热图分辨率高于输入（scale < 1）时可能把相邻的
    不同峰误判为同一点。
    """
    best_by_key = {}
    for index, peak in enumerate(peaks):
        key = (int(round(peak.x_heat)), int(round(peak.y_heat)))
        current = best_by_key.get(key)
        if current is None:
            best_by_key[key] = index
            continue
        other = peaks[current]
        if (peak.score, peak.label) > (other.score, other.label):
            best_by_key[key] = index
    keep = set(best_by_key.values())
    return [peak for index, peak in enumerate(peaks) if index in keep]


class PeakExtractor:
    """固定峰值提取器（推理与训练奖励共用，不含随机性）。

    Args:
        config: 可选 dict。顶层可给 ``sigma`` / ``h`` / ``threshold`` /
            ``min_distance``；每个值可以是标量（两通道共用）、逐通道映射
            （如 ``{"positive": 0.3, "negative": 0.25}``），或
            ``(positive, negative)`` 二元组。通道覆盖也可放在 ``channels``
            中（也接受 ``positive`` / ``negative``、``pos`` / ``neg``、
            ``0`` / ``1``）。默认值见 :data:`DEFAULT_PEAK_CONFIG`。
        **overrides: 也可直接以关键字给出顶层参数，例如
            ``PeakExtractor(sigma=0.5, h=0.05)``。同名顶层键由关键字覆盖
            ``config``；``config`` 中显式的通道块仍可对该通道做更具体的
            覆盖。

    Example:
        >>> extractor = PeakExtractor(
        ...     {"sigma": 0.5, "h": 0.05, "threshold": 0.3,
        ...      "min_distance": 4, "negative": {"threshold": 0.25}}
        ... )
        >>> points = extractor(heatmaps, image_size=1024,
        ...                    valid_sizes=[(1024, 1024)])
        >>> coords, labels = points[0]  # coords: K x 2 (x, y)
    """

    def __init__(
        self,
        config: Optional[Mapping[str, Any]] = None,
        **overrides: Any,
    ) -> None:
        if overrides:
            if config is None:
                config = {}
            if not isinstance(config, Mapping):
                raise TypeError("PeakExtractor 配置必须是 dict 或 None")
            merged = dict(config)
            merged.update(overrides)
            config = merged
        self._channels = _normalise_config(config)

    @property
    def config(self) -> dict:
        """返回完整、已归一化的逐通道配置（只读用途，修改不会生效）。"""
        return {
            "channels": {
                name: asdict(channel)
                for name, channel in self._channels.items()
            }
        }

    @property
    def positive_config(self) -> dict:
        return asdict(self._channels["positive"])

    @property
    def negative_config(self) -> dict:
        return asdict(self._channels["negative"])

    def __repr__(self) -> str:
        return (
            "PeakExtractor("
            f"positive={self.positive_config}, "
            f"negative={self.negative_config})"
        )

    def _extract_numpy(
        self,
        heatmaps: np.ndarray,
        image_size: int,
        valid_sizes: Any,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        array = _as_numpy_heatmaps(heatmaps)
        image_size = int(image_size)
        if image_size <= 0:
            raise ValueError("image_size 必须为正")
        batch_size, _, heatmap_height, heatmap_width = array.shape
        if heatmap_height <= 0 or heatmap_width <= 0:
            raise ValueError("提示热图的空间尺寸必须为正")
        if heatmap_height != heatmap_width:
            raise ValueError("提示热图必须是正方形")
        scale = image_size / float(heatmap_height)
        sizes = _normalise_valid_sizes(valid_sizes, batch_size, image_size)

        results: List[Tuple[np.ndarray, np.ndarray]] = []
        for batch_index in range(batch_size):
            channel_peaks: List[_MappedPeak] = []
            for channel_index, name in enumerate(("positive", "negative")):
                config = self._channels[name]
                peaks = _detect_peaks(array[batch_index, channel_index], config)
                mapped = _map_and_filter_peaks(
                    peaks,
                    scale=scale,
                    valid_size=sizes[batch_index],
                    image_size=image_size,
                    label=1 if channel_index == 0 else 0,
                )
                # NMS 在过滤填充区域之后进行：填充区峰值不能压掉有效峰。
                channel_peaks.extend(_nms_peaks(mapped, config.min_distance))

            final = _resolve_cross_channel(channel_peaks)
            if final:
                coords = np.asarray(
                    [[peak.x_image, peak.y_image] for peak in final],
                    dtype=np.float32,
                ).reshape(-1, 2)
                labels = np.asarray(
                    [peak.label for peak in final], dtype=np.int64
                )
            else:
                coords = np.empty((0, 2), dtype=np.float32)
                labels = np.empty((0,), dtype=np.int64)
            results.append((coords, labels))
        return results

    def __call__(
        self,
        heatmaps: Union[np.ndarray, torch.Tensor],
        image_size: int = 1024,
        valid_sizes: Any = None,
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """提取一批提示点。

        Args:
            heatmaps: ``B x 2 x h x w``（也接受单张 ``2 x h x w``）。
            image_size: SAM 输入边长，默认 1024。
            valid_sizes: 每张图填充前的 ``(height, width)``；``None`` 表示
                整张图有效。可直接传 ``DeepGlobeDataset`` 的 ``resized_size``。

        Returns:
            长度为 ``B`` 的列表，每项为 ``(coords, labels)``：

            * ``coords``: ``K x 2`` 的 ``float32``，顺序为 ``(x, y)``；
            * ``labels``: ``K`` 的 ``int64``，1 为正点、0 为负点。

            点数 ``K`` 完全由有效峰决定，没有固定上限。
        """
        device = heatmaps.device if torch.is_tensor(heatmaps) else torch.device("cpu")
        array = _as_numpy_heatmaps(heatmaps)
        results = self._extract_numpy(array, image_size, valid_sizes)
        return [
            (
                torch.as_tensor(coords, dtype=torch.float32, device=device),
                torch.as_tensor(labels, dtype=torch.int64, device=device),
            )
            for coords, labels in results
        ]


def extract_numpy(
    heatmaps: Union[np.ndarray, torch.Tensor],
    image_size: int = 1024,
    valid_sizes: Any = None,
    config: Optional[Union[Mapping[str, Any], PeakExtractor]] = None,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """``PeakExtractor`` 的纯 Numpy 接口，便于离线诊断/评估。

    Returns:
        每项为 ``(coords float32 K x 2 (x, y), labels int64 K)``。
    """
    if isinstance(config, PeakExtractor):
        extractor = config
    else:
        extractor = PeakExtractor(config)
    array = _as_numpy_heatmaps(heatmaps)
    return extractor._extract_numpy(array, image_size, valid_sizes)


def _as_array(value: Any, name: str) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _normalise_points_batch(
    value: Any,
    batch_size: int,
    name: str,
    width: int = 2,
) -> List[np.ndarray]:
    """把 list / tensor / ndarray 统一成逐样本数组列表。"""
    if isinstance(value, (list, tuple)):
        if batch_size == 1:
            # 单样本既可能是 "K 个点/标签的列表"，也可能是 "含一个样本的列表"。
            # 后者可能包含 torch.Tensor，直接对整表 np.asarray 会出错。
            if len(value) == 1:
                return [_as_array(value[0], name).reshape(-1, width)]
            return [_as_array(value, name).reshape(-1, width)]
        if len(value) != batch_size:
            raise ValueError(
                f"{name} 的样本数 {len(value)} 与 batch_size {batch_size} 不一致"
            )
        return [_as_array(item, name).reshape(-1, width) for item in value]

    array = _as_array(value, name)
    if batch_size == 1:
        return [array.reshape(-1, width)]
    if array.ndim == 1:
        if width == 1 and array.shape[0] == batch_size:
            return [array[index].reshape(1, 1) for index in range(batch_size)]
        raise ValueError(f"{name} 形状无法对应 batch_size={batch_size}: {array.shape}")
    if array.ndim == 2:
        # 标签可以是 (B, K)；坐标的二维数组始终是单样本 (K, 2)，
        # 因为逐样本坐标必须是 (B, K, 2)。避免把 (K, 2) 误拆成 K 个样本。
        if width == 1 and array.shape[0] == batch_size:
            return [array[index].reshape(-1, 1) for index in range(batch_size)]
        raise ValueError(
            f"{name} 形状无法对应 batch_size={batch_size}: {array.shape}"
        )
    if array.ndim == 3:
        if array.shape[0] != batch_size:
            raise ValueError(
                f"{name} 第一维 {array.shape[0]} 与 batch_size {batch_size} 不一致"
            )
        if array.shape[2] != width:
            # 避免把 (B, K, 3) 之类的错误输入静默 reshape 成错误的点对。
            raise ValueError(
                f"{name} 最后一维应为 {width}，实际为 {array.shape[2]}"
            )
        return [array[index].reshape(-1, width) for index in range(batch_size)]
    raise ValueError(f"{name} 形状不支持: {array.shape}")


def _match_points(
    expected: np.ndarray,
    predicted: np.ndarray,
    radius: float,
) -> Tuple[int, List[float]]:
    """一对一贪心匹配；按距离从近到远，确定性。"""
    if len(expected) == 0 or len(predicted) == 0:
        return 0, []
    distance = np.linalg.norm(
        expected[:, None, :] - predicted[None, :, :], axis=2
    )
    candidate_pairs = np.argwhere(distance <= radius)
    if len(candidate_pairs) == 0:
        return 0, []
    pair_distances = distance[candidate_pairs[:, 0], candidate_pairs[:, 1]]
    order = np.lexsort(
        (
            candidate_pairs[:, 1],
            candidate_pairs[:, 0],
            pair_distances,
        )
    )
    used_expected = set()
    used_predicted = set()
    matched_distances: List[float] = []
    for pair_index in order:
        expected_index = int(candidate_pairs[pair_index, 0])
        predicted_index = int(candidate_pairs[pair_index, 1])
        if expected_index in used_expected or predicted_index in used_predicted:
            continue
        used_expected.add(expected_index)
        used_predicted.add(predicted_index)
        matched_distances.append(float(pair_distances[pair_index]))
    return len(matched_distances), matched_distances


def _summary(
    expected: int,
    predicted: int,
    matched: int,
    distances: Sequence[float],
) -> dict:
    missed = expected - matched
    extra = predicted - matched
    if predicted > 0:
        precision = matched / predicted
    else:
        precision = 1.0 if expected == 0 else 0.0
    if expected > 0:
        recall = matched / expected
    else:
        recall = 1.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )
    return {
        "expected": int(expected),
        "predicted": int(predicted),
        "matched": int(matched),
        "missed": int(missed),
        "extra": int(extra),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "mean_match_distance": (
            float(np.mean(distances)) if len(distances) else None
        ),
    }


def target_recovery_report(
    heatmaps: Union[np.ndarray, torch.Tensor],
    expected_coords: Any,
    expected_labels: Any,
    extractor: Optional[Union[PeakExtractor, Mapping[str, Any]]] = None,
    image_size: int = 1024,
    valid_sizes: Any = None,
    match_radius: Optional[float] = None,
    expected_space: str = "image",
    heatmap_size: Optional[int] = None,
) -> dict:
    """检查目标热图能否被**最终峰值提取器**回提。

    这是数据集构造阶段和实验排查用的诊断函数：把目标热图送进固定的
    :class:`PeakExtractor`，再与构造目标时的期望点逐一匹配。它不参与损失。

    Args:
        heatmaps: ``B x 2 x h x w`` 或单张 ``2 x h x w``。
        expected_coords: 期望点。形式为逐样本列表、``B x K x 2`` 数组，
            或 ``batch_size=1`` 时的 ``K x 2``；坐标顺序为 ``(x, y)``。
        expected_labels: 对应的 1/0 标签，形状对应 ``expected_coords``。
        extractor: ``PeakExtractor``、配置 dict 或 ``None``（默认配置）。
        image_size: SAM 输入边长。
        valid_sizes: 每张图填充前的 ``(height, width)``。
        match_radius: 匹配半径（SAM 输入像素）。``None``（默认）时按热图
            量化误差自动取值：``max(2.0, 0.75 * image_size / heatmap_size)``。
            max-高斯目标的实际峰值只能落在整数热图像素上，因此期望点与
            回提点之间最多有约 ``0.5 * sqrt(2)`` 个热图像素的偏移；默认值
            会覆盖该偏移，避免把正常量化误差误报为漏检。
        expected_space: ``"image"``（默认，SAM 坐标）或 ``"heatmap"``。
        heatmap_size: ``expected_space="heatmap"`` 时用于换算的热图边长；
            默认从 ``heatmaps`` 推断。

    Returns:
        报告 dict，包含整体的 precision / recall / F1、匹配距离，以及
        ``positive``、``negative`` 两个通道和逐样本明细。``recovered`` 为
        ``True`` 表示所有期望点都在半径内被找回且没有多余峰。
    """
    array = _as_numpy_heatmaps(heatmaps)
    batch_size, _, heatmap_height, _ = array.shape
    if heatmap_size is None:
        heatmap_size = heatmap_height
    image_size = int(image_size)
    heatmap_size = int(heatmap_size)
    if image_size <= 0 or heatmap_size <= 0:
        raise ValueError("image_size 与 heatmap_size 必须为正")
    if expected_space not in {"image", "heatmap"}:
        raise ValueError("expected_space 必须是 'image' 或 'heatmap'")
    if match_radius is None:
        # 热图量化：0.5 像素/轴的最大偏移，对角线再乘 sqrt(2)；0.75 是
        # 该上界的保守近似，并保留 2 个输入像素的下限以兼容旧默认值。
        match_radius = max(2.0, 0.75 * image_size / float(heatmap_size))
    if match_radius <= 0:
        raise ValueError("match_radius 必须为正")

    if extractor is None:
        extractor = PeakExtractor()
    elif isinstance(extractor, Mapping):
        extractor = PeakExtractor(extractor)
    elif not isinstance(extractor, PeakExtractor):
        raise TypeError("extractor 必须是 PeakExtractor、dict 或 None")

    predictions = extractor._extract_numpy(array, image_size, valid_sizes)
    points_batch = _normalise_points_batch(
        expected_coords, batch_size, "expected_coords", width=2
    )
    labels_batch = _normalise_points_batch(
        expected_labels, batch_size, "expected_labels", width=1
    )
    if expected_space == "heatmap":
        scale = image_size / float(heatmap_size)
        converted = []
        for points in points_batch:
            if len(points) == 0:
                converted.append(points.reshape(0, 2))
                continue
            image_points = np.empty_like(points, dtype=np.float64)
            image_points[:, 0] = (points[:, 0] + 0.5) * scale - 0.5
            image_points[:, 1] = (points[:, 1] + 0.5) * scale - 0.5
            converted.append(image_points)
        points_batch = converted

    totals = {
        "positive": {"expected": 0, "predicted": 0, "matched": 0, "distances": []},
        "negative": {"expected": 0, "predicted": 0, "matched": 0, "distances": []},
    }
    per_sample = []
    empty_correct = 0
    empty_count = 0

    for sample_index in range(batch_size):
        expected = points_batch[sample_index]
        labels = labels_batch[sample_index].reshape(-1)
        if len(expected) != len(labels):
            raise ValueError(
                f"样本 {sample_index} 的 expected_coords 与 expected_labels 数量不一致"
            )
        if len(labels) and not np.isin(labels, [0, 1]).all():
            raise ValueError("expected_labels 只能包含 0 和 1")
        predicted_coords, predicted_labels = predictions[sample_index]
        predicted_labels = np.asarray(predicted_labels).reshape(-1)

        sample_report = {"index": sample_index}
        sample_expected_total = 0
        sample_predicted_total = 0
        sample_matched_total = 0
        for class_id, name in ((1, "positive"), (0, "negative")):
            expected_class = expected[labels == class_id]
            predicted_class = predicted_coords[predicted_labels == class_id]
            matched, distances = _match_points(
                expected_class, predicted_class, float(match_radius)
            )
            totals[name]["expected"] += len(expected_class)
            totals[name]["predicted"] += len(predicted_class)
            totals[name]["matched"] += matched
            totals[name]["distances"].extend(distances)
            sample_report[name] = {
                "expected": int(len(expected_class)),
                "predicted": int(len(predicted_class)),
                "matched": int(matched),
                "missed": int(len(expected_class) - matched),
                "extra": int(len(predicted_class) - matched),
                "mean_match_distance": (
                    float(np.mean(distances)) if distances else None
                ),
            }
            sample_expected_total += len(expected_class)
            sample_predicted_total += len(predicted_class)
            sample_matched_total += matched

        sample_report["expected_total"] = int(sample_expected_total)
        sample_report["predicted_total"] = int(sample_predicted_total)
        sample_report["matched_total"] = int(sample_matched_total)
        sample_report["empty_correct"] = (
            sample_expected_total == 0 and sample_predicted_total == 0
        )
        if sample_expected_total == 0:
            empty_count += 1
            if sample_predicted_total == 0:
                empty_correct += 1
        per_sample.append(sample_report)

    summaries = {}
    for name in ("positive", "negative"):
        values = totals[name]
        summaries[name] = _summary(
            values["expected"],
            values["predicted"],
            values["matched"],
            values["distances"],
        )

    expected_total = sum(item["expected"] for item in summaries.values())
    predicted_total = sum(item["predicted"] for item in summaries.values())
    matched_total = sum(item["matched"] for item in summaries.values())
    all_distances = (
        totals["positive"]["distances"] + totals["negative"]["distances"]
    )
    report = _summary(
        expected_total, predicted_total, matched_total, all_distances
    )
    report.update(
        {
            "num_samples": int(batch_size),
            "image_size": int(image_size),
            "heatmap_size": int(heatmap_size),
            "match_radius": float(match_radius),
            "expected_space": expected_space,
            "positive": summaries["positive"],
            "negative": summaries["negative"],
            "empty_sample_count": int(empty_count),
            "empty_correct_count": int(empty_correct),
            "recovered": bool(
                report["missed"] == 0 and report["extra"] == 0
            ),
            "per_sample": per_sample,
        }
    )
    return report
