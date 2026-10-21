"""DeepGlobe 数据接口；轻量几何和划分工具不强制加载图像科学依赖。"""
from __future__ import annotations

from importlib import import_module

from .geometry import ImageGeometry, build_geometry
from .splits import (
    load_split_entries,
    mask_path_for_image,
    resolve_under_root,
    sample_id_from_image_path,
)

_SAMPLING_EXPORTS = {
    "HardNegativeProvider", "NegativeSamplingConfig", "PositiveSamplingConfig",
    "RoadTargetSampler", "SampledTargets", "render_gaussian_heatmap", "select_spread_points",
}

__all__ = [
    "DeepGlobeDataset", "ImageGeometry", "build_geometry",
    "HardNegativeProvider", "NegativeSamplingConfig", "PositiveSamplingConfig",
    "RoadTargetSampler", "SampledTargets", "render_gaussian_heatmap", "select_spread_points",
    "load_split_entries", "mask_path_for_image", "resolve_under_root", "sample_id_from_image_path",
]


def __getattr__(name: str):
    if name == "DeepGlobeDataset":
        value = getattr(import_module(".deepglobe", __name__), name)
    elif name in _SAMPLING_EXPORTS:
        value = getattr(import_module(".sampling", __name__), name)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value
