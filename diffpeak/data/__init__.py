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

# 山脊场符号同样延迟导入，避免轻量几何工具被迫加载图像科学依赖。
_RIDGE_EXPORTS = {
    "BACKGROUND_VALUE", "RIDGE_VALUE", "PADDING_VALUE", "FIELD_TARGET_VERSION",
    "REGION_CENTER", "REGION_REST_ROAD", "REGION_OUTSIDE_BAND",
    "REGION_FAR_BACKGROUND", "REGION_PADDING", "REGION_NAMES",
    "RidgeFieldConfig", "RidgeField", "coerce_ridge_config",
    "build_source_ridge_field", "build_ridge_field",
    "regions_to_one_hot", "regions_from_one_hot", "source_retention_statistics",
    "field_statistics",
}

__all__ = [
    "DeepGlobeDataset", "ImageGeometry", "build_geometry",
    "HardNegativeProvider", "NegativeSamplingConfig", "PositiveSamplingConfig",
    "RoadTargetSampler", "SampledTargets", "render_gaussian_heatmap", "select_spread_points",
    "BACKGROUND_VALUE", "RIDGE_VALUE", "PADDING_VALUE", "FIELD_TARGET_VERSION",
    "REGION_CENTER", "REGION_REST_ROAD", "REGION_OUTSIDE_BAND",
    "REGION_FAR_BACKGROUND", "REGION_PADDING", "REGION_NAMES",
    "RidgeFieldConfig", "RidgeField", "coerce_ridge_config",
    "build_source_ridge_field", "build_ridge_field",
    "regions_to_one_hot", "regions_from_one_hot", "source_retention_statistics",
    "field_statistics",
    "load_split_entries", "mask_path_for_image", "resolve_under_root", "sample_id_from_image_path",
]


def __getattr__(name: str):
    if name == "DeepGlobeDataset":
        value = getattr(import_module(".deepglobe", __name__), name)
    elif name in _SAMPLING_EXPORTS:
        value = getattr(import_module(".sampling", __name__), name)
    elif name in _RIDGE_EXPORTS:
        value = getattr(import_module(".ridge", __name__), name)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value
