"""划分文件与 DeepGlobe 命名的解析工具。

``split_file`` 的正式格式是 JSON 字符串列表，每一项为相对数据根目录的
**影像**路径，例如 ``"train/100034_sat.jpg"``；掩码由影像名派生为同目录的
``"train/100034_mask.png"``。这里刻意不接受“自动扫目录”作为训练划分，
避免把没有标注的官方 valid/test 影像误当成监督样本。
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List, Sequence, Union

__all__ = [
    "load_split_entries",
    "resolve_under_root",
    "mask_path_for_image",
    "sample_id_from_image_path",
]

_SAT_SUFFIX = re.compile(r"_sat$", re.IGNORECASE)


def load_split_entries(split_file: Union[str, Path, Sequence[str]]) -> List[str]:
    """读取 JSON 数组，或直接返回传入的字符串序列。

    Args:
        split_file: JSON 文件路径，或已经是字符串列表的划分。

    Returns:
        相对路径字符串列表；保持文件中的顺序。
    """
    if isinstance(split_file, (list, tuple)):
        entries = list(split_file)
    elif isinstance(split_file, (str, Path)):
        path = Path(split_file).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"split_file 不存在：{path}")
        with path.open("r", encoding="utf-8") as handle:
            entries = json.load(handle)
    else:
        raise TypeError("split_file 必须是 JSON 文件路径或字符串列表")

    if not isinstance(entries, list) or not all(
        isinstance(entry, str) for entry in entries
    ):
        raise ValueError("split_file 内容必须是 JSON 字符串数组")
    if not entries:
        raise ValueError("split_file 不能为空")
    return entries


def resolve_under_root(root: Union[str, Path], relative_path: str) -> Path:
    """把划分项解析为绝对路径，并阻止 ``..`` 逃逸数据根目录。"""
    root = Path(root).expanduser().resolve()
    candidate = Path(relative_path)
    if candidate.is_absolute():
        raise ValueError(f"划分项必须使用相对路径：{relative_path}")
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"划分项逃逸出数据根目录：{relative_path}") from exc
    return resolved


def mask_path_for_image(image_path: Union[str, Path]) -> Path:
    """按 ``xxx_sat.jpg`` -> 同目录 ``xxx_mask.png`` 的规则派生掩码路径。"""
    image_path = Path(image_path)
    stem = image_path.stem
    if not _SAT_SUFFIX.search(stem):
        raise ValueError(
            f"影像名不以 _sat 结尾，无法按约定派生掩码：{image_path.name}"
        )
    return image_path.with_name(_SAT_SUFFIX.sub("", stem) + "_mask.png")


def sample_id_from_image_path(image_path: Union[str, Path]) -> str:
    """返回稳定的样本 id，例如 ``train/100034_sat.jpg`` -> ``100034``。"""
    stem = Path(image_path).stem
    return _SAT_SUFFIX.sub("", stem)
