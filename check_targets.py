"""检查 DeepGlobe 单通道有符号山脊场目标；依赖安装后按需运行。

输出每个样本一行的 JSON 统计：有效/padding 比例、场分布、四个监督区域的
像素数，以及基于源分辨率几何投影的细道路/骨架保留诊断。检查的是目标场
本身，不再调用旧峰值提取器或点采样器。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from diffpeak.data import DeepGlobeDataset
from diffpeak.inference import resolve_data_root, resolve_split_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/3090.yaml"))
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument(
        "--split",
        choices=("train", "val"),
        default="train",
        help="使用配置中的 train_split 或 val_split。",
    )
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--heatmap-size", type=int, default=None)
    args = parser.parse_args()
    if args.samples <= 0:
        parser.error("samples 必须为正整数")

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    config_dir = args.config.expanduser().resolve().parent
    data = config["data"]
    split_key = "val_split" if args.split == "val" else "train_split"
    split_file = data.get(split_key)
    if not split_file:
        parser.error(f"配置缺少 data.{split_key}")
    # 与 engine / inference 使用同一套相对路径解析规则，而不是依赖当前
    # 工作目录。
    data_root = resolve_data_root(config, config_dir)
    split_path = resolve_split_file(
        split_file, data_root, config_dir, must_exist=True
    )
    dataset = DeepGlobeDataset(
        str(data_root),
        str(split_path),
        augment=False,
        image_size=args.image_size or int(data.get("image_size", 1024)),
        heatmap_size=args.heatmap_size or int(data.get("heatmap_size", 256)),
        seed=int(config.get("seed", 42)),
        ridge_config=data.get("ridge_config"),
    )
    for index in range(min(args.samples, len(dataset))):
        print(
            json.dumps(dataset.field_report(index), ensure_ascii=False)
        )


if __name__ == "__main__":
    main()
