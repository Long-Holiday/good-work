"""用最终峰值提取器检查随机监督热图；由用户在依赖安装后按需运行。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from diffpeak.data import DeepGlobeDataset
from diffpeak.prompts import PeakExtractor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/3090.yaml"))
    parser.add_argument("--samples", type=int, default=16)
    args = parser.parse_args()
    if args.samples <= 0:
        parser.error("samples 必须为正整数")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    data = config["data"]
    dataset = DeepGlobeDataset(
        data["root"], data["train_split"], seed=config["seed"],
        sampling_config=data.get("sampling_config"),
        heatmap_sigma=data.get("heatmap_sigma", 1.0),
        target_min_distance=data.get("target_min_distance", 4.0),
    )
    extractor = PeakExtractor(**config["peaks"])
    for index in range(min(args.samples, len(dataset))):
        print(json.dumps(dataset.recovery_report(index, extractor=extractor), ensure_ascii=False))


if __name__ == "__main__":
    main()
