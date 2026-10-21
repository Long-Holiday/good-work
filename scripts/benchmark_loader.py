"""只测真实数据预处理吞吐，使用固定 train 子集；不代表 GPU 训练吞吐。"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

# 子进程继承线程限制；防止每个 worker 再创建整机规模的 BLAS 线程池。
for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[variable] = "1"

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from diffpeak.data import DeepGlobeDataset
from diffpeak.engine import _collate_samples, _seed_worker
from diffpeak.inference import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/3090.yaml")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--train-split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, nargs="+", default=[0, 4, 8, 12])
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--samples", type=int, default=64)
    args = parser.parse_args()
    if args.samples < 1 or args.warmup < 0 or any(n < 0 for n in args.workers):
        parser.error("samples > 0，warmup/workers >= 0")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    entries = json.loads(args.train_split.read_text())
    random.Random(2026).shuffle(entries)
    entries = entries[:args.warmup + args.samples]
    if len(entries) < args.warmup + args.samples:
        parser.error("train 划分不足以完成预热和测量")
    config = load_config(args.config)
    data = config["data"]
    # 记录实际使用的脚本/配置修订；worker 循环内共用同一份元数据。
    config_sha256 = hashlib.sha256(args.config.read_bytes()).hexdigest()
    script_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    manifest_sha256 = hashlib.sha256(args.train_split.read_bytes()).hexdigest()
    # 独占结果路径，拒绝覆盖已有测量。
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        for workers in args.workers:
            dataset = DeepGlobeDataset(
                root=str(args.data_root), split_file=entries, augment=True,
                seed=2026, sampling_config=data.get("sampling_config"),
                heatmap_sigma=data.get("heatmap_sigma", 1.0),
                target_min_distance=data.get("target_min_distance", 4.0),
            )
            generator = torch.Generator().manual_seed(2026)
            loader = DataLoader(
                dataset, batch_size=1, shuffle=False, num_workers=workers,
                collate_fn=_collate_samples, worker_init_fn=_seed_worker,
                generator=generator, pin_memory=True,
            )
            started = time.perf_counter()
            iterator = iter(loader)
            for _ in range(args.warmup):
                next(iterator)
            warmup_seconds = time.perf_counter() - started
            started = time.perf_counter()
            for _ in range(args.samples):
                next(iterator)
            seconds = time.perf_counter() - started
            row = {
                "workers": workers, "cpu_threads": 1, "samples": args.samples,
                "warmup_samples": args.warmup, "warmup_seconds": warmup_seconds,
                "seconds": seconds, "images_per_second": args.samples / seconds,
                "seed": 2026, "train_manifest_sha256": manifest_sha256,
                "subset_sha256": hashlib.sha256(json.dumps(entries).encode()).hexdigest(),
                "config_path": str(args.config.resolve()),
                "config_sha256": config_sha256,
                "script_sha256": script_sha256,
                "torch_version": torch.__version__,
                "scope": "CPU loader only; worker count changes sampling RNG",
            }
            print(json.dumps(row), flush=True)
            handle.write(json.dumps(row) + "\n")
            handle.flush()
            del iterator, loader, dataset


if __name__ == "__main__":
    main()
