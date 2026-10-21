"""从有标注影像生成可复现划分；绝不修改原始数据。"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("DeepGlobe_Road"))
    parser.add_argument("--output", type=Path, default=Path("splits"))
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.0,
                        help="从有标注 train 中另留独立测试集；默认 0 保持原有两分划分。")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 0 < args.val_fraction < 1:
        parser.error("val-fraction 必须在 (0, 1) 内")
    if not 0 <= args.test_fraction < 1 or args.val_fraction + args.test_fraction >= 1:
        parser.error("test-fraction 必须在 [0, 1) 内，且 val/test 比例之和小于 1")
    if not args.root.is_dir():
        parser.error(f"数据目录不存在：{args.root}")
    candidates = sorted((args.root / "train").glob("*_sat.jpg"))
    missing = [p for p in candidates if not p.with_name(p.name.replace("_sat.jpg", "_mask.png")).is_file()]
    if missing:
        parser.error(f"发现 {len(missing)} 张缺少标注的训练影像，例如 {missing[0]}")
    if len(candidates) < 2:
        parser.error("至少需要两张有标注影像")
    items = [p.relative_to(args.root).as_posix() for p in candidates]
    random.Random(args.seed).shuffle(items)
    n_val = min(len(items) - 1, max(1, round(len(items) * args.val_fraction)))
    n_test = max(1, round(len(items) * args.test_fraction)) if args.test_fraction > 0 else 0
    if n_val + n_test >= len(items):
        parser.error("样本数不足，无法在所选 val/test 比例下保留非空训练集")
    manifests = {"train.json": sorted(items[n_val + n_test:]), "val.json": sorted(items[:n_val])}
    if n_test:
        manifests["test.json"] = sorted(items[n_val:n_val + n_test])
    inference = sorted(p.relative_to(args.root).as_posix() for split in ("valid", "test")
                       for p in (args.root / split).glob("*_sat.jpg"))
    manifests["unlabeled.json"] = inference
    manifests["protocol.json"] = {
        "seed": args.seed, "val_fraction": args.val_fraction,
        "source": "labeled train only", "train_count": len(items) - n_val - n_test,
        "val_count": n_val, "test_count": n_test, "test_fraction": args.test_fraction,
        "warning": "随机按影像划分不保证地理独立性；有地理分组时请替换为区域隔离划分。",
    }
    for name in manifests:
        if (args.output / name).exists():
            parser.error(f"拒绝覆盖已有划分：{args.output / name}，请指定新的 --output")
    args.output.mkdir(parents=True, exist_ok=True)
    for name, contents in manifests.items():
        (args.output / name).write_text(json.dumps(contents, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"划分已写入 {args.output}：train={len(items) - n_val - n_test}, "
          f"val={n_val}, test={n_test}, unlabeled={len(inference)}")


if __name__ == "__main__":
    main()
