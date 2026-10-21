"""DiffPeak-SAM 有标注划分评估入口。

只接受带掩码的划分（通常为 ``splits/val.json`` 或 ``splits/train.json``）。
每张图独立推理：影像 -> 条件编码 -> 扩散采样 -> 峰值提取 -> SAM 解码，
之后才读取掩码计算指标。预测过程不接触掩码或骨架。

指标同时报告：
* 逐图宏平均 ``iou`` / ``f1`` / ``cldice`` / ``reward``；
* 累加 TP/FP/FN 后的 ``micro_iou`` / ``micro_f1``；
* 正负提示点数与空预测/空标注比例。

本脚本总是使用**生成提示**，因此应评估 B/C/D 阶段快照。A 阶段尚未训练
扩散生成器，其“真实提示”验证在 ``train.py`` 的验证循环内完成。

分辨率说明：``engine`` 验证在 1024² 画布上按缩放后的有效区域计算指标，本脚本
则先把预测恢复到原图尺寸再读取同尺寸标注。当前 DeepGlobe 影像均为 1024²，
两者一致；换用其他分辨率时不要直接比较这两套指标。

示例::

    python evaluate.py --checkpoint runs/diffpeak/best.pt --split val
    python evaluate.py --config configs/3090.yaml --checkpoint runs/diffpeak/best.pt \
        --split val --save-predictions --save-points
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from diffpeak.inference import (
    Predictor,
    aggregate_metrics,
    compute_image_metrics,
    config_from_checkpoint,
    load_config,
    load_labeled_split,
    load_mask_file,
    make_reward_config,
    read_checkpoint,
    resolve_data_root,
    resolve_output_dir,
    resolve_sam_checkpoint,
    resolve_split_file,
    save_mask_file,
    save_points_file,
    unique_output_path,
)


def _load_config_and_checkpoint(
    args: argparse.Namespace,
) -> Tuple[Dict[str, Any], Path, Optional[Path], Optional[Dict[str, Any]]]:
    """显式 --config 覆盖；否则默认使用 checkpoint 内保存的训练配置。"""
    if args.config is not None:
        config = load_config(args.config)
        config_dir = Path(args.config).expanduser().parent
        if args.checkpoint is not None:
            checkpoint = Path(args.checkpoint)
        else:
            checkpoint = _find_default_checkpoint(
                resolve_output_dir(config, config_dir)
            )
        return config, checkpoint, config_dir, None

    if args.checkpoint is not None:
        checkpoint = Path(args.checkpoint)
        payload = read_checkpoint(checkpoint)
        config = config_from_checkpoint(checkpoint, payload=payload)
        return config, checkpoint, None, payload

    bootstrap = load_config(None)
    checkpoint = _find_default_checkpoint(resolve_output_dir(bootstrap))
    payload = read_checkpoint(checkpoint)
    config = config_from_checkpoint(checkpoint, payload=payload)
    return config, checkpoint, None, payload


def _find_default_checkpoint(output_dir: Path) -> Path:
    for name in ("best.pt", "last.pt"):
        candidate = output_dir / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"{output_dir} 下找不到 best.pt/last.pt，请通过 --checkpoint 指定"
    )


def _resolve_eval_split(
    args: argparse.Namespace,
    config: Dict[str, Any],
    config_dir: Optional[Path],
    root: Path,
) -> Tuple[Path, str]:
    if args.split_file is not None:
        path = resolve_split_file(
            args.split_file, root, config_dir, must_exist=True
        )
        return path, Path(args.split_file).stem
    key = f"{args.split}_split"
    raw = config.get("data", {}).get(key)
    if not raw:
        raise ValueError(
            f"配置缺少 data.{key}；请用 --split-file 指定有标注划分"
        )
    path = resolve_split_file(raw, root, config_dir, must_exist=True)
    return path, str(args.split)


def _write_json(
    path: Path,
    payload: Any,
    *,
    overwrite: bool = False,
) -> None:
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"结果文件已存在，拒绝覆盖：{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _check_output_conflicts(output_dir: Path) -> None:
    """推理前检查旧结果，避免跑完整个评估后才在写文件时失败。"""
    if output_dir.exists() and not output_dir.is_dir():
        raise NotADirectoryError(f"评估输出路径不是目录：{output_dir}")
    conflicts = [
        output_dir / name
        for name in ("metrics.json", "per_image.json")
        if (output_dir / name).exists()
    ]
    if conflicts:
        raise FileExistsError(
            "输出目录已有旧评估结果，拒绝覆盖："
            + ", ".join(str(path) for path in conflicts)
            + "；请通过 --output 指定新目录（脚本不会自动覆盖历史指标）。"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default=None, help="显式 YAML；省略时读取 checkpoint 配置。")
    parser.add_argument("--checkpoint", type=str, default=None, help="训练快照；省略时尝试 output_dir/best.pt。")
    parser.add_argument("--split", type=str, default="val", help="train 或 val（读取 data.<split>_split）。")
    parser.add_argument("--split-file", type=str, default=None, help="显式有标注划分文件。")
    parser.add_argument("--output", type=str, default=None, help="评估输出目录。")
    parser.add_argument("--threshold", type=float, default=None, help="覆盖 inference.threshold。")
    parser.add_argument("--device", type=str, default=None, help="cuda、cuda:0 或 cpu。")
    parser.add_argument("--seed", type=int, default=None, help="显式覆盖评估种子。")
    parser.add_argument("--save-predictions", action="store_true", help="保存 0/255 预测掩码。")
    parser.add_argument("--save-points", action="store_true", help="保存提示点 JSON。")
    parser.add_argument("--limit", type=int, default=None, help="最多评估的影像数。")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit 必须是正整数")

    config, checkpoint, config_dir, preloaded = _load_config_and_checkpoint(args)
    if args.seed is not None:
        config["seed"] = int(args.seed)
    seed = int(config.get("seed", 42))

    root = resolve_data_root(config, config_dir)
    split_file, split_name = _resolve_eval_split(
        args, config, config_dir, root
    )
    entries = load_labeled_split(split_file, root)

    # 有标注评估必须完整：任何缺失都直接报错，不能静默跳过拉高指标。
    missing: List[str] = []
    for entry in entries:
        image_ok = Path(entry["image_path"]).is_file()
        mask = entry.get("mask_path")
        mask_ok = mask is not None and Path(mask).is_file()
        if not image_ok or not mask_ok:
            missing.append(str(entry.get("relative_path", entry.get("id"))))
    if missing:
        preview = "\n  ".join(missing[:8])
        raise FileNotFoundError(
            f"划分中有 {len(missing)} 个样本缺少影像或掩码：\n  {preview}"
            + ("\n  ……" if len(missing) > 8 else "")
            + "\n官方 valid/test 无掩码，不能用于有标注评估；"
            "请使用 train 划分出的 val.json 或 --split-file。"
        )

    split_total = len(entries)
    usable = list(entries)
    # 稳定排序，且逐图处理；指标与会话顺序无关。
    usable.sort(key=lambda item: (str(item["id"]), str(item["relative_path"])))
    if args.limit is not None:
        usable = usable[: int(args.limit)]
    if not usable:
        raise ValueError("--limit 截取后没有可评估样本")
    subset = len(usable) < split_total

    output_dir = (
        Path(args.output).expanduser()
        if args.output is not None
        else resolve_output_dir(config, config_dir) / f"eval_{split_name}"
    )
    # 在构建模型/开始推理之前检查旧结果，避免长评估结束后才写失败。
    _check_output_conflicts(output_dir)

    predictor = Predictor(
        config,
        checkpoint=checkpoint,
        device=args.device,
        config_dir=config_dir,
        threshold=args.threshold,
        seed=seed,
        payload=preloaded,
    )
    preloaded = None
    # 提前校验 SAM 权重路径。
    resolve_sam_checkpoint(config, config_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    produced = set()
    save_points = bool(args.save_points)
    reward_config = make_reward_config(config)
    threshold = predictor.threshold

    rows: List[Dict[str, Any]] = []
    point_counts: List[float] = []
    empty_predictions = 0
    empty_targets = 0

    for index, entry in enumerate(usable, start=1):
        image_path = Path(entry["image_path"])
        # 先推理，后读取掩码，确保生成提示不接触 GT。
        prediction = predictor.predict_file(
            image_path, image_id=entry["id"], seed=seed
        )
        target = load_mask_file(Path(entry["mask_path"]))
        prediction_bool = np.asarray(prediction["mask"]) > 0

        metrics = compute_image_metrics(
            prediction_bool,
            target,
            prediction["point_count"],
            reward_config,
        )
        row: Dict[str, Any] = {
            "id": str(entry["id"]),
            "relative_path": str(entry["relative_path"]),
            **metrics,
            "point_count": int(prediction["point_count"]),
            "positive_count": int(prediction["positive_count"]),
            "negative_count": int(prediction["negative_count"]),
            "original_size": list(prediction["original_size"]),
            "resized_size": list(prediction["resized_size"]),
        }
        rows.append(row)
        point_counts.append(float(prediction["point_count"]))
        if not prediction_bool.any():
            empty_predictions += 1
        if not target.any():
            empty_targets += 1

        if args.save_predictions or save_points:
            mask_path = unique_output_path(
                output_dir,
                entry["relative_path"],
                suffix=".png",
                produced=produced,
                source_path=image_path,
            )
            if args.save_predictions:
                save_mask_file(prediction["mask"], mask_path)
            if save_points:
                points_payload = {
                    "image_id": row["id"],
                    "source_path": str(image_path),
                    "original_size": row["original_size"],
                    "resized_size": row["resized_size"],
                    "threshold": threshold,
                    **prediction["points"],
                }
                save_points_file(
                    points_payload, mask_path.with_suffix(".points.json")
                )

        if index % 50 == 0 or index == len(usable):
            print(f"已评估 {index}/{len(usable)}", flush=True)

    summary = aggregate_metrics(rows)
    summary.update(
        {
            "split": split_name,
            "split_file": str(split_file),
            "checkpoint": str(checkpoint),
            "seed": seed,
            "threshold": float(threshold),
            "count": len(rows),
            "split_total": split_total,
            "subset": bool(subset),
            "mean_point_count": float(
                sum(point_counts) / max(len(point_counts), 1)
            ),
            "empty_prediction_rate": float(empty_predictions) / float(len(rows)),
            "empty_target_rate": float(empty_targets) / float(len(rows)),
        }
    )

    _write_json(output_dir / "metrics.json", summary)
    _write_json(output_dir / "per_image.json", rows)

    print(
        "评估完成：\n"
        f"  images={summary['count']}/{summary['split_total']}  "
        f"subset={summary['subset']}  "
        f"macro_iou={summary['iou']:.4f}  "
        f"macro_f1={summary['f1']:.4f}  clDice={summary['cldice']:.4f}\n"
        f"  micro_iou={summary['micro_iou']:.4f}  "
        f"micro_f1={summary['micro_f1']:.4f}  "
        f"reward={summary['reward']:.4f}  "
        f"points={summary['mean_point_count']:.2f}\n"
        f"  结果目录：{output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
