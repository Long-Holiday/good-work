"""DiffPeak-SAM 无标注推理入口（signed_ridge_v1）。

示例（在项目根目录执行）::

    # 单张影像；省略 --config 时使用 checkpoint 内保存的训练配置
    python predict.py --checkpoint runs/diffpeak/best.pt \
        --input DeepGlobe_Road/valid/100794_sat.jpg --output predictions/100794.png

    # 目录推理，保留相对路径与文件名，避免覆盖源文件
    python predict.py --checkpoint runs/diffpeak/best.pt \
        --input DeepGlobe_Road/valid --output predictions/valid --save-field

预测流程固定为：等比 resize + 右下 pad 到 1024 -> 条件编码 -> 20 步扩散采样
单通道有符号道路山脊场 -> 固定尺度/终端映射 -> 冻结 SAM decode_heatmaps ->
裁 padding -> 恢复原图尺寸。不再提取峰值点。

旧双通道 checkpoint 不能直接用于推理（协议严格拒绝）；如需只迁移形状兼容
权重，显式使用 ``--migrate-weights``（不恢复优化器状态）。

分辨率说明：``engine`` 验证在 1024² 画布上按缩放后有效区域计算指标，而本脚本
把概率图恢复到原图尺寸后再保存/评估。当前 DeepGlobe 影像均为 1024²，两者等价；
换用其他分辨率时两类指标不应直接互相比较。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from diffpeak.inference import (
    IMAGE_EXTENSIONS,
    Predictor,
    config_from_checkpoint,
    load_config,
    read_checkpoint,
    resolve_output_dir,
    resolve_sam_checkpoint,
    save_mask_file,
    unique_output_path,
)


def _load_config_and_checkpoint(
    args: argparse.Namespace,
) -> Tuple[Dict[str, Any], Path, Optional[Path], Optional[Dict[str, Any]]]:
    """显式 --config 覆盖；否则默认读取 checkpoint 内保存的配置。

    返回 ``(config, checkpoint, config_dir, preloaded_payload)``。当配置来自
    checkpoint 时同时返回已加载快照，避免 Predictor 再次读取整份权重。
    """
    if args.config is not None:
        config = load_config(args.config)
        config_dir = Path(args.config).expanduser().parent
        if args.checkpoint is not None:
            checkpoint = Path(args.checkpoint)
        else:
            output_dir = resolve_output_dir(config, config_dir)
            checkpoint = _find_default_checkpoint(output_dir)
        return config, checkpoint, config_dir, None

    if args.checkpoint is not None:
        checkpoint = Path(args.checkpoint)
        payload = read_checkpoint(checkpoint)
        config = config_from_checkpoint(checkpoint, payload=payload)
        return config, checkpoint, None, payload

    # 既没有 --config 也没有 --checkpoint：只用默认配置定位输出目录，
    # 最终配置仍以找到的 checkpoint 内配置为准。
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


def _save_result(
    result: Dict[str, Any],
    mask_path: Path,
    source_path: Path,
    save_field: bool,
) -> None:
    save_mask_file(result["mask"], mask_path)
    if save_field:
        if "field" not in result:
            raise ValueError("--save-field 需要 Predictor 返回有符号场")
        field_path = mask_path.with_suffix(".field.npy")
        if field_path.exists():
            raise FileExistsError(f"输出文件已存在，拒绝覆盖：{field_path}")
        np.save(field_path, np.asarray(result["field"], dtype=np.float32))


def _predict_single(
    predictor: Predictor,
    input_path: Path,
    output: Optional[Path],
    save_field: bool,
) -> Path:
    if output is None:
        mask_path = resolve_output_dir(
            predictor.config, predictor.config_dir
        ) / "predictions" / (input_path.stem + ".png")
    elif output.exists() and output.is_dir():
        mask_path = output / (input_path.stem + ".png")
    elif output.suffix.lower() in IMAGE_EXTENSIONS:
        mask_path = output.with_suffix(".png")
    else:
        mask_path = output / (input_path.stem + ".png")

    mask_path = unique_output_path(
        mask_path.parent,
        mask_path.name,
        suffix=".png",
        source_path=input_path,
    )
    result = predictor.predict_file(input_path, return_field=save_field)
    _save_result(result, mask_path, input_path, save_field)
    probability = np.asarray(result["probability"])
    print(
        f"{input_path} -> {mask_path} "
        f"(positive_pixels={int((probability >= result['threshold']).sum())}, "
        f"field_scale={result['prompt_scale']})",
        flush=True,
    )
    return mask_path


def _predict_directory(
    predictor: Predictor,
    input_dir: Path,
    output: Optional[Path],
    save_field: bool,
    limit: Optional[int],
) -> int:
    if output is None:
        output_dir = resolve_output_dir(
            predictor.config, predictor.config_dir
        ) / "predictions" / input_dir.name
    else:
        output_dir = output
    output_dir = output_dir.resolve()
    if output_dir == input_dir.resolve():
        raise ValueError("输出目录不能与输入目录相同，避免覆盖源影像")
    output_dir.mkdir(parents=True, exist_ok=True)

    images: List[Path] = []
    for path in sorted(input_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        # 跳过 DeepGlobe 的标注掩码，避免把 * _mask.png 当成影像预测。
        if path.stem.lower().endswith("_mask"):
            continue
        if output_dir in path.resolve().parents:
            continue
        images.append(path)
    if limit is not None:
        images = images[: int(limit)]
    if not images:
        raise ValueError(f"输入目录中没有可预测影像：{input_dir}")

    produced = set()
    failures = 0
    for index, image_path in enumerate(images, start=1):
        relative = image_path.relative_to(input_dir)
        mask_path = unique_output_path(
            output_dir,
            relative,
            suffix=".png",
            produced=produced,
            source_path=image_path,
        )
        try:
            # 用相对路径作为 image_id；canonical_image_id 会剥离目录与 _sat，
            # 与验证/评估对同一张图的扩散种子保持一致。
            result = predictor.predict_file(
                image_path, image_id=str(relative), return_field=save_field
            )
            _save_result(result, mask_path, image_path, save_field)
        except Exception as exc:  # 单图失败不终止整个目录
            failures += 1
            print(f"[失败] {image_path}: {exc}", file=sys.stderr, flush=True)
        if index % 50 == 0 or index == len(images):
            print(f"已处理 {index}/{len(images)}，失败 {failures}", flush=True)

    print(f"目录推理完成：{len(images) - failures} 成功，{failures} 失败", flush=True)
    if failures and failures == len(images):
        raise SystemExit(1)
    return failures


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default=None, help="显式 YAML；省略时读取 checkpoint 配置。")
    parser.add_argument("--checkpoint", type=str, default=None, help="训练快照；省略时尝试 output_dir/best.pt。")
    parser.add_argument("--input", type=str, required=True, help="影像文件或目录。")
    parser.add_argument("--output", type=str, default=None, help="输出文件或目录。")
    parser.add_argument("--threshold", type=float, default=None, help="覆盖 inference.threshold。")
    parser.add_argument("--device", type=str, default=None, help="cuda、cuda:0 或 cpu。")
    parser.add_argument("--seed", type=int, default=None, help="显式覆盖评估种子。")
    parser.add_argument(
        "--save-field",
        action="store_true",
        help="同时保存单通道有符号场（.field.npy）。",
    )
    parser.add_argument(
        "--migrate-weights",
        action="store_true",
        help="显式迁移旧快照的形状兼容权重（跳过 SAM/输入输出头，不恢复优化器）。",
    )
    parser.add_argument("--limit", type=int, default=None, help="目录模式下最多处理的影像数。")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit 必须是正整数")

    config, checkpoint, config_dir, preloaded = _load_config_and_checkpoint(args)
    if args.seed is not None:
        config["seed"] = int(args.seed)

    predictor = Predictor(
        config,
        checkpoint=checkpoint,
        device=args.device,
        config_dir=config_dir,
        threshold=args.threshold,
        payload=preloaded,
        migrate_weights=bool(args.migrate_weights),
    )
    preloaded = None
    save_field = bool(args.save_field)

    input_path = Path(args.input).expanduser()
    if not input_path.exists():
        parser.error(f"输入不存在：{input_path}")
    output = Path(args.output).expanduser() if args.output is not None else None

    # 提前校验 SAM 权重路径，给出比模型初始化更清晰的错误。
    resolve_sam_checkpoint(config, config_dir)

    if input_path.is_file():
        _predict_single(predictor, input_path, output, save_field)
    elif input_path.is_dir():
        _predict_directory(
            predictor,
            input_path,
            output,
            save_field,
            args.limit,
        )
    else:
        parser.error(f"输入既不是文件也不是目录：{input_path}")


if __name__ == "__main__":
    main()
