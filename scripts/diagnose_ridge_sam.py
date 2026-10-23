"""Oracle 提示接口诊断：标注山脊场/二值场 -> 冻结 SAM -> 分割指标。

该脚本对应方案第 4.2/10 节的第一步：在扩散训练之前，先确认目标表示与
固定尺度映射能被冻结 SAM 有效解释。它比较：

* oracle ``ridge``（方案公式构造的有符号山脊场）与 ``binary``（道路 +1）提示；
* 多个固定提示尺度 ``U = clamp(scale * H)``；
* ``none`` / ``zero`` / ``shuffled`` / ``local`` 扰动下的输出敏感性。

模式
----
``--mock``（默认）不加载任何权重，用 CPU 合成 mask 与简单 decode 验证诊断
链路可运行；``--image/--mask`` 模式需要 SAM2 权重，用于真实接口实验。

示例::

    python scripts/diagnose_ridge_sam.py --mock --size 64

    python scripts/diagnose_ridge_sam.py --image DeepGlobe_Road/train/1_sat.jpg \
        --mask DeepGlobe_Road/train/1_mask.png \
        --config configs/3090.yaml --device cuda --scales 0.5 1 2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diffpeak.diagnostics import (  # noqa: E402
    build_oracle_prompt_fields,
    oracle_prompt_diagnostics,
)
from diffpeak.data.geometry import build_geometry, resize_and_pad  # noqa: E402
from diffpeak.data.ridge import build_ridge_field  # noqa: E402
from diffpeak.inference import (  # noqa: E402
    load_config,
    load_image_file,
    load_mask_file,
    prepare_image,
    resolve_prompt_mapping,
    resolve_sam_checkpoint,
)


def _mock_mask(size: int) -> np.ndarray:
    mask = np.zeros((size, size), dtype=bool)
    mid = size // 2
    mask[mid - 1 : mid + 2, 2 : size - 2] = True
    mask[2 : size - 2, mid - 1 : mid + 2] = True
    return mask


def _mock_decode(prompt: torch.Tensor) -> torch.Tensor:
    """CPU mock：logits = 6 * prompt - 0.5，响应随提示强度和结构变化。"""
    return prompt * 6.0 - 0.5


def _run_mock(size: int, scales: list[float], threshold: float) -> Dict[str, Any]:
    mask = _mock_mask(size)
    fields = build_oracle_prompt_fields(mask)
    result = oracle_prompt_diagnostics(
        fields,
        _mock_decode,
        mask,
        scales=scales,
        threshold=threshold,
    )
    return {
        "mode": "mock",
        "size": size,
        "target_positive_rate": float(mask.mean()),
        **result,
    }


def _run_real(args: argparse.Namespace) -> Dict[str, Any]:
    from diffpeak.models import RoadSAM

    config = load_config(args.config) if args.config else load_config(None)
    config_dir = Path(args.config).expanduser().parent if args.config else None
    prompt_scale, terminal_mapping = resolve_prompt_mapping(config)
    checkpoint = resolve_sam_checkpoint(config, config_dir)
    model_cfg = config.get("model", {})
    data_cfg = config.get("data", {})
    device = torch.device(
        args.device
        if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    # 诊断尺度由本脚本外部施加；模型内部 scale 固定为 1，避免重复缩放。
    sam = RoadSAM(
        str(checkpoint),
        config_name=str(
            model_cfg.get("sam_config", "configs/sam2.1/sam2.1_hiera_b+.yaml")
        ),
        architecture=str(
            model_cfg.get("sam_architecture", "sam2.1_hiera_base_plus")
        ),
        prompt_scale=1.0,
    ).to(device).eval()

    image = load_image_file(args.image)
    original_mask = load_mask_file(args.mask)
    if tuple(original_mask.shape) != tuple(image.shape[:2]):
        raise ValueError(
            "影像与 mask 尺寸不一致："
            f"{image.shape[:2]} vs {original_mask.shape}"
        )
    tensor, original_size, resized_size = prepare_image(image, image_size=1024)
    tensor = tensor.to(device)
    geometry = build_geometry(
        original_size[0], original_size[1], image_size=1024, heatmap_size=256
    )
    if geometry.valid_size != resized_size:
        raise ValueError(
            "prepare_image 与 build_geometry 的有效尺寸不一致："
            f"{resized_size} vs {geometry.valid_size}"
        )
    valid_sizes = [geometry.valid_size]
    with torch.no_grad():
        embeddings = sam.encode(tensor, valid_sizes=valid_sizes)

    # 与训练完全相同的目标构造：源分辨率构造 -> 热图分辨率池化，
    # padding 固定为 0。二值 oracle 使用同一 road_mask 覆盖定义。
    ridge = build_ridge_field(
        original_mask,
        geometry,
        data_cfg.get("ridge_config"),
    )
    binary = np.where(
        ridge.road_mask, 1.0, -1.0
    ).astype(np.float32)
    binary[~ridge.valid_mask] = 0.0
    fields = {"ridge": ridge.field, "binary": binary}

    # 评分只在有效图像区域进行；padding 不参与指标。
    target_canvas = resize_and_pad(
        original_mask, geometry, resample="nearest", fill=0
    ).astype(bool)
    valid_canvas = np.zeros(
        (geometry.image_size, geometry.image_size), dtype=bool
    )
    valid_canvas[
        : geometry.resized_height, : geometry.resized_width
    ] = True

    def decode(prompt: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return sam.decode_heatmaps(
                embeddings,
                prompt.to(device),
                output_size=(geometry.image_size, geometry.image_size),
                valid_sizes=valid_sizes,
            )

    result = oracle_prompt_diagnostics(
        fields,
        decode,
        target_canvas,
        scales=args.scales,
        threshold=args.threshold,
        prompt_valid_mask=ridge.valid_mask,
        valid_mask=valid_canvas,
    )
    return {
        "mode": "real",
        "image": str(args.image),
        "mask": str(args.mask),
        "checkpoint": str(checkpoint),
        "prompt_scale_in_config": prompt_scale,
        "terminal_mapping": terminal_mapping,
        "heatmap_size": 256,
        "valid_size": list(geometry.valid_size),
        **result,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mock", action="store_true", help="使用 CPU 合成数据与 mock decode（默认）。"
    )
    parser.add_argument("--size", type=int, default=64, help="mock 掩码边长。")
    parser.add_argument("--image", type=str, default=None, help="真实影像路径。")
    parser.add_argument("--mask", type=str, default=None, help="真实二值标注路径。")
    parser.add_argument("--config", type=str, default=None, help="YAML 配置。")
    parser.add_argument("--device", type=str, default=None, help="cpu 或 cuda。")
    parser.add_argument(
        "--scales",
        type=float,
        nargs="+",
        default=[0.5, 1.0, 2.0],
        help="固定提示尺度列表。",
    )
    parser.add_argument("--threshold", type=float, default=0.5, help="概率阈值。")
    parser.add_argument("--output", type=str, default=None, help="可选 JSON 输出路径。")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if any(scale <= 0 for scale in args.scales):
        parser.error("--scales 必须是正数")
    if args.image or args.mask:
        if not (args.image and args.mask):
            parser.error("真实模式必须同时提供 --image 与 --mask")
        result = _run_real(args)
    else:
        if args.size <= 0:
            parser.error("--size 必须是正整数")
        result = _run_mock(int(args.size), list(args.scales), args.threshold)

    text = json.dumps(result, ensure_ascii=False, indent=2, default=str)
    print(text)
    if args.output:
        output = Path(args.output).expanduser()
        if output.exists():
            raise FileExistsError(f"输出文件已存在，拒绝覆盖：{output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
