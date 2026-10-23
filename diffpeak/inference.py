"""DiffPeak-SAM 的推理、配置与几何工具（signed_ridge_v1）。

推理主链路只使用单通道有符号道路山脊场：

``影像 -> 条件编码 -> 扩散采样 H -> 固定尺度/终端映射 -> 冻结 SAM decode_heatmaps``

本协议不再提取离散峰值点。旧的 ``PeakExtractor`` / 点提示工具仅为兼容历史
纯函数而保留，不参与 ``Predictor``、``predict.py`` 或 ``evaluate.py``。

坐标与填充约定
--------------
* 图像按 ``scale = min(1024 / H, 1024 / W)`` 等比缩放，放在左上角，右下用
  零填充到 ``1024 x 1024``；
* ``resized_size = (h, w)`` 是填充前的有效尺寸，作为 ``valid_sizes`` 同时传给
  ``RoadSAM.encode`` 与 ``RoadSAM.decode_heatmaps``。

确定性
------
推理链路中的随机性只来自扩散采样。每张图的生成器种子由
``stable_seed(base_seed, image_id)`` 计算，与 batch 大小、目录遍历顺序和
DataLoader 顺序无关。

分辨率说明
----------
训练/``engine`` 验证在 ``1024 x 1024`` 填充画布上按 ``valid_sizes``（缩放后
尺寸）裁剪预测与标注后计算指标；``predict.py`` / ``evaluate.py`` 则裁掉
padding 后把概率图双线性恢复到**原图尺寸**，并在原图空间保存掩码、计算指标。
当前 DeepGlobe_Road 影像全部是 ``1024 x 1024``，两者完全等价；若换用其他
分辨率，训练验证与推理/评估的像素空间不同，指标不应直接互相比较。
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import random
import warnings
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from diffpeak.models import ConditionEncoder, ConditionalDiT, GaussianDiffusion, RoadSAM
from diffpeak.models.sam import (
    DEFAULT_SAM2_ARCHITECTURE,
    DEFAULT_SAM2_CONFIG,
    SAM2ImageFeatures,
    low_res_features,
)
from diffpeak.prompts import PeakExtractor

__all__ = [
    "CHECKPOINT_FORMAT_VERSION",
    "CheckpointProtocolError",
    "DEFAULT_CONFIG",
    "JOINT_ABC_PROTOCOL",
    "PROMPT_TERMINAL_MAPPINGS",
    "Predictor",
    "SIGNED_RIDGE_PROTOCOL",
    "TARGET_REPRESENTATION",
    "aggregate_metrics",
    "apply_terminal_mapping",
    "build_models",
    "canonical_image_id",
    "compute_image_metrics",
    "compute_resized_size",
    "config_from_checkpoint",
    "decode_with_prompts",
    "encode_images",
    "evaluation_seed",
    "find_config_path",
    "find_existing_path",
    "get_points",
    "load_checkpoint_models",
    "load_compatible_weights",
    "load_config",
    "load_image_file",
    "load_labeled_split",
    "load_mask_file",
    "make_generator",
    "make_reward_config",
    "map_points_to_original",
    "normalize_valid_sizes",
    "points_to_prompt",
    "prepare_image",
    "prepare_image_file",
    "read_checkpoint",
    "resolve_data_root",
    "resolve_output_dir",
    "resolve_prompt_mapping",
    "resolve_sam_checkpoint",
    "resolve_split_file",
    "save_mask_file",
    "save_points_file",
    "seed_everything",
    "stable_seed",
    "unique_output_path",
    "uses_joint_abc_protocol",
    "uses_signed_ridge_protocol",
    "validate_checkpoint_protocol",
    "validate_signed_ridge_config",
    "warn_protocol_differences",
]

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}

# 唯一训练/推理协议。旧 ``joint_abc_v1``（双通道点提示热图 + 可训练 SAM）
# 与新协议不兼容；``JOINT_ABC_PROTOCOL`` 仅作为导入兼容别名保留。
SIGNED_RIDGE_PROTOCOL = "signed_ridge_v1"
JOINT_ABC_PROTOCOL = SIGNED_RIDGE_PROTOCOL

# 目标表示版本：单通道有符号道路山脊场。
TARGET_REPRESENTATION = "signed_ridge_v1"

# 新快照格式版本；旧快照（v4 及更早）不能 --resume。
CHECKPOINT_FORMAT_VERSION = 5

# 终端映射：第一版只允许 ``clamp``。扩散采样、RoadSAM 内部 prompt_scale 与
# 推理都必须使用同一映射；``identity`` 会引入训练/推理不一致，暂不支持。
PROMPT_TERMINAL_MAPPINGS: Tuple[str, ...] = ("clamp",)


class CheckpointProtocolError(ValueError):
    """checkpoint 协议/元数据与 signed_ridge_v1 不兼容。"""


# 根配置的完整默认值。YAML 中缺少的可选项会自动补齐。
DEFAULT_CONFIG: Dict[str, Any] = {
    "seed": 42,
    # 唯一协议标记；推理、验证与快照读取都使用该字段。
    "training_protocol": SIGNED_RIDGE_PROTOCOL,
    "target_representation": TARGET_REPRESENTATION,
    # SAM 图像编码器、提示编码器与掩码解码器全阶段冻结。
    "sam_frozen": True,
    "output_dir": "runs/diffpeak",
    "logging": {"tensorboard": True, "log_every": 10, "flush_secs": 30},
    "data": {
        "root": "DeepGlobe_Road",
        "train_split": "splits/train.json",
        "val_split": "splits/val.json",
        "num_workers": 4,
        "augment": False,
        # 山脊场构造参数；改变它们会改变监督目标，因此 checkpoint
        # 协议比较包含该字段（旧快照不能 --resume）。
        "ridge_config": {
            "center_radius": 1.0,
            "boundary_band": 2.0,
            "epsilon": 1e-6,
        },
    },
    "model": {
        "sam_checkpoint": "checkpoints/sam2.1_hiera_base_plus.pt",
        "sam_config": DEFAULT_SAM2_CONFIG,
        "sam_architecture": DEFAULT_SAM2_ARCHITECTURE,
        "sam_apply_postprocessing": False,
        # SAM 图像编码器、提示编码器与掩码解码器全阶段冻结。
        "sam_frozen": True,
        # 默认去噪器：ConditionalDiT（单通道有符号场上的 patch Transformer）。
        "dit": {
            "in_channels": 1,
            "out_channels": 1,
            "patch_size": 8,
            "hidden_size": 256,
            "depth": 6,
            "num_heads": 8,
            "mlp_ratio": 4.0,
            # 与 ConditionEncoder 的四尺度输出一致：256/128/64/32。
            "condition_channels": (64, 128, 256, 256),
            "gradient_checkpointing": True,
        },
    },
    "diffusion": {
        "timesteps": 1000,
        "sampling_steps": 20,
        "min_std": 0.02,
        # 显式写出 DDIM/损失调度参数，避免快照记录与实际构造不一致。
        "eta": 0.0,
        "beta_start": 1e-4,
        "beta_end": 0.02,
        "heat_loss_max_timestep": None,
        # 单通道有符号场；与 ConditionalDiT 输入/输出通道一致。
        "latent_channels": 1,
    },
    # 有符号场 H -> SAM 密集提示 U 的固定映射：U = scale * clamp(H)。
    # scale 在 clamp 之后施加，允许 s > 1 时保留 >1 的提示幅度。
    "prompt": {"scale": 1.0, "terminal_mapping": "clamp"},
    "train": {
        "batch_size": 1,
        "accumulation_steps": 4,
        "amp": True,
        "weight_decay": 0.01,
        "grad_clip": 1.0,
        # engine 仍以 heat_weight 作为场重建损失权重；不要再加重复别名，
        # 否则会被静默忽略并造成配置歧义。
        "heat_weight": 0.1,
        "aux_weight": 0.2,
        "save_every": 1,
        # 线性 warmup + cosine 退火；显式 warmup_steps 优先于 warmup_ratio。
        "warmup_steps": 0,
        "warmup_ratio": 0.05,
        "min_lr_ratio": 0.05,
        "log_every": 10,
        "val_every": 1,
        "val_limit": None,
        "deterministic": False,
    },
    "stages": {
        # ABC：条件编码器与 DiT 训练，SAM 全阶段冻结。
        "ABC": {
            "epochs": 60,
            "lr": 0.0001,
            "batch_size": 1,
            "accumulation_steps": 4,
        },
        # D 只训练扩散 DiT；更小的学习率见 README。
        "D": {
            "epochs": 10,
            "lr": 0.0000003,
            "batch_size": 1,
        },
    },
    "rl": {
        # 当前只实现 GRPO；显式算法名便于拒绝旧 DDPO 配置。
        "algorithm": "grpo",
        "group_size": 4,
        # 沿用 ppo_clip 键名以兼容旧配置；语义是 GRPO 的裁剪半径 epsilon。
        "ppo_clip": 0.0001,
        "kl_weight": 0.01,
        # 可选监督锚定（目标山脊场上的扩散 MSE）；0 表示消融关闭。
        "anchor_weight": 0.01,
        "update_epochs": 1,
        # 组内奖励标准差下限，只用于优势分母 clamp；
        # 非零低方差组仍参与学习，奖励完全相同的组优势为零。
        "advantage_std_delta": 0.0001,
        # 高维联合 log-prob 的策略梯度较大，D 使用更保守的 AMP 初始 scale。
        "amp_init_scale": 64.0,
    },
    "reward": {
        "iou_weight": 0.6,
        "cldice_weight": 0.4,
        "path_weight": 0.0,
        # 密集有符号场提示不消耗离散点。
        "point_cost": 0.0,
        "path_samples": 32,
        "path_match_radius": 8,
    },
    "inference": {"threshold": 0.5, "save_points": False},
}


# ---------------------------------------------------------------------------
# 配置、路径与随机种子
# ---------------------------------------------------------------------------
def _deep_update(base: Dict[str, Any], update: Mapping[str, Any]) -> Dict[str, Any]:
    """递归合并配置；``update`` 中的值优先。"""
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), Mapping):
            base[key] = _deep_update(dict(base[key]), value)
        else:
            base[key] = value
    return base


def find_config_path(path: Optional[Union[str, Path]] = None) -> Path:
    """定位 YAML 配置文件。

    未显式指定时依次尝试 ``configs/3090.yaml``、``config.yaml`` 等常见位置。
    """
    if path is not None:
        candidate = Path(path).expanduser()
        if not candidate.is_file():
            raise FileNotFoundError(f"配置文件不存在：{candidate}")
        return candidate
    for name in (
        "configs/3090.yaml",
        "config.yaml",
        "config.yml",
        "configs/config.yaml",
        "configs/diffpeak.yaml",
    ):
        candidate = Path(name)
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "找不到配置文件，请通过 --config 指定，例如 configs/3090.yaml"
    )


def load_config(path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """读取 YAML 并补齐默认项。"""
    config_path = find_config_path(path)
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - 依赖安装提示
        raise ImportError("读取 YAML 需要 PyYAML，请先安装 requirements.txt") from exc
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, Mapping):
        raise ValueError(f"配置文件顶层必须是映射：{config_path}")
    result = _deep_update(json.loads(json.dumps(DEFAULT_CONFIG)), loaded)
    # 历史四阶段 YAML 不能被新增默认 marker 静默改成密集提示模型。
    if "training_protocol" not in loaded and any(
        stage in loaded.get("stages", {}) for stage in ("A", "B", "C")
    ):
        result.pop("training_protocol", None)
    return result


def _normalise_protocol(value: Any) -> Optional[str]:
    """把协议字符串规范为小写并去空白；非字符串返回 ``None``。"""
    if not isinstance(value, str):
        return None
    text = value.strip().lower()
    return text or None


def uses_signed_ridge_protocol(config: Mapping[str, Any]) -> bool:
    """判断配置是否声明 ``signed_ridge_v1`` 单通道有符号场协议。

    标记位于配置根部而非 ``train`` 下，训练快照、推理 YAML 与评测入口
    读取的是同一个字段。缺省或旧协议返回 ``False``。
    """
    return _normalise_protocol(config.get("training_protocol")) == SIGNED_RIDGE_PROTOCOL


def uses_joint_abc_protocol(config: Mapping[str, Any]) -> bool:
    """历史名称兼容：现在等价于 :func:`uses_signed_ridge_protocol`。"""
    return uses_signed_ridge_protocol(config)


def _positive_finite(value: Any, name: str) -> float:
    """校验并返回正有限浮点数。"""
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} 必须是正有限数，实际为 {value!r}") from exc
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} 必须是正有限数，实际为 {value!r}")
    return number


def _normalise_terminal_mapping(value: Any) -> str:
    """把终端映射配置规范为 ``clamp``（第一版唯一支持的值）。"""
    if isinstance(value, Mapping):
        value = value.get("mode", value.get("type", "clamp"))
    text = _normalise_protocol(value)
    if text is None:
        raise ValueError("prompt.terminal_mapping 必须是非空字符串")
    if text not in PROMPT_TERMINAL_MAPPINGS:
        raise ValueError(
            "prompt.terminal_mapping 第一版只支持 "
            + "、".join(PROMPT_TERMINAL_MAPPINGS)
            + f"，实际为 {value!r}"
        )
    return text


def resolve_prompt_mapping(config: Mapping[str, Any]) -> Tuple[float, str]:
    """解析固定提示映射 ``U = scale * clamp(H)``。

    Returns:
        ``(scale, terminal_mapping)``。``scale`` 必须是有限正数；
        ``terminal_mapping`` 第一版只接受 ``clamp``。
    """
    raw = config.get("prompt", DEFAULT_CONFIG["prompt"])
    if not isinstance(raw, Mapping):
        raise ValueError("prompt 配置必须是映射")
    scale = _positive_finite(raw.get("scale", 1.0), "prompt.scale")
    mapping = _normalise_terminal_mapping(
        raw.get("terminal_mapping", "clamp")
    )
    return scale, mapping


def apply_terminal_mapping(
    field: torch.Tensor,
    terminal_mapping: str = "clamp",
) -> torch.Tensor:
    """对场应用终端映射；第一版唯一支持 ``clamp`` 到 ``[-1, 1]``。

    注意映射顺序固定为 ``U = scale * clamp(H)``，不是 ``clamp(scale * H)``：
    ``s > 1`` 时必须保留超过 1 的提示幅度（例如 ``H=0.8, s=2`` 得到
    ``U=1.6``，而不是 1.0）。预测器先调用本函数裁剪，再乘以固定尺度；
    非 ``clamp`` 配置会被显式拒绝。
    """
    if not torch.is_tensor(field):
        raise TypeError(f"field 必须是张量，实际为 {type(field)!r}")
    _normalise_terminal_mapping(terminal_mapping)
    return field.clamp(-1.0, 1.0)


def validate_signed_ridge_config(
    config: Mapping[str, Any],
    *,
    require_protocol: bool = True,
) -> Dict[str, Any]:
    """校验 signed_ridge_v1 配置中的协议、通道、冻结与提示映射。

    Args:
        config: 根配置。
        require_protocol: 为 ``True``（推理/新训练）时要求
            ``training_protocol == signed_ridge_v1``。显式迁移旧 checkpoints
            时可设为 ``False``，但仍会校验任何显式出现的通道/冻结字段。

    Returns:
        规范化后的关键字段字典。

    Raises:
        ValueError: 协议、目标表示、通道、冻结标记或提示映射不兼容。
    """
    errors: List[str] = []
    protocol = _normalise_protocol(config.get("training_protocol"))
    if require_protocol and protocol != SIGNED_RIDGE_PROTOCOL:
        errors.append(
            f"training_protocol={config.get('training_protocol')!r}，"
            f"要求 {SIGNED_RIDGE_PROTOCOL!r}"
        )
    representation = _normalise_protocol(config.get("target_representation"))
    if representation is not None and representation != TARGET_REPRESENTATION:
        errors.append(
            f"target_representation={config.get('target_representation')!r}，"
            f"要求 {TARGET_REPRESENTATION!r}"
        )

    model_cfg = config.get("model", {})
    model_cfg = model_cfg if isinstance(model_cfg, Mapping) else {}
    # YAML 把 sam_frozen 写在根级；兼容旧布局的 model.sam_frozen。
    sam_frozen = _first_value(
        config.get("sam_frozen"), model_cfg.get("sam_frozen")
    )
    if sam_frozen is not _MISSING_PROTOCOL_VALUE and sam_frozen is not True:
        errors.append("sam_frozen 必须为 true（SAM 全阶段冻结）")

    dit_cfg = model_cfg.get("dit", {})
    dit_cfg = dit_cfg if isinstance(dit_cfg, Mapping) else {}
    for key in ("in_channels", "out_channels"):
        if key in dit_cfg and int(dit_cfg[key]) != 1:
            errors.append(f"model.dit.{key}={dit_cfg[key]!r}，单通道协议要求 1")

    diffusion_cfg = config.get("diffusion", {})
    diffusion_cfg = diffusion_cfg if isinstance(diffusion_cfg, Mapping) else {}
    if "latent_channels" in diffusion_cfg and int(diffusion_cfg["latent_channels"]) != 1:
        errors.append(
            f"diffusion.latent_channels={diffusion_cfg['latent_channels']!r}，"
            "单通道协议要求 1"
        )

    try:
        scale, mapping = resolve_prompt_mapping(config)
    except ValueError as exc:
        errors.append(str(exc))
        scale, mapping = float("nan"), ""
    if errors:
        raise ValueError("signed_ridge_v1 配置不兼容：" + "；".join(errors))
    return {
        "training_protocol": SIGNED_RIDGE_PROTOCOL,
        "target_representation": TARGET_REPRESENTATION,
        "latent_channels": 1,
        "sam_frozen": True,
        "prompt_scale": scale,
        "terminal_mapping": mapping,
    }


def find_existing_path(
    value: Union[str, Path],
    base_dirs: Optional[Sequence[Optional[Union[str, Path]]]] = None,
    *,
    must_exist: bool = False,
) -> Path:
    """在若干基准目录下解析相对路径；优先返回第一个真实存在的候选。"""
    expanded = os.path.expanduser(str(value))
    path = Path(expanded)
    if path.is_absolute():
        if must_exist and not path.exists():
            raise FileNotFoundError(f"路径不存在：{path}")
        return path

    candidates: List[Path] = []
    seen = set()
    for base in list(base_dirs or []) + [Path.cwd()]:
        if base is None:
            continue
        candidate = Path(base) / path
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        candidates.append(candidate)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    if must_exist:
        searched = "\n  ".join(str(item) for item in candidates)
        raise FileNotFoundError(f"找不到 {value}，已尝试：\n  {searched}")
    return candidates[0] if candidates else path


def _config_base_dirs(config_dir: Optional[Union[str, Path]]) -> List[Path]:
    if config_dir is None:
        return [Path.cwd()]
    directory = Path(config_dir).expanduser().resolve()
    return [directory, directory.parent, Path.cwd()]


def resolve_data_root(
    config: Mapping[str, Any],
    config_dir: Optional[Union[str, Path]] = None,
) -> Path:
    """解析 ``data.root``。"""
    raw = config.get("data", {}).get("root", DEFAULT_CONFIG["data"]["root"])
    return find_existing_path(raw, _config_base_dirs(config_dir))


def resolve_sam_checkpoint(
    config: Mapping[str, Any],
    config_dir: Optional[Union[str, Path]] = None,
) -> Path:
    """解析并校验 SAM 权重路径。"""
    raw = config.get("model", {}).get(
        "sam_checkpoint", DEFAULT_CONFIG["model"]["sam_checkpoint"]
    )
    return find_existing_path(raw, _config_base_dirs(config_dir), must_exist=True)


def resolve_split_file(
    value: Union[str, Path],
    root: Optional[Union[str, Path]] = None,
    config_dir: Optional[Union[str, Path]] = None,
    *,
    must_exist: bool = True,
) -> Path:
    """解析划分文件（默认相对项目根目录，其次数据根目录）。"""
    bases: List[Optional[Union[str, Path]]] = []
    if config_dir is not None:
        directory = Path(config_dir).expanduser().resolve()
        bases.extend([directory, directory.parent])
    if root is not None:
        root_path = Path(root)
        bases.extend([root_path, root_path.parent])
    bases.append(Path.cwd())
    return find_existing_path(value, bases, must_exist=must_exist)


def _config_output_base(config_dir: Optional[Union[str, Path]]) -> Path:
    """输出目录的相对路径基准。

    统一以项目根目录为基准：``configs/3090.yaml`` 的配置目录 basename 是
    ``configs``，因此取其父目录；其他配置目录则直接使用配置所在目录。
    未提供 ``config_dir`` 时仍按当前工作目录解释。
    """
    if config_dir is None:
        return Path.cwd()
    directory = Path(config_dir).expanduser().resolve()
    if directory.name == "configs":
        return directory.parent
    return directory


def resolve_output_dir(
    config: Mapping[str, Any],
    config_dir: Optional[Union[str, Path]] = None,
) -> Path:
    """解析输出目录。

    相对路径以配置所在位置为基准：``configs/*.yaml`` 视为项目根的配置，
    输出相对于项目根；其他目录中的配置则相对于该配置目录；无 ``config_dir``
    时相对于当前工作目录。绝对路径原样返回。
    """
    raw = config.get("output_dir", DEFAULT_CONFIG["output_dir"])
    path = Path(os.path.expanduser(str(raw)))
    if path.is_absolute():
        return path
    return _config_output_base(config_dir) / path


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """设置 Python / NumPy / PyTorch 随机种子。"""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    if deterministic:
        # 显式要求 cuDNN 选择确定的卷积算法；与 use_deterministic_algorithms
        # 一起使用，避免依赖全局默认值。
        torch.backends.cudnn.deterministic = True
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass


def stable_seed(base_seed: int, *parts: Any) -> int:
    """由基础种子和任意标识计算稳定的 63 位种子（与 Python hash 无关）。"""
    digest = hashlib.blake2b(digest_size=8)
    digest.update(str(int(base_seed)).encode("utf-8"))
    for part in parts:
        digest.update(b"\x00")
        digest.update(str(part).encode("utf-8"))
    return int.from_bytes(digest.digest(), "little") & ((1 << 63) - 1)


def canonical_image_id(image_id: Any) -> str:
    """把路径/文件名规范为数据集的样本 id。

    ``"train/100034_sat.jpg"``、``"100034_sat"`` 与 ``"100034"`` 都会得到
    ``"100034"``。这样训练验证、``evaluate.py`` 与 ``predict.py`` 对同一张
    影像使用完全相同的扩散噪声种子。
    """
    if image_id is None:
        return ""
    text = str(image_id).replace("\\", "/")
    name = text.rsplit("/", 1)[-1]
    stem = Path(name).stem
    if stem.lower().endswith("_sat"):
        stem = stem[:-4]
    return stem


def evaluation_seed(base_seed: int, image_id: Any) -> int:
    """评估统一种子：``stable_seed(seed, 'evaluation', canonical_image_id)``。"""
    return stable_seed(int(base_seed), "evaluation", canonical_image_id(image_id))


def config_from_checkpoint(
    path: Union[str, Path],
    payload: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """读取 checkpoint 内保存的完整训练配置。

    CLI 未显式传 ``--config`` 时默认使用它，避免用仓库中的示例 YAML 意外
    覆盖训练时修改过的 peaks / diffusion / inference 协议。若调用方已经用
    ``read_checkpoint`` 加载过快照，可通过 ``payload`` 传入以免重复读取。
    """
    if payload is None:
        payload = read_checkpoint(path)
    if not isinstance(payload, Mapping) or not isinstance(
        payload.get("config"), Mapping
    ):
        raise ValueError(f"checkpoint 中没有保存 config：{path}")
    return copy.deepcopy(dict(payload["config"]))


# 影响推理结果的协议字段：显式配置可以覆盖它们，但与训练快照不同时需提示。
_PROTOCOL_KEYS: Tuple[str, ...] = (
    "training_protocol",
    "target_representation",
    "data.ridge_config",
    "model.sam_frozen",
    "model.sam_config",
    "model.sam_architecture",
    "model.sam_apply_postprocessing",
    "model.dit.in_channels",
    "model.dit.out_channels",
    "diffusion.latent_channels",
    "diffusion.timesteps",
    "diffusion.sampling_steps",
    "diffusion.min_std",
    "diffusion.eta",
    "diffusion.beta_start",
    "diffusion.beta_end",
    "diffusion.heat_loss_max_timestep",
    "prompt",
    "train.amp",
    "inference.threshold",
)

_MISSING = object()


def _config_value(config: Mapping[str, Any], dotted: str) -> Any:
    node: Any = config
    for part in dotted.split("."):
        if not isinstance(node, Mapping) or part not in node:
            return _MISSING
        node = node[part]
    return node


def _format_config_value(value: Any) -> str:
    if value is _MISSING:
        return "<missing>"
    if isinstance(value, Mapping):
        return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    return repr(value)


def warn_protocol_differences(
    config: Mapping[str, Any],
    saved_config: Mapping[str, Any],
) -> List[str]:
    """比较当前配置与 checkpoint 保存配置中的推理协议字段。

    显式 ``--config`` 覆盖是允许的，但采样配置、峰值提取、AMP 等变化会让
    推理结果偏离训练时的验证协议，因此发出警告而不是静默继续。返回发生
    变化的字段名列表，便于调用方记录/测试。
    """
    changed: List[str] = []
    for key in _PROTOCOL_KEYS:
        if _format_config_value(_config_value(config, key)) != _format_config_value(
            _config_value(saved_config, key)
        ):
            changed.append(key)
    if changed:
        warnings.warn(
            "当前配置与 checkpoint 保存的训练协议存在差异（"
            + ", ".join(changed)
            + "）；推理会按当前配置执行，结果可能与训练验证不一致。",
            RuntimeWarning,
            stacklevel=3,
        )
    return changed


def make_generator(seed: int) -> torch.Generator:
    """创建 CPU 随机生成器。

    ``GaussianDiffusion._randn`` 支持 CPU 生成器配合 GPU 张量；固定 CPU 生成器
    可以让同一 ``(seed, image_id)`` 在不同设备布局下更容易复现。
    """
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) % (1 << 63))
    return generator


# ---------------------------------------------------------------------------
# 模型构建与权重加载
# ---------------------------------------------------------------------------
_SAM2_ARCHITECTURES = {
    "sam2.1_hiera_t.yaml": "sam2.1_hiera_tiny",
    "sam2.1_hiera_s.yaml": "sam2.1_hiera_small",
    "sam2.1_hiera_b+.yaml": "sam2.1_hiera_base_plus",
    "sam2.1_hiera_l.yaml": "sam2.1_hiera_large",
}


def _validate_sam_architecture(sam_config: str, architecture: str) -> None:
    """拒绝配置名与记录的骨干标识不一致，避免快照元数据静默错记。"""
    expected = _SAM2_ARCHITECTURES.get(Path(sam_config).name.lower())
    if expected is not None and architecture != expected:
        raise ValueError(
            f"model.sam_config={sam_config!r} 对应 {expected!r}，与 "
            f"model.sam_architecture={architecture!r} 不一致。"
        )


def build_models(
    config: Mapping[str, Any],
    device: Union[str, torch.device],
    sam_checkpoint: Optional[Union[str, Path]] = None,
    *,
    config_dir: Optional[Union[str, Path]] = None,
) -> Tuple[Dict[str, torch.nn.Module], None]:
    """构建 signed_ridge_v1 的全部模型组件。

    主链路只包含 ``sam``、``condition_encoder``、``dit``、``diffusion``。
    返回元组第二项固定为 ``None``，仅为兼容 engine 的 ``(models, second)``
    API；旧 ``PeakExtractor`` 已从新协议主链路移除。

    ``sam_checkpoint`` 未给出时按与配置相同的路径基准解析
    ``model.sam_checkpoint``，并校验文件真实存在。DiT 输入/输出通道与扩散
    潜变量通道固定为 1；``RoadSAM`` 的 ``prompt_scale`` 来自
    ``prompt.scale``（默认 1.0，必须有限正数）。

    Returns:
        ``(models, None)``。
    """
    # 构建即进入新协议：旧协议/双通道/未冻结 SAM 配置必须立即失败。
    validate_signed_ridge_config(config)
    device = torch.device(device)
    if sam_checkpoint is not None:
        checkpoint = Path(os.path.expanduser(str(sam_checkpoint)))
        if not checkpoint.is_absolute():
            checkpoint = find_existing_path(
                checkpoint, _config_base_dirs(config_dir), must_exist=True
            )
    else:
        checkpoint = resolve_sam_checkpoint(config, config_dir)
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"SAM2 权重不存在：{checkpoint}。请按 README 下载 "
            "sam2.1_hiera_base_plus.pt。"
        )

    model_cfg = config.get("model", {})
    diffusion_cfg = config.get("diffusion", {})
    dit_cfg = model_cfg.get("dit", {})
    if not isinstance(dit_cfg, Mapping):
        raise ValueError("model.dit 必须是映射")
    sam_config = str(model_cfg.get("sam_config", DEFAULT_SAM2_CONFIG))
    sam_architecture = str(
        model_cfg.get("sam_architecture", DEFAULT_SAM2_ARCHITECTURE)
    )
    _validate_sam_architecture(sam_config, sam_architecture)
    # 梯度检查点只由 model.dit 控制，默认开启。DEFAULT_CONFIG 始终包含该
    # 嵌套键，因此不再回退到旧 model.gradient_checkpointing（避免 merge 后
    # 旧键看似可覆盖、实际被嵌套默认值遮蔽）。
    dit_gradient_checkpointing = bool(
        dit_cfg.get("gradient_checkpointing", True)
    )
    condition_channels = tuple(
        int(channels)
        for channels in dit_cfg.get(
            "condition_channels", (64, 128, 256, 256)
        )
    )
    in_channels = int(dit_cfg.get("in_channels", 1))
    out_channels = int(dit_cfg.get("out_channels", 1))
    latent_channels = int(diffusion_cfg.get("latent_channels", 1))
    if (in_channels, out_channels, latent_channels) != (1, 1, 1):
        raise ValueError(
            "signed_ridge_v1 要求 model.dit.in_channels=1、"
            "model.dit.out_channels=1、diffusion.latent_channels=1，实际为 "
            f"{(in_channels, out_channels, latent_channels)}"
        )
    prompt_scale, _ = resolve_prompt_mapping(config)

    sam_kwargs: Dict[str, Any] = {
        "config_name": sam_config,
        "apply_postprocessing": bool(
            model_cfg.get("sam_apply_postprocessing", False)
        ),
        "architecture": sam_architecture,
    }
    # RoadSAM 在 decode_heatmaps 内部执行 U = prompt_scale * clamp(H)；
    # 尺度直接来自 prompt.scale，禁止静默省略。
    sam_kwargs["prompt_scale"] = prompt_scale

    models: Dict[str, torch.nn.Module] = {
        "sam": RoadSAM(str(checkpoint), **sam_kwargs),
        # 条件编码器与 DiT 使用同一组四尺度通道，非默认配置也不会静默错配。
        "condition_encoder": ConditionEncoder(
            condition_channels=condition_channels
        ),
        "dit": ConditionalDiT(
            in_channels=in_channels,
            out_channels=out_channels,
            patch_size=int(dit_cfg.get("patch_size", 8)),
            hidden_size=int(dit_cfg.get("hidden_size", 256)),
            depth=int(dit_cfg.get("depth", 6)),
            num_heads=int(dit_cfg.get("num_heads", 8)),
            mlp_ratio=float(dit_cfg.get("mlp_ratio", 4.0)),
            condition_channels=condition_channels,
            gradient_checkpointing=dit_gradient_checkpointing,
        ),
        "diffusion": GaussianDiffusion(
            timesteps=int(diffusion_cfg.get("timesteps", 1000)),
            sampling_steps=int(diffusion_cfg.get("sampling_steps", 20)),
            min_std=float(diffusion_cfg.get("min_std", 0.02)),
            beta_start=float(diffusion_cfg.get("beta_start", 1e-4)),
            beta_end=float(diffusion_cfg.get("beta_end", 0.02)),
            eta=float(diffusion_cfg.get("eta", 0.0)),
            latent_channels=latent_channels,
            heat_loss_max_timestep=(
                None
                if diffusion_cfg.get("heat_loss_max_timestep") is None
                else int(diffusion_cfg["heat_loss_max_timestep"])
            ),
        ),
    }
    for name, module in models.items():
        models[name] = module.to(device)

    # 第二项保持 ``(models, second)`` 元组形状以兼容 engine；新协议主链路
    # 不再需要 PeakExtractor。
    return models, None


_MISSING_PROTOCOL_VALUE = object()

# 恢复时必须与 checkpoint 保存值一致的协议字段。差异意味着训练期间的
# 采样调度、RL 超参数或奖励阈值已改变，继续 --resume 会悄悄改变优化目标。
PROTOCOL_COMPARISON_KEYS: Tuple[str, ...] = (
    "training_protocol",
    "target_representation",
    # 目标构造参数变化会改变监督语义，必须拒绝 resume。
    "data.ridge_config",
    "model.sam_frozen",
    "model.dit.in_channels",
    "model.dit.out_channels",
    "diffusion.latent_channels",
    "diffusion.timesteps",
    "diffusion.sampling_steps",
    "diffusion.min_std",
    "diffusion.eta",
    "diffusion.beta_start",
    "diffusion.beta_end",
    "diffusion.heat_loss_max_timestep",
    "prompt",
    "rl.algorithm",
    "rl.group_size",
    "rl.ppo_clip",
    "rl.kl_weight",
    "rl.anchor_weight",
    "rl.update_epochs",
    "rl.advantage_std_delta",
    "reward.iou_weight",
    "reward.cldice_weight",
    "reward.path_weight",
    "reward.point_cost",
    "reward.path_samples",
    "reward.path_match_radius",
    "inference.threshold",
)


def _checkpoint_config(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    config = payload.get("config")
    return config if isinstance(config, Mapping) else {}


def _first_value(*values: Any, default: Any = _MISSING_PROTOCOL_VALUE) -> Any:
    """返回第一个非 ``None`` 的值；全部缺失时返回 ``default``。"""
    for value in values:
        if value is not None:
            return value
    return default


def _prompt_candidate(value: Any) -> Optional[Mapping[str, Any]]:
    """把空/无效 prompt 视为缺失，避免根级空 dict 遮蔽 config 中的有效值。"""
    return value if isinstance(value, Mapping) and value else None


def _checkpoint_sources(payload: Mapping[str, Any]) -> Dict[str, List[Any]]:
    """收集根级与 ``config`` 中的协议字段，用于一致性校验。

    新快照同时写根级与 ``config``。任一字段在多个位置出现且值不同即为
    矛盾，必须拒绝，而不能静默采用“第一个非空值”。
    """
    config = _checkpoint_config(payload)
    model_cfg = config.get("model", {})
    model_cfg = model_cfg if isinstance(model_cfg, Mapping) else {}
    diffusion_cfg = config.get("diffusion", {})
    diffusion_cfg = diffusion_cfg if isinstance(diffusion_cfg, Mapping) else {}
    dit_cfg = model_cfg.get("dit", {})
    dit_cfg = dit_cfg if isinstance(dit_cfg, Mapping) else {}

    return {
        "training_protocol": [
            _normalise_protocol(payload.get("training_protocol")),
            _normalise_protocol(config.get("training_protocol")),
        ],
        "target_representation": [
            _normalise_protocol(payload.get("target_representation")),
            _normalise_protocol(config.get("target_representation")),
        ],
        "latent_channels": [
            payload.get("latent_channels"),
            diffusion_cfg.get("latent_channels"),
        ],
        "dit_in_channels": [dit_cfg.get("in_channels")],
        "dit_out_channels": [dit_cfg.get("out_channels")],
        "sam_frozen": [
            payload.get("sam_frozen"),
            payload.get("abc_sam_frozen"),
            config.get("sam_frozen"),
            model_cfg.get("sam_frozen"),
        ],
        "prompt": [
            _prompt_candidate(payload.get("prompt")),
            _prompt_candidate(config.get("prompt")),
        ],
    }


def _normalised_scalar(value: Any) -> Any:
    """协议字段的可比较规范化：bool/整数/字符串/浮点。"""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower()
    if isinstance(value, (int, float)):
        return float(value)
    return value


def _metadata_contradictions(payload: Mapping[str, Any]) -> List[str]:
    sources = _checkpoint_sources(payload)
    errors: List[str] = []
    for key in (
        "training_protocol",
        "target_representation",
        "latent_channels",
        "dit_in_channels",
        "dit_out_channels",
        "sam_frozen",
    ):
        present = [
            _normalised_scalar(value)
            for value in sources[key]
            if value is not None
        ]
        if len({json.dumps(value, sort_keys=True, default=str) for value in present}) > 1:
            errors.append(f"{key} 在根级/config 中不一致：{present}")
    prompt_values = [value for value in sources["prompt"] if value is not None]
    if len(prompt_values) > 1:
        canonical = {
            json.dumps(dict(value), sort_keys=True, default=str)
            for value in prompt_values
        }
        if len(canonical) > 1:
            errors.append(f"prompt 在根级/config 中不一致：{prompt_values}")
    return errors


def _checkpoint_metadata(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """从快照提取并规范化 signed_ridge_v1 元数据（不做严格校验）。

    新快照把关键字段同时写在根级与 ``config`` 中；这里读取两者。
    :func:`validate_checkpoint_protocol` 会先检查两者是否矛盾。
    """
    config = _checkpoint_config(payload)
    model_cfg = config.get("model", {})
    model_cfg = model_cfg if isinstance(model_cfg, Mapping) else {}
    diffusion_cfg = config.get("diffusion", {})
    diffusion_cfg = diffusion_cfg if isinstance(diffusion_cfg, Mapping) else {}
    dit_cfg = model_cfg.get("dit", {})
    dit_cfg = dit_cfg if isinstance(dit_cfg, Mapping) else {}

    prompt = _first_value(
        _prompt_candidate(payload.get("prompt")),
        _prompt_candidate(config.get("prompt")),
        default={},
    )
    try:
        prompt_scale, terminal_mapping = resolve_prompt_mapping({"prompt": prompt})
    except ValueError:
        prompt_scale, terminal_mapping = float("nan"), ""

    latent_channels = _first_value(
        payload.get("latent_channels"),
        diffusion_cfg.get("latent_channels"),
        dit_cfg.get("in_channels"),
    )
    sam_frozen = _first_value(
        payload.get("sam_frozen"),
        payload.get("abc_sam_frozen"),
        config.get("sam_frozen"),
        model_cfg.get("sam_frozen"),
    )
    return {
        "format_version": payload.get("format_version"),
        "training_protocol": _normalise_protocol(
            _first_value(
                payload.get("training_protocol"),
                config.get("training_protocol"),
            )
        ),
        "target_representation": _normalise_protocol(
            _first_value(
                payload.get("target_representation"),
                config.get("target_representation"),
            )
        ),
        "latent_channels": latent_channels,
        "sam_frozen": sam_frozen,
        "prompt_scale": prompt_scale,
        "terminal_mapping": terminal_mapping,
        "dit_in_channels": _first_value(
            dit_cfg.get("in_channels"), default=None
        ),
        "dit_out_channels": _first_value(
            dit_cfg.get("out_channels"), default=None
        ),
    }


def validate_checkpoint_protocol(
    payload: Mapping[str, Any],
    config: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """严格校验 checkpoint 是否为 signed_ridge_v1 快照。

    Engine 的调用约定为 ``validate_checkpoint_protocol(payload, self.config)``：
    除根级/``config`` 元数据外，还会比较当前配置中的采样、RL 与奖励阈值；
    任一协议字段不同即拒绝恢复（而不是静默用新设置继续训练）。

    Args:
        payload: 训练快照（``torch.load`` 结果）。
        config: 当前运行配置；提供时逐项比较协议相关字段。

    Returns:
        规范化后的 checkpoint 元数据字典。

    Raises:
        CheckpointProtocolError: 旧协议、通道/冻结/提示映射不兼容，或当前
            配置与快照保存的训练协议不一致。
    """
    if not isinstance(payload, Mapping):
        raise CheckpointProtocolError("checkpoint 顶层必须是映射")
    meta = _checkpoint_metadata(payload)
    errors: List[str] = _metadata_contradictions(payload)

    version = meta["format_version"]
    try:
        version_int = int(version)
    except (TypeError, ValueError):
        version_int = -1
    if version_int < CHECKPOINT_FORMAT_VERSION:
        errors.append(
            f"format_version={version!r}，{SIGNED_RIDGE_PROTOCOL} 要求 "
            f">= {CHECKPOINT_FORMAT_VERSION}"
        )

    if meta["training_protocol"] != SIGNED_RIDGE_PROTOCOL:
        errors.append(
            f"training_protocol={meta['training_protocol']!r}，要求 "
            f"{SIGNED_RIDGE_PROTOCOL!r}"
        )
    if meta["target_representation"] != TARGET_REPRESENTATION:
        errors.append(
            f"target_representation={meta['target_representation']!r}，要求 "
            f"{TARGET_REPRESENTATION!r}"
        )
    try:
        if int(meta["latent_channels"]) != 1:
            errors.append(
                f"latent_channels={meta['latent_channels']!r}，要求 1"
            )
    except (TypeError, ValueError):
        errors.append(f"latent_channels={meta['latent_channels']!r} 缺失或无效")
    for channel_key in ("dit_in_channels", "dit_out_channels"):
        value = meta[channel_key]
        if value is not None:
            try:
                if int(value) != 1:
                    errors.append(f"{channel_key}={value!r}，要求 1")
            except (TypeError, ValueError):
                errors.append(f"{channel_key}={value!r} 无效")
    if meta["sam_frozen"] is not True:
        errors.append(
            f"sam_frozen={meta['sam_frozen']!r}，要求 true（SAM 全阶段冻结）"
        )
    if not math.isfinite(meta["prompt_scale"]) or meta["prompt_scale"] <= 0.0:
        errors.append("prompt.scale 缺失或不是有限正数")
    if meta["terminal_mapping"] not in PROMPT_TERMINAL_MAPPINGS:
        errors.append(
            f"prompt.terminal_mapping={meta['terminal_mapping']!r} 不受支持"
        )

    # 当前配置与快照保存配置逐字段比较。缺失值不比较，避免旧 YAML 的
    # 可选字段造成误报；显式出现的差异一律拒绝。
    saved_config = _checkpoint_config(payload)
    if config is not None and saved_config:
        for key in PROTOCOL_COMPARISON_KEYS:
            current = _config_value(config, key)
            saved = _config_value(saved_config, key)
            if current is _MISSING or saved is _MISSING:
                continue
            if _format_config_value(current) != _format_config_value(saved):
                errors.append(
                    f"{key} 当前={_format_config_value(current)} "
                    f"与快照={_format_config_value(saved)} 不一致"
                )
    if errors:
        raise CheckpointProtocolError(
            "checkpoint 与 signed_ridge_v1 训练协议不兼容："
            + "；".join(errors)
            + "。旧双通道/SAM 联合微调快照不能 --resume；"
            "如需只迁移形状兼容权重，请显式使用 "
            "load_compatible_weights（或 CLI --migrate-weights），"
            "它不会恢复优化器状态。"
        )
    return meta


def _load_state(module: torch.nn.Module, state: Mapping[str, Any], name: str) -> None:
    """严格加载权重；缺失/多余键直接报错，避免静默使用错误 checkpoint。"""
    try:
        module.load_state_dict(state, strict=True)
    except (RuntimeError, ValueError) as exc:
        if name == "dit":
            raise ValueError(
                "checkpoint.models.dit 与当前 ConditionalDiT 不兼容："
                f"{exc}。请使用相同 DiT 配置生成的训练快照。"
            ) from exc
        raise


def _expects_sam2_state(module: torch.nn.Module) -> bool:
    """目标 SAM 是否使用 SAM2 的参数命名（``sam.sam_mask_decoder`` 等）。"""
    inner = getattr(module, "sam", None)
    return inner is not None and hasattr(inner, "sam_mask_decoder")


def _looks_like_sam2_state(state: Mapping[str, Any]) -> bool:
    """通过默认键前缀识别 SAM2 权重，显式拒绝旧 SAM1 ViT-B 快照。"""
    prefixes = (
        "sam.sam_mask_decoder.",
        "sam.sam_prompt_encoder.",
        "sam.image_encoder.trunk.",
    )
    return any(str(key).startswith(prefixes) for key in state)


def load_checkpoint_models(
    models: Mapping[str, torch.nn.Module],
    payload: Mapping[str, Any],
    *,
    include_ref: bool = False,
    validate_protocol: bool = True,
) -> Dict[str, Any]:
    """从本项目生成的训练快照中严格加载模型权重（不含优化器状态）。

    本项目的 ``last.pt`` / ``best.pt`` / ``stage_X.pth`` 必须包含
    ``models.{sam, condition_encoder, dit, diffusion}``。
    缺字段或键不匹配会立即失败，避免用随机或错误权重继续预测。
    ``ref_dit`` 只存在于 D
    阶段快照，由 ``include_ref`` 控制是否加载。

    所有快照都会先经过 :func:`validate_checkpoint_protocol`，旧协议或缺少
    完整 signed_ridge_v1 元数据的 payload 一律拒绝。迁移只能显式调用
    :func:`load_compatible_weights`（不恢复优化器）。
    """
    if not isinstance(payload, Mapping) or "models" not in payload:
        raise ValueError("checkpoint 缺少顶层 models 字段，不是本项目的训练快照")
    if validate_protocol:
        # 生产路径不存在“无协议标记即可放行”的测试绕行：旧快照必须被
        # 拒绝，迁移只能走显式 load_compatible_weights。
        validate_checkpoint_protocol(payload)
    source = payload["models"]
    if not isinstance(source, Mapping):
        raise ValueError("checkpoint.models 必须是映射")
    required = ("sam", "condition_encoder", "dit", "diffusion")
    missing = [name for name in required if name not in source]
    if missing:
        raise ValueError(
            "checkpoint.models 缺少必需模块："
            + ", ".join(missing)
            + "；请使用本项目 train.py 生成的快照"
        )

    sam_source = source["sam"]
    if (
        isinstance(sam_source, Mapping)
        and _expects_sam2_state(models.get("sam"))
        and not _looks_like_sam2_state(sam_source)
    ):
        raise ValueError(
            "checkpoint.models.sam 不是 SAM2 权重：缺少 sam.sam_mask_decoder / "
            "sam.sam_prompt_encoder / sam.image_encoder.trunk 等键。旧 SAM1 "
            "ViT-B 训练快照与 SAM2.1 Hiera-B+ 不兼容，不能 --resume/--init "
            "到 SAM2，请从 A 阶段重新训练。"
        )

    # 快照内记录的骨干必须与当前构建的 SAM2 配置一致，避免静默混用不同
    # Hiera 尺寸（例如把 B+ 快照加载到 T/S/L）。旧快照没有该字段时跳过，
    # 由上面的键前缀检查负责拒绝 SAM1。
    saved_config = payload.get("config")
    if isinstance(saved_config, Mapping):
        saved_model_cfg = saved_config.get("model", {})
        if isinstance(saved_model_cfg, Mapping):
            # 注意力头数不改变权重形状，strict=True 无法发现这种语义错配。
            saved_dit = saved_model_cfg.get("dit", {})
            target_dit = models.get("dit")
            if isinstance(saved_dit, Mapping) and isinstance(target_dit, ConditionalDiT):
                saved_heads = saved_dit.get("num_heads")
                if saved_heads is not None and int(saved_heads) != target_dit.num_heads:
                    raise ValueError(
                        "checkpoint 的 DiT num_heads 与当前配置不一致："
                        f"{saved_heads} != {target_dit.num_heads}；"
                        "注意力头数必须与训练快照匹配。"
                    )
            saved_architecture = saved_model_cfg.get("sam_architecture")
            target_architecture = getattr(models.get("sam"), "architecture", None)
            if (
                saved_architecture
                and target_architecture
                and str(saved_architecture) != str(target_architecture)
            ):
                raise ValueError(
                    "checkpoint 的 SAM 骨干与当前配置不一致："
                    f"{saved_architecture!r} != {target_architecture!r}。"
                    "请使用匹配的 model.sam_architecture / sam_config 与权重。"
                )

    for name in ("sam", "condition_encoder", "dit"):
        if name not in models:
            raise ValueError(f"目标模型集合缺少 {name}")
        _load_state(models[name], source[name], name)
    if "diffusion" not in models:
        raise ValueError("目标模型集合缺少 diffusion")
    _load_state(models["diffusion"], source["diffusion"], "diffusion")

    if include_ref and "ref_dit" in source:
        if "ref_dit" not in models:
            raise ValueError("需要加载 ref_dit，但目标模型集合未创建参考 DiT")
        _load_state(models["ref_dit"], source["ref_dit"], "ref_dit")
    return {}


# Legacy -> signed_ridge_v1 迁移时允许加载的 DiT 参数前缀。输入投影与输出头
# 的语义随单通道协议改变，即使形状偶然相同也不能复用。
_DIT_MIGRATION_SKIP_PREFIXES: Tuple[str, ...] = (
    "patch_embed.",
    "final_layer.",
)


def load_compatible_weights(
    models: Mapping[str, torch.nn.Module],
    payload: Mapping[str, Any],
    *,
    include_ref: bool = False,
) -> Dict[str, Any]:
    """显式迁移旧快照的形状兼容权重，并报告跳过/重新初始化的参数。

    迁移规则（不可用 ``--resume`` 替代）：

    * ``sam`` 与 ``diffusion`` 一律不加载。冻结 SAM 应保持官方预训练权重，
      旧快照中的 SAM 状态（尤其旧 SAM1/联合微调解码器）不能混入。
    * ``condition_encoder`` 加载所有名称与形状匹配的参数。
    * ``dit`` 只加载内部块（attention/MLP/adaLN 等）；``patch_embed.*`` 与
      ``final_layer.*`` 因输入输出通道语义改变而始终跳过。
    * 形状不匹配、目标缺失或来源多余的键都记录在报告中；目标模块保留当前
      初始化值（即“重新初始化”）。
    * **不恢复优化器、GradScaler、调度器或 RNG 状态。**

    Args:
        models: 已构建的目标模型集合。
        payload: 旧/新快照；只需包含 ``models``。
        include_ref: 是否迁移 ``ref_dit``（若快照与目标模型都提供）。

    Returns:
        ``{"migration": True, "optimizer_restored": False, "loaded":
        {module: [key]}, "skipped": {module: {key: reason}}, "reinitialized":
        {module: [key]}, "skipped_modules": [...], ...}``。
    """
    if not isinstance(payload, Mapping) or not isinstance(
        payload.get("models"), Mapping
    ):
        raise ValueError("迁移需要 checkpoint.models 映射")
    source = payload["models"]
    report: Dict[str, Any] = {
        "migration": True,
        "optimizer_restored": False,
        "source_protocol": _checkpoint_metadata(payload)["training_protocol"],
        "target_protocol": SIGNED_RIDGE_PROTOCOL,
        "loaded": {},
        "skipped": {},
        "reinitialized": {},
        "skipped_modules": [],
        "missing_modules": [],
        "counts": {"loaded": 0, "skipped": 0, "reinitialized": 0},
    }

    def _migrate_module(
        name: str,
        module: torch.nn.Module,
        *,
        skip_prefixes: Sequence[str] = (),
    ) -> None:
        state = source.get(name)
        target_state = module.state_dict()
        loaded: List[str] = []
        skipped: Dict[str, str] = {}
        if not isinstance(state, Mapping):
            report["missing_modules"].append(name)
            state = {}

        for key, value in state.items():
            key = str(key)
            if any(key.startswith(prefix) for prefix in skip_prefixes):
                skipped[key] = "input_output_head"
                continue
            if key not in target_state:
                skipped[key] = "unexpected_key"
                continue
            if not torch.is_tensor(value):
                skipped[key] = "not_a_tensor"
                continue
            target_value = target_state[key]
            if tuple(value.shape) != tuple(target_value.shape):
                skipped[key] = "shape_mismatch"
                continue
            loaded.append(key)

        loaded_set = set(loaded)
        loadable = {key: state[key] for key in loaded}
        result = module.load_state_dict(loadable, strict=False)
        reinitialized = [str(key) for key in result.missing_keys]
        # 理论上 loadable 已被过滤；保留 unexpected 记录以防御 dtype/类型问题。
        for key in result.unexpected_keys:
            skipped[str(key)] = "load_rejected"
        report["loaded"][name] = loaded
        report["skipped"][name] = skipped
        report["reinitialized"][name] = reinitialized
        report["counts"]["loaded"] += len(loaded)
        report["counts"]["skipped"] += len(skipped)
        report["counts"]["reinitialized"] += len(reinitialized)

    for name in ("sam", "diffusion"):
        if name not in models:
            continue
        source_state = source.get(name)
        if not isinstance(source_state, Mapping):
            # 目标存在但旧快照没有该模块：全部参数保持新初始化。
            report["missing_modules"].append(name)
            source_state = {}
        else:
            report["skipped_modules"].append(name)
        report["loaded"][name] = []
        report["skipped"][name] = {
            str(key): (
                "frozen_sam_preserved"
                if name == "sam"
                else "protocol_buffer_not_migrated"
            )
            for key in source_state
        }
        report["reinitialized"][name] = [
            str(key) for key in models[name].state_dict()
        ]
        report["counts"]["skipped"] += len(report["skipped"][name])
        report["counts"]["reinitialized"] += len(report["reinitialized"][name])

    if "condition_encoder" in models:
        _migrate_module("condition_encoder", models["condition_encoder"])
    if "dit" in models:
        _migrate_module(
            "dit", models["dit"], skip_prefixes=_DIT_MIGRATION_SKIP_PREFIXES
        )
    if include_ref and "ref_dit" in models and "ref_dit" in source:
        _migrate_module(
            "ref_dit",
            models["ref_dit"],
            skip_prefixes=_DIT_MIGRATION_SKIP_PREFIXES,
        )
    return report


def read_checkpoint(path: Union[str, Path]) -> Any:
    """读取 PyTorch 快照；兼容新旧 ``weights_only`` 默认值。"""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint 不存在：{path}")
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover - 旧版 PyTorch
        return torch.load(path, map_location="cpu")


# ---------------------------------------------------------------------------
# 尺寸、图像与提示
# ---------------------------------------------------------------------------
def compute_resized_size(
    height: int,
    width: int,
    image_size: int = 1024,
) -> Tuple[int, int]:
    """与 ``diffpeak.data.geometry.build_geometry`` 完全一致的缩放尺寸。"""
    if height <= 0 or width <= 0:
        raise ValueError(f"图像尺寸必须为正，实际为 {(height, width)}")
    scale = min(image_size / float(height), image_size / float(width))
    resized_height = int(min(image_size, max(1, round(height * scale))))
    resized_width = int(min(image_size, max(1, round(width * scale))))
    return resized_height, resized_width


def normalize_valid_sizes(
    value: Any,
    batch_size: int,
    image_size: int = 1024,
) -> List[Optional[Tuple[int, int]]]:
    """把 DataLoader 中的 ``resized_size`` 整理为逐样本 ``(h, w)`` 列表。

    支持 ``[(h, w), ...]``、单个 ``(h, w)``（仅 B=1）、以及默认 collate
    可能产生的 ``[h_tensor, w_tensor]`` 两种形式。
    """
    if value is None:
        return [None] * batch_size

    if torch.is_tensor(value):
        if value.ndim == 1 and value.numel() == 2 and batch_size == 1:
            height, width = value.detach().cpu().tolist()
            return [(int(height), int(width))]
        if value.ndim == 2 and value.shape[0] == batch_size:
            return [
                (int(row[0]), int(row[1]))
                for row in value.detach().cpu().tolist()
            ]

    if isinstance(value, (list, tuple)):
        # 默认 collate 对 tuple 可能返回 [h 的张量, w 的张量]。
        if (
            len(value) == 2
            and all(torch.is_tensor(item) and item.ndim == 1 for item in value)
            and int(value[0].numel()) == batch_size
            and int(value[1].numel()) == batch_size
        ):
            heights = value[0].detach().cpu().tolist()
            widths = value[1].detach().cpu().tolist()
            return [(int(h), int(w)) for h, w in zip(heights, widths)]
        if (
            len(value) == 2
            and all(np.isscalar(item) for item in value)
            and batch_size == 1
        ):
            return [(int(value[0]), int(value[1]))]
        if len(value) == batch_size:
            sizes: List[Optional[Tuple[int, int]]] = []
            for item in value:
                if item is None:
                    sizes.append(None)
                    continue
                if torch.is_tensor(item):
                    item = item.detach().cpu().tolist()
                if not isinstance(item, (list, tuple)) or len(item) != 2:
                    raise ValueError(f"无效的 resized_size：{item!r}")
                sizes.append((int(item[0]), int(item[1])))
            return sizes

    raise ValueError(f"无法解析 resized_size：{type(value)!r} / {value!r}")


def encode_images(
    sam: RoadSAM,
    images: torch.Tensor,
    valid_sizes: Optional[Sequence[Optional[Tuple[int, int]]]] = None,
) -> Union[torch.Tensor, SAM2ImageFeatures]:
    """调用 ``RoadSAM.encode``，并显式传入填充前有效尺寸。

    ``valid_sizes`` 中每项是右下填充前的 ``(height, width)``；模型会据此在
    标准化后把 padding 特征置零。真实 ``RoadSAM`` 返回
    :class:`SAM2ImageFeatures`；测试用的轻量替身也可返回裸张量。
    """
    if valid_sizes is not None:
        valid_sizes = [
            None if item is None else (int(item[0]), int(item[1]))
            for item in valid_sizes
        ]
    return sam.encode(images, valid_sizes=valid_sizes)


def get_points(
    extractor: PeakExtractor,
    heatmaps: Union[torch.Tensor, np.ndarray],
    image_size: int = 1024,
    valid_sizes: Optional[Sequence[Optional[Tuple[int, int]]]] = None,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """在 CPU 上执行固定峰值提取，并保留逐样本有效尺寸过滤。"""
    if torch.is_tensor(heatmaps):
        heatmaps_cpu = heatmaps.detach().float().cpu()
    else:
        heatmaps_cpu = torch.as_tensor(np.asarray(heatmaps), dtype=torch.float32)
    if heatmaps_cpu.ndim == 3:
        heatmaps_cpu = heatmaps_cpu.unsqueeze(0)

    sizes = None
    if valid_sizes is not None:
        sizes = [
            None if item is None else (int(item[0]), int(item[1]))
            for item in valid_sizes
        ]
    results = extractor(heatmaps_cpu, image_size=int(image_size), valid_sizes=sizes)
    cleaned: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for coords, labels in results:
        coords = torch.as_tensor(coords, dtype=torch.float32).reshape(-1, 2)
        labels = torch.as_tensor(labels, dtype=torch.int64).reshape(-1)
        if coords.shape[0] != labels.shape[0]:
            raise ValueError("峰值提取返回的坐标与标签数量不一致")
        cleaned.append((coords, labels))
    return cleaned


def points_to_prompt(
    points: Tuple[torch.Tensor, torch.Tensor],
    device: Union[str, torch.device],
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """把点集转换为 SAM 提示；没有正点时返回 ``None``（输出空掩码）。"""
    coords, labels = points
    labels = labels.reshape(-1)
    if labels.numel() == 0 or not bool((labels == 1).any()):
        return None
    return (
        coords.reshape(-1, 2).to(device=device, dtype=torch.float32),
        labels.to(device=device, dtype=torch.long),
    )


def decode_with_prompts(
    sam: RoadSAM,
    embeddings: Union[torch.Tensor, SAM2ImageFeatures],
    prompts: Sequence[Optional[Tuple[torch.Tensor, torch.Tensor]]],
    output_size: Tuple[int, int] = (1024, 1024),
) -> Tuple[torch.Tensor, List[bool]]:
    """逐图调用 ``RoadSAM.decode``。

    逐图解码规避官方 SAM2 在 ``repeat_image=True`` 时对 batch 做
    ``repeat_interleave`` 导致 B² 放大的问题，也便于跳过没有正点的样本。
    返回 ``(logits, active)``；``active[b]`` 表示该图实际参与了解码。
    """
    view = low_res_features(embeddings)
    batch_size = int(view.shape[0])
    if len(prompts) != batch_size:
        raise ValueError(f"prompts 数量 {len(prompts)} 与 batch size {batch_size} 不一致")

    empty_logit = float(getattr(sam, "no_positive_logit", -20.0))
    pieces: List[torch.Tensor] = []
    active: List[bool] = []
    for index in range(batch_size):
        prompt = prompts[index]
        valid = False
        coords: Optional[torch.Tensor] = None
        labels: Optional[torch.Tensor] = None
        if prompt is not None:
            coords, labels = prompt
            coords = coords.to(device=view.device, dtype=torch.float32).reshape(-1, 2)
            labels = labels.to(device=view.device, dtype=torch.long).reshape(-1)
            valid = labels.numel() > 0 and bool((labels == 1).any())

        if valid and coords is not None and labels is not None:
            sample = (
                embeddings.select(index)
                if isinstance(embeddings, SAM2ImageFeatures)
                else embeddings[index : index + 1]
            )
            sample_logits = sam.decode(
                sample,
                [(coords, labels)],
                output_size=output_size,
            )
            if not torch.is_tensor(sample_logits):
                raise TypeError(
                    "RoadSAM.decode 必须返回 B x 1 x H x W 的 Tensor，"
                    f"实际为 {type(sample_logits)!r}"
                )
            if sample_logits.ndim != 4 or sample_logits.shape[1] != 1:
                raise ValueError(
                    "RoadSAM.decode 输出必须为 B x 1 x H x W，"
                    f"实际为 {tuple(sample_logits.shape)}"
                )
            pieces.append(sample_logits.float())
            active.append(True)
        else:
            pieces.append(
                torch.full(
                    (1, 1, int(output_size[0]), int(output_size[1])),
                    empty_logit,
                    dtype=torch.float32,
                    device=view.device,
                )
            )
            active.append(False)
    return torch.cat(pieces, dim=0), active


def prepare_image(
    image: Union[np.ndarray, torch.Tensor, Image.Image],
    image_size: int = 1024,
) -> Tuple[torch.Tensor, Tuple[int, int], Tuple[int, int]]:
    """等比缩放 + 右下零填充到 ``image_size``。

    Returns:
        ``(image_tensor[1,3,S,S] float32 [0,1], original_size=(H,W),
        resized_size=(h,w))``。
    """
    if isinstance(image, Image.Image):
        array = np.asarray(image.convert("RGB"), dtype=np.uint8)
    elif torch.is_tensor(image):
        array = image.detach().cpu().numpy()
        if array.ndim == 3 and array.shape[0] in (1, 3, 4) and array.shape[-1] not in (1, 3, 4):
            array = np.transpose(array, (1, 2, 0))
    else:
        array = np.asarray(image)

    if array.ndim == 2:
        array = np.repeat(array[:, :, None], 3, axis=2)
    if array.ndim != 3:
        raise ValueError(f"图像必须是 HxW 或 HxWxC，实际为 {array.shape}")
    if array.shape[2] == 1:
        array = np.repeat(array, 3, axis=2)
    if array.shape[2] >= 3:
        array = array[:, :, :3]
    else:
        raise ValueError(f"图像通道数不足：{array.shape}")

    if np.issubdtype(array.dtype, np.floating):
        maximum = float(np.nanmax(array)) if array.size else 0.0
        if maximum <= 1.0001:
            array = array * 255.0
        array = np.nan_to_num(array, nan=0.0, posinf=255.0, neginf=0.0)
    array = np.clip(array, 0, 255).astype(np.uint8)

    height, width = int(array.shape[0]), int(array.shape[1])
    resized_height, resized_width = compute_resized_size(height, width, image_size)
    pil = Image.fromarray(array, mode="RGB")
    resized = np.asarray(
        pil.resize((resized_width, resized_height), Image.BILINEAR),
        dtype=np.uint8,
    )
    canvas = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    canvas[:resized_height, :resized_width] = resized
    tensor = torch.from_numpy(canvas).permute(2, 0, 1).float().div_(255.0).unsqueeze(0)
    return tensor, (height, width), (resized_height, resized_width)


def prepare_image_file(
    path: Union[str, Path],
    image_size: int = 1024,
) -> Tuple[torch.Tensor, Tuple[int, int], Tuple[int, int]]:
    """从磁盘读取 RGB 图像并完成几何预处理。"""
    return prepare_image(load_image_file(path), image_size=image_size)


def load_image_file(path: Union[str, Path]) -> np.ndarray:
    """读取 RGB 图像为 ``H x W x 3`` 的 ``uint8`` 数组。"""
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def load_mask_file(path: Union[str, Path]) -> np.ndarray:
    """读取二值道路标注，返回 ``H x W bool``。"""
    with Image.open(path) as image:
        if image.mode in ("RGB", "RGBA"):
            array = np.asarray(image.convert("RGB"), dtype=np.uint8)
            return np.any(array > 127, axis=2)
        array = np.asarray(image.convert("L"), dtype=np.uint8)
    return array > 127


def crop_resize_probability(
    probability: torch.Tensor,
    original_size: Tuple[int, int],
    resized_size: Tuple[int, int],
) -> np.ndarray:
    """裁掉右下 padding，再把概率图双线性恢复到原始尺寸。"""
    if probability.ndim == 3:
        probability = probability[0]
    original_height, original_width = int(original_size[0]), int(original_size[1])
    resized_height, resized_width = int(resized_size[0]), int(resized_size[1])

    cropped = probability[:resized_height, :resized_width].float()
    if (resized_height, resized_width) != (original_height, original_width):
        cropped = F.interpolate(
            cropped[None, None],
            size=(original_height, original_width),
            mode="bilinear",
            align_corners=False,
        )[0, 0]
    return cropped.detach().cpu().numpy().astype(np.float32)


def map_points_to_original(
    points: Tuple[Union[torch.Tensor, np.ndarray], Union[torch.Tensor, np.ndarray]],
    original_size: Tuple[int, int],
    resized_size: Tuple[int, int],
) -> Dict[str, Any]:
    """把 SAM 坐标系下的点映射回原图坐标，供 JSON 保存。"""
    coords, labels = points
    if torch.is_tensor(coords):
        coords = coords.detach().cpu().numpy()
    if torch.is_tensor(labels):
        labels = labels.detach().cpu().numpy()
    coords = np.asarray(coords, dtype=np.float64).reshape(-1, 2)
    labels = np.asarray(labels).reshape(-1)

    height, width = int(original_size[0]), int(original_size[1])
    resized_height, resized_width = int(resized_size[0]), int(resized_size[1])
    scale_x = width / float(max(resized_width, 1))
    scale_y = height / float(max(resized_height, 1))

    positive: List[List[float]] = []
    negative: List[List[float]] = []
    for (x, y), label in zip(coords, labels):
        # 与数据几何的像素中心对齐约定保持一致。
        original_x = (float(x) + 0.5) * scale_x - 0.5
        original_y = (float(y) + 0.5) * scale_y - 0.5
        original_x = float(np.clip(original_x, 0.0, max(width - 1.0, 0.0)))
        original_y = float(np.clip(original_y, 0.0, max(height - 1.0, 0.0)))
        item = [round(original_x, 4), round(original_y, 4)]
        if int(label) == 1:
            positive.append(item)
        else:
            negative.append(item)
    return {
        "positive": positive,
        "negative": negative,
        "positive_count": len(positive),
        "negative_count": len(negative),
        "point_count": len(positive) + len(negative),
        "coordinate_space": "original_image_xy",
    }


# ---------------------------------------------------------------------------
# 保存与指标
# ---------------------------------------------------------------------------
def save_mask_file(
    mask: Union[np.ndarray, torch.Tensor],
    path: Union[str, Path],
    *,
    overwrite: bool = False,
) -> Path:
    """保存 0/255 单通道二值 PNG；默认拒绝覆盖已有文件。"""
    if torch.is_tensor(mask):
        mask = mask.detach().cpu().numpy()
    array = np.asarray(mask)
    if array.dtype != np.bool_:
        array = array > 0
    output = Path(path)
    if output.exists() and not overwrite:
        raise FileExistsError(f"输出文件已存在，拒绝覆盖：{output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((array.astype(np.uint8) * 255), mode="L").save(output)
    return output


def save_points_file(
    points: Mapping[str, Any],
    path: Union[str, Path],
    *,
    overwrite: bool = False,
) -> Path:
    """保存提示点 JSON；默认拒绝覆盖已有文件。"""
    output = Path(path)
    if output.exists() and not overwrite:
        raise FileExistsError(f"输出文件已存在，拒绝覆盖：{output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(dict(points), handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return output


def unique_output_path(
    output_root: Union[str, Path],
    relative_path: Union[str, Path],
    *,
    suffix: str = ".png",
    produced: Optional[set] = None,
    source_path: Optional[Union[str, Path]] = None,
) -> Path:
    """为目录推理生成不覆盖磁盘文件、其他输入或源文件的输出路径。

    目标已存在、已由本次运行产生或与源文件相同的时候，依次追加
    ``__1``、``__2``……直到得到空闲路径。
    """
    output_root = Path(output_root)
    relative = Path(relative_path)
    if relative.is_absolute():
        relative = Path(relative.name)
    target = output_root / relative.with_suffix(suffix)
    source = Path(source_path).resolve() if source_path is not None else None

    def _taken(candidate: Path) -> bool:
        if source is not None and candidate.resolve() == source:
            return True
        if candidate.exists():
            return True
        return produced is not None and str(candidate.resolve()) in produced

    base = target.with_suffix("")
    candidate = target
    counter = 1
    while _taken(candidate):
        candidate = Path(f"{base}__{counter}{suffix}")
        counter += 1
    if produced is not None:
        produced.add(str(candidate.resolve()))
    return candidate


def make_reward_config(config: Mapping[str, Any]):
    """从根配置构造 ``RewardConfig``（只传入 dataclass 认识的字段）。"""
    from dataclasses import fields

    from diffpeak.rewards import RewardConfig

    raw = dict(config.get("reward", DEFAULT_CONFIG["reward"]))
    valid_names = {field.name for field in fields(RewardConfig)}
    keyword = {key: value for key, value in raw.items() if key in valid_names}
    return RewardConfig(**keyword)


def compute_image_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    point_count: int,
    reward_config: Any = None,
) -> Dict[str, float]:
    """计算单张图的 macro IoU/F1、clDice、reward 与混淆计数。

    预测与标注必须位于同一像素空间；尺寸不一致说明调用方没有恢复原图几何，
    直接报错而不是悄悄缩放 GT（那会在错误的网格上计算指标）。
    """
    from diffpeak.rewards import compute_reward, image_confusion_metrics

    prediction = np.asarray(prediction).astype(bool)
    target = np.asarray(target).astype(bool)
    if prediction.shape != target.shape:
        raise ValueError(
            "预测与标注尺寸不一致："
            f"prediction={prediction.shape}, target={target.shape}；"
            "请先把预测恢复到标注所在的原图分辨率"
        )
    if reward_config is None:
        reward_config = make_reward_config({})
    result = compute_reward(prediction, target, int(point_count), reward_config)
    confusion = image_confusion_metrics(prediction, target)
    return {
        "reward": float(result["reward"]),
        "iou": float(result["iou"]),
        "f1": float(result["f1"]),
        "cldice": float(result["cldice"]),
        "path": float(result["path"]),
        "tp": float(confusion["tp"]),
        "fp": float(confusion["fp"]),
        "fn": float(confusion["fn"]),
        "tn": float(confusion["tn"]),
    }


def aggregate_metrics(rows: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
    """聚合逐图指标。

    ``iou`` / ``f1`` / ``cldice`` / ``reward`` / ``path`` 是宏平均；另外由
    累加的 TP/FP/FN 计算 ``micro_iou`` 与 ``micro_f1``。
    """
    keys = ("reward", "iou", "f1", "cldice", "path")
    summary: Dict[str, float] = {}
    for key in keys:
        values = []
        for row in rows:
            value = row.get(key)
            if value is None:
                continue
            value = float(value)
            if math.isfinite(value):
                values.append(value)
        summary[key] = float(np.mean(values)) if values else float("nan")
    summary["count"] = int(len(rows))

    # micro 指标依赖每张图的 TP/FP/FN 计数。任一图缺少计数（或 rows 为空）时，
    # 无法区分“没有正样本”和“没有统计”，因此返回 NaN 而不是默认 0 ——
    # 否则空输入会被误判成 micro_iou == micro_f1 == 1.0 的满分。
    counts: List[Tuple[float, float, float]] = []
    for row in rows:
        try:
            counts.append(tuple(float(row[key]) for key in ("tp", "fp", "fn")))
        except (KeyError, TypeError, ValueError):
            counts = []
            break
    counts_complete = bool(rows) and len(counts) == len(rows) and all(
        math.isfinite(value) for item in counts for value in item
    )
    if counts_complete:
        tp = float(sum(item[0] for item in counts))
        fp = float(sum(item[1] for item in counts))
        fn = float(sum(item[2] for item in counts))
        union = tp + fp + fn
        f1_denominator = 2.0 * tp + fp + fn
        summary["tp"] = tp
        summary["fp"] = fp
        summary["fn"] = fn
        summary["micro_iou"] = 1.0 if union == 0 else tp / union
        summary["micro_f1"] = (
            1.0 if f1_denominator == 0 else 2.0 * tp / f1_denominator
        )
    else:
        summary["tp"] = float("nan")
        summary["fp"] = float("nan")
        summary["fn"] = float("nan")
        summary["micro_iou"] = float("nan")
        summary["micro_f1"] = float("nan")
    if math.isfinite(summary.get("iou", float("nan"))):
        summary["score"] = summary["iou"]
    else:
        summary["score"] = summary.get("reward", float("-inf"))
    return summary


# ---------------------------------------------------------------------------
# 划分文件
# ---------------------------------------------------------------------------
def load_labeled_split(
    split_file: Union[str, Path, Sequence[str]],
    root: Union[str, Path],
) -> List[Dict[str, Any]]:
    """读取有标注划分并解析影像/掩码路径。

    正式划分是相对数据根目录的 JSON 字符串数组，例如
    ``"train/100034_sat.jpg"``；掩码按 ``xxx_sat.jpg -> xxx_mask.png`` 派生。
    """
    root = Path(root)
    if isinstance(split_file, (list, tuple)):
        entries = list(split_file)
    else:
        path = Path(split_file)
        if not path.is_file():
            raise FileNotFoundError(f"划分文件不存在：{path}")
        with path.open("r", encoding="utf-8") as handle:
            entries = json.load(handle)
    if not isinstance(entries, list) or not entries:
        raise ValueError("划分文件内容必须是非空数组")
    if not all(isinstance(entry, str) for entry in entries):
        raise ValueError("划分文件必须是相对数据根目录的字符串数组")

    parsed: List[Dict[str, Any]] = []
    for relative in entries:
        image_path = Path(relative)
        if image_path.is_absolute():
            raise ValueError(f"划分项必须使用相对路径：{relative}")
        image_path = (root / image_path).expanduser()
        stem = image_path.stem
        if stem.lower().endswith("_sat"):
            stem = stem[:-4]
        mask_path = image_path.with_name(f"{stem}_mask.png")
        try:
            relative_path = image_path.relative_to(root).as_posix()
        except ValueError:
            relative_path = image_path.name
        parsed.append(
            {
                "id": stem,
                "image_path": image_path,
                "mask_path": mask_path,
                "relative_path": relative_path,
            }
        )
    return parsed


# ---------------------------------------------------------------------------
# 推理器
# ---------------------------------------------------------------------------
class Predictor:
    """加载训练快照，执行“无标注影像 -> 有符号场 -> 道路掩码”的推理。

    推理只读取影像，不读取掩码/骨架。主链路直接使用扩散采样的单通道有符号
    道路山脊场，并调用 ``RoadSAM.decode_heatmaps``；不构建或调用
    ``PeakExtractor``。生成器种子按 ``image_id`` 固定，因此同一影像的结果与
    目录遍历顺序、batch 划分无关。
    """

    def __init__(
        self,
        config: Mapping[str, Any],
        checkpoint: Optional[Union[str, Path]] = None,
        device: Optional[Union[str, torch.device]] = None,
        config_dir: Optional[Union[str, Path]] = None,
        threshold: Optional[float] = None,
        seed: Optional[int] = None,
        *,
        require_checkpoint: bool = True,
        image_size: int = 1024,
        payload: Optional[Mapping[str, Any]] = None,
        migrate_weights: bool = False,
    ) -> None:
        self.config = dict(config)
        self.config_dir = (
            Path(config_dir).expanduser().resolve()
            if config_dir is not None
            else None
        )
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.seed = int(seed if seed is not None else config.get("seed", 42))
        self.image_size = int(image_size)
        self.amp = bool(
            config.get("train", {}).get("amp", True)
            and self.device.type == "cuda"
        )
        self.threshold = float(
            threshold
            if threshold is not None
            else config.get("inference", {}).get("threshold", 0.5)
        )
        self.migrate_weights = bool(migrate_weights)
        self.prompt_scale, self.terminal_mapping = resolve_prompt_mapping(config)
        self.prompt_mode = "signed_ridge"
        # 历史属性名保留，便于外部诊断代码识别新协议（恒为 True）。
        self.joint_abc = True

        # 先读取并校验 checkpoint 协议，避免为旧快照构建/加载模型。
        checkpoint_payload: Optional[Mapping[str, Any]] = None
        if payload is not None:
            if not isinstance(payload, Mapping):
                raise ValueError("payload 必须是映射")
            checkpoint_payload = payload
        elif checkpoint is not None:
            loaded = read_checkpoint(checkpoint)
            if not isinstance(loaded, Mapping):
                raise ValueError(f"checkpoint 顶层必须是映射：{checkpoint}")
            checkpoint_payload = loaded
        elif require_checkpoint:
            raise FileNotFoundError(
                "推理需要训练 checkpoint；请通过 --checkpoint 指定 best.pt/last.pt"
            )

        if checkpoint_payload is not None and not self.migrate_weights:
            validate_checkpoint_protocol(checkpoint_payload, self.config)
        if self.migrate_weights and not uses_signed_ridge_protocol(self.config):
            # 显式迁移允许旧 checkpoint 内的旧配置，但模型结构必须按新协议
            # 构建；只补协议字段，不迁移优化器/旧点提示设置。
            warnings.warn(
                "--migrate-weights：旧配置缺少 signed_ridge_v1 协议字段，"
                "将按新协议构建模型并只迁移形状兼容权重。",
                RuntimeWarning,
                stacklevel=2,
            )
            self.config = copy.deepcopy(self.config)
            self.config["training_protocol"] = SIGNED_RIDGE_PROTOCOL
            self.config.setdefault("target_representation", TARGET_REPRESENTATION)
            self.config.setdefault("sam_frozen", True)
        validate_signed_ridge_config(
            self.config, require_protocol=not self.migrate_weights
        )

        checkpoint_path = resolve_sam_checkpoint(config, config_dir)
        self.models, self.aux_model = build_models(
            config,
            self.device,
            sam_checkpoint=checkpoint_path,
            config_dir=config_dir,
        )
        self.checkpoint_report: Dict[str, Any] = {}
        if checkpoint_payload is not None:
            if self.migrate_weights:
                self.checkpoint_report = load_compatible_weights(
                    self.models, checkpoint_payload, include_ref=False
                )
            else:
                self.checkpoint_report = load_checkpoint_models(
                    self.models,
                    checkpoint_payload,
                    include_ref=False,
                    validate_protocol=False,
                )
            saved_config = checkpoint_payload.get("config")
            if isinstance(saved_config, Mapping) and not self.migrate_weights:
                warn_protocol_differences(self.config, saved_config)

        for module in self.models.values():
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad_(False)

        self.sam = self.models["sam"]
        self.condition_encoder = self.models["condition_encoder"]
        self.dit = self.models["dit"]
        self.diffusion = self.models["diffusion"]
        # 真实 RoadSAM 在 decode_heatmaps 内部执行 U = prompt_scale * H；
        # Predictor 只负责先 clamp(H)，由模型施加尺度。轻量测试替身没有
        # prompt_scale 时由 Predictor 自己乘以配置尺度。
        self.sam_applies_prompt_scale = hasattr(self.sam, "prompt_scale")

    def _autocast(self):
        """与训练/验证一致的 AMP 协议：CUDA 上覆盖 encode/condition/sample/decode。"""
        if not self.amp:
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=torch.float16)

    def _prompt_field(self, field: torch.Tensor) -> torch.Tensor:
        """构造 ``U = scale * clamp(H)``；先裁剪再由 SAM/预测器施加尺度。"""
        if field.ndim != 4 or int(field.shape[1]) != 1:
            raise ValueError(
                "扩散输出必须是 B x 1 x H x W 的单通道有符号场，"
                f"实际为 {tuple(field.shape)}"
            )
        prompt = apply_terminal_mapping(field, self.terminal_mapping)
        if not getattr(self, "sam_applies_prompt_scale", False):
            prompt = prompt * float(self.prompt_scale)
        return prompt

    @torch.no_grad()
    def predict_array(
        self,
        image: Union[np.ndarray, torch.Tensor, Image.Image],
        image_id: Optional[str] = None,
        seed: Optional[int] = None,
        *,
        return_field: bool = True,
    ) -> Dict[str, Any]:
        """对单张原图执行完整推理。

        Returns:
            包含 ``mask``、``probability``、``field``（``1 x h x w`` 有符号场）
            等字段；``point_count`` 在新协议下恒为 0。
        """
        tensor, original_size, resized_size = prepare_image(
            image, image_size=self.image_size
        )
        tensor = tensor.to(self.device)
        valid_sizes = [(int(resized_size[0]), int(resized_size[1]))]

        base_seed = int(seed if seed is not None else self.seed)
        identifier = canonical_image_id(image_id)
        generator = make_generator(evaluation_seed(base_seed, identifier))

        with self._autocast():
            embeddings = encode_images(self.sam, tensor, valid_sizes)
            # 传入完整特征容器：开启高分辨率融合时条件编码器直接使用
            # SAM2 的原始 FPN 1/4、1/8 特征。
            conditions, _ = self.condition_encoder(tensor, embeddings)
            conditions = [item.detach() for item in conditions]
            field = self.diffusion.sample(
                self.dit,
                conditions,
                generator=generator,
                record=False,
            )
            prompt_field = self._prompt_field(field)
            # 与训练同一路径：单通道有符号场作为冻结 SAM 提示编码器的密集
            # 掩码提示，所有样本（包括空场）都参与解码。
            logits = self.sam.decode_heatmaps(
                embeddings,
                prompt_field,
                output_size=(self.image_size, self.image_size),
                valid_sizes=valid_sizes,
            )
            probability = torch.sigmoid(logits[0, 0])

        probability_np = crop_resize_probability(
            probability, original_size, resized_size
        )
        # 与 D/engine 的确定性评分一致：概率 >= threshold 记为前景。
        mask = probability_np >= self.threshold

        result: Dict[str, Any] = {
            "image_id": identifier,
            "mask": (mask.astype(np.uint8) * 255),
            "probability": probability_np,
            # 密集有符号场提示不再有离散点计数（点成本恒为 0）。
            "original_size": (int(original_size[0]), int(original_size[1])),
            "resized_size": (int(resized_size[0]), int(resized_size[1])),
            "padding": (
                int(self.image_size - resized_size[1]),
                int(self.image_size - resized_size[0]),
            ),
            "threshold": float(self.threshold),
            "prompt_mode": self.prompt_mode,
            "target_representation": TARGET_REPRESENTATION,
            "prompt_scale": float(self.prompt_scale),
            "terminal_mapping": self.terminal_mapping,
        }
        if return_field:
            # 单样本去除 batch 维：字段诊断统一使用 1 x H x W 的热图约定。
            result["field"] = field[0].detach().float().cpu().numpy()
            result["prompt_field"] = (
                prompt_field[0].detach().float().cpu().numpy()
            )
        return result

    def predict_file(
        self,
        path: Union[str, Path],
        image_id: Optional[str] = None,
        seed: Optional[int] = None,
        *,
        return_field: bool = True,
    ) -> Dict[str, Any]:
        """从文件读取影像并推理；不会读取同名掩码。"""
        path = Path(path)
        image = load_image_file(path)
        # 与训练验证/evaluate 使用同一 canonical id，避免 ``scene_sat.jpg``
        # 和 ``scene.jpg`` 得到不同扩散种子。
        identifier = (
            str(image_id) if image_id is not None else canonical_image_id(path.stem)
        )
        result = self.predict_array(
            image,
            image_id=identifier,
            seed=seed,
            return_field=return_field,
        )
        result["source_path"] = str(path)
        return result
