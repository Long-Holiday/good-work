"""DiffPeak-SAM 训练入口。

用法示例（在项目根目录执行）::

    # 顺序执行 A -> B -> C -> D
    python train.py --config configs/3090.yaml --stage all

    # 只跑 B，并用 A 阶段权重初始化
    python train.py --config configs/3090.yaml --stage B --init runs/diffpeak/stage_A.pth

    # 恢复某阶段中断前的 last.pt（不传 --config 时使用 checkpoint 内配置）
    python train.py --resume runs/diffpeak/last.pt

    # 显式修改路径等配置；resume 的优化器及学习率曲线仍以 checkpoint 为准
    python train.py --config configs/3090.yaml --resume runs/diffpeak/last.pt

``--resume`` 恢复模型、优化器、GradScaler、学习率调度、epoch 与 RNG；
``--init`` 只加载模型权重，从指定阶段的第 0 个 epoch 开始。
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from diffpeak.engine import STAGE_ORDER, DiffPeakEngine
from diffpeak.experiment_logging import ExperimentLogger
from diffpeak.inference import (
    config_from_checkpoint,
    find_config_path,
    load_config,
    read_checkpoint,
    resolve_output_dir,
    seed_everything,
)


def _positive_int(value: str) -> int:
    """argparse 类型：只接受正整数。"""
    try:
        number = int(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError(f"必须是正整数，收到：{value!r}")
    if number <= 0:
        raise argparse.ArgumentTypeError(f"必须是正整数，收到：{value!r}")
    return number


def _apply_thread_settings(
    cpu_threads: Optional[int], interop_threads: Optional[int]
) -> Dict[str, Optional[int]]:
    """按需设置 torch 线程数，并返回用于实验元数据的请求值/实际值。

    仅在显式传入参数时调用对应的 torch setter，因此不传 flag 时保持
    PyTorch 默认行为不变。此函数必须在模型/engine 构造前调用，尤其是
    ``set_num_interop_threads`` 需要在任何并行工作开始前设置。
    """
    if cpu_threads is not None:
        torch.set_num_threads(cpu_threads)
    if interop_threads is not None:
        torch.set_num_interop_threads(interop_threads)
    return {
        "cpu_threads_requested": cpu_threads,
        "cpu_threads_actual": int(torch.get_num_threads()),
        "interop_threads_requested": interop_threads,
        "interop_threads_actual": int(torch.get_num_interop_threads()),
    }


def _parse_stages(value: str) -> List[str]:
    if value is None or value.strip().lower() == "all":
        return list(STAGE_ORDER)
    requested = {
        part.strip().upper() for part in value.split(",") if part.strip()
    }
    unknown = requested - set(STAGE_ORDER)
    if unknown:
        raise ValueError(f"未知阶段：{sorted(unknown)}")
    # 固定按 A/B/C/D 顺序执行，避免跨阶段依赖被颠倒。
    return [stage for stage in STAGE_ORDER if stage in requested]


def _load_training_config(args: argparse.Namespace):
    """显式 --config 优先；否则 resume/init 使用 checkpoint 内配置。

    返回 ``(config, config_dir, preloaded_payload)``；从 checkpoint 读取配置时
    顺便返回已加载快照，避免训练入口再读取一次整份权重。
    """
    if args.config is not None:
        return load_config(args.config), Path(args.config).expanduser().parent, None
    if args.resume is not None:
        payload = read_checkpoint(args.resume)
        config = config_from_checkpoint(args.resume, payload=payload)
        return config, None, payload
    if args.init is not None:
        payload = read_checkpoint(args.init)
        config = config_from_checkpoint(args.init, payload=payload)
        return config, None, payload
    return load_config(None), find_config_path(None).parent, None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="YAML 配置；省略时 resume/init 读取 checkpoint 内配置，否则使用默认配置。",
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        help="all 或逗号分隔的阶段子集，例如 A、B,C、all。",
    )
    parser.add_argument("--resume", type=str, default=None, help="完整快照，用于阶段内恢复。")
    parser.add_argument("--init", type=str, default=None, help="仅加载模型权重的上一阶段快照。")
    parser.add_argument("--device", type=str, default=None, help="例如 cuda、cuda:0、cpu。")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="显式覆盖配置中的 seed（默认不覆盖）。",
    )
    parser.add_argument(
        "--cpu-threads",
        type=_positive_int,
        default=None,
        metavar="N",
        help="设置 torch 的 CPU 运算线程数（默认沿用 PyTorch 的自动选择）。",
    )
    parser.add_argument(
        "--interop-threads",
        type=_positive_int,
        default=None,
        metavar="N",
        help="设置 torch 的 inter-op 线程数（默认沿用 PyTorch 的自动选择）。",
    )
    parser.add_argument("--output-dir", type=str, default=None, help="覆盖输出目录（相对于当前工作目录）。")
    parser.add_argument("--experiment-name", type=str, default=None, help="在输出目录下创建命名实验子目录。")
    parser.add_argument("--no-tensorboard", action="store_true", help="关闭 TensorBoard，仍保存文本和 JSONL 日志。")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.resume is not None and args.init is not None:
        parser.error("--resume 与 --init 不能同时使用")

    # 必须在 ExperimentLogger / engine（及其模型）构造前设置 torch 线程数。
    thread_metadata = _apply_thread_settings(args.cpu_threads, args.interop_threads)

    config, config_dir, preloaded = _load_training_config(args)
    if args.seed is not None:
        config["seed"] = int(args.seed)

    seed = int(config.get("seed", 42))
    seed_everything(seed, deterministic=bool(config.get("train", {}).get("deterministic", False)))

    if args.output_dir is not None:
        config["output_dir"] = str(Path(args.output_dir).expanduser().resolve())
    if args.experiment_name is not None:
        name = args.experiment_name.strip()
        if not name or name in (".", "..") or "/" in name or "\\" in name:
            parser.error("--experiment-name 必须是非空目录名，不能包含路径分隔符")
        if args.resume is not None:
            parser.error("--resume 请直接使用 checkpoint 中的实验目录，或通过 --output-dir 指定目录")
        config["output_dir"] = str(resolve_output_dir(config, config_dir) / name)
    if args.no_tensorboard:
        config.setdefault("logging", {})["tensorboard"] = False

    with ExperimentLogger(
        resolve_output_dir(config, config_dir), config,
        metadata={"resume": args.resume, "init": args.init, "requested_stages": args.stage,
                  "config_source": args.config, "device_requested": args.device,
                  **thread_metadata},
    ) as experiment:
        engine = DiffPeakEngine(config, device=args.device, config_dir=config_dir)
        engine.experiment_logger = experiment
        experiment.save_config(engine._resolved_config())
        experiment.metadata.update(device=str(engine.device), seed=seed)
        experiment.event("engine_ready", device=str(engine.device), seed=seed)
        _run_training(parser, args, config, preloaded, engine, experiment)


def _run_training(parser, args, config, preloaded, engine, experiment) -> None:
    resume_payload: Optional[Dict[str, Any]] = None
    resume_stage: Optional[str] = None
    resume_epoch = 0

    if args.init is not None:
        if preloaded is not None and args.config is None:
            engine.load_models_from_payload(preloaded, include_ref=True)
            print(f"已从 {args.init} 初始化模型权重（不含优化器状态）", flush=True)
            preloaded = None
        else:
            engine.load_init_weights(args.init)
    if args.resume is not None:
        if preloaded is not None and args.config is None:
            resume_payload = preloaded
        else:
            resume_payload = engine.read_checkpoint(args.resume)
        if not isinstance(resume_payload, dict):
            raise ValueError("--resume 必须指向本项目生成的完整训练快照")
        # Also validate already-completed stages, which bypass run_stage below.
        engine._validate_resume_algorithm(resume_payload)
        # run_stage 自身会严格加载模型权重（含 D 的 ref_dit）；
        # 这里不要预加载或删除 payload.models，避免重复加载/破坏严格恢复。
        resume_stage = str(resume_payload.get("stage", "")).upper()
        resume_epoch = int(resume_payload.get("epoch", 0))
        experiment.metadata.update(resume_stage=resume_stage, resume_epoch=resume_epoch,
                                   resume_global_step=resume_payload.get("global_step"))
        experiment.event("resume", checkpoint=args.resume, stage=resume_stage, epoch=resume_epoch,
                         global_step=resume_payload.get("global_step"))
        if resume_stage not in STAGE_ORDER:
            raise ValueError(f"checkpoint 中的阶段无效：{resume_stage!r}")

    selected = _parse_stages(args.stage)
    if resume_stage is not None:
        if resume_stage not in selected:
            if args.stage.strip().lower() != "all":
                parser.error(
                    f"checkpoint 属于阶段 {resume_stage}，不在 --stage {args.stage} 中"
                )
            selected = list(STAGE_ORDER)
        selected = [stage for stage in selected if STAGE_ORDER.index(stage) >= STAGE_ORDER.index(resume_stage)]

    experiment.metadata["selected_stages"] = selected
    print(
        f"配置来源：{'显式 --config' if args.config is not None else ('checkpoint' if args.resume or args.init else '默认配置')}；"
        f"输出目录：{engine.output_dir}",
        flush=True,
    )
    print(f"执行阶段：{','.join(selected)}", flush=True)

    has_weights = args.init is not None or args.resume is not None
    if not has_weights and selected and selected[0] != "A":
        warnings.warn(
            f"阶段 {selected[0]} 没有 --init/--resume，将从当前初始化权重开始。",
            RuntimeWarning,
        )

    try:
        for stage in selected:
            if resume_stage is not None and stage == resume_stage:
                epochs = int(config.get("stages", {}).get(stage, {}).get("epochs", 0))
                if resume_epoch >= epochs:
                    # 该阶段已完成，但后续阶段仍必须从它的权重继续；
                    # engine.run_stage 不会被调用，所以这里先恢复一次模型。
                    if resume_payload is None:
                        raise RuntimeError("已完成的 resume 阶段缺少模型 payload")
                    engine.load_models_from_payload(
                        resume_payload, include_ref=True
                    )
                    # 下一阶段使用新优化器，但保留历史指标与 RNG 连续性。
                    engine.history = list(resume_payload.get("history", []))
                    engine._restore_rng(resume_payload.get("rng"))
                    resume_payload = None
                    resume_stage = None
                    print(
                        f"[{stage}] checkpoint 已完成该阶段（epoch={resume_epoch}/{epochs}），"
                        "已恢复权重并继续后续阶段",
                        flush=True,
                    )
                    continue
                engine.run_stage(
                    stage, start_epoch=resume_epoch, resume_payload=resume_payload
                )
                resume_payload = None
            else:
                engine.run_stage(stage, start_epoch=0)
    except KeyboardInterrupt:
        # 不把当前 epoch 的部分更新写成 last.pt：last.pt 仍对应上一个完整
        # epoch，其优化器、调度器与 RNG 状态一致，可安全 --resume。
        print(
            "训练被中断。last.pt 保留上一个完整 epoch 的检查点；"
            "当前 epoch 已进行的更新将丢失（最多一个 epoch）。",
            flush=True,
        )
        raise SystemExit(130)

    print(
        f"训练结束。checkpoint：{engine.output_dir / 'last.pt'}、"
        f"{engine.output_dir / 'best.pt'}、stage_X.pth",
        flush=True,
    )


if __name__ == "__main__":
    main()
