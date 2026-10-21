"""RTX 3090 single-candidate hyperparameter probe using the real engine.

The probe runs a bounded number of optimizer updates through the actual
``DiffPeakEngine._prepare_stage`` and ``_train_epoch_supervised`` /
``_train_epoch_grpo`` methods. It only trains on a deterministic subset of the
existing canonical training split; validation/test paths are rejected. Each run
writes to a new, non-empty-rejecting output directory so a previous experiment
is never overwritten.

Example::

    python scripts/probe_3090.py --output-dir runs/probes/b1 \
        --config configs/3090.yaml --stage B --batch-size 1 \
        --accumulation-steps 4 --num-workers 2 --warmup-updates 2 \
        --measure-updates 4

CPU thread defaults (2 intra-op, 1 inter-op) avoid OpenMP oversubscription when
the host cgroup exposes far more CPUs than the allocation really allows.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence

PROBE_FORMAT_VERSION = 1
GPU_TELEMETRY_INTERVAL_SECONDS = 0.25
FORBIDDEN_TRAIN_COMPONENTS = {"test", "tests", "valid", "validation", "val"}
THREAD_ENV_KEYS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def _prescan_threads(argv: Sequence[str]) -> tuple[int, int]:
    """Read thread flags before importing torch so env defaults are early."""
    def value(flag: str, default: int) -> int:
        for index, argument in enumerate(argv):
            raw = None
            if argument == flag and index + 1 < len(argv):
                raw = argv[index + 1]
            elif argument.startswith(flag + "="):
                raw = argument.split("=", 1)[1]
            if raw is not None:
                try:
                    return max(1, int(raw))
                except ValueError:
                    return default
        return default

    return value("--cpu-threads", 2), value("--interop-threads", 1)


_PRESCAN_CPU, _PRESCAN_INTEROP = _prescan_threads(sys.argv[1:])
for _key in THREAD_ENV_KEYS:
    # Early default so torch/OpenMP does not latch the host-wide value during
    # import. configure_threads() later applies the CLI values (including their
    # defaults, which are deliberate) to both the environment and torch itself.
    os.environ.setdefault(_key, str(_PRESCAN_CPU))

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import torch  # noqa: E402

from gpu_telemetry import GpuTelemetrySampler, unavailable_summary  # noqa: E402

from diffpeak.data.splits import load_split_entries, sample_id_from_image_path  # noqa: E402
from diffpeak.engine import STAGE_ORDER, DiffPeakEngine  # noqa: E402
from diffpeak.inference import (  # noqa: E402
    load_config,
    resolve_data_root,
    resolve_sam_checkpoint,
    resolve_split_file,
)


# ---------------------------------------------------------------------------
# JSON / logging helpers
# ---------------------------------------------------------------------------
def jsonable(value: Any) -> Any:
    """Convert arbitrary values to strict-JSON-safe data (NaN/inf -> null)."""
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(item) for item in value]
    if isinstance(value, (str, bool)) or value is None:
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        try:
            return jsonable(value.item())
        except (TypeError, ValueError, RuntimeError):
            # Multi-element tensors/arrays cannot be converted to a scalar.
            pass
    return str(value)


def write_json(path: Path, payload: Any) -> None:
    """Atomically write strict JSON (never allows NaN/Infinity literals)."""
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(jsonable(payload), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    temporary.replace(path)


class RawLog:
    """Append-only JSONL event log; used for raw batch/fetch timing records."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a", encoding="utf-8")

    def event(self, name: str, **fields: Any) -> None:
        record = {"event": name, "wall_time": time.time(), **fields}
        self._handle.write(
            json.dumps(jsonable(record), sort_keys=True, allow_nan=False) + "\n"
        )
        self._handle.flush()

    def sink(self, record: Mapping[str, Any]) -> None:
        self.event("batch", **dict(record))

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> "RawLog":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Bounded loader and split guardrails
# ---------------------------------------------------------------------------
class BoundedIterable:
    """Yield at most ``max_items`` items, optionally cycling, and time fetches.

    ``fetch_seconds`` separates dataloader waiting from model compute. The
    wrapper is the single place that bounds a probe phase; the engine still
    performs its normal micro-batch grouping and gradient accumulation.
    """

    def __init__(
        self,
        source: Iterable[Any],
        max_items: int,
        *,
        cycle: bool = True,
        phase: str = "",
        sink: Optional[Callable[[Mapping[str, Any]], None]] = None,
    ) -> None:
        if max_items < 0:
            raise ValueError("max_items 不能为负")
        self.source = source
        self.max_items = int(max_items)
        self.cycle = bool(cycle)
        self.phase = phase
        self.sink = sink
        self.fetch_seconds = 0.0
        self.items_yielded = 0
        self.samples_yielded = 0
        self._samples_detected = False

    def counted_samples(self, fallback_batch_size: int) -> int:
        """Return images yielded, falling back to items * batch size.

        Real engine batches are mappings with an ``image`` tensor, so partial
        final batches are counted exactly. Test sources without images keep
        the older items * batch_size estimate.
        """
        if self._samples_detected:
            return self.samples_yielded
        return self.items_yielded * max(1, int(fallback_batch_size))

    def __len__(self) -> int:
        return self.max_items

    def __iter__(self) -> Iterator[Any]:
        produced = 0
        empty_restarts = 0
        source_iterator: Optional[Iterator[Any]] = None
        while produced < self.max_items:
            if source_iterator is None:
                source_iterator = iter(self.source)
            started = time.perf_counter()
            try:
                item = next(source_iterator)
            except StopIteration:
                if not self.cycle:
                    break
                source_iterator = None
                empty_restarts += 1
                if empty_restarts > 1:
                    raise ValueError("BoundedIterable 的源为空，无法循环取样")
                continue
            self.fetch_seconds += time.perf_counter() - started
            produced += 1
            self.items_yielded = produced
            image = item.get("image") if isinstance(item, dict) else None
            if torch.is_tensor(image) and image.ndim >= 1:
                self._samples_detected = True
                self.samples_yielded += int(image.shape[0])
            empty_restarts = 0
            if self.sink is not None:
                self.sink(
                    {
                        "phase": self.phase,
                        "index": produced,
                        "fetch_seconds": self.fetch_seconds,
                    }
                )
            yield item


def _entry_key(entry: str) -> str:
    """Normalize separators and a leading ``./`` only.

    Leading ``/`` and ``..`` segments are deliberately preserved so that
    :func:`assert_train_entries_allowed` can still reject absolute paths and
    parent-directory escapes. ``str.lstrip("./")`` would silently strip them
    and turn an absolute entry into a path relative to the data root.
    """
    text = str(entry).replace("\\", "/")
    while text.startswith("./"):
        text = text[2:]
    return text


def _sample_key(entry: str) -> str:
    try:
        return sample_id_from_image_path(Path(_entry_key(entry)))
    except ValueError:
        return _entry_key(entry)


def assert_train_entries_allowed(entries: Sequence[str]) -> None:
    """Reject training entries that point into official test/validation dirs."""
    for entry in entries:
        text = str(entry).replace("\\", "/")
        path = Path(text)
        if path.is_absolute():
            raise ValueError(f"训练划分项必须是相对路径：{entry}")
        if ".." in path.parts:
            raise ValueError(f"训练划分项不能包含 '..'：{entry}")
        forbidden = [
            part for part in path.parts if part.lower() in FORBIDDEN_TRAIN_COMPONENTS
        ]
        if forbidden:
            raise ValueError(
                f"训练划分包含 test/validation 路径分量 {forbidden}：{entry}。"
                "官方 valid/test 只有影像没有掩码，且绝不能用于训练。"
            )


def select_train_subset(
    entries: Sequence[str],
    count: int,
    seed: int,
    val_entries: Sequence[str] = (),
) -> List[str]:
    """Deterministically pick ``count`` canonical train entries.

    Selection uses a seeded BLAKE2b rank, independent of filesystem order.
    The function refuses absolute paths, forbidden directory components, and
    any overlap with the validation split.
    """
    if count < 1:
        raise ValueError("train_samples 必须 >= 1")
    normalized = [_entry_key(entry) for entry in entries]
    assert_train_entries_allowed(normalized)

    val_paths = {_entry_key(entry) for entry in val_entries}
    val_ids = {_sample_key(entry) for entry in val_entries}
    overlap = [
        entry
        for entry in normalized
        if _entry_key(entry) in val_paths or _sample_key(entry) in val_ids
    ]
    if overlap:
        raise ValueError(
            "训练划分与验证划分重叠，拒绝训练：" + ", ".join(overlap[:5])
        )
    if int(count) >= len(normalized):
        # Preserve the canonical JSON order when the request clamps to all.
        return normalized

    def rank(entry: str) -> bytes:
        payload = f"{int(seed)}\x00{entry}".encode("utf-8")
        return hashlib.blake2b(payload, digest_size=8).digest()

    ordered = sorted(normalized, key=lambda entry: (rank(entry), entry))
    return ordered[: min(int(count), len(ordered))]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_entries(entries: Sequence[str]) -> str:
    payload = json.dumps(list(entries), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def prepare_output_dir(path: Path) -> Path:
    """Create a unique output directory; refuse to touch prior evidence."""
    output = Path(path).expanduser()
    if output.exists():
        if not output.is_dir():
            raise FileExistsError(f"输出路径已存在且不是目录：{output}")
        if any(output.iterdir()):
            raise FileExistsError(
                f"输出目录非空，拒绝覆盖已有实验证据：{output.resolve()}"
            )
    output.mkdir(parents=True, exist_ok=True)
    return output.resolve()


# ---------------------------------------------------------------------------
# CLI and configuration
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Single-candidate RTX 3090 probe on the real training engine.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output-dir", required=True, help="新的唯一输出目录（非空则拒绝）")
    parser.add_argument("--config", default="configs/3090.yaml", help="基础 YAML 配置")
    parser.add_argument("--data-root", default=None, help="覆盖 data.root")
    parser.add_argument("--sam-checkpoint", default=None, help="覆盖 model.sam_checkpoint")
    parser.add_argument("--stage", required=True, choices=STAGE_ORDER, help="只测一个阶段")
    parser.add_argument("--batch-size", type=int, default=None, help="覆盖 train.batch_size")
    parser.add_argument("--accumulation-steps", type=int, default=None, help="覆盖 train.accumulation_steps（仅 A/B/C）")
    parser.add_argument("--num-workers", type=int, default=None, help="覆盖 data.num_workers")
    parser.add_argument("--gradient-checkpointing", choices=("on", "off"), default=None, help="覆盖 model.gradient_checkpointing")
    parser.add_argument("--lr", type=float, default=None, help="覆盖 stages.<stage>.lr")
    parser.add_argument("--amp-init-scale", type=float, default=None, help="覆盖 AMP 初始 scale（D 用 rl，其余用 train）")
    parser.add_argument("--group-size", type=int, default=None, help="覆盖 rl.group_size（仅 D）")
    parser.add_argument("--sampling-steps", type=int, default=None, help="覆盖 diffusion.sampling_steps")
    parser.add_argument("--init", default=None, help="仅加载模型权重的上一阶段快照")
    parser.add_argument("--seed", type=int, default=None, help="覆盖 seed")
    parser.add_argument("--warmup-updates", type=int, default=2, help="测量前的预热优化器更新数（D 为轨迹组数）")
    parser.add_argument("--measure-updates", type=int, default=4, help="计入报告的更新预算（A/B/C 为优化器组数，D 为轨迹组数）")
    parser.add_argument("--train-split", default=None, help="覆盖 data.train_split（仍会做训练来源校验）")
    parser.add_argument("--val-split", default=None, help="覆盖 data.val_split")
    parser.add_argument("--train-samples", type=int, default=32, help="从 canonical train JSON 确定性抽取的样本数")
    parser.add_argument("--validate", action="store_true", help="在固定验证子集上调用 engine.validate")
    parser.add_argument("--val-samples", type=int, default=16, help="验证子集样本数（配合 --validate；只迭代该子集）")
    parser.add_argument(
        "--save-weights",
        default=None,
        metavar="PATH",
        help="可选：探针结束保存 model-only checkpoint（models=engine._model_states()，"
        "D 保留 ref_unet）；相对路径相对于 --output-dir，供 --init 串联开发 bootstrap。",
    )
    parser.add_argument("--device", default=None, help="torch 设备，默认 cuda（不可用时 cpu）")
    parser.add_argument("--cpu-threads", type=int, default=2, help="torch 主进程 intra-op 线程数")
    parser.add_argument("--interop-threads", type=int, default=1, help="torch inter-op 线程数")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.train_samples < 1:
        raise ValueError("--train-samples 必须 >= 1")
    if args.val_samples < 1:
        raise ValueError("--val-samples 必须 >= 1")
    if args.warmup_updates < 0:
        raise ValueError("--warmup-updates 必须 >= 0")
    if args.measure_updates < 1:
        raise ValueError("--measure-updates 必须 >= 1")
    for name in ("batch_size", "accumulation_steps", "group_size"):
        value = getattr(args, name)
        if value is not None and value < 1:
            raise ValueError(f"--{name.replace('_', '-')} 必须 >= 1")
    if args.num_workers is not None and args.num_workers < 0:
        raise ValueError("--num-workers 必须 >= 0")
    if args.sampling_steps is not None and args.sampling_steps < 1:
        raise ValueError("--sampling-steps 必须 >= 1")
    if args.cpu_threads < 1 or args.interop_threads < 1:
        raise ValueError("线程数必须 >= 1")
    if args.lr is not None and (not math.isfinite(args.lr) or args.lr < 0):
        raise ValueError("--lr 必须是非负有限数")
    if args.amp_init_scale is not None and (
        not math.isfinite(args.amp_init_scale) or args.amp_init_scale <= 0
    ):
        raise ValueError("--amp-init-scale 必须是正有限数")


def apply_overrides(config: Mapping[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Apply CLI overrides in memory only; config files on disk are untouched."""
    resolved = copy.deepcopy(dict(config))
    data = resolved.setdefault("data", {})
    train = resolved.setdefault("train", {})
    model = resolved.setdefault("model", {})
    diffusion = resolved.setdefault("diffusion", {})
    rl = resolved.setdefault("rl", {})

    if args.seed is not None:
        resolved["seed"] = int(args.seed)
    if args.data_root is not None:
        data["root"] = str(args.data_root)
    if args.sam_checkpoint is not None:
        model["sam_checkpoint"] = str(args.sam_checkpoint)
    if args.train_split is not None:
        data["train_split"] = str(args.train_split)
    if args.val_split is not None:
        data["val_split"] = str(args.val_split)
    if args.num_workers is not None:
        data["num_workers"] = int(args.num_workers)
    if args.batch_size is not None:
        train["batch_size"] = int(args.batch_size)
    if args.accumulation_steps is not None:
        train["accumulation_steps"] = int(args.accumulation_steps)
    if args.gradient_checkpointing is not None:
        model["gradient_checkpointing"] = args.gradient_checkpointing == "on"
    if args.sampling_steps is not None:
        diffusion["sampling_steps"] = int(args.sampling_steps)
    if args.group_size is not None:
        rl["group_size"] = int(args.group_size)
    if args.amp_init_scale is not None:
        target = rl if args.stage == "D" else train
        target["amp_init_scale"] = float(args.amp_init_scale)
    if args.lr is not None:
        stages = resolved.setdefault("stages", {})
        stages.setdefault(args.stage, {})["lr"] = float(args.lr)
    return resolved


def configure_threads(args: argparse.Namespace) -> Dict[str, Any]:
    """Bound CPU threading and record the actual runtime values."""
    for key in THREAD_ENV_KEYS:
        os.environ[key] = str(int(args.cpu_threads))
    torch.set_num_threads(int(args.cpu_threads))
    interop_error = None
    try:
        torch.set_num_interop_threads(int(args.interop_threads))
    except RuntimeError as exc:  # already initialized by a previous library
        interop_error = str(exc)
    return {
        "requested": {
            "cpu_threads": int(args.cpu_threads),
            "interop_threads": int(args.interop_threads),
        },
        "actual": {
            "intra_op_threads": int(torch.get_num_threads()),
            "interop_threads": int(torch.get_num_interop_threads()),
        },
        "environment": {key: os.environ.get(key) for key in THREAD_ENV_KEYS},
        "interop_error": interop_error,
    }


# ---------------------------------------------------------------------------
# Inputs, splits, environment metadata
# ---------------------------------------------------------------------------
def resolve_inputs(
    config: Mapping[str, Any],
    config_dir: Optional[Path],
    args: argparse.Namespace,
    output_dir: Path,
    raw: Optional[RawLog] = None,
) -> Dict[str, Any]:
    """Resolve paths and materialize guarded train/validation subsets."""
    data_root = Path(resolve_data_root(config, config_dir)).expanduser().resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"数据根目录不存在：{data_root}")
    sam_checkpoint = Path(
        resolve_sam_checkpoint(config, config_dir)
    ).expanduser().resolve()

    train_split = Path(
        resolve_split_file(
            config["data"]["train_split"], data_root, config_dir, must_exist=True
        )
    ).expanduser().resolve()
    raw_val = config.get("data", {}).get("val_split")
    val_split: Optional[Path] = None
    if raw_val not in (None, ""):
        val_split = Path(
            resolve_split_file(raw_val, data_root, config_dir, must_exist=True)
        ).expanduser().resolve()
        if val_split == train_split:
            raise ValueError(
                "data.train_split 与 data.val_split 指向同一文件，拒绝训练/验证混用"
            )
    if args.validate and val_split is None:
        raise ValueError("--validate 需要可用的 data.val_split 或 --val-split")

    train_entries = load_split_entries(train_split)
    val_entries = load_split_entries(val_split) if val_split is not None else []
    subset = select_train_subset(
        train_entries, args.train_samples, int(config.get("seed", 42)), val_entries
    )

    train_subset_path = output_dir / "train_subset.json"
    write_json(train_subset_path, subset)
    config["data"]["train_split"] = str(train_subset_path)

    val_subset: Optional[List[str]] = None
    val_subset_path: Optional[Path] = None
    if val_split is not None:
        val_subset = select_train_subset(
            val_entries, min(args.val_samples, len(val_entries)),
            int(config.get("seed", 42)),
        )
        val_subset_path = output_dir / "val_subset.json"
        write_json(val_subset_path, val_subset)
        config["data"]["val_split"] = str(val_subset_path)

    split_record: Dict[str, Any] = {
        "train": {
            "source_path": str(train_split),
            "source_sha256": sha256_file(train_split),
            "source_count": len(train_entries),
            "subset_count": len(subset),
            "subset_path": str(train_subset_path),
            # File hash so the recorded value can be checked against the
            # written artifact; the logical entry-list hash is kept separately.
            "subset_sha256": sha256_file(train_subset_path),
            "subset_entries_sha256": sha256_entries(subset),
            "subset_ids": [_sample_key(entry) for entry in subset],
            "selection": "seeded_blake2b_rank",
            "seed": int(config.get("seed", 42)),
        },
        "val": (
            {
                "source_path": str(val_split),
                "source_sha256": sha256_file(val_split),
                "source_count": len(val_entries),
                "subset_count": len(val_subset),
                "subset_path": str(val_subset_path),
                "subset_sha256": sha256_file(val_subset_path),
                "subset_entries_sha256": sha256_entries(val_subset),
                "subset_ids": [_sample_key(entry) for entry in val_subset],
                "selection": "seeded_blake2b_rank",
                "seed": int(config.get("seed", 42)),
                "subset_used_for_training": False,
            }
            if val_split is not None and val_subset is not None and val_subset_path
            else None
        ),
    }
    write_json(output_dir / "splits.json", split_record)
    if raw is not None:
        raw.event(
            "splits_ready",
            train_source=str(train_split),
            train_source_count=len(train_entries),
            train_subset_count=len(subset),
            val_source=None if val_split is None else str(val_split),
        )
    return {
        "data_root": data_root,
        "sam_checkpoint": sam_checkpoint,
        "train_split": train_split,
        "val_split": val_split,
        "train_entries": train_entries,
        "val_entries": val_entries,
        "train_subset": subset,
        "val_subset": val_subset,
        "splits": split_record,
    }


def _read_text(path: str) -> Optional[str]:
    try:
        return Path(path).read_text(encoding="utf-8").strip()
    except OSError:
        return None


def resource_metadata() -> Dict[str, Any]:
    """Best-effort cgroup/affinity/RSS metadata for oversubscription audits."""
    result: Dict[str, Any] = {}
    cpu_max = _read_text("/sys/fs/cgroup/cpu.max")
    if cpu_max:
        parts = cpu_max.split()
        result["cpu_max"] = cpu_max
        if len(parts) == 2 and parts[0] != "max":
            try:
                result["cpu_quota_cores"] = int(parts[0]) / int(parts[1])
            except (ValueError, ZeroDivisionError):
                pass
    for key, path in (
        ("memory_max_bytes", "/sys/fs/cgroup/memory.max"),
        ("memory_current_bytes", "/sys/fs/cgroup/memory.current"),
        ("memory_peak_bytes", "/sys/fs/cgroup/memory.peak"),
    ):
        text = _read_text(path)
        if text is not None and text != "max":
            try:
                result[key] = int(text)
            except ValueError:
                pass
    status = _read_text("/proc/self/status") or ""
    for line in status.splitlines():
        if line.startswith("VmRSS:"):
            try:
                result["rss_bytes"] = int(line.split()[1]) * 1024
            except (ValueError, IndexError):
                pass
    if hasattr(os, "sched_getaffinity"):
        try:
            result["affinity_cpus"] = len(os.sched_getaffinity(0))
        except OSError:
            pass
    result["os_cpu_count"] = os.cpu_count()
    return result


def environment_info(args: argparse.Namespace) -> Dict[str, Any]:
    git: Dict[str, Any] = {"commit": None, "branch": None, "dirty": None}
    try:
        git["commit"] = subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            text=True, stderr=subprocess.DEVNULL, timeout=5,
        ).strip()
        git["branch"] = subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "--abbrev-ref", "HEAD"],
            text=True, stderr=subprocess.DEVNULL, timeout=5,
        ).strip()
        git["dirty"] = bool(
            subprocess.check_output(
                ["git", "-C", str(ROOT), "status", "--porcelain"],
                text=True, stderr=subprocess.DEVNULL, timeout=5,
            ).strip()
        )
    except (OSError, subprocess.SubprocessError):
        pass

    cuda: Dict[str, Any] = {
        "available": bool(torch.cuda.is_available()),
        "runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "device_count": torch.cuda.device_count(),
    }
    if cuda["available"]:
        try:
            index = torch.cuda.current_device()
            properties = torch.cuda.get_device_properties(index)
            cuda.update(
                {
                    "device_index": index,
                    "device_name": torch.cuda.get_device_name(index),
                    "capability": list(torch.cuda.get_device_capability(index)),
                    "total_memory_bytes": int(properties.total_memory),
                }
            )
        except Exception as exc:  # pragma: no cover - driver edge cases
            cuda["error"] = f"{type(exc).__name__}: {exc}"

    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": cuda,
        "deterministic_algorithms": (
            torch.are_deterministic_algorithms_enabled()
            if hasattr(torch, "are_deterministic_algorithms_enabled")
            else None
        ),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "git": git,
        "source_sha256": {
            str(path.relative_to(ROOT)): sha256_file(path)
            for path in sorted((ROOT / "diffpeak").rglob("*.py"))
        } | {
            "scripts/probe_3090.py": sha256_file(Path(__file__).resolve()),
            "scripts/gpu_telemetry.py": sha256_file(
                Path(__file__).resolve().with_name("gpu_telemetry.py")
            ),
        },
        "argv": list(sys.argv),
        "threading": {
            "torch_num_threads": int(torch.get_num_threads()),
            "torch_num_interop_threads": int(torch.get_num_interop_threads()),
            "environment": {key: os.environ.get(key) for key in THREAD_ENV_KEYS},
        },
        "resources": resource_metadata(),
        "requested_device": args.device,
    }


# ---------------------------------------------------------------------------
# Probe execution
# ---------------------------------------------------------------------------
def _phase_batches(
    stage: str,
    updates: int,
    accumulation_steps: int,
    batch_size: int = 1,
) -> int:
    """Translate an update budget into bounded loader items (batches).

    A/B/C: one optimizer update is one gradient-accumulation group, so the
    budget is ``updates * accumulation_steps`` micro-batches.

    D: one optimizer update is one GRPO batch (``update_epochs`` steps per
    batch), while each image in that batch is an independent trajectory group.
    ``updates`` therefore denotes trajectory groups and the item budget is
    ``ceil(updates / batch_size)``. A batch cannot be split, so the final
    partial batch may attempt up to ``batch_size - 1`` extra groups; the
    measured summary reports the actual number of groups processed.
    """
    if updates <= 0:
        return 0
    if stage == "D":
        return int(math.ceil(int(updates) / max(1, int(batch_size))))
    return int(updates) * max(1, int(accumulation_steps))


def scaler_state(engine: Any) -> Dict[str, Any]:
    scaler = getattr(engine, "scaler", None)
    if scaler is None:
        return {"present": False}
    try:
        return {
            "present": True,
            "enabled": bool(scaler.is_enabled()),
            "scale": float(scaler.get_scale()),
        }
    except Exception as exc:  # pragma: no cover - defensive
        return {"present": True, "error": f"{type(exc).__name__}: {exc}"}


def _finite_metrics(metrics: Mapping[str, Any]) -> Dict[str, bool]:
    return {
        key: math.isfinite(float(value))
        for key, value in metrics.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }


def _applicable_metric_keys(
    stage: str, engine: Any, stats: Mapping[str, Any]
) -> List[str]:
    """Metrics this stage is expected to produce; absent ones are not failures.

    The engine reports means over per-batch records, so a stage that never
    computes ``aux``/``diff``/``heat`` still leaves those keys as NaN. Those are
    intentionally absent, not non-finite training evidence. Only metrics the
    stage actually computes (and, for D, only enabled KL/anchor/policy terms)
    participate in ``all_metrics_finite``.
    """
    if stage == "A":
        return ["loss", "seg", "points"]
    if stage == "B":
        return ["loss", "diff", "heat", "aux"]
    if stage == "C":
        return ["loss", "seg", "points"]
    keys = [
        "reward", "reward_std", "reward_variance", "iou", "f1", "cldice",
        "point_count", "positive_point_count", "negative_point_count",
        "point_set_diversity", "unique_point_sets", "point_set_identical_ratio",
        "point_cost_scale", "point_cost", "lr", "updates", "skipped_updates",
        "policy_terms",
    ]
    if float(stats.get("policy_terms", 0.0) or 0.0) > 0:
        keys.extend(
            [
                "grpo", "clip_fraction", "logratio_abs_mean",
                "logratio_abs_max", "advantage_abs_mean",
                "active_group_rate", "inactive_group_rate",
            ]
        )
    if float(getattr(engine, "kl_weight", 0.0) or 0.0) > 0:
        keys.append("kl")
    if float(getattr(engine, "anchor_weight", 0.0) or 0.0) > 0:
        keys.append("anchor")
    return keys


def _phase_summary(
    stage: str,
    stats: Mapping[str, Any],
    elapsed: float,
    loader: BoundedIterable,
    accumulation_steps: int,
    batch_size: int,
    peak_memory: Optional[Mapping[str, float]],
    amp: Mapping[str, Any],
    applicable_metrics: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    numeric = {
        key: (float(value) if math.isfinite(float(value)) else None)
        for key, value in stats.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }
    finite = _finite_metrics(stats)
    applicable = list(applicable_metrics or [])
    applicable_present = [key for key in applicable if key in finite]
    non_applicable_nonfinite = sorted(
        key for key, ok in finite.items() if not ok and key not in applicable
    )
    core_keys = (
        ("reward", "grpo", "iou") if stage == "D"
        else ("loss", "seg") if stage in ("A", "C")
        else ("loss", "diff", "heat", "aux")
    )
    core_present = {key: finite[key] for key in core_keys if key in finite}
    success = float(stats.get("optimizer_steps", 0.0) or 0.0)
    items = int(loader.items_yielded)
    samples = loader.counted_samples(batch_size)
    if stage == "D":
        # One loader item is a batch of images, and each image forms its own
        # GRPO group (reward_tensor is [G, B]). Reporting items as groups would
        # undercount by the batch size whenever batch_size > 1.
        attempted_batches = items
        attempted_groups = samples
        optimizer = {
            "attempted_batches": attempted_batches,
            "attempted_groups": attempted_groups,
            "successful_updates": success,
            "amp_skipped_updates": float(stats.get("skipped_update_steps", 0.0) or 0.0),
            "zero_update_groups": float(stats.get("skipped_groups", 0.0) or 0.0),
        }
    else:
        skipped = float(stats.get("skipped_groups", 0.0) or 0.0)
        optimizer = {
            "attempted_updates": success + skipped,
            "successful_updates": success,
            "skipped_updates": skipped,
        }
    elapsed = max(float(elapsed), 1e-12)
    summary: Dict[str, Any] = {
        "elapsed_seconds": float(elapsed),
        "dataloader_seconds": float(loader.fetch_seconds),
        "compute_seconds": max(0.0, float(elapsed) - float(loader.fetch_seconds)),
        "batches": items,
        "samples": samples,
        "optimizer": optimizer,
        "throughput": {
            "batches_per_second": items / elapsed,
            "samples_per_second": samples / elapsed,
            "successful_updates_per_second": success / elapsed,
            **(
                {"groups_per_second": attempted_groups / elapsed}
                if stage == "D"
                else {}
            ),
        },
        "metrics": numeric,
        "stats": numeric,
        "finite_metrics": finite,
        "finite": finite,
        "applicable_metrics": applicable,
        "applicable_metrics_present": applicable_present,
        "non_applicable_nonfinite_metrics": non_applicable_nonfinite,
        "intentionally_absent_metrics": non_applicable_nonfinite,
        "core_metrics_finite": (
            all(core_present.values()) if core_present else None
        ),
        # Only applicable, present metrics decide this flag. Absent stage
        # outputs (e.g. aux/diff/heat in stage A) are intentionally NaN.
        "all_metrics_finite": (
            all(finite[key] for key in applicable_present)
            if applicable_present
            else None
        ),
        "amp": dict(amp),
    }
    if peak_memory is not None:
        summary["memory_mib"] = dict(peak_memory)
    return summary


def _selected_cuda_uuid(device: Any) -> Optional[str]:
    """Best-effort authoritative UUID of the selected CUDA device.

    Used only by GPU telemetry: torch knows which physical GPU backs the
    logical device even when CUDA's enumeration order differs from nvidia-smi.
    """
    try:
        index = getattr(device, "index", None)
        if index is None:
            index = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(int(index))
        raw = getattr(properties, "uuid", None)
    except Exception:  # pragma: no cover - CPU/driver edge cases
        return None
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    if text.startswith("GPU-") or text.startswith("MIG-"):
        return text
    return "GPU-" + text


def run_phase(
    engine: Any,
    stage: str,
    updates: int,
    phase: str,
    raw: RawLog,
    *,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run one bounded phase through the exact real engine epoch method.

    Warmup and measurement both iterate the same guarded training subset and
    continue from the same model state; each call consumes a *distinct* update
    budget (warmup updates are not counted in the measurement budget). For
    A/B/C the budget counts optimizer updates; for D it counts trajectory
    groups and is translated into batches by :func:`_phase_batches`. The
    loader is rebuilt per phase, so worker startup is included in that phase's
    dataloader timing.

    On a real CUDA device, best-effort ``nvidia-smi`` sampling runs only around
    this phase's training loop, in its own thread and raw JSONL file. It is
    stopped in a ``finally`` so initialization and validation can never leak
    into the measurement samples, and any telemetry failure is recorded rather
    than raised.
    """
    accumulation_steps = max(1, int(getattr(engine, "accumulation_steps", 1)))
    batch_size = max(1, int(getattr(engine, "train_cfg", {}).get("batch_size", 1)))
    budget = _phase_batches(stage, updates, accumulation_steps, batch_size)
    base_loader = engine._build_train_loader(0)
    loader = BoundedIterable(
        base_loader, budget, cycle=True, phase=phase, sink=raw.sink
    )
    device = getattr(engine, "device", torch.device("cpu"))
    is_cuda = getattr(device, "type", str(device)) == "cuda"
    output_root = (
        Path(output_dir).resolve()
        if output_dir is not None
        else Path(raw.path).resolve().parent
    )

    telemetry: Optional[GpuTelemetrySampler] = None
    telemetry_summary = unavailable_summary(
        "not_cuda" if not is_cuda else "empty_phase",
        interval_seconds=GPU_TELEMETRY_INTERVAL_SECONDS,
    )
    if is_cuda and budget > 0:
        try:
            telemetry = GpuTelemetrySampler(
                output_root / f"gpu_telemetry_{phase}.jsonl",
                device=device,
                device_uuid=_selected_cuda_uuid(device),
                interval=GPU_TELEMETRY_INTERVAL_SECONDS,
            )
            telemetry.start()
            telemetry_summary = telemetry.summary()
        except Exception as exc:
            telemetry = None
            telemetry_summary = unavailable_summary(
                f"telemetry_setup_failed: {type(exc).__name__}: {exc}",
                interval_seconds=GPU_TELEMETRY_INTERVAL_SECONDS,
            )

    if is_cuda:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    amp_before = scaler_state(engine)
    raw.event(phase + "_start", updates=updates, batch_budget=budget)
    started = time.perf_counter()
    try:
        if stage == "D":
            stats = engine._train_epoch_grpo(stage, 0, loader)
        else:
            stats = engine._train_epoch_supervised(stage, 0, loader)
        if is_cuda:
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
    finally:
        # The sampling thread must be closed with the phase, even on OOM.
        if telemetry is not None:
            try:
                telemetry_summary = telemetry.stop()
            except Exception as exc:  # pragma: no cover - defensive
                telemetry_summary = unavailable_summary(
                    f"telemetry_stop_failed: {type(exc).__name__}: {exc}",
                    interval_seconds=GPU_TELEMETRY_INTERVAL_SECONDS,
                )
    amp_after = scaler_state(engine)

    peak_memory = None
    if is_cuda:
        peak_memory = {
            "peak_allocated_mib": float(torch.cuda.max_memory_allocated(device)) / 1024**2,
            "peak_reserved_mib": float(torch.cuda.max_memory_reserved(device)) / 1024**2,
        }
    summary = _phase_summary(
        stage, stats, elapsed, loader, accumulation_steps, batch_size,
        peak_memory, {"before": amp_before, "after": amp_after},
        applicable_metrics=_applicable_metric_keys(stage, engine, stats),
    )
    summary["gpu_telemetry"] = telemetry_summary
    raw.event(phase + "_end", **summary)
    return summary


def parameters_finite(engine: Any) -> Dict[str, Any]:
    """Check trainable parameters after the measured phase (outside timing)."""
    try:
        parameters = list(engine._trainable_parameters())
    except Exception:
        parameters = []
    count = 0
    nonfinite = 0
    with torch.no_grad():
        for parameter in parameters:
            count += 1
            if not bool(torch.isfinite(parameter.detach()).all()):
                nonfinite += 1
    device = getattr(engine, "device", torch.device("cpu"))
    if getattr(device, "type", str(device)) == "cuda":
        torch.cuda.synchronize(device)
    return {
        "trainable_parameter_tensors": count,
        "nonfinite_parameter_tensors": nonfinite,
        "all_finite": nonfinite == 0,
    }


def _is_oom(exc: BaseException) -> bool:
    oom_type = getattr(torch.cuda, "OutOfMemoryError", ())
    return isinstance(exc, oom_type) or "out of memory" in str(exc).lower()


def write_error(
    output_dir: Path,
    exc: BaseException,
    phase: str,
    stage: str,
    filename: str = "error.json",
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "format_version": PROBE_FORMAT_VERSION,
        "status": "failed",
        "stage": stage,
        "phase": phase,
        "error_type": type(exc).__name__,
        "message": str(exc),
        "traceback": traceback.format_exc(),
        "out_of_memory": _is_oom(exc),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "resources": resource_metadata(),
    }
    if torch.cuda.is_available():
        try:
            device = torch.device("cuda")
            payload["memory_mib"] = {
                "allocated": float(torch.cuda.memory_allocated(device)) / 1024**2,
                "reserved": float(torch.cuda.memory_reserved(device)) / 1024**2,
                "peak_allocated": float(torch.cuda.max_memory_allocated(device)) / 1024**2,
                "peak_reserved": float(torch.cuda.max_memory_reserved(device)) / 1024**2,
            }
        except Exception:  # pragma: no cover - driver edge cases
            pass
    write_json(output_dir / filename, payload)
    return payload


def save_probe_weights(
    engine: Any,
    args: argparse.Namespace,
    output_dir: Path,
    inputs: Mapping[str, Any],
) -> Path:
    """Save a model-only snapshot for development bootstrap (``--init`` only).

    Unlike ``last.pt`` this has no optimizer/scheduler/RNG, so it is not
    resumable; D keeps the normal ``ref_unet`` snapshot.
    """
    path = Path(args.save_weights).expanduser()
    if not path.is_absolute():
        path = output_dir / path
    path = path.resolve()
    if path.exists():
        raise FileExistsError(
            f"--save-weights 目标已存在，拒绝覆盖已有快照：{path}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    sam = getattr(engine, "sam", None)
    payload = {
        "format_version": 3,
        "sam": {
            "architecture": str(getattr(sam, "architecture", "")),
            "config": str(getattr(sam, "config_name", "")),
            "apply_postprocessing": bool(getattr(sam, "apply_postprocessing", False)),
        },
        "algorithm": {
            "name": str(getattr(engine, "rl_algorithm", "grpo")),
            "version": 2,
        },
        "config": engine._resolved_config(),
        "stage": args.stage,
        "epoch": 1,
        "global_step": int(getattr(engine, "global_step", 0)),
        "best_metric": None,
        "history": [],
        "models": engine._model_states(),
        "probe": {
            "kind": "development_probe",
            "note": (
                "单候选有界探针权重；未完成完整训练，不声称收敛或质量；"
                "仅用于串联开发 bootstrap（--init）。"
            ),
            "script": str(Path(__file__).resolve()),
            "output_dir": str(output_dir),
            "stage": args.stage,
            "warmup_updates": int(args.warmup_updates),
            "measure_updates": int(args.measure_updates),
            "train_samples": len(inputs.get("train_subset", [])),
            "seed": int(engine.config.get("seed", 42)),
            "created_utc": datetime.now(timezone.utc).isoformat(),
        },
    }
    engine._save_payload(path, payload)
    return path


def run_session(args: argparse.Namespace) -> int:
    """Execute one probe; every failure leaves prior files in place."""
    validate_args(args)
    output_dir = prepare_output_dir(Path(args.output_dir))
    raw = RawLog(output_dir / "raw.jsonl")
    phase = "setup"
    stage = args.stage
    try:
        raw.event("probe_start", stage=stage, output_dir=str(output_dir))

        config_path = Path(args.config).expanduser()
        config = load_config(config_path)
        config_dir = config_path.resolve().parent
        config = apply_overrides(config, args)
        config["output_dir"] = str(output_dir)
        config.setdefault("logging", {})["tensorboard"] = False

        phase = "inputs"
        inputs = resolve_inputs(config, config_dir, args, output_dir, raw)
        val_subset_size = len(inputs.get("val_subset") or [])

        threading = configure_threads(args)
        write_json(output_dir / "environment.json", environment_info(args))
        write_json(
            output_dir / "threading.json", threading
        )
        if threading.get("interop_error"):
            raw.event("thread_warning", message=threading["interop_error"])

        phase = "engine"
        engine = DiffPeakEngine(config, device=args.device, config_dir=config_dir)
        write_json(output_dir / "resolved_config.json", engine._resolved_config())
        raw.event("engine_ready", device=str(engine.device))

        if args.init is not None:
            phase = "init"
            engine.load_init_weights(args.init)
            raw.event("init_loaded", path=str(args.init))

        phase = "prepare"
        update_epochs = max(1, int(getattr(engine, "update_epochs", 1)))
        batch_size = max(
            1, int(getattr(engine, "train_cfg", {}).get("batch_size", 1))
        )
        accumulation_steps = max(
            1, int(getattr(engine, "accumulation_steps", 1))
        )
        if stage == "D":
            # Each D batch performs ``update_epochs`` optimizer steps; the
            # group budget is converted to batches so the cosine schedule is
            # sized for the steps that will actually run.
            warmup_batches = _phase_batches(
                stage, args.warmup_updates, accumulation_steps, batch_size
            )
            measure_batches = _phase_batches(
                stage, args.measure_updates, accumulation_steps, batch_size
            )
            total_steps = max(
                1, (warmup_batches + measure_batches) * update_epochs
            )
        else:
            # One optimizer step per accumulated group.
            total_steps = max(
                1, int(args.warmup_updates) + int(args.measure_updates)
            )
        engine._prepare_stage(stage, 1, total_steps)

        phase = "warmup"
        warmup = run_phase(
            engine, stage, int(args.warmup_updates), "warmup", raw,
            output_dir=output_dir,
        )

        phase = "measure"
        measurement = run_phase(
            engine, stage, int(args.measure_updates), "measurement", raw,
            output_dir=output_dir,
        )

        phase = "parameters"
        parameter_check = parameters_finite(engine)

        weights_path: Optional[Path] = None
        if args.save_weights is not None:
            phase = "weights"
            weights_path = save_probe_weights(engine, args, output_dir, inputs)
            raw.event(
                "weights_saved", path=str(weights_path), kind="model_only",
                stage=stage,
            )

        validated = stage in ("C", "D") and args.init is None
        notes: List[str] = []
        if validated:
            notes.append(
                "阶段 C/D 未提供 --init：验证结果仅为未训练模型的容量 smoke，"
                "不代表质量或收敛。"
            )
        if torch.cuda.is_available() and measurement.get("memory_mib"):
            notes.append(
                "peak 显存为 measurement 阶段（重置 peak 后）的实测值；"
                "不包含 --validate。"
            )

        results: Dict[str, Any] = {
            "format_version": PROBE_FORMAT_VERSION,
            "status": "ok",
            "stage": stage,
            "quality_claim": False,
            "purpose": "capacity_smoke" if validated else "performance_probe",
            "warmup": warmup,
            "measurement": measurement,
            "parameters": parameter_check,
            "weights": (
                None
                if weights_path is None
                else {
                    "path": str(weights_path),
                    "kind": "model_only",
                    "resumable": False,
                    "note": "供 --init 串联开发 bootstrap；不声称收敛。",
                }
            ),
            "notes": notes,
            "outputs": {
                "environment": "environment.json",
                "resolved_config": "resolved_config.json",
                "splits": "splits.json",
                "raw_log": "raw.jsonl",
                "validation": "validation.json" if args.validate else None,
                "gpu_telemetry": {
                    "warmup": warmup.get("gpu_telemetry", {}).get("raw_jsonl"),
                    "measurement": measurement.get("gpu_telemetry", {}).get(
                        "raw_jsonl"
                    ),
                },
            },
            "probe": {
                "warmup_updates": int(args.warmup_updates),
                "measure_updates": int(args.measure_updates),
                "train_samples": int(args.train_samples),
                "val_samples": val_subset_size if args.validate else None,
                "batch_size": int(getattr(engine, "train_cfg", {}).get("batch_size", 1)),
                "accumulation_steps": int(engine.accumulation_steps),
                "group_size": int(getattr(engine, "group_size", 1)),
                "stage_lr": engine.scheduler_stage_lr,
                "device": str(engine.device),
            },
        }
        write_json(output_dir / "results.json", results)
        raw.event("results_written", path=str(output_dir / "results.json"))

        if args.validate:
            phase = "validate"
            try:
                started = time.perf_counter()
                validation = engine.validate(stage)
                elapsed = time.perf_counter() - started
                validation_record = {
                    "format_version": PROBE_FORMAT_VERSION,
                    "status": "ok",
                    "stage": stage,
                    "purpose": "capacity_smoke" if validated else "limited_val_subset",
                    "quality_claim": False,
                    # Actual materialized subset, not the requested --val-samples.
                    "subset_count": val_subset_size,
                    "val_limit": getattr(engine, "val_limit", None),
                    "elapsed_seconds": elapsed,
                    "metrics": validation,
                    "finite_metrics": _finite_metrics(validation or {}),
                    "warning": (
                        "未训练模型（无 --init）的 C/D 验证只验证管线容量，"
                        "不能用于质量结论。"
                        if validated
                        else "固定验证子集仅用于 smoke，不代表完整验证集指标。"
                    ),
                }
                write_json(output_dir / "validation.json", validation_record)
                raw.event("validation_end", **validation_record)
            except BaseException as exc:
                write_error(
                    output_dir, exc, "validate", stage,
                    filename="validation_error.json",
                )
                raw.event(
                    "validation_error", error_type=type(exc).__name__, message=str(exc)
                )
                return 1
        raw.event("probe_end", status="ok")
        print(f"probe 完成：{output_dir}", flush=True)
        return 0
    except BaseException as exc:
        write_error(output_dir, exc, phase, stage)
        try:
            raw.event(
                "probe_end",
                status="failed",
                phase=phase,
                error_type=type(exc).__name__,
                message=str(exc),
            )
        except Exception:
            pass
        print(
            f"probe 失败（{type(exc).__name__}, phase={phase}）：{exc}",
            file=sys.stderr,
            flush=True,
        )
        return 1
    finally:
        raw.close()


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    raise SystemExit(run_session(args))


if __name__ == "__main__":
    main()
