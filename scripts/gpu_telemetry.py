"""Best-effort per-GPU telemetry sampling for the single-candidate probe.

This module intentionally depends only on the standard library so it can be
unit-tested on CPU-only hosts. It never imports torch and never raises into a
training loop: every failure is recorded in the returned summary instead.

The sampler resolves the *one* physical GPU backing the selected CUDA logical
device, honouring ``CUDA_VISIBLE_DEVICES`` (plain indices or GPU UUIDs), and
queries only that GPU with ``nvidia-smi -i``. Machine-wide aggregates are never
used. When the mapping cannot be established reliably (for example a MIG UUID,
an invalid visibility list, or an unavailable ``nvidia-smi``) the summary is
marked unavailable with a machine-readable reason.

Each sampler writes timestamped raw samples to its own JSONL file, so warmup,
measurement, initialization, and validation can never share a file.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

QUERY_FIELDS = (
    "index",
    "uuid",
    "utilization.gpu",
    "memory.used",
    "power.draw",
    "temperature.gpu",
)
DISCOVERY_FIELDS = ("index", "uuid", "name")

# Summary metric name -> nvidia-smi field name.
METRIC_FIELDS = {
    "utilization_gpu_percent": "utilization.gpu",
    "memory_used_mib": "memory.used",
    "power_draw_w": "power.draw",
    "temperature_c": "temperature.gpu",
}

DEFAULT_INTERVAL_SECONDS = 0.25
DEFAULT_TIMEOUT_SECONDS = 5.0

_AUTO = object()


@dataclass(frozen=True)
class GpuRow:
    """One row from ``nvidia-smi --query-gpu=index,uuid,name``."""

    index: int
    uuid: str
    name: str = ""


@dataclass(frozen=True)
class GpuTarget:
    """The physical GPU that backs a CUDA logical device."""

    physical_index: int
    uuid: str
    name: str
    logical_index: int
    selector: str


@dataclass(frozen=True)
class TargetResolution:
    target: Optional[GpuTarget]
    reason: Optional[str]


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
def parse_gpu_rows(text: str) -> List[GpuRow]:
    """Parse discovery output; malformed rows are skipped, never fatal."""
    rows: List[GpuRow] = []
    for line in str(text or "").splitlines():
        if not line.strip():
            continue
        # The name may itself contain commas, so keep the remainder intact.
        parts = [part.strip() for part in line.split(",", 2)]
        if len(parts) < 2:
            continue
        try:
            index = int(parts[0])
        except ValueError:
            continue
        uuid = parts[1]
        if not uuid:
            continue
        name = parts[2] if len(parts) > 2 else ""
        rows.append(GpuRow(index=index, uuid=uuid, name=name))
    return rows


def _parse_metric(raw: Any) -> Optional[float]:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    if text.startswith("[") and text.endswith("]"):
        return None
    if text.lower() in {"n/a", "not supported", "unknown", "none"}:
        return None
    try:
        value = float(text)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def parse_sample_line(text: str) -> Optional[Dict[str, Any]]:
    """Parse one metric row; return ``None`` when the row is unusable."""
    line = next(
        (item for item in str(text or "").splitlines() if item.strip()), None
    )
    if line is None:
        return None
    parts = [part.strip() for part in line.split(",")]
    if len(parts) < len(QUERY_FIELDS):
        return None
    try:
        index = int(parts[0])
    except ValueError:
        return None
    return {
        "index": index,
        "uuid": parts[1],
        "utilization_gpu_percent": _parse_metric(parts[2]),
        "memory_used_mib": _parse_metric(parts[3]),
        "power_draw_w": _parse_metric(parts[4]),
        "temperature_c": _parse_metric(parts[5]),
    }


def summarize_samples(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Min/max/mean/last per metric; absent values are reported as count 0."""
    summary: Dict[str, Any] = {}
    for key in METRIC_FIELDS:
        values = [
            float(record[key])
            for record in records
            if isinstance(record.get(key), (int, float))
            and not isinstance(record.get(key), bool)
            and math.isfinite(float(record[key]))
        ]
        summary[key] = {
            "count": len(values),
            "min": min(values) if values else None,
            "max": max(values) if values else None,
            "mean": (sum(values) / len(values)) if values else None,
            "last": values[-1] if values else None,
        }
    return summary


# ---------------------------------------------------------------------------
# CUDA_VISIBLE_DEVICES mapping
# ---------------------------------------------------------------------------
def _logical_index(device: Any) -> Tuple[Optional[int], Optional[str]]:
    if isinstance(device, bool):
        return None, "invalid_device"
    if isinstance(device, int):
        return (device, None) if device >= 0 else (None, "invalid_device")
    text = str(device if device is not None else "cuda").strip().lower()
    if not text.startswith("cuda"):
        return None, "not_cuda"
    if ":" not in text:
        return 0, None
    suffix = text.rsplit(":", 1)[1]
    if not suffix.isdigit():
        return None, "invalid_device"
    return int(suffix), None


def _visible_tokens(
    cuda_visible_devices: Any,
) -> Tuple[Optional[List[str]], Optional[str]]:
    """Return tokens, ``None`` for unset, or a reason for an invalid value."""
    if cuda_visible_devices is None:
        return None, None
    text = str(cuda_visible_devices).strip()
    if text in {"", "-1"}:
        return [], None
    tokens = [token.strip() for token in text.split(",")]
    if any(not token for token in tokens):
        return None, "invalid_cuda_visible_devices"
    return tokens, None


def _match_uuid(
    token: str, rows: Sequence[GpuRow]
) -> Tuple[Optional[GpuRow], Optional[str]]:
    exact = [row for row in rows if row.uuid == token]
    if len(exact) == 1:
        return exact[0], None
    if len(exact) > 1:  # pragma: no cover - UUID duplicates are not expected
        return None, "ambiguous_uuid"
    # Accept an unambiguous UUID prefix (nvidia-smi prints the full UUID).
    prefixed = [
        row for row in rows if row.uuid.startswith(token) or token.startswith(row.uuid)
    ]
    if len(prefixed) == 1:
        return prefixed[0], None
    if not prefixed:
        return None, "uuid_not_found"
    return None, "ambiguous_uuid"


def resolve_target_device(
    device: Any,
    cuda_visible_devices: Optional[str],
    rows: Sequence[GpuRow],
    device_uuid: Optional[str] = None,
) -> TargetResolution:
    """Map a torch-style CUDA device to exactly one physical GPU.

    ``device_uuid`` (from ``torch.cuda.get_device_properties().uuid``) is the
    authoritative identity when available: it is unaffected by CUDA device
    enumeration order. Without it, ``CUDA_VISIBLE_DEVICES`` is honored (plain
    physical indices or GPU UUIDs). Because CUDA's default ``FASTEST_FIRST``
    order need not match nvidia-smi's PCI order, an unset visibility list is
    mapped by ascending physical index only as a best-effort fallback.

    MIG UUIDs and malformed lists are treated as unmappable because a
    physical-GPU query could otherwise silently describe the wrong device.
    With neither a UUID nor a visibility list, only a single reported GPU is
    accepted (``single_gpu_default``); multiple GPUs yield
    ``ambiguous_enumeration_order`` rather than an index guess.
    """
    logical, reason = _logical_index(device)
    if logical is None:
        return TargetResolution(None, reason)
    if not rows:
        return TargetResolution(None, "nvidia_smi_unavailable")

    if device_uuid:
        identifier = str(device_uuid).strip()
        if identifier.startswith("MIG-"):
            return TargetResolution(None, "unmapped_visible_device")
        if identifier and not identifier.startswith("GPU-"):
            identifier = "GPU-" + identifier
        match, match_reason = _match_uuid(identifier, rows)
        if match is None:
            return TargetResolution(None, match_reason or "device_uuid_not_found")
        return TargetResolution(
            GpuTarget(
                physical_index=match.index,
                uuid=match.uuid,
                name=match.name,
                logical_index=logical,
                selector="torch_device_uuid",
            ),
            None,
        )

    tokens, reason = _visible_tokens(cuda_visible_devices)
    if reason is not None:
        return TargetResolution(None, reason)

    if tokens is None:
        # CUDA_VISIBLE_DEVICES unset: CUDA's default FASTEST_FIRST order need
        # not match nvidia-smi's PCI index order. Only a single reported GPU is
        # unambiguous; with multiple GPUs this would be a guess, so report
        # unavailable instead of silently sampling the wrong device.
        if len(rows) == 1:
            if logical != 0:
                return TargetResolution(None, "logical_device_out_of_range")
            match = rows[0]
            return TargetResolution(
                GpuTarget(
                    physical_index=match.index,
                    uuid=match.uuid,
                    name=match.name,
                    logical_index=logical,
                    selector="single_gpu_default",
                ),
                None,
            )
        return TargetResolution(None, "ambiguous_enumeration_order")

    if not tokens:
        return TargetResolution(None, "no_visible_devices")
    if logical >= len(tokens):
        return TargetResolution(None, "logical_device_out_of_range")

    token = tokens[logical]
    if token.isdigit():
        physical = int(token)
        match = next((row for row in rows if row.index == physical), None)
        if match is None:
            return TargetResolution(None, "physical_index_not_found")
        return TargetResolution(
            GpuTarget(
                physical_index=match.index,
                uuid=match.uuid,
                name=match.name,
                logical_index=logical,
                selector="cuda_visible_devices_index",
            ),
            None,
        )

    if token.startswith("GPU-"):
        match, reason = _match_uuid(token, rows)
        if match is None:
            return TargetResolution(None, reason)
        return TargetResolution(
            GpuTarget(
                physical_index=match.index,
                uuid=match.uuid,
                name=match.name,
                logical_index=logical,
                selector="cuda_visible_devices_uuid",
            ),
            None,
        )

    # MIG-UUID, an unsupported spelling, or a negative index.
    return TargetResolution(None, "unmapped_visible_device")


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------
def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _iso_from_epoch(timestamp: float) -> str:
    return datetime.fromtimestamp(float(timestamp), timezone.utc).isoformat()


def default_smi_runner(
    args: Sequence[str],
    *,
    smi_path: str = "nvidia-smi",
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> str:
    """Run one nvidia-smi query; raise on any failure (callers contain it)."""
    completed = subprocess.run(
        [str(smi_path), *[str(argument) for argument in args]],
        capture_output=True,
        text=True,
        timeout=float(timeout),
        check=False,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        raise RuntimeError(
            f"nvidia-smi 退出码 {completed.returncode}: {detail[:400]}"
        )
    return completed.stdout


def unavailable_summary(
    reason: str, *, interval_seconds: Optional[float] = None
) -> Dict[str, Any]:
    """Uniform summary for CPU runs, empty phases, or unmappable devices."""
    return {
        "available": False,
        "reason": str(reason),
        "source": "nvidia-smi",
        "device": None,
        "cuda_visible_devices": None,
        "interval_seconds": interval_seconds,
        "started_utc": None,
        "ended_utc": None,
        "duration_seconds": 0.0,
        "sample_count": 0,
        "error_count": 0,
        "last_error": None,
        "raw_jsonl": None,
        "metrics": summarize_samples([]),
    }


class GpuTelemetrySampler:
    """Sample one GPU in a background thread; failures never propagate.

    ``start()`` resolves the target and opens the raw JSONL file synchronously
    (outside the training timer); the thread then takes a sample immediately and
    every ``interval`` seconds. ``stop()`` closes the thread and returns the
    summary, and is safe to call when ``start()`` failed entirely.
    """

    def __init__(
        self,
        output_path: Path,
        *,
        device: Any = None,
        cuda_visible_devices: Any = _AUTO,
        device_uuid: Optional[str] = None,
        interval: float = DEFAULT_INTERVAL_SECONDS,
        runner: Optional[Callable[[Sequence[str]], str]] = None,
        smi_path: str = "nvidia-smi",
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        clock: Any = time.time,
        monotonic: Any = time.monotonic,
    ) -> None:
        if callable(runner):
            self._runner = runner
        else:
            self._runner = lambda args: default_smi_runner(
                args, smi_path=smi_path, timeout=timeout
            )
        self.output_path = Path(output_path)
        self.device = device
        self.device_uuid = device_uuid
        self.cuda_visible_devices = (
            os.environ.get("CUDA_VISIBLE_DEVICES")
            if cuda_visible_devices is _AUTO
            else cuda_visible_devices
        )
        self.interval = max(0.001, float(interval))
        self._clock = clock
        self._monotonic = monotonic
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._handle: Any = None
        self._raw_jsonl: Optional[str] = None
        self._records: List[Dict[str, Any]] = []
        self._errors: List[str] = []
        self._target: Optional[GpuTarget] = None
        self._reason: Optional[str] = "not_started"
        self._last_error: Optional[str] = None
        self._started_mono: Optional[float] = None
        self._started_utc: Optional[str] = None

    def _record_error(self, exc: BaseException) -> None:
        message = f"{type(exc).__name__}: {exc}"
        with self._lock:
            self._errors.append(message)
            self._last_error = message

    def start(self) -> "GpuTelemetrySampler":
        """Resolve the device and start sampling; never raises."""
        self._started_mono = float(self._monotonic())
        self._started_utc = _utc_now()
        try:
            discovery = self._runner(
                [
                    f"--query-gpu={','.join(DISCOVERY_FIELDS)}",
                    "--format=csv,noheader",
                ]
            )
            rows = parse_gpu_rows(discovery)
            resolution = resolve_target_device(
                self.device,
                self.cuda_visible_devices,
                rows,
                device_uuid=self.device_uuid,
            )
            if resolution.target is None:
                self._reason = resolution.reason or "unmapped_device"
                return self
            self._target = resolution.target
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = self.output_path.open("w", encoding="utf-8")
            self._raw_jsonl = self.output_path.name
            self._reason = None
            self._thread = threading.Thread(
                target=self._loop,
                name=f"gpu-telemetry-{self.output_path.stem}",
                daemon=True,
            )
            self._thread.start()
        except BaseException as exc:  # noqa: BLE001 - best effort only
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            self._target = None
            self._reason = f"nvidia_smi_unavailable: {type(exc).__name__}: {exc}"
            handle = self._handle
            self._handle = None
            self._raw_jsonl = None
            if handle is not None:
                try:
                    handle.close()
                except Exception:  # pragma: no cover - close edge cases
                    pass
            self._record_error(exc)
        return self

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            self.sample_once()
            if self._stop_event.wait(self.interval):
                break

    def sample_once(self) -> Optional[Dict[str, Any]]:
        """Take one sample; return the raw record or ``None`` on failure."""
        target = self._target
        if target is None:
            return None
        selector = target.uuid or str(target.physical_index)
        try:
            text = self._runner(
                [
                    f"--id={selector}",
                    f"--query-gpu={','.join(QUERY_FIELDS)}",
                    "--format=csv,noheader,nounits",
                ]
            )
            parsed = parse_sample_line(text)
            if parsed is None:
                raise ValueError("无法解析 nvidia-smi 采样输出")
            if target.uuid and parsed.get("uuid") != target.uuid:
                raise ValueError(
                    "nvidia-smi 返回了非目标 GPU："
                    f"期望 {target.uuid}，实际 {parsed.get('uuid')}"
                )
            with self._lock:
                timestamp = float(self._clock())
                elapsed = max(
                    0.0,
                    float(self._monotonic()) - float(self._started_mono or 0.0),
                )
                record = {
                    "time": timestamp,
                    "time_utc": _iso_from_epoch(timestamp),
                    "elapsed_seconds": elapsed,
                    **parsed,
                }
                self._records.append(record)
                handle = self._handle
                if handle is not None:
                    handle.write(
                        json.dumps(record, sort_keys=True, allow_nan=False) + "\n"
                    )
                    handle.flush()
            return record
        except BaseException as exc:  # noqa: BLE001 - best effort only
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            self._record_error(exc)
            return None

    def stop(self, timeout: float = DEFAULT_TIMEOUT_SECONDS) -> Dict[str, Any]:
        """Stop the thread, close the raw file, and return the summary."""
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(float(timeout))
            if thread.is_alive():
                self._record_error(RuntimeError("gpu telemetry 采样线程未及时退出"))
        handle = None
        with self._lock:
            handle = self._handle
            self._handle = None
        if handle is not None:
            try:
                handle.close()
            except Exception as exc:  # pragma: no cover - close edge cases
                self._record_error(exc)
        return self.summary()

    def summary(self) -> Dict[str, Any]:
        """Snapshot of device mapping, raw sample statistics, and errors."""
        with self._lock:
            records = list(self._records)
            errors = list(self._errors)
            last_error = self._last_error
        available = self._target is not None
        duration = 0.0
        if self._started_mono is not None:
            duration = max(0.0, float(self._monotonic()) - float(self._started_mono))
        return {
            "available": available,
            "reason": None if available else (self._reason or "unavailable"),
            "source": "nvidia-smi",
            "device": (
                None
                if self._target is None
                else {
                    "logical_index": int(self._target.logical_index),
                    "physical_index": int(self._target.physical_index),
                    "uuid": self._target.uuid,
                    "name": self._target.name,
                    "selector": self._target.selector,
                }
            ),
            "cuda_visible_devices": self.cuda_visible_devices,
            "interval_seconds": float(self.interval),
            "started_utc": self._started_utc,
            "ended_utc": _utc_now(),
            "duration_seconds": duration,
            "sample_count": len(records),
            "error_count": len(errors),
            "last_error": last_error,
            "raw_jsonl": self._raw_jsonl if available else None,
            "metrics": summarize_samples(records),
        }
