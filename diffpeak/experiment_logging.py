"""训练实验的配置快照、文本/JSONL 日志与 TensorBoard 标量。"""
from __future__ import annotations

import json
import math
import platform
import subprocess
import sys
import time
import traceback
import uuid
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


class _Tee:
    def __init__(self, stream, logfile):
        self.stream = stream
        self.logfile = logfile

    def write(self, text):
        result = self.stream.write(text)
        self.logfile.write(text)
        self.logfile.flush()
        return result

    def flush(self):
        self.stream.flush()
        self.logfile.flush()

    def __getattr__(self, name):
        return getattr(self.stream, name)


class ExperimentLogger:
    """每次调用使用独立会话；不占用训练用的 Python/NumPy/Torch RNG。"""

    def __init__(self, output_dir: Path, config: Mapping[str, Any], metadata=None):
        settings = config.get("logging", {})
        self.log_every = max(1, int(settings.get("log_every", 10)))
        self.flush_secs = max(1, int(settings.get("flush_secs", 30)))
        self.tensorboard = bool(settings.get("tensorboard", True))
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        self.session_id = f"{timestamp}_{uuid.uuid4().hex[:8]}"
        self.directory = Path(output_dir) / "experiments" / self.session_id
        self.config = dict(config)
        self.metadata = dict(metadata or {})
        self.metadata.update(session_id=self.session_id, status="running")
        self.writer = None
        self._stack = ExitStack()
        self._metrics = None
        self._started = time.monotonic()

    def _write_json(self, name, value):
        path = self.directory / name
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(_json_value(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)

    def __enter__(self):
        self.directory.mkdir(parents=True, exist_ok=False)
        self._started = time.monotonic()
        try:
            logfile = self._stack.enter_context(
                (self.directory / "train.log").open("a", encoding="utf-8", buffering=1)
            )
            self._logfile = logfile
            self._metrics = self._stack.enter_context(
                (self.directory / "metrics.jsonl").open("a", encoding="utf-8", buffering=1)
            )
            self._stack.enter_context(redirect_stdout(_Tee(sys.stdout, logfile)))
            self._stack.enter_context(redirect_stderr(_Tee(sys.stderr, logfile)))
            if self.tensorboard:
                try:
                    from torch.utils.tensorboard import SummaryWriter
                except ImportError as exc:
                    raise RuntimeError(
                        "TensorBoard 未安装：请在 pytorch 环境执行 python -m pip install tensorboard，"
                        "或使用 --no-tensorboard。"
                    ) from exc
                self.writer = SummaryWriter(
                    log_dir=str(self.directory / "tensorboard"), flush_secs=self.flush_secs
                )
                self._stack.callback(self.writer.close)
            self.metadata.update(
                started_at=datetime.now(timezone.utc).isoformat(),
                command=list(sys.argv),
                python=platform.python_version(),
                platform=platform.platform(),
                pytorch=torch.__version__,
                cuda=torch.version.cuda,
                conda_env=sys.prefix,
                tensorboard=self.tensorboard,
            )
            project = Path(__file__).resolve().parent.parent
            for name, command in (
                ("git_revision", ["git", "rev-parse", "HEAD"]),
                ("git_status", ["git", "status", "--porcelain"]),
            ):
                try:
                    result = subprocess.run(
                        command, cwd=project, capture_output=True, text=True, timeout=3, check=True
                    )
                    self.metadata[name] = result.stdout.strip()
                except (OSError, subprocess.SubprocessError):
                    self.metadata[name] = None
            self.save_config(self.config)
            self._write_json("metadata.json", self.metadata)
            self.event("session_start", **self.metadata)
            print(f"实验日志：{self.directory}", flush=True)
            return self
        except BaseException:
            if self._metrics is not None:
                self.__exit__(*sys.exc_info())
            else:
                self._stack.close()
            raise

    def save_config(self, config: Mapping[str, Any]):
        self.config = dict(config)
        self._write_json("config.json", self.config)

    def event(self, event: str, **fields):
        record = {"event": event, "time": datetime.now(timezone.utc).isoformat(), **fields}
        self._metrics.write(json.dumps(_json_value(record), ensure_ascii=False, allow_nan=False) + "\n")

    def log_scalars(self, stage: str, scope: str, metrics: Mapping[str, Any], step: int, epoch: int):
        self.event("metrics", stage=stage, scope=scope, step=int(step), epoch=int(epoch), metrics=metrics)
        if self.writer is not None:
            for name, value in metrics.items():
                if isinstance(value, (int, float)) and math.isfinite(value):
                    self.writer.add_scalar(f"{stage}/{scope}/{name}", value, int(step))

    def flush(self):
        self._metrics.flush()
        if self.writer is not None:
            self.writer.flush()

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None or (isinstance(exc, SystemExit) and exc.code in (None, 0)):
            status = "completed"
        elif isinstance(exc, KeyboardInterrupt) or (isinstance(exc, SystemExit) and exc.code == 130):
            status = "interrupted"
        else:
            status = "failed"
        try:
            if exc is not None:
                traceback.print_exception(exc_type, exc, tb, file=self._logfile)
                self.metadata["error"] = f"{exc_type.__name__}: {exc}"
            self.metadata.update(
                status=status,
                finished_at=datetime.now(timezone.utc).isoformat(),
                duration_seconds=time.monotonic() - self._started,
            )
            self.event("session_end", status=status, duration_seconds=self.metadata["duration_seconds"])
            self._write_json("metadata.json", self.metadata)
            self.flush()
        finally:
            self._stack.close()
        return False
