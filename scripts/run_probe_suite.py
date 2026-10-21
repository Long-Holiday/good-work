"""按顺序执行固定预算开发探针；独占 GPU，逐候选落盘并继续记录失败。"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time


# ---------------------------------------------------------------------------
# 候选有效性、父子选择与证据记录
# ---------------------------------------------------------------------------
def _load_json(path: Path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _finite_iou(validation) -> float | None:
    """validation.json -> 有限的 macro IoU；缺失/非有限返回 None。"""
    metrics = (validation or {}).get("metrics")
    if not isinstance(metrics, dict):
        return None
    value = metrics.get("iou")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _successful_updates(result) -> int:
    measurement = (result or {}).get("measurement") or {}
    optimizer = measurement.get("optimizer") or {}
    value = optimizer.get("successful_updates")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0
    return max(0, int(value))


def _candidate(
    directory: Path,
    *,
    require_validation: bool = True,
    require_weights: bool = True,
) -> dict:
    """Read one probe output and decide whether it may enter a comparison.

    File presence alone is not success: a probe can write validation/weights
    after zero optimizer updates or with non-finite parameters.  Candidates
    must also share the same train/val subset, otherwise their IoUs are not
    comparable.
    """
    directory = Path(directory)
    result = _load_json(directory / "results.json")
    validation = _load_json(directory / "validation.json")
    splits = _load_json(directory / "splits.json")
    parameters = (result or {}).get("parameters") or {}
    update_count = _successful_updates(result)
    iou = _finite_iou(validation)
    train = (splits or {}).get("train") or {}
    val = (splits or {}).get("val") or {}
    split_key = (
        train.get("seed"),
        train.get("subset_entries_sha256"),
        val.get("seed"),
        val.get("subset_entries_sha256"),
    )
    reasons = []
    if not result or result.get("status") != "ok":
        reasons.append("results_missing_or_failed")
    if update_count <= 0:
        reasons.append("no_successful_updates")
    if parameters.get("all_finite") is not True:
        reasons.append("parameters_not_finite")
    if require_validation and iou is None:
        reasons.append("validation_iou_missing_or_nonfinite")
    if require_weights and not (directory / "weights.pt").is_file():
        reasons.append("weights_missing")
    if splits is None or any(part is None for part in split_key):
        reasons.append("splits_missing")
    return {
        "name": directory.name,
        "directory": directory,
        "iou": iou,
        "successful_updates": update_count,
        "split_key": split_key,
        "valid": not reasons,
        "reasons": reasons,
    }


def _lr_key(name: str) -> float:
    """候选名末尾的学习率，用于固定的低 LR 优先平局规则。"""
    try:
        return float(name.rsplit("_", 1)[-1])
    except ValueError:
        return math.inf


def _select_candidate(directories) -> dict:
    """Pick the best valid candidate; fail only when every candidate failed.

    Ties use the declared candidate order (ascending learning rate), matching
    the protocol's "prefer the earlier candidate" tie rule.  The tie is
    recorded instead of being hidden in a lexicographic Path comparison.
    """
    rows = [_candidate(path) for path in directories]
    rows.sort(key=lambda row: (_lr_key(row["name"]), row["name"]))
    valid = [row for row in rows if row["valid"]]
    if not valid:
        details = "; ".join(
            f"{row['name']}: {','.join(row['reasons'])}" for row in rows
        ) or "无候选目录"
        raise RuntimeError("所有候选均未通过有效性检查，停止后续串联：" + details)
    keys = {row["split_key"] for row in valid}
    if len(keys) != 1:
        raise RuntimeError("候选使用的 train/val 子集不一致，拒绝按 IoU 比较")
    best = max(row["iou"] for row in valid)
    tied = [row for row in valid if row["iou"] == best]
    selected = tied[0]
    return {
        "selected": selected,
        "candidates": rows,
        "tie": len(tied) > 1,
        "tied": [row["name"] for row in tied],
    }


def _candidate_summary(row: dict) -> dict:
    return {
        "name": row["name"],
        "iou": row["iou"],
        "successful_updates": row["successful_updates"],
        "valid": row["valid"],
        "reasons": row["reasons"],
    }


def _selection_event(stage: str, selection: dict, weights: Path | None = None) -> dict:
    event = {
        "event": "select",
        "stage": stage,
        "selected": selection["selected"]["name"],
        "iou": selection["selected"]["iou"],
        "successful_updates": selection["selected"]["successful_updates"],
        "tie": selection["tie"],
        "tied": selection["tied"],
        "candidates": [
            _candidate_summary(row) for row in selection["candidates"]
        ],
    }
    if weights is not None:
        try:
            event["parent_weights_sha256"] = _sha256(weights)
        except OSError:
            event["parent_weights_sha256"] = None
    return event


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fingerprint(path: Path) -> dict:
    path = Path(path)
    try:
        digest: str | None = _sha256(path)
    except OSError:
        digest = None
    return {"path": str(path), "sha256": digest}


def _link_selected(link: Path, directory: Path) -> None:
    """Create a relative parent link; never replace existing evidence."""
    if link.is_symlink() or link.exists():
        raise FileExistsError(f"父权重链接已存在，拒绝覆盖：{link}")
    link.symlink_to(directory.name, target_is_directory=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--phase", choices=["capacity", "learning"], required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    results = root / "results_sam2"
    suite = root / f"suite_sam2_{args.phase}.jsonl"
    # 先检查独占路径，再写任何派生配置或链接。
    if suite.is_symlink() or suite.exists():
        raise FileExistsError(f"suite 记录已存在，拒绝覆盖：{suite}")

    probe_script = root / "scripts/probe_3090.py"
    checkpoint = root / "checkpoints/sam2_download/sam2.1_hiera_base_plus.pt"
    train_split = root / "splits/protocol_v1/train.json"
    val_split = root / "splits/protocol_v1/val.json"
    data_root = root.parent / "DeepGlobe_Road"
    base_config = root / "configs/3090.yaml"

    base = [sys.executable, str(probe_script),
            "--config", str(base_config),
            "--data-root", str(data_root),
            "--sam-checkpoint", str(checkpoint),
            "--train-split", str(train_split),
            "--val-split", str(val_split),
            "--seed", "2026", "--cpu-threads", "2", "--interop-threads", "1",
            "--amp-init-scale", "1024", "--train-samples", "256"]
    env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1")
    candidates = []
    active_config = base_config
    if args.phase == "capacity":
        for workers, checkpointing, batch, accumulation in (
            (4, "on", 1, 4), (12, "on", 1, 4), (8, "off", 1, 4),
            (12, "off", 1, 4), (8, "off", 2, 2), (8, "off", 4, 1),
        ):
            name = f"B_system_w{workers}_{checkpointing}_b{batch}"
            candidates.append((name, ["--stage", "B", "--num-workers", str(workers),
                "--gradient-checkpointing", checkpointing, "--batch-size", str(batch),
                "--accumulation-steps", str(accumulation), "--warmup-updates", "3",
                "--measure-updates", "16", "--init", str(results / "A_initial/weights.pt")]))
    else:
        # 所有 LR 候选共享样本、父权重、恒定 LR 曲线及验证机会。
        import yaml
        selected_link = results / "A_selected"
        if selected_link.is_symlink() or selected_link.exists():
            raise FileExistsError(
                f"A_selected 已存在，拒绝复用旧父权重或覆盖：{selected_link}"
            )
        config = yaml.safe_load(base_config.read_text())
        for stage in ("A", "B", "C", "D"):
            config["stages"][stage].update(warmup_steps=0, warmup_ratio=0, min_lr_ratio=1)
        active_config = root / "configs/probe_sam2_constant_lr.yaml"
        if active_config.is_symlink() or active_config.exists():
            raise FileExistsError(f"恒定 LR 配置已存在，拒绝覆盖：{active_config}")
        with active_config.open("x") as handle:
            yaml.safe_dump(config, handle, sort_keys=False)
        base[base.index("--config") + 1] = str(active_config)
        for stage, lrs, updates in (("A", [3e-5, 1e-4, 3e-4], 32),
                                    ("B", [3e-5, 1e-4, 3e-4], 128)):
            for lr in lrs:
                name = f"{stage}_lr_{lr:g}"
                extra = ["--stage", stage, "--num-workers", "8", "--gradient-checkpointing", "off",
                         "--batch-size", "1", "--accumulation-steps", "4", "--lr", str(lr),
                         "--warmup-updates", "0", "--measure-updates", str(updates),
                         "--validate", "--val-samples", "32", "--save-weights", "weights.pt"]
                if stage == "B":
                    extra += ["--init", str(results / "A_selected/weights.pt")]
                candidates.append((name, extra))

    header = {
        "event": "suite_start",
        "phase": args.phase,
        "wall_time": time.time(),
        "probe_script": _fingerprint(probe_script),
        "suite_script": _fingerprint(Path(__file__).resolve()),
        "base_config": _fingerprint(base_config),
        "config": _fingerprint(active_config),
        "sam_checkpoint": _fingerprint(checkpoint),
        "data_root": str(data_root),
        "train_split": _fingerprint(train_split),
        "val_split": _fingerprint(val_split),
    }
    with suite.open("x") as record:
        record.write(json.dumps(header) + "\n")
        record.flush()
        for name, extra in candidates:
            if args.phase == "learning" and name.startswith("B_") and not (results / "A_selected").exists():
                selection = _select_candidate(results.glob("A_lr_*"))
                _link_selected(results / "A_selected", selection["selected"]["directory"])
                event = _selection_event(
                    "A", selection,
                    selection["selected"]["directory"] / "weights.pt",
                )
                record.write(json.dumps(event) + "\n")
                record.flush()
                print(f"Selected A: {selection['selected']['name']}", flush=True)
            output = results / name
            log = root / f"sam2_{name}.log"
            command = base + ["--output-dir", str(output)] + extra
            print(f"START {name}", flush=True)
            started = time.perf_counter()
            with log.open("x") as stream:
                completed = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, env=env, cwd=root)
            row = {"name": name, "returncode": completed.returncode,
                   "wall_seconds": time.perf_counter() - started, "command": command}
            path = output / "results.json"
            if path.exists():
                result = json.loads(path.read_text())
                row.update(measurement=result.get("measurement"), parameters=result.get("parameters"))
            validation = output / "validation.json"
            if validation.exists():
                row["validation"] = json.loads(validation.read_text())
            record.write(json.dumps(row) + "\n")
            record.flush()
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
