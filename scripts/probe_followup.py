"""从等预算 B 筛选结果串联 C/D 开发探针，保留全部候选与失败。"""
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

    Validation/weights files alone are not success: a probe can write them
    after zero optimizer updates or with non-finite parameters.  Compared
    candidates must also share the same train/val subset.
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

    A partially failed sweep can still be compared among its valid members;
    the failure reasons are recorded with the selection.  Ties use the
    declared candidate order (ascending learning rate), matching the
    protocol's "prefer the earlier candidate" rule, and are recorded rather
    than resolved silently by Path ordering.
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
    args = parser.parse_args()
    root = args.root.resolve()
    results = root / "results_sam2"
    suite = root / "suite_sam2_followup.jsonl"
    config_path = root / "configs/probe_sam2_constant_lr.yaml"
    # 先验证输入与独占路径，再做任何链接/文件写入。
    if not config_path.is_file():
        raise FileNotFoundError(
            f"缺少恒定 LR 配置：{config_path}；请先运行 run_probe_suite.py --phase learning"
        )
    if suite.is_symlink() or suite.exists():
        raise FileExistsError(f"suite 记录已存在，拒绝覆盖：{suite}")
    for link in (results / "B_selected", results / "C_selected"):
        # 在启动任何昂贵候选前检查父权重链接，避免跑到一半才发现被占用。
        if link.is_symlink() or link.exists():
            raise FileExistsError(f"父权重链接已存在，拒绝覆盖：{link}")
    selection = _select_candidate(results.glob("B_lr_*"))

    base = [sys.executable, str(root / "scripts/probe_3090.py"),
            "--config", str(config_path),
            "--data-root", str(root.parent / "DeepGlobe_Road"),
            "--sam-checkpoint", str(root / "checkpoints/sam2_download/sam2.1_hiera_base_plus.pt"),
            "--train-split", str(root / "splits/protocol_v1/train.json"),
            "--val-split", str(root / "splits/protocol_v1/val.json"),
            "--seed", "2026", "--cpu-threads", "2", "--interop-threads", "1",
            "--num-workers", "8", "--batch-size", "1", "--train-samples", "256",
            "--gradient-checkpointing", "off"]
    env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1")
    header = {
        "event": "suite_start",
        "wall_time": time.time(),
        "probe_script": _fingerprint(root / "scripts/probe_3090.py"),
        "followup_script": _fingerprint(Path(__file__).resolve()),
        "config": _fingerprint(config_path),
        "sam_checkpoint": _fingerprint(
            root / "checkpoints/sam2_download/sam2.1_hiera_base_plus.pt"
        ),
        "train_split": _fingerprint(root / "splits/protocol_v1/train.json"),
        "val_split": _fingerprint(root / "splits/protocol_v1/val.json"),
    }
    with suite.open("x") as record:
        record.write(json.dumps(header) + "\n")
        _link_selected(results / "B_selected", selection["selected"]["directory"])
        record.write(json.dumps(_selection_event(
            "B", selection, selection["selected"]["directory"] / "weights.pt"
        )) + "\n")
        record.flush()
        print(f"Selected B by fixed-subset automatic IoU: {selection['selected']['name']}", flush=True)

        def run(name, extra):
            command = base + ["--output-dir", str(results / name)] + extra
            print(f"START {name}", flush=True)
            started = time.perf_counter()
            with (root / f"sam2_{name}.log").open("x") as stream:
                completed = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, env=env, cwd=root)
            row = {"name": name, "returncode": completed.returncode,
                   "wall_seconds": time.perf_counter() - started, "command": command}
            for key, filename in (("result", "results.json"), ("validation", "validation.json"), ("error", "error.json")):
                path = results / name / filename
                if path.exists():
                    row[key] = json.loads(path.read_text())
            record.write(json.dumps(row) + "\n")
            record.flush()
            print(json.dumps(row), flush=True)
            return row

        for lr in (2e-5, 5e-5, 1e-4):
            run(f"C_lr_{lr:g}", ["--stage", "C", "--init", str(selection["selected"]["directory"] / "weights.pt"),
                "--lr", str(lr), "--amp-init-scale", "1024", "--accumulation-steps", "4",
                "--warmup-updates", "0", "--measure-updates", "16",
                "--validate", "--val-samples", "32", "--save-weights", "weights.pt"])
        c_selection = _select_candidate(results.glob("C_lr_*"))
        _link_selected(results / "C_selected", c_selection["selected"]["directory"])
        record.write(json.dumps(_selection_event(
            "C", c_selection, c_selection["selected"]["directory"] / "weights.pt"
        )) + "\n")
        record.flush()
        print(f"Selected C: {c_selection['selected']['name']}", flush=True)
        # 先确认真实 G=4/20 步重放能实际更新，再扩大每个候选预算。
        d_system = {}
        for checkpointing in ("off", "on"):
            d_system[checkpointing] = run(
                f"D_system_{checkpointing}",
                ["--stage", "D", "--init", str(c_selection["selected"]["directory"] / "weights.pt"),
                 "--lr", "1e-6", "--amp-init-scale", "64", "--group-size", "4",
                 "--gradient-checkpointing", checkpointing, "--warmup-updates", "1",
                 "--measure-updates", "4"],
            )
        system_candidates = {
            key: _candidate(results / f"D_system_{key}",
                            require_validation=False, require_weights=False)
            for key in ("off", "on")
        }
        record.write(json.dumps({
            "event": "gate", "stage": "D_system",
            "candidates": [_candidate_summary(row) for row in system_candidates.values()],
        }) + "\n")
        record.flush()
        # D_lr 固定关闭梯度检查点；只有这条真实配置确认更新后才扩大预算。
        if not system_candidates["off"]["valid"]:
            raise RuntimeError(
                "D 系统探针未确认真实更新，停止 D_lr："
                + ",".join(system_candidates["off"]["reasons"])
            )
        for lr in (3e-7, 1e-6, 3e-6):
            run(f"D_lr_{lr:g}", ["--stage", "D", "--init", str(c_selection["selected"]["directory"] / "weights.pt"),
                "--lr", str(lr), "--amp-init-scale", "64", "--group-size", "4",
                "--warmup-updates", "0", "--measure-updates", "32",
                "--validate", "--val-samples", "32", "--save-weights", "weights.pt"])


if __name__ == "__main__":
    main()
