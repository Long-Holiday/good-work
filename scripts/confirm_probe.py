"""从 C 父权重确认选中配方，包含 D 点成本完整预热和稳态组。"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--lr", type=float, default=1e-6)
    args = parser.parse_args()
    root = args.root.resolve()
    output = root / "results_sam2/D_confirm_128"
    log_path = root / "sam2_D_confirm_128.log"
    if output.exists() or log_path.exists():
        parser.error("拒绝覆盖已有确认实验")
    command = [sys.executable, str(root / "scripts/probe_3090.py"),
        "--config", str(root / "configs/probe_sam2_constant_lr.yaml"),
        "--data-root", str(root.parent / "DeepGlobe_Road"),
        "--sam-checkpoint", str(root / "checkpoints/sam2_download/sam2.1_hiera_base_plus.pt"),
        "--train-split", str(root / "splits/protocol_v1/train.json"),
        "--val-split", str(root / "splits/protocol_v1/val.json"),
        "--stage", "D", "--output-dir", str(output),
        "--init", str(root / "results_sam2/C_selected/weights.pt"),
        "--num-workers", "8", "--cpu-threads", "2", "--interop-threads", "1",
        "--gradient-checkpointing", "off", "--batch-size", "1", "--group-size", "4",
        "--amp-init-scale", "64", "--lr", str(args.lr), "--seed", "2026",
        "--train-samples", "256", "--warmup-updates", "100", "--measure-updates", "28",
        "--validate", "--val-samples", "32", "--save-weights", "weights.pt"]
    print(json.dumps({"scope": "100 training warmup groups then 28 full-cost measurement groups", "command": command}), flush=True)
    env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1")
    with log_path.open("x") as stream:
        result = subprocess.run(command, cwd=root, env=env, stdout=stream, stderr=subprocess.STDOUT)
    print(json.dumps({"returncode": result.returncode, "output": str(output)}), flush=True)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
