"""Apply a confirmed throughput winner and restart the full training protocol."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import yaml

repo = Path('/root/good-work')
probe = Path('/root/autodl-tmp/abc_probe_20261022')
winner = json.loads((probe / 'winner.json').read_text())
assert winner['healthy'] and winner['name'].endswith('_confirm')
assert winner['batch'] * winner['accumulation'] == 4
assert winner['stats']['optimizer_steps'] == 30
assert winner['peak_reserved_mib'] < 22000
path = repo / 'configs/3090.yaml'
backup = probe / '3090_before.yaml'
if backup.exists():
    raise RuntimeError('Configuration was already applied; refusing a duplicate restart')
shutil.copy2(path, backup)
text = path.read_text()
old = '  ABC: {epochs: 60, lr: 0.0001, batch_size: 1, accumulation_steps: 4}'
new = f"  ABC: {{epochs: 60, lr: 0.0001, batch_size: {winner['batch']}, accumulation_steps: {winner['accumulation']}, num_workers: {winner['workers']}}}"
assert text.count(old) == 1
text = text.replace(old, new)
old_ck = '    gradient_checkpointing: true'
assert text.count(old_ck) == 1
text = text.replace(old_ck, f"    gradient_checkpointing: {str(winner['checkpoint']).lower()}")
yaml.safe_load(text)
path.write_text(text)
shutil.copy2(path, probe / '3090_selected.yaml')
stamp = time.strftime('%Y%m%dT%H%M%S')
output = f'/root/autodl-tmp/runs/diffpeak/throughput_selected_{stamp}'
log = f'/root/good-work/train_session_{stamp}.log'
threads = winner['threads']
command = f'cd /root/good-work && OMP_NUM_THREADS={threads} MKL_NUM_THREADS={threads} OPENBLAS_NUM_THREADS=1 /root/miniconda3/bin/python train.py --config configs/3090.yaml --stage all --output-dir {output} --cpu-threads {threads} --interop-threads 1 > {log} 2>&1'
subprocess.run(['tmux', 'new-session', '-d', '-s', f'diffpeak_optimized_{stamp}', command], check=True)
(probe / 'restart.json').write_text(json.dumps(dict(winner=winner, output=output, log=log, command=command), indent=2))
print((probe / 'restart.json').read_text())
