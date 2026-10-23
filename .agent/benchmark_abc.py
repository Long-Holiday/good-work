"""Bounded throughput probes using the production ABC training loop."""
import argparse
import itertools
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


def candidate(args):
    import torch
    sys.path.insert(0, str(Path(args.repo).resolve()))
    from diffpeak.engine import DiffPeakEngine
    from diffpeak.inference import load_config
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    config = load_config(str(Path(args.repo) / 'configs/3090.yaml'))
    config['output_dir'] = str(Path(args.output) / args.name)
    config['model']['unet']['gradient_checkpointing'] = bool(args.checkpoint)
    config['stages']['ABC'].update(batch_size=args.batch, accumulation_steps=4 // args.batch, num_workers=args.workers)
    engine = DiffPeakEngine(config, config_dir=Path(args.repo) / 'configs')
    engine._apply_stage_config('ABC')
    loader = engine._build_train_loader(0)
    engine._prepare_stage('ABC', 60, math.ceil(len(loader) / engine.accumulation_steps))
    engine.log_every = 1000000
    iterator = iter(loader)
    engine._train_epoch_supervised('ABC', 0, list(itertools.islice(iterator, 3 * engine.accumulation_steps)))
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    class BoundedLoader:
        def __len__(self):
            return args.updates * engine.accumulation_steps
        def __iter__(self):
            return itertools.islice(iterator, len(self))
    stats = engine._train_epoch_supervised('ABC', 0, BoundedLoader())
    torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    finite = all(bool(torch.isfinite(p).all()) for p in engine._trainable_parameters())
    result = dict(name=args.name, batch=args.batch, accumulation=4 // args.batch, workers=args.workers, threads=args.threads, checkpoint=bool(args.checkpoint), seconds=seconds, images=args.updates * 4, images_per_second=args.updates * 4 / seconds, peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20, peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20, params_finite=finite, stats=stats)
    result['healthy'] = finite and stats['optimizer_steps'] == args.updates and stats.get('skipped_groups', 0) == 0 and all(math.isfinite(stats[k]) for k in ('loss', 'flow', 'seg', 'sampled_seg', 'heat', 'aux'))
    Path(args.result).write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


def suite(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    results = []
    def run(batch, checkpoint, workers=4, threads=2, updates=10, suffix=''):
        name = f'b{batch}_ck{checkpoint}_w{workers}_t{threads}{suffix}'
        result_path = output / f'{name}.json'
        cmd = [sys.executable, __file__, '--repo', args.repo, '--output', str(output), '--name', name, '--batch', str(batch), '--checkpoint', str(checkpoint), '--workers', str(workers), '--threads', str(threads), '--updates', str(updates), '--result', str(result_path)]
        env = dict(os.environ, OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads), OPENBLAS_NUM_THREADS='1')
        with (output / f'{name}.log').open('w') as log:
            try:
                proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, timeout=600)
                row = json.loads(result_path.read_text()) if proc.returncode == 0 and result_path.exists() else dict(name=name, healthy=False, returncode=proc.returncode)
            except subprocess.TimeoutExpired:
                row = dict(name=name, healthy=False, error='timeout')
        results.append(row)
        (output / 'suite.json').write_text(json.dumps(results, indent=2))
        print(json.dumps(row), flush=True)
        return row
    for checkpoint in (1, 0):
        for batch in (1, 2, 4):
            run(batch, checkpoint)
    good = [r for r in results if r.get('healthy')]
    if not good:
        raise RuntimeError('No healthy candidate')
    best = max(good, key=lambda r: r['images_per_second'])
    run(best['batch'], int(best['checkpoint']), 8, 2)
    run(best['batch'], int(best['checkpoint']), 4, 4)
    best = max((r for r in results if r.get('healthy')), key=lambda r: r['images_per_second'])
    for r in sorted((r for r in results if r.get('healthy')), key=lambda r: r['images_per_second'], reverse=True)[:2]:
        run(r['batch'], int(r['checkpoint']), r['workers'], r['threads'], 30, '_confirm')
    confirmed = [r for r in results if r.get('healthy') and r['name'].endswith('_confirm') and r['peak_reserved_mib'] < 22000]
    if not confirmed:
        raise RuntimeError('No confirmed candidate with memory headroom')
    best = max(confirmed, key=lambda r: r['images_per_second'])
    (output / 'winner.json').write_text(json.dumps(best, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--suite', action='store_true')
    p.add_argument('--repo', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--name')
    p.add_argument('--result')
    p.add_argument('--batch', type=int, default=1)
    p.add_argument('--checkpoint', type=int, default=1)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--updates', type=int, default=10)
    args = p.parse_args()
    try:
        suite(args) if args.suite else candidate(args)
    except Exception:
        traceback.print_exc()
        sys.exit(1)
