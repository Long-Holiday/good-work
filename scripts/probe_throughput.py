"""Portable A/B/C/D capacity sweep; dry-run needs only Python and PyYAML."""
from __future__ import annotations

import argparse
import copy
import hashlib
import fcntl
import json
import math
import os
import signal
from pathlib import Path
import statistics
import subprocess
import sys
import time

import yaml

ROOT = Path(__file__).resolve().parents[1]


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=ROOT / 'configs/3090.yaml')
    p.add_argument('--data-root', type=Path)
    p.add_argument('--sam-checkpoint', type=Path)
    p.add_argument('--train-split', type=Path)
    p.add_argument('--val-split', type=Path)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--stages', nargs='+', choices=list('ABCD'), default=list('ABCD'))
    p.add_argument('--workers', nargs='+', type=int, default=[4, 8])
    p.add_argument('--batches', nargs='+', type=int, default=[1, 2, 4])
    p.add_argument('--effective-batch', type=int, default=4)
    p.add_argument('--checkpointing', nargs='+', choices=['off', 'on'], default=['off', 'on'])
    p.add_argument('--d-batches', nargs='+', type=int, default=[1, 2])
    p.add_argument('--repeats', type=int, default=2)
    p.add_argument('--warmup-updates', type=int, default=3)
    p.add_argument('--measure-updates', type=int, default=16)
    p.add_argument('--d-warmup-groups', type=int, default=2)
    p.add_argument('--d-measure-groups', type=int, default=4)
    p.add_argument('--train-samples', type=int, default=256)
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--cpu-threads', type=int, default=2)
    p.add_argument('--interop-threads', type=int, default=1)
    p.add_argument('--candidate-timeout', type=float, default=1800)
    p.add_argument('--max-seconds', type=float, default=10800)
    for stage in 'ABCD':
        p.add_argument(f'--init-{stage.lower()}', type=Path)
    p.add_argument('--bootstrap', action='store_true', help='Use first healthy baseline weights as next-stage parent; capacity claims only')
    p.add_argument('--dry-run', action='store_true', help='Print plan without creating files or importing torch')
    p.add_argument('--check-inputs', action='store_true', help='Also require weights and explicit parent files during dry-run')
    return p


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def build_plan(args):
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text())
    if not isinstance(config, dict):
        raise ValueError('Config must be a mapping')
    for key in ('repeats', 'effective_batch', 'measure_updates', 'd_measure_groups',
                'train_samples', 'cpu_threads', 'interop_threads'):
        if getattr(args, key) < 1:
            raise ValueError(f'{key} must be positive')
    if args.warmup_updates < 0 or args.d_warmup_groups < 0:
        raise ValueError('Warmup budgets must be nonnegative')
    if any(w < 0 for w in args.workers):
        raise ValueError('Workers must be nonnegative')
    if any(b < 1 or args.effective_batch % b for b in args.batches):
        raise ValueError('Each A/B/C batch must divide effective_batch')
    if any(b < 1 or args.d_measure_groups % b or args.d_warmup_groups % b for b in args.d_batches):
        raise ValueError('D batch must divide both group budgets to avoid rounding')
    if args.train_samples < max(args.batches + args.d_batches):
        raise ValueError('train_samples must cover largest batch')
    if any(not math.isfinite(v) or v <= 0 for v in (args.candidate_timeout, args.max_seconds)):
        raise ValueError('Timeouts must be positive and finite')
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f'Refusing existing output directory: {output}')
    # Config-relative defaults are project paths, not configs/ paths.
    project = ROOT
    def resolve(value):
        value = Path(value)
        return (value if value.is_absolute() else project / value).resolve()
    inputs = {
        'data_root': resolve(args.data_root or config['data']['root']),
        'sam_checkpoint': resolve(args.sam_checkpoint or config['model']['sam_checkpoint']),
        'train_split': resolve(args.train_split or config['data']['train_split']),
        'val_split': resolve(args.val_split or config['data']['val_split']),
    }
    warnings = []
    for name, path in inputs.items():
        if not path.exists():
            if not args.dry_run or args.check_inputs or name in ('data_root', 'train_split', 'val_split'):
                raise FileNotFoundError(f'{name}: {path}')
            warnings.append(f'Missing {name}: {path}; required before execution')
    # Validate full manifests without reading validation pixels or touching test data.
    sets = {}
    for name in ('train_split', 'val_split'):
        entries = json.loads(inputs[name].read_text())
        if not isinstance(entries, list) or not entries or any(not isinstance(e, str) for e in entries):
            raise ValueError(f'{name} must be a nonempty list of image paths')
        if len(set(entries)) != len(entries):
            raise ValueError(f'Duplicate entries in {name}')
        paths = set()
        for entry in entries:
            image = (inputs['data_root'] / entry).resolve()
            if not image.is_relative_to(inputs['data_root']) or not entry.startswith('train/') or not entry.endswith('_sat.jpg'):
                raise ValueError(f'Unsafe/noncanonical training source: {entry}')
            mask = image.with_name(image.name.replace('_sat.jpg', '_mask.png'))
            if not image.is_file() or not mask.is_file():
                raise FileNotFoundError(f'Missing image/mask pair: {image}')
            paths.add(str(image))
        sets[name] = paths
    if sets['train_split'] & sets['val_split']:
        raise ValueError('Train and validation overlap')
    if args.train_samples > len(sets['train_split']):
        raise ValueError('train_samples exceeds train manifest')
    stages = [s for s in 'ABCD' if s in args.stages]
    candidates = []
    parents = {}
    for stage in stages:
        parent = getattr(args, f'init_{stage.lower()}')
        if parent:
            parent = parent.resolve()
            if not parent.is_file():
                if not args.dry_run or args.check_inputs:
                    raise FileNotFoundError(parent)
                warnings.append(f'Missing parent: {parent}')
        elif stage != 'A':
            previous = 'ABCD'['ABCD'.index(stage) - 1]
            if args.bootstrap and previous in parents:
                parent = parents[previous]
            else:
                raise ValueError(f'Stage {stage} requires --init-{stage.lower()} or --bootstrap with preceding stage')
        stage_rows = []
        batches = args.d_batches if stage == 'D' else args.batches
        for batch in dict.fromkeys(batches):
            for workers in dict.fromkeys(args.workers):
                for checkpointing in dict.fromkeys(args.checkpointing):
                    name = f'b{batch}_w{workers}_{checkpointing}'
                    for repeat in range(args.repeats):
                        directory = output / stage / f'{name}_r{repeat}'
                        save = not stage_rows and stage != 'D' and args.bootstrap
                        cmd = [sys.executable, str(ROOT / 'scripts/probe_3090.py'),
                               '--config', str(config_path), '--stage', stage,
                               '--output-dir', str(directory), '--device', 'cuda',
                               '--batch-size', str(batch), '--num-workers', str(workers),
                               '--gradient-checkpointing', checkpointing,
                               '--seed', str(args.seed), '--train-samples', str(args.train_samples),
                               '--cpu-threads', str(args.cpu_threads), '--interop-threads', str(args.interop_threads),
                               '--warmup-updates', str(args.d_warmup_groups if stage == 'D' else args.warmup_updates),
                               '--measure-updates', str(args.d_measure_groups if stage == 'D' else args.measure_updates)]
                        for key, value in inputs.items():
                            cmd += ['--' + key.replace('_', '-'), str(value)]
                        if stage != 'D':
                            cmd += ['--accumulation-steps', str(args.effective_batch // batch)]
                        if parent:
                            cmd += ['--init', str(parent)]
                        if save:
                            cmd += ['--save-weights', 'weights.pt']
                            parents[stage] = directory / 'weights.pt'
                        row = dict(stage=stage, name=name, repeat=repeat, directory=str(directory),
                                   batch_size=batch, accumulation_steps=None if stage == 'D' else args.effective_batch // batch,
                                   workers=workers, checkpointing=checkpointing, parent=str(parent) if parent else None,
                                   expected_samples=(args.d_measure_groups if stage == 'D' else args.measure_updates * args.effective_batch),
                                   command=cmd)
                        stage_rows.append(row)
        candidates.extend(stage_rows)
    if 'D' in stages and args.d_warmup_groups < config.get('rl', {}).get('point_cost_warmup_steps', 0):
        warnings.append('D is a short capacity sweep before point-cost warmup completes; no steady-state or quality claim')
    return dict(format_version=1, purpose='capacity_throughput_only', config=str(config_path),
                config_sha256=digest(config_path), inputs={k: str(v) for k, v in inputs.items()},
                input_sha256={k: digest(v) for k, v in inputs.items() if v.is_file()},
                output_dir=str(output), stages=stages, warnings=warnings,
                d_protocol={k: config.get(k) for k in ('rl', 'diffusion', 'reward')}, candidates=candidates)


def load_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def assess(row, returncode):
    directory = Path(row['directory'])
    result = load_json(directory / 'results.json')
    measurement = result.get('measurement', {})
    optimizer = measurement.get('optimizer', {})
    throughput = measurement.get('throughput', {})
    splits = load_json(directory / 'splits.json')
    speed = throughput.get('samples_per_second')
    reasons = []
    if returncode != 0 or result.get('status') != 'ok':
        reasons.append('failed')
    if result.get('parameters', {}).get('all_finite') is not True or measurement.get('all_metrics_finite') is not True:
        reasons.append('nonfinite_or_missing_metrics')
    if optimizer.get('successful_updates', 0) <= 0:
        reasons.append('no_successful_updates')
    if row.get('expected_samples') is not None and measurement.get('samples') != row['expected_samples']:
        reasons.append('incomplete_sample_budget')
    if optimizer.get('amp_skipped_updates', optimizer.get('skipped_updates', 0)):
        reasons.append('skipped_updates')
    if not isinstance(speed, (int, float)) or isinstance(speed, bool) or not math.isfinite(speed) or speed <= 0:
        reasons.append('invalid_throughput')
    split_key = tuple(splits.get(s, {}).get('subset_entries_sha256') for s in ('train', 'val'))
    if any(v is None for v in split_key):
        reasons.append('missing_subset_hash')
    return row | dict(returncode=returncode, valid=not reasons, reasons=reasons,
                      samples_per_second=speed, split_key=split_key, measurement=measurement,
                      error=load_json(directory / 'error.json'))


def summarize(rows, repeats):
    summary = {}
    for stage in 'ABCD':
        stage_rows = [r for r in rows if r['stage'] == stage]
        if not stage_rows:
            continue
        groups = {}
        for row in stage_rows:
            groups.setdefault(row['name'], []).append(row)
        ranked = []
        keys = {tuple(r['split_key']) for r in stage_rows if r['valid']}
        if len(keys) > 1:
            summary[stage] = dict(status='incomparable_subsets', selected=None)
            continue
        for name, group in groups.items():
            if len(group) == repeats and all(r['valid'] for r in group):
                ranked.append(dict(name=name, median_samples_per_second=statistics.median(r['samples_per_second'] for r in group),
                                   min_samples_per_second=min(r['samples_per_second'] for r in group),
                                   max_samples_per_second=max(r['samples_per_second'] for r in group),
                                   settings={k: group[0][k] for k in ('batch_size', 'accumulation_steps', 'workers', 'checkpointing')}))
        ranked.sort(key=lambda r: -r['median_samples_per_second'])
        summary[stage] = dict(status='ok' if ranked else 'no_valid_candidate', selected=ranked[0] if ranked else None, ranked=ranked)
    return summary


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def run_candidate(command, handle, env, timeout):
    # Workers inherit a new session so a timeout cannot leave DataLoader children.
    with subprocess.Popen(command, cwd=ROOT, env=env, stdout=handle,
                          stderr=subprocess.STDOUT, start_new_session=True) as process:
        try:
            return process.wait(timeout=timeout)
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
            raise


def execute(plan, args):
    output = Path(plan['output_dir'])
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'plan.json', plan)
    start = time.monotonic()
    rows = []
    env = dict(os.environ, OMP_NUM_THREADS=str(args.cpu_threads), MKL_NUM_THREADS=str(args.cpu_threads), OPENBLAS_NUM_THREADS='1')
    with (output / 'suite.jsonl').open('x') as ledger:
        for row in plan['candidates']:
            remaining = args.max_seconds - (time.monotonic() - start)
            if remaining <= 0:
                break
            print(f"{row['stage']}/{row['name']} repeat={row['repeat']}", flush=True)
            # Missing bootstrap parents cannot silently fall back to random weights.
            parent_run = next((r for r in rows if str(Path(r['directory']) / 'weights.pt') == row['parent']), None)
            if row['parent'] and (not Path(row['parent']).is_file() or (parent_run is not None and not parent_run['valid'])):
                code = 125
            else:
                log = output / f"{row['stage']}_{row['name']}_r{row['repeat']}.log"
                with log.open('x') as handle:
                    try:
                        code = run_candidate(row['command'], handle, env, min(args.candidate_timeout, remaining))
                    except subprocess.TimeoutExpired:
                        code = 124
                    except OSError as exc:
                        handle.write(f'Candidate launch failed: {exc}\n')
                        code = 126
            measured = assess(row, code)
            rows.append(measured)
            ledger.write(json.dumps(measured, allow_nan=False) + '\n')
            ledger.flush()
            write_json(output / 'summary.json', summarize(rows, args.repeats))
    summary = summarize(rows, args.repeats)
    write_json(output / 'summary.json', summary)
    config = yaml.safe_load(Path(plan['config']).read_text())
    for stage, selected in summary.items():
        if not selected['selected']:
            continue
        settings = selected['selected']['settings']
        derived = copy.deepcopy(config)
        derived['data']['root'] = plan['inputs']['data_root']
        derived['data']['train_split'] = plan['inputs']['train_split']
        derived['data']['val_split'] = plan['inputs']['val_split']
        derived['model']['sam_checkpoint'] = plan['inputs']['sam_checkpoint']
        derived['data']['num_workers'] = settings['workers']
        derived['train']['batch_size'] = settings['batch_size']
        if stage != 'D':
            derived['train']['accumulation_steps'] = settings['accumulation_steps']
        derived['model']['gradient_checkpointing'] = settings['checkpointing'] == 'on'
        (output / f'recommended_{stage}.yaml').write_text('# Capacity recommendation; validate training quality separately.\n' + yaml.safe_dump(derived, sort_keys=False))
    complete = len(rows) == len(plan['candidates']) and all(summary.get(s, {}).get('status') == 'ok' for s in plan['stages'])
    write_json(output / 'completion.json', dict(complete=complete, executed=len(rows), planned=len(plan['candidates']), elapsed_seconds=time.monotonic() - start))
    return 0 if complete else 1


def main():
    args = parser().parse_args()
    try:
        plan = build_plan(args)
        if args.dry_run:
            print(json.dumps(plan, indent=2, allow_nan=False))
            return 0
        # Serialize cooperating local suites on the visible GPU set.
        lock_name = hashlib.sha256(os.environ.get('CUDA_VISIBLE_DEVICES', 'all').encode()).hexdigest()[:16]
        with open(f'/tmp/diffpeak-throughput-{os.getuid()}-{lock_name}.lock', 'a') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise ValueError('Another throughput suite holds the GPU lock')
            return execute(plan, args)
    except (ValueError, OSError, KeyError) as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
