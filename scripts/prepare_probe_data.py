"""Copy a small deterministic train/val probe dataset without touching test."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import shutil

ROOT = Path(__file__).resolve().parents[1]


def select_entries(manifest, count, seed, root):
    entries = json.loads(manifest.read_text())
    if not isinstance(entries, list) or any(not isinstance(e, str) for e in entries):
        raise ValueError('Manifest must be a list of strings')
    if len(entries) != len(set(entries)) or count < 1 or count > len(entries):
        raise ValueError('Duplicate entries or invalid subset size')
    # Validate full source list before sampling so escaping entries cannot hide.
    for entry in entries:
        image = (root / entry).resolve()
        if not entry.startswith('train/') or not entry.endswith('_sat.jpg') or not image.is_relative_to(root):
            raise ValueError(f'Unsafe training source: {entry}')
    chosen = sorted(random.Random(seed).sample(sorted(entries), count))
    for entry in chosen:
        image = root / entry
        mask = image.with_name(image.name.replace('_sat.jpg', '_mask.png'))
        if not image.is_file() or not mask.is_file() or not mask.resolve().is_relative_to(root):
            raise FileNotFoundError(f'Missing/unsafe image pair: {image}')
    return entries, chosen


def prepare(root, train_manifest, val_manifest, output, train_samples=256, val_samples=32, seed=2026):
    root, output = Path(root).resolve(), Path(output).resolve()
    train_manifest, val_manifest = Path(train_manifest).resolve(), Path(val_manifest).resolve()
    if output.exists():
        raise FileExistsError(f'Refusing existing output: {output}')
    if output.is_relative_to(root):
        raise ValueError('Output must be outside source dataset')
    train_all, train = select_entries(train_manifest, train_samples, seed, root)
    val_all, val = select_entries(val_manifest, val_samples, seed, root)
    train_paths = {(root / e).resolve() for e in train_all}
    if train_paths & {(root / e).resolve() for e in val_all}:
        raise ValueError('Train/val source manifests overlap')
    output.mkdir(parents=True)
    files, total = [], 0
    for entry in train + val:
        for relative in (Path(entry), Path(entry.replace('_sat.jpg', '_mask.png'))):
            source = root / relative
            target = output / 'data' / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            size = target.stat().st_size
            total += size
            files.append(dict(path=str(Path('data') / relative), bytes=size, sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
    splits = output / 'splits'
    splits.mkdir()
    for name, entries in [('train', train), ('val', val)]:
        (splits / f'{name}.json').write_text(json.dumps(entries, indent=2) + '\n')
    metadata = dict(purpose='capacity_probe_only', seed=seed, train_samples=len(train), val_samples=len(val),
                    source_manifest_sha256={name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in [('train', train_manifest), ('val', val_manifest)]},
                    copied_bytes=total, files=files)
    (output / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    return metadata


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, default=ROOT / 'DeepGlobe_Road')
    p.add_argument('--train-split', type=Path, default=ROOT / 'splits/protocol_v1/train.json')
    p.add_argument('--val-split', type=Path, default=ROOT / 'splits/protocol_v1/val.json')
    p.add_argument('--output-dir', required=True, type=Path)
    p.add_argument('--train-samples', type=int, default=256)
    p.add_argument('--val-samples', type=int, default=32)
    p.add_argument('--seed', type=int, default=2026)
    a = p.parse_args()
    try:
        result = prepare(a.data_root, a.train_split, a.val_split, a.output_dir, a.train_samples, a.val_samples, a.seed)
    except (ValueError, OSError) as exc:
        p.error(str(exc))
    print(json.dumps({k: v for k, v in result.items() if k != 'files'}, indent=2))


if __name__ == '__main__':
    main()
