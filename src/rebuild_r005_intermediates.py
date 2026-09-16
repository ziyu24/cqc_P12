"""Recreate declared historical intermediates without overwriting retained runs.

This is an on-demand recovery command, not part of the active training entry.
It replays the exact checkpoint-embedded configurations, including the old
35-epoch B2 budget. Rebuilt stochastic weights are not byte-identical evidence.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

if __package__:
    from .training_resources import training_gpu_environment
else:
    from training_resources import training_gpu_environment

ROOT = Path(__file__).resolve().parents[1]
TRAIN = Path('/home/rspip/zy/study/third_party/ai4rs/tools/train.py')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/r005_checkpoint_rebuild.json')
    parser.add_argument('--output-root', default='runs/r005/rebuilt_intermediates')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    plan = json.loads((ROOT / args.config).read_text(encoding='utf-8'))
    output_root = (ROOT / args.output_root).resolve()
    if ROOT / 'runs' not in output_root.parents:
        raise ValueError('recovery work directory must be inside this project runs/')
    jobs = []
    for group in plan['groups']:
        config = ROOT / group['configuration']
        actual = hashlib.sha256(config.read_text(encoding='utf-8').encode()).hexdigest()
        if actual != group['configuration_sha256']:
            raise RuntimeError(f'historical generating configuration changed: {config}')
        original = (ROOT / group['work_directory']).resolve()
        if ROOT / 'runs/r005/confirmation' not in original.parents:
            raise ValueError('invalid historical work directory')
        work = output_root / group['dataset'] / f"{group['arm']}_seed{group['seed']}"
        if work.exists() or work == original:
            raise FileExistsError(f'recovery requires a fresh output directory: {work}')
        targets = [(work / f'epoch_{epoch}.pth', original / f'epoch_{epoch}.pth')
                   for epoch in group['checkpoint_epochs']]
        if not args.dry_run and any(target.exists() for _, target in targets):
            raise FileExistsError('recovery never replaces an existing historical checkpoint')
        argv = [str(Path(sys.executable).with_name('torchrun')), '--standalone',
                '--nproc_per_node=2', str(TRAIN), str(config),
                '--launcher', 'pytorch', '--work-dir', str(work)]
        jobs.append((group, argv, targets))
        print(json.dumps({'dataset': group['dataset'], 'arm': group['arm'],
                          'seed': group['seed'], 'epochs': group['epochs'],
                          'gpus': 2, 'restore_files': len(targets), 'command': argv}), flush=True)
    if args.dry_run:
        return 0
    for group, argv, targets in jobs:
        env = os.environ.copy()
        env.update(PYTHONPATH=str(ROOT),
                   TORCH_HOME=str(ROOT / 'runs/r005/.cache/torch'))
        env.update(training_gpu_environment())
        # These literal snapshots preserve the original native training and
        # requested intermediate outputs. They are never used by active r005.
        subprocess.run(argv, cwd=ROOT, env=env, check=True)
        for source, target in targets:
            if not source.is_file():
                raise FileNotFoundError(source)
            # Exclusive create prevents replacing current/retained evidence.
            with source.open('rb') as incoming, target.open('xb') as outgoing:
                shutil.copyfileobj(incoming, outgoing)
                outgoing.flush()
                os.fsync(outgoing.fileno())
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
