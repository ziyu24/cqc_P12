"""Run the frozen r005 post-development confirmation matrix.

HRSC and DOTA: B1/B2/B3/M1 on their frozen split for seed 17 only, with the
same frozen 36-epoch budget.  The user cancelled every not-yet-started
seed-29/43 arm for both datasets.  All evaluation cells use the frozen 4x4
degradation grid; B3 additionally keeps its expanded target.
"""
from __future__ import annotations

import json, math, os, re, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

if __package__:
    from .training_resources import training_gpu_environment
else:
    from training_resources import training_gpu_environment

ROOT = Path(__file__).resolve().parents[1]
MMROTATE = Path('/home/rspip/zy/study/third_party/ai4rs')
OUT = ROOT / 'runs/r005/artifacts/r005_confirmation.json'
GRID = [(s, f) for s in (0., .8, 1.6, 3.2) for f in (1, 2, 4, 8)]
AP = re.compile(r'r005/(AP50|AP75|AR100):\s*([0-9.]+)')
ARMS = {'B1': ('B1', 36), 'B2': ('B2', 36), 'B3': ('B3', 36), 'M1': ('M1', 36)}
# This is deliberately shared by both datasets: it is the user-authorized
# post-cancellation scope, not an inference from existing artifact folders.
ACTIVE_SEEDS = (17,)
HRSC_SEEDS = ACTIVE_SEEDS
DOTA_SEEDS = ACTIVE_SEEDS
SAMPLER_PROVENANCE = 'DefaultSampler(shuffle=True), distributed rank sharding'
DOTA_ROOT = Path('/home/rspip/zy/data/dataset/dota/dota1.0/split_ss_dota10')
DOTA_CLASSES = {'plane', 'baseball-diamond', 'bridge', 'ground-track-field',
                'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
                'basketball-court', 'storage-tank', 'soccer-ball-field',
                'roundabout', 'harbor', 'swimming-pool', 'helicopter'}

def invoke(argv, env):
    done = subprocess.run(argv, cwd=ROOT, env=env, text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, check=False)
    if done.returncode:
        raise RuntimeError('$ ' + ' '.join(map(str, argv)) + '\n' + done.stdout)
    return done.stdout

def audit_dota_inputs():
    """Assert the frozen train->val tile identity and annotation mapping."""
    splits = {}
    for split in ('train', 'val'):
        image_dir, ann_dir = DOTA_ROOT / split / 'images', DOTA_ROOT / split / 'annfiles'
        if not image_dir.is_dir() or not ann_dir.is_dir():
            raise RuntimeError(f'missing DOTA {split} input directory')
        images = [p for p in image_dir.iterdir() if p.is_file()]
        annotations = [p for p in ann_dir.glob('*.txt') if p.is_file()]
        image_by_id = {p.stem: p for p in images}
        ann_by_id = {p.stem: p for p in annotations}
        if len(image_by_id) != len(images) or len(ann_by_id) != len(annotations):
            raise RuntimeError(f'DOTA {split} has duplicate tile identities')
        missing_ann, missing_image = sorted(set(image_by_id) - set(ann_by_id)), sorted(set(ann_by_id) - set(image_by_id))
        if missing_ann or missing_image:
            raise RuntimeError(f'DOTA {split} image/annotation coverage mismatch: missing_ann={missing_ann[:3]} missing_image={missing_image[:3]}')
        def count_valid_boxes(ann):
            count = 0
            for line_no, line in enumerate(ann.read_text().splitlines(), 1):
                fields = line.split()
                if len(fields) != 10 or fields[8] not in DOTA_CLASSES:
                    raise RuntimeError(f'invalid DOTA annotation {ann}:{line_no}')
                try:
                    coordinates = [float(v) for v in fields[:8]]
                    difficulty = int(fields[9])
                except ValueError as exc:
                    raise RuntimeError(f'non-numeric DOTA annotation {ann}:{line_no}') from exc
                if not all(math.isfinite(v) for v in coordinates) or difficulty not in (0, 1, 2):
                    raise RuntimeError(f'invalid DOTA geometry/difficulty {ann}:{line_no}')
                count += 1
            return count
        # These are independent, immutable text files on shared storage.  A
        # bounded pool retains whole-corpus validation while avoiding one
        # network round trip per file in sequence.
        with ThreadPoolExecutor(max_workers=16) as pool:
            boxes = sum(pool.map(count_valid_boxes, ann_by_id.values()))
        splits[split] = {'tiles': len(image_by_id), 'boxes': boxes,
                         'original_ids': {tile_id.split('__', 1)[0] for tile_id in image_by_id}}
    crossing = sorted(splits['train']['original_ids'] & splits['val']['original_ids'])
    if crossing:
        raise RuntimeError(f'DOTA train/val original-image crossing: {crossing[:3]}')
    audit = {'root': str(DOTA_ROOT), 'train_tiles': splits['train']['tiles'], 'val_tiles': splits['val']['tiles'],
             'train_boxes': splits['train']['boxes'], 'val_boxes': splits['val']['boxes'],
             'train_original_images': len(splits['train']['original_ids']),
             'val_original_images': len(splits['val']['original_ids']), 'crossing_original_images': 0}
    (OUT.parent / 'dota_input_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    return audit

def work_dir(dataset, key, seed):
    return ROOT / 'runs/r005/confirmation' / dataset / f'{key}_seed{seed}'

def environment(dataset, key, seed, fixed_epoch):
    method, _ = ARMS[key]
    work = work_dir(dataset, key, seed)
    env = os.environ.copy()
    env.update({'PYTHONPATH': str(ROOT), 'R005_DATASET': dataset, 'R005_ARM': key,
                'R005_METHOD_ARM': method, 'R005_SEED': str(seed),
                'R005_WORK_DIR': str(work), 'R005_PER_GPU_BATCH': '1',
                'R005_COVARIANCE_SCALE': '.5', 'R005_AMBIGUITY_THRESHOLD': '.2',
                'R005_FIXED_EPOCH': str(fixed_epoch)})
    env.update(training_gpu_environment())
    return env, work

def checkpoint(work, fixed_epoch=None):
    if fixed_epoch is not None:
        exact = work / f'epoch_{fixed_epoch}.pth'
        if not exact.is_file(): raise RuntimeError(f'missing frozen epoch checkpoint {exact}')
        return exact
    picks = sorted(work.glob('best_*.pth') if any(work.glob('best_*.pth')) else work.glob('epoch_*.pth'), key=lambda p: p.stat().st_mtime)
    if not picks: raise RuntimeError(f'no checkpoint in {work}')
    return picks[-1]

def evaluate(env, ckpt, b3):
    grid, expanded = {}, {}
    for sigma, factor in GRID:
        e = env.copy(); e.update({'R005_EVAL_SIGMA':str(sigma), 'R005_EVAL_FACTOR':str(factor)})
        config = ROOT / ('configs/r005_confirmation.py' if e['R005_DATASET'] == 'hrsc' else 'configs/r005_confirmation_dota.py')
        raw_root = Path(e['R005_WORK_DIR']) / 'per_image'
        raw_root.mkdir(parents=True, exist_ok=True)
        e['R005_DUMP_PATH'] = str(raw_root / f'sigma{sigma:g}_factor{factor}_original.pkl')
        out = invoke([sys.executable, str(MMROTATE/'tools/test.py'), str(config), str(ckpt), '--launcher','none'], e)
        metrics = {m:float(v) for m,v in AP.findall(out)}
        if set(metrics) != {'AP50','AP75','AR100'}: raise RuntimeError(f'missing primary metrics {sigma}/{factor}')
        if not Path(e['R005_DUMP_PATH']).is_file(): raise RuntimeError(f'missing per-image output {e["R005_DUMP_PATH"]}')
        grid[f'{sigma:g}/{factor}'] = metrics
        if b3:
            e['R005_EVAL_TARGET'] = 'expanded'
            e['R005_DUMP_PATH'] = str(raw_root / f'sigma{sigma:g}_factor{factor}_expanded.pkl')
            out = invoke([sys.executable, str(MMROTATE/'tools/test.py'), str(config), str(ckpt), '--launcher','none'], e)
            metrics = {m:float(v) for m,v in AP.findall(out)}
            if set(metrics) != {'AP50','AP75','AR100'}: raise RuntimeError(f'missing expanded metrics {sigma}/{factor}')
            if not Path(e['R005_DUMP_PATH']).is_file(): raise RuntimeError(f'missing per-image output {e["R005_DUMP_PATH"]}')
            expanded[f'{sigma:g}/{factor}'] = metrics
    return grid, expanded

def main():
    results = json.loads(OUT.read_text()) if OUT.exists() else {}
    for dataset, seeds in (('hrsc', HRSC_SEEDS), ('dota', DOTA_SEEDS)):
        for seed in seeds:
            for key, (_, epoch) in ARMS.items():
                name = f'{dataset}/{key}/seed{seed}'
                # Results generated before the confirmation loader explicitly
                # restored DefaultSampler were unsharded under DDP, doubling
                # optimizer steps per epoch.  They remain on disk for audit
                # but cannot satisfy this frozen confirmation protocol.
                work = work_dir(dataset, key, seed)
                target = str((work / f'epoch_{epoch}.pth').relative_to(ROOT))
                if (name in results and results[name].get('sampler_provenance') == SAMPLER_PROVENANCE
                        and results[name].get('checkpoint') == target):
                    continue
                results.pop(name, None)
                if dataset == 'dota':
                    print('DOTA input audit:', json.dumps(audit_dota_inputs(), sort_keys=True), flush=True)
                env, work = environment(dataset, key, seed, epoch)
                if dataset == 'hrsc': env.update({'R005_HRSC_TRAIN_SPLIT':'trainval','R005_HRSC_EVAL_SPLIT':'test'})
                config = ROOT / ('configs/r005_confirmation.py' if dataset == 'hrsc' else 'configs/r005_confirmation_dota.py')
                # Both datasets use the arm-specific epoch frozen during HRSC
                # development.  A failed grid evaluation can therefore retry
                # from that exact checkpoint without selecting on test/val.
                frozen = work / f'epoch_{epoch}.pth'
                if not frozen.is_file():
                    train = ['torchrun', '--standalone', '--nproc_per_node=2',
                             str(MMROTATE/'tools/train.py'), str(config), '--launcher', 'pytorch']
                    if any(work.glob('epoch_*.pth')):
                        train.append('--resume')
                    invoke(train, env)
                ckpt = checkpoint(work, epoch)
                grid, expanded = evaluate(env, ckpt, key == 'B3')
                results[name] = {'dataset':dataset,'arm':key,'seed':seed,'checkpoint':str(ckpt.relative_to(ROOT)),
                                 'sampler_provenance': SAMPLER_PROVENANCE,
                                 'gpu_physical': env['R005_GPU_PHYSICAL'], 'grid':grid,
                                 **({'expanded_target_grid':expanded} if expanded else {})}
                OUT.parent.mkdir(parents=True, exist_ok=True)
                OUT.write_text(json.dumps(results, indent=2)+'\n')
    return 0

if __name__ == '__main__': raise SystemExit(main())
