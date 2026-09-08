"""Run the frozen r005 post-development confirmation matrix.

HRSC: B1/B2/B3/M1 on trainval->test, seeds 17/29/43, fixed development
epochs. DOTA: same arms and seeds on train->val.  All evaluation cells use
the frozen 4x4 degradation grid; B3 additionally keeps its expanded target.
"""
from __future__ import annotations

import json, os, re, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MMROTATE = Path('/home/rspip/zy/study/third_party/ai4rs')
OUT = ROOT / 'runs/r005/artifacts/r005_confirmation.json'
GRID = [(s, f) for s in (0., .8, 1.6, 3.2) for f in (1, 2, 4, 8)]
AP = re.compile(r'r005/(AP50|AP75|AR100):\s*([0-9.]+)')
ARMS = {'B1': ('B1', 1), 'B2': ('B2', 35), 'B3': ('B3', 1), 'M1': ('M1', 36)}
SEEDS = (17, 29, 43)

def invoke(argv, env):
    done = subprocess.run(argv, cwd=ROOT, env=env, text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, check=False)
    if done.returncode:
        raise RuntimeError('$ ' + ' '.join(map(str, argv)) + '\n' + done.stdout)
    return done.stdout

def environment(dataset, key, seed, fixed_epoch):
    method, _ = ARMS[key]
    work = ROOT / 'runs/r005/confirmation' / dataset / f'{key}_seed{seed}'
    env = os.environ.copy()
    env.update({'PYTHONPATH': str(ROOT), 'R005_DATASET': dataset, 'R005_ARM': key,
                'R005_METHOD_ARM': method, 'R005_SEED': str(seed),
                'R005_WORK_DIR': str(work), 'R005_PER_GPU_BATCH': '1',
                'R005_COVARIANCE_SCALE': '.5', 'R005_AMBIGUITY_THRESHOLD': '.2',
                'R005_FIXED_EPOCH': str(fixed_epoch)})
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
    for dataset in ('hrsc','dota'):
        for seed in SEEDS:
            for key, (_, epoch) in ARMS.items():
                name = f'{dataset}/{key}/seed{seed}'
                if name in results: continue
                env, work = environment(dataset, key, seed, epoch)
                if dataset == 'hrsc': env.update({'R005_HRSC_TRAIN_SPLIT':'trainval','R005_HRSC_EVAL_SPLIT':'test'})
                config = ROOT / ('configs/r005_confirmation.py' if dataset == 'hrsc' else 'configs/r005_confirmation_dota.py')
                # A failed post-training evaluation may be safely retried from
                # the frozen HRSC epoch checkpoint.  Never reuse DOTA's
                # validation-selected checkpoints, whose selection must run
                # with the current training attempt.
                frozen = work / f'epoch_{epoch}.pth' if dataset == 'hrsc' else None
                if frozen is None or not frozen.is_file():
                    invoke(['torchrun','--standalone','--nproc_per_node=2',str(MMROTATE/'tools/train.py'),str(config),'--launcher','pytorch'], env)
                ckpt = checkpoint(work, epoch if dataset == 'hrsc' else None)
                grid, expanded = evaluate(env, ckpt, key == 'B3')
                results[name] = {'dataset':dataset,'arm':key,'seed':seed,'checkpoint':str(ckpt.relative_to(ROOT)), 'grid':grid,
                                 **({'expanded_target_grid':expanded} if expanded else {})}
                OUT.parent.mkdir(parents=True, exist_ok=True)
                OUT.write_text(json.dumps(results, indent=2)+'\n')
    return 0

if __name__ == '__main__': raise SystemExit(main())
