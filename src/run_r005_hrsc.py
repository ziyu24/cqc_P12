"""r005 HRSC development executor: train arms then score every frozen grid cell.

The script is deliberately a small process orchestrator, not an alternate
training loop. It invokes the checked-in MMRotate config for each arm and
records stdout-derived AP50/AP75 for all 16 fixed validation degradations.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MMROTATE = Path('/home/rspip/zy/study/third_party/ai4rs')
CONFIG = ROOT / 'configs/r005_hrsc.py'
OUT = ROOT / 'runs/r005/artifacts/hrsc_development.json'
GRID = [(sigma, factor) for sigma in (0., .8, 1.6, 3.2) for factor in (1, 2, 4, 8)]
AP = re.compile(r"r005/(AP50|AP75):\s*([0-9.]+)")


def env_for(arm: str, covariance: float = 1., threshold: float = .0) -> dict[str, str]:
    env = os.environ.copy()
    env.update({'PYTHONPATH': str(ROOT), 'R005_ARM': arm,
                'R005_WORK_DIR': str(ROOT / 'runs/r005/work_dirs'),
                'R005_COVARIANCE_SCALE': str(covariance),
                'R005_AMBIGUITY_THRESHOLD': str(threshold),
                # DDP uses two cards with unchanged global batch 2.
                'R005_PER_GPU_BATCH': '1'})
    return env


def run(argv: list[str], env: dict[str, str]) -> str:
    completed = subprocess.run(argv, cwd=ROOT, env=env, text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               check=True)
    return completed.stdout


def checkpoint(work: Path) -> Path:
    choices = sorted(work.glob('best_*.pth'), key=lambda item: item.stat().st_mtime)
    if not choices:
        raise RuntimeError(f'no best checkpoint in {work}')
    return choices[-1]


def train_and_grid(name: str, arm: str, covariance=1., threshold=.0) -> dict:
    env = env_for(arm, covariance, threshold)
    work = ROOT / 'runs/r005/work_dirs' / name
    env['R005_WORK_DIR'] = str(work.parent)
    # `arm` controls config work_dir, so name is made unique by a symlink-free
    # environment suffix in config's root below.
    env['R005_ARM'] = name
    # Map named M1 settings back to implementation arm through a second var.
    env['R005_METHOD_ARM'] = arm
    # Config recognizes the canonical arm; use a separate work root per name.
    env['R005_WORK_DIR'] = str(ROOT / 'runs/r005/work_dirs')
    train = ['torchrun', '--standalone', '--nproc_per_node=2', str(MMROTATE/'tools/train.py'), str(CONFIG), '--launcher', 'pytorch']
    log = run(train, env)
    ckpt = checkpoint(ROOT / 'runs/r005/work_dirs' / name)
    cells = {}
    for sigma, factor in GRID:
        eval_env = env.copy()
        eval_env.update({'R005_EVAL_SIGMA': str(sigma), 'R005_EVAL_FACTOR': str(factor)})
        output = run([sys.executable, str(MMROTATE/'tools/test.py'), str(CONFIG), str(ckpt), '--launcher', 'none'], eval_env)
        values = {metric: float(value) for metric, value in AP.findall(output)}
        if set(values) != {'AP50', 'AP75'}:
            raise RuntimeError(f'missing AP metric for {name} {sigma}/{factor}')
        cells[f'{sigma:g}/{factor}'] = values
    return {'arm': arm, 'covariance_scale': covariance, 'ambiguity_threshold': threshold,
            'checkpoint': str(ckpt.relative_to(ROOT)), 'grid': cells}


def mean_degraded(result: dict, metric='AP75') -> float:
    values = [value[metric] for key, value in result['grid'].items() if key != '0/1']
    return sum(values) / len(values)


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict] = {}
    # Fixed budget non-method controls and stage-one covariance choice, with
    # ambiguity threshold zero fixed throughout this first comparison.
    for name, arm, cov in [('B0','B0',1.), ('B1','B1',1.), ('B2','B2',1.),
                           ('B3','B3',1.), ('M0','M0',1.),
                           ('M1_cov05','M1',.5), ('M1_cov10','M1',1.), ('M1_cov20','M1',2.)]:
        results[name] = train_and_grid(name, arm, cov, 0.)
        OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    chosen_cov = max(('M1_cov05','M1_cov10','M1_cov20'), key=lambda key: mean_degraded(results[key]))
    covariance = results[chosen_cov]['covariance_scale']
    # Only after covariance is frozen vary ambiguity threshold.
    for threshold in (.1, .2):
        name = f'M1_thr{threshold:g}'.replace('.', '')
        results[name] = train_and_grid(name, 'M1', covariance, threshold)
        OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    winner = max(('M1_cov05','M1_cov10','M1_cov20','M1_thr01','M1_thr02'), key=lambda key: mean_degraded(results[key]))
    # Mechanism ablations use the frozen full-M1 settings.
    for name, arm in [('M1_no_condition','M1_no_condition'), ('M1_no_competition','M1_no_competition'), ('M1_expanded_regression','M1_expanded_regression')]:
        results[name] = train_and_grid(name, arm, covariance, results[winner]['ambiguity_threshold'])
        OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    results['_selection'] = {'covariance_stage_winner': chosen_cov, 'full_m1_winner': winner,
                             'mean_degraded_ap75': {key: mean_degraded(value) for key, value in results.items() if not key.startswith('_')}}
    OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
