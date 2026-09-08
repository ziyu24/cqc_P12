"""Compile the frozen r005 4x4 confirmation matrix and decision inputs.

This is a descriptive post-processing step.  It refuses incomplete or
unsharded entries, keeps original-OBB cells separate from B3's companion
expanded-target score, and reports all per-seed values before any mean.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


GRID = [(sigma, factor) for sigma in (0., .8, 1.6, 3.2) for factor in (1, 2, 4, 8)]
ARMS = ('B1', 'B2', 'B3', 'M1')
SEEDS = (17, 29, 43)
PROVENANCE = 'DefaultSampler(shuffle=True), distributed rank sharding'


def key(sigma: float, factor: int) -> str:
    return f'{sigma:g}/{factor}'


def cells(kind: str) -> list[tuple[float, int]]:
    if kind == 'clear':
        return [(0., 1)]
    if kind == 'degraded':
        return [(s, f) for s, f in GRID if (s, f) != (0., 1)]
    if kind == 'blur':
        return [(s, 1) for s in (0.8, 1.6, 3.2)]
    if kind == 'sampling':
        return [(0., f) for f in (2, 4, 8)]
    if kind == 'combination':
        return [(s, f) for s in (0.8, 1.6, 3.2) for f in (2, 4, 8)]
    raise ValueError(kind)


def summarise(entry: dict) -> dict:
    grid = entry['grid']
    if set(grid) != {key(s, f) for s, f in GRID}:
        raise ValueError(f'incomplete original-OBB grid for {entry["dataset"]}/{entry["arm"]}')
    answer = {'grid': grid, 'aggregates': {}}
    for subset in ('clear', 'degraded', 'blur', 'sampling', 'combination'):
        values = [grid[key(s, f)] for s, f in cells(subset)]
        answer['aggregates'][subset] = {metric: float(np.mean([row[metric] for row in values]))
                                        for metric in ('AP50', 'AP75', 'AR100')}
    return answer


def mean_sd(rows: list[dict]) -> dict:
    return {metric: {'mean': float(np.mean([row[metric] for row in rows])),
                     'seed_sd': float(np.std([row[metric] for row in rows], ddof=1)),
                     'per_seed': [row[metric] for row in rows]}
            for metric in ('AP50', 'AP75', 'AR100')}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--dataset', choices=('hrsc', 'dota'), required=True)
    parser.add_argument('--bootstrap', type=Path,
                        help='optional paired M1-vs-strongest-baseline bootstrap JSON')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.summary.read_text())
    entries, report = {}, {'dataset': args.dataset, 'arms': {}, 'comparisons': {}}
    for arm in ARMS:
        per_seed = []
        for seed in SEEDS:
            name = f'{args.dataset}/{arm}/seed{seed}'
            item = source.get(name)
            if item is None or item.get('sampler_provenance') != PROVENANCE:
                raise ValueError(f'missing or unsharded confirmation entry: {name}')
            entry = summarise(item); entries[arm, seed] = entry; per_seed.append(entry)
        report['arms'][arm] = {
            'per_seed': {str(seed): entry['aggregates'] for seed, entry in zip(SEEDS, per_seed)},
            'mean_and_seed_sd': {subset: mean_sd([entry['aggregates'][subset] for entry in per_seed])
                                 for subset in ('clear', 'degraded', 'blur', 'sampling', 'combination')}}
    # Strongest is defined only after all final 15-cell values exist, by the
    # frozen primary endpoint rather than by clear or expanded-target scores.
    baselines = ('B1', 'B2', 'B3')
    strongest = max(baselines, key=lambda arm: report['arms'][arm]['mean_and_seed_sd']['degraded']['AP75']['mean'])
    report['strongest_baseline'] = strongest
    for arm in baselines:
        comparison = {}
        for subset in ('clear', 'degraded', 'blur', 'sampling', 'combination'):
            comparison[subset] = {}
            for metric in ('AP50', 'AP75', 'AR100'):
                left = [entries['M1', seed]['aggregates'][subset][metric] for seed in SEEDS]
                right = [entries[arm, seed]['aggregates'][subset][metric] for seed in SEEDS]
                delta = np.asarray(left) - np.asarray(right)
                comparison[subset][metric] = {'mean': float(delta.mean()),
                                               'seed_sd': float(delta.std(ddof=1)),
                                               'per_seed': [float(x) for x in delta]}
        report['comparisons'][f'M1_minus_{arm}'] = comparison
    primary = report['comparisons'][f'M1_minus_{strongest}']
    gate = {'primary_ap75_ge_1pp': primary['degraded']['AP75']['mean'] >= .01,
            'all_three_seed_ap75_positive': all(x > 0 for x in primary['degraded']['AP75']['per_seed']),
            'ar100_same_direction': primary['degraded']['AR100']['mean'] > 0,
            'clear_ap75_drop_no_more_than_0_5pp': primary['clear']['AP75']['mean'] >= -.005,
            'two_or_more_degradation_families_positive': sum(
                primary[subset]['AP75']['mean'] > 0 for subset in ('blur', 'sampling', 'combination')) >= 2}
    if args.bootstrap:
        bootstrap = json.loads(args.bootstrap.read_text())
        if bootstrap.get('dataset') != args.dataset or bootstrap.get('left') != 'M1' or bootstrap.get('right') != strongest:
            raise ValueError('bootstrap does not match final M1-vs-strongest comparison')
        report['bootstrap'] = bootstrap
        gate['paired_ci95_lower_gt_zero'] = bootstrap['ci95'][0] > 0
    report['dota_continue_gate_inputs'] = gate if args.dataset == 'dota' else None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'dataset': args.dataset, 'strongest_baseline': strongest,
                      'output': str(args.output)}, indent=2))


if __name__ == '__main__':
    main()
