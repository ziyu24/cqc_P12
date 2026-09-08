"""Paired original-image-cluster bootstrap for r005 AP75 differences.

The r005 evaluator writes a pickle with GT/prediction pairs per evaluation
image.  This utility reproduces its area-AP@.75 calculation from those pairs,
but samples original-image clusters (all DOTA tiles sharing ``__`` prefix) as
the independent unit.  One draw sequence is shared by every arm, seed, and
the complete 15-cell degraded grid, as fixed in the r005 protocol.
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from mmdet.evaluation.functional import average_precision
from mmrotate.evaluation.functional.mean_ap import tpfp_default

GRID = [(sigma, factor) for sigma in (0., .8, 1.6, 3.2) for factor in (1, 2, 4, 8)]
SAMPLER_PROVENANCE = 'DefaultSampler(shuffle=True), distributed rank sharding'


def cluster_id(pred: dict, dataset: str) -> str:
    image = str(pred.get('img_id', pred.get('img_path', '')))
    return image.split('__', 1)[0] if dataset == 'dota' else image


def load_clusters(path: Path, dataset: str) -> tuple[tuple[str, ...], dict[str, list[tuple[dict, dict]]]]:
    with path.open('rb') as handle:
        payload = pickle.load(handle)
    groups: dict[str, list[tuple[dict, dict]]] = {}
    for gt, pred in payload['results']:
        groups.setdefault(cluster_id(pred, dataset), []).append((gt, pred))
    return tuple(payload['classes']), groups


def _ap75(records: list[tuple[dict, dict]], classes: tuple[str, ...]) -> float:
    aps = []
    for label in range(len(classes)):
        scores, tps, fps, num_gts = [], [], [], 0
        for gt, pred in records:
            det = np.asarray(pred['pred_bbox_scores'][label])
            boxes = np.asarray(gt['bboxes'])
            labels = np.asarray(gt['labels'])
            ignored = np.asarray(gt['bboxes_ignore'])
            ignored_labels = np.asarray(gt['labels_ignore'])
            selected_gt = boxes[labels == label]
            selected_ignored = ignored[ignored_labels == label]
            tp, fp = tpfp_default(det, selected_gt, selected_ignored, iou_thr=.75, box_type='rbox')
            scores.append(det[:, -1])
            tps.append(tp[0])
            fps.append(fp[0])
            num_gts += len(selected_gt)
        if not num_gts:
            continue
        score = np.concatenate(scores)
        order = np.argsort(-score)
        tp, fp = np.cumsum(np.concatenate(tps)[order]), np.cumsum(np.concatenate(fps)[order])
        aps.append(float(average_precision(tp / num_gts, tp / np.maximum(tp + fp, np.finfo(np.float32).eps), 'area')))
    return float(np.mean(aps)) if aps else 0.


def paired_delta(left: Path, right: Path, dataset: str, draws: list[np.ndarray],
                 expected_clusters: tuple[str, ...]) -> np.ndarray:
    left_classes, left_groups = load_clusters(left, dataset)
    right_classes, right_groups = load_clusters(right, dataset)
    clusters = tuple(left_groups)
    if (left_classes != right_classes or clusters != tuple(right_groups)
            or clusters != expected_clusters):
        raise ValueError(f'paired records differ: {left} vs {right}')
    values = []
    for draw in draws:
        chosen = [clusters[index] for index in draw]
        a = [record for key in chosen for record in left_groups[key]]
        b = [record for key in chosen for record in right_groups[key]]
        values.append(_ap75(a, left_classes) - _ap75(b, left_classes))
    return np.asarray(values)


def record_path(root: Path, dataset: str, arm: str, seed: int, sigma: float, factor: int) -> Path:
    return root / dataset / f'{arm}_seed{seed}' / 'per_image' / f'sigma{sigma:g}_factor{factor}_original.pkl'


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--dataset', choices=('hrsc', 'dota'), required=True)
    parser.add_argument('--left', required=True, help='normally M1')
    parser.add_argument('--right', required=True, help='frozen strongest baseline')
    parser.add_argument('--seeds', nargs='+', type=int, default=(17, 29, 43))
    parser.add_argument('--replicates', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=20260907)
    parser.add_argument('--summary', type=Path, required=True,
                        help='confirmation summary used to reject unsharded records')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    confirmed = json.loads(args.summary.read_text())
    for seed in args.seeds:
        for arm in (args.left, args.right):
            name = f'{args.dataset}/{arm}/seed{seed}'
            if confirmed.get(name, {}).get('sampler_provenance') != SAMPLER_PROVENANCE:
                raise ValueError(f'unsharded or unknown confirmation provenance: {name}')
    rng = np.random.default_rng(args.seed)
    per_seed, shared_clusters, shared_draws = [], None, None
    for seed in args.seeds:
        cell_deltas = []
        for sigma, factor in GRID:
            if sigma == 0 and factor == 1:
                continue
            left = record_path(args.root, args.dataset, args.left, seed, sigma, factor)
            right = record_path(args.root, args.dataset, args.right, seed, sigma, factor)
            _, groups = load_clusters(left, args.dataset)
            clusters = tuple(groups)
            if shared_clusters is None:
                shared_clusters = clusters
                shared_draws = [np.arange(len(clusters), dtype=int)] + [
                    rng.integers(0, len(clusters), len(clusters)) for _ in range(args.replicates)]
            elif clusters != shared_clusters:
                raise ValueError(f'cluster identity/order differs at {left}')
            cell_deltas.append(paired_delta(left, right, args.dataset, shared_draws, shared_clusters))
        per_seed.append(np.mean(cell_deltas, axis=0))
    per_seed = np.asarray(per_seed)
    mean = per_seed.mean(axis=0)
    payload = dict(dataset=args.dataset, left=args.left, right=args.right, seeds=args.seeds,
                   replicates=args.replicates, point=float(mean[0]),
                   ci95=[float(v) for v in np.quantile(mean, [.025, .975])],
                   seed_point_estimates=[float(v[0]) for v in per_seed],
                   seed_sd=float(np.std(per_seed[:, 0], ddof=1)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + '\n')
    print(json.dumps(payload, indent=2))


if __name__ == '__main__':
    main()
