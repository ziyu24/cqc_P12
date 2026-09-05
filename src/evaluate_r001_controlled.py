#!/usr/bin/env python3
"""Run the frozen r001 controlled-degradation group analysis on HRSC2016.

The script deliberately treats the original image as the bootstrap cluster.
It writes every group-level decision, so aggregate claims can be audited rather
than reconstructed from a chart.
"""
from __future__ import annotations

import argparse
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
from mmdet.apis import inference_detector, init_detector
from shapely.geometry import Polygon


ROOT = Path('/home/rspip/zy/data/dataset/HRSC2016')


def rbox_poly(cx, cy, w, h, angle):
    pts = np.array([[-w/2, -h/2], [w/2, -h/2], [w/2, h/2], [-w/2, h/2]])
    c, s = math.cos(angle), math.sin(angle)
    rot = np.array([[c, -s], [s, c]])
    return Polygon(pts @ rot.T + np.array([cx, cy]))


def truth(image_id):
    root = ET.parse(ROOT / 'annfiles' / f'{image_id}.xml').getroot()
    ans = []
    for obj in root.findall('./HRSC_Objects/HRSC_Object'):
        vals = [float(obj.findtext(k)) for k in ('mbox_cx', 'mbox_cy', 'mbox_w', 'mbox_h', 'mbox_ang')]
        ans.append(rbox_poly(*vals))
    return ans


def degrade(image, level):
    if level == 0:
        return image
    sigma, area = {1: (.8, 2), 2: (1.6, 4), 3: (3.2, 8)}[level]
    blur = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma, sigmaY=sigma)
    scale = 1 / math.sqrt(area)
    small = cv2.resize(blur, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_CUBIC)


def iou(a, b):
    inter = a.intersection(b).area
    return inter / (a.area + b.area - inter) if inter else 0.0


def predict(model, image, score):
    data = inference_detector(model, image).pred_instances
    boxes, scores = data.bboxes.detach().cpu().numpy(), data.scores.detach().cpu().numpy()
    return [(rbox_poly(*box), float(conf)) for box, conf in zip(boxes, scores) if conf >= score]


def events(gt, pred):
    """Group-level multi-label event taxonomy specified by r001."""
    overlaps = np.array([[iou(g, p) for p, _ in pred] for g in gt]) if pred else np.zeros((len(gt), 0))
    # maximum-cardinality greedy matching is sufficient at the prescribed 0.5
    # threshold because each edge is explicitly stored for audit.
    edges = sorted(((overlaps[i, j], i, j) for i in range(len(gt)) for j in range(len(pred)) if overlaps[i, j] >= .5), reverse=True)
    used_g, used_p, matches = set(), set(), []
    for _, gi, pi in edges:
        if gi not in used_g and pi not in used_p:
            used_g.add(gi); used_p.add(pi); matches.append([gi, pi])
    merge = []
    for pi, (p, _) in enumerate(pred):
        covered = [gi for gi, g in enumerate(gt) if p.intersection(g).area / g.area >= .5]
        if len(covered) >= 2:
            merge.append({'prediction': pi, 'truth': covered})
    merged_truth = {gi for m in merge for gi in m['truth']}
    duplicate = [gi for gi in range(len(gt)) if (overlaps[gi] >= .3).sum() >= 2]
    miss = [gi for gi in range(len(gt)) if gi not in used_g and gi not in merged_truth]
    # predictions plausibly belonging to this group are those with IoU>=.3;
    # retaining the count makes cardinality decisions fully inspectable.
    relevant = {pi for pi in range(len(pred)) if any(overlaps[gi, pi] >= .3 for gi in range(len(gt)))}
    return {
        'resolved': len(used_g) == len(gt) and len(relevant) == len(gt),
        'cardinality_error': len(relevant) != len(gt),
        'merge': bool(merge), 'duplicate': bool(duplicate), 'miss': bool(miss),
        'matches': sorted(matches),
        'merge_detail': merge, 'duplicate_truth': duplicate, 'miss_truth': miss,
        'relevant_prediction_count': len(relevant),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--population', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--score', type=float, default=.25)
    ap.add_argument('--nms', type=float, default=None,
                    help='frozen shared rotated-NMS IoU threshold for sensitivity scans')
    ap.add_argument('--orcnn-config', type=Path, required=True)
    ap.add_argument('--orcnn-ckpt', type=Path, required=True)
    ap.add_argument('--retina-config', type=Path, required=True)
    ap.add_argument('--retina-ckpt', type=Path, required=True)
    args = ap.parse_args()
    pop = json.loads(args.population.read_text())
    groups = pop['audit_order'][:pop['manual_audit']['accepted_candidates']]
    isolated = pop['isolated_candidates']
    # Freeze scale-matched pseudo-groups before seeing degradation outcomes.
    pseudo, used = [], set()
    for group in groups:
        candidates = sorted((x for x in isolated if x['image_id'] not in used),
                            key=lambda x: abs(x['short_side'] - group['min_short_side']))
        picked = candidates[:group['size']]
        if len(picked) == group['size']:
            pseudo.append(picked); used.update(x['image_id'] for x in picked)
        else:
            pseudo.append([])
    models = {
        'oriented_rcnn': init_detector(str(args.orcnn_config), str(args.orcnn_ckpt), device='cuda:0'),
        'rotated_retinanet': init_detector(str(args.retina_config), str(args.retina_ckpt), device='cuda:0'),
    }
    if args.nms is not None:
        for model in models.values():
            # RetinaNet exposes `test_cfg.nms`; two-stage Oriented R-CNN
            # exposes the same public setting under `test_cfg.rcnn.nms`.
            nms_cfg = model.test_cfg.nms if 'nms' in model.test_cfg else model.test_cfg.rcnn.nms
            nms_cfg.iou_threshold = args.nms
    images = {g['image_id'] for g in groups} | {x['image_id'] for row in pseudo for x in row}
    cache = {name: {} for name in models}
    for image_id in sorted(images):
        im = cv2.imread(str(ROOT / 'images' / f'{image_id}.bmp'))
        if im is None:
            raise RuntimeError(f'missing image {image_id}')
        for level in range(4):
            d = degrade(im, level)
            for name, model in models.items():
                cache[name][(image_id, level)] = predict(model, d, args.score)
    rows = []
    for idx, (group, control) in enumerate(zip(groups, pseudo)):
        gt = truth(group['image_id'])
        group_gt = [gt[i] for i in group['members']]
        control_gt = [] if not control else [truth(x['image_id'])[x['object_index']] for x in control]
        for name in models:
            for level in range(4):
                row = {'group_id': idx, 'image_id': group['image_id'], 'size': group['size'],
                       'level': level, 'detector': name,
                       'adjacent': events(group_gt, cache[name][(group['image_id'], level)])}
                # A pseudo-group consists of isolated instances possibly from
                # different source images; its bootstrap cluster ids are kept.
                row['control_image_ids'] = [x['image_id'] for x in control]
                row['control'] = [events([g], cache[name][(x['image_id'], level)])
                                  for x, g in zip(control, control_gt)]
                rows.append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({'score': args.score, 'nms_iou_threshold': args.nms,
                                       'levels': ['clear', 'sigma0.8_area2', 'sigma1.6_area4', 'sigma3.2_area8'],
                                       'groups': groups, 'pseudo_groups': pseudo, 'rows': rows}, indent=2) + '\n')
    print(f'wrote {len(rows)} detector/group/level rows to {args.output}')


if __name__ == '__main__':
    main()
