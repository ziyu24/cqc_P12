#!/usr/bin/env python3
"""Audit frozen HRSC observers with the evaluator used by DOTAMetric.

This performs clear-image inference only. It never overwrites the reviewed
experiment's predictions, changes shared weights, or reopens a completed run.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from mmdet.apis import inference_detector, init_detector
from mmrotate.datasets import HRSCDataset
from mmrotate.evaluation import eval_rbbox_map
from mmrotate.structures.bbox import qbox2rbox

from audit_controlled_evidence import load_archived_checkpoint
from run_r003_evidence import events, polygon


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--retina-config', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    # Refuse to replace either a historical artifact or a previous audit.
    args.out.mkdir(parents=True, exist_ok=False)
    cfg = json.loads(args.config.read_text())
    root = Path(cfg['dataset_root'])
    dataset = HRSCDataset(
        ann_file=str(root / 'splits/test.txt'),
        data_prefix={'sub_data_root': str(root)},
        img_subdir='images', ann_subdir='annfiles',
        test_mode=True, pipeline=[])
    infos = [dataset.get_data_info(i) for i in range(len(dataset))]
    annotations = []
    for info in infos:
        instances = info['instances']
        boxes = torch.tensor([x['bbox'] for x in instances], dtype=torch.float32).reshape(-1, 8)
        boxes = qbox2rbox(boxes).numpy()
        ignored = np.asarray([x['ignore_flag'] for x in instances], dtype=bool)
        annotations.append({
            'bboxes': boxes[~ignored], 'labels': np.zeros((~ignored).sum(), dtype=np.int64),
            'bboxes_ignore': boxes[ignored], 'labels_ignore': np.zeros(ignored.sum(), dtype=np.int64)})

    population = json.loads(Path(cfg['population']).read_text())
    groups = population['audit_order'][:cfg['accepted_groups']]
    image_index = {info['img_id']: i for i, info in enumerate(infos)}
    output = {
        'purpose': 'clear_observer_audit_only_not_corrected_degradation_effect',
        'images': len(infos), 'evaluator': 'mmrotate.evaluation.eval_rbbox_map',
        'eval_mode': '11points', 'iou_thr': 0.5,
        'config_sha256': digest(args.config), 'population_sha256': digest(cfg['population']),
        'models': {},
    }
    resolved_sets = []
    for name, config, checkpoint_path in (
        ('orcnn', cfg['orcnn_config'], cfg['orcnn_checkpoint']),
        ('retina', str(args.retina_config), cfg['retina_checkpoint']),
    ):
        checkpoint = load_archived_checkpoint(checkpoint_path)
        model = init_detector(config, checkpoint=None, device=args.device)
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        model.dataset_meta = dataset.metainfo
        predictions = []
        for i, info in enumerate(infos):
            result = inference_detector(model, info['img_path']).pred_instances
            boxes = result.bboxes.detach().cpu().numpy()
            scores = result.scores.detach().cpu().numpy()
            predictions.append([np.column_stack((boxes, scores)).astype(np.float32)])
            if (i + 1) % 100 == 0 or i + 1 == len(infos):
                print(json.dumps({'observer': name, 'clear_images_done': i + 1}), flush=True)
        mean_ap, details = eval_rbbox_map(
            predictions, annotations, iou_thr=.5, use_07_metric=True,
            box_type='rbox', dataset=('ship',), nproc=2)
        resolved = []
        group_records = []
        for gi, group in enumerate(groups):
            index = image_index[group['image_id']]
            # HRSC test loader keeps every instance, in original annotation order.
            truth = [polygon(annotations[index]['bboxes'][member]) for member in group['members']]
            pred = [{'id': j, 'box': row[:5].tolist(), 'score': float(row[5])}
                    for j, row in enumerate(predictions[index][0]) if row[5] >= .25]
            event = events(truth, pred)
            group_records.append({'group': gi, 'image': group['image_id'], 'event': event})
            if event['resolved']:
                resolved.append(gi)
        resolved_sets.append(set(resolved))
        payload = {'image_ids': [info['img_id'] for info in infos],
                   'predictions': [pred[0].tolist() for pred in predictions]}
        pred_path = args.out / f'{name}_clear_predictions.json.gz'
        with gzip.open(pred_path, 'wt') as stream:
            json.dump(payload, stream, allow_nan=False)
        output['models'][name] = {
            'ap50': float(mean_ap), 'gt_count': int(details[0]['num_gts']),
            'detections': int(details[0]['num_dets']),
            'strict_load': True, 'checkpoint_sha256': digest(checkpoint_path),
            'model_config_sha256': digest(config),
            'anchor_angle_version': model.bbox_head.prior_generator.angle_version if name == 'retina' else None,
            'bbox_coder_angle_version': model.bbox_head.bbox_coder.angle_version if name == 'retina' else 'le90',
            'clear_resolved_groups': len(resolved), 'resolved_group_ids': resolved,
            'group_records': group_records, 'predictions_sha256': digest(pred_path),
        }
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    common = resolved_sets[0] & resolved_sets[1]
    output['common_groups'] = len(common)
    output['common_images'] = len({groups[i]['image_id'] for i in common})
    output['common_group_ids'] = sorted(common)
    (args.out / 'observer_review.json').write_text(json.dumps(output, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'ap50': {name: value['ap50'] for name, value in output['models'].items()},
                      'common_groups': output['common_groups'], 'common_images': output['common_images']}), flush=True)


if __name__ == '__main__':
    main()
