"""Minimal executable check for the r005 AP/AR100 evaluator."""
import numpy as np
import pickle
import tempfile
from pathlib import Path

from src.r005_components import R005DOTAMetric


def main():
    gt = dict(labels=np.array([0]),
              bboxes=np.array([[10., 10., 8., 4., 0.]], dtype=np.float32),
              bboxes_ignore=np.empty((0, 5), dtype=np.float32),
              labels_ignore=np.empty((0,), dtype=np.int64))
    pred = dict(img_id='synthetic',
                bboxes=np.array([[10., 10., 8., 4., 0.]], dtype=np.float32),
                scores=np.array([.9], dtype=np.float32), labels=np.array([0]),
                pred_bbox_scores=[np.array([[10., 10., 8., 4., 0., .9]], dtype=np.float32)])
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / 'per_image.pkl'
        metric = R005DOTAMetric(metric='mAP', iou_thrs=[.5, .75], eval_mode='area', dump_path=str(output))
        metric.dataset_meta = {'classes': ('ship',)}
        values = metric.compute_metrics([(gt, pred)])
        assert values['AR100'] == 1.0, values
        assert values['AP50'] == 1.0 and values['AP75'] == 1.0, values
        with output.open('rb') as handle:
            saved = pickle.load(handle)
        assert saved['classes'] == ('ship',) and len(saved['results']) == 1
    print('r005 metric: PASS')


if __name__ == '__main__':
    main()
