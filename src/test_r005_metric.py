"""Minimal executable check for the r005 AP/AR100 evaluator."""
import numpy as np

from src.r005_components import R005DOTAMetric


def main():
    metric = R005DOTAMetric(metric='mAP', iou_thrs=[.5, .75], eval_mode='area')
    metric.dataset_meta = {'classes': ('ship',)}
    gt = dict(labels=np.array([0]),
              bboxes=np.array([[10., 10., 8., 4., 0.]], dtype=np.float32),
              bboxes_ignore=np.empty((0, 5), dtype=np.float32),
              labels_ignore=np.empty((0,), dtype=np.int64))
    pred = dict(img_id='synthetic',
                bboxes=np.array([[10., 10., 8., 4., 0.]], dtype=np.float32),
                scores=np.array([.9], dtype=np.float32), labels=np.array([0]),
                pred_bbox_scores=[np.array([[10., 10., 8., 4., 0., .9]], dtype=np.float32)])
    values = metric.compute_metrics([(gt, pred)])
    assert values['AR100'] == 1.0, values
    assert values['AP50'] == 1.0 and values['AP75'] == 1.0, values
    print('r005 metric: PASS')


if __name__ == '__main__':
    main()
