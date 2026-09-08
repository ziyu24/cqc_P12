"""Render the pre-fixed r005 HRSC examples without selecting on outcomes.

The script maps the twelve IDs frozen in ``configs/r005_visual_samples.json``
to the M1 seed-17 severe-cell record.  It redraws the exact evaluation
degradation, original GT OBBs and the model's original-OBB predictions.  The
blue/yellow lattice marks reproduce the frozen M1 assigner's positive support
and ambiguity weight from its Gaussian responsibility rule.  They are a
training-side explanation only: predictions are neither changed nor filtered
for this rendering.
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


def corners(box: np.ndarray) -> list[tuple[float, float]]:
    cx, cy, width, height, angle = map(float, box)
    c, s = math.cos(angle), math.sin(angle)
    return [(cx + x*c-y*s, cy+x*s+y*c)
            for x, y in ((-width/2, -height/2), (width/2, -height/2),
                         (width/2, height/2), (-width/2, height/2))]


def degraded(image: np.ndarray, sigma: float, factor: int) -> np.ndarray:
    if sigma:
        kernel = 2 * math.ceil(3 * sigma) + 1
        image = cv2.GaussianBlur(image, (kernel, kernel), sigmaX=sigma, sigmaY=sigma)
    if factor > 1:
        height, width = image.shape[:2]
        image = cv2.resize(image, (max(1, width // factor), max(1, height // factor)), interpolation=cv2.INTER_AREA)
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    return image


def covariance(boxes: np.ndarray, extra: float) -> np.ndarray:
    angle = boxes[:, 4]
    c, s = np.cos(angle), np.sin(angle)
    vx, vy = (boxes[:, 2] / 2) ** 2, (boxes[:, 3] / 2) ** 2
    return np.stack((c*c*vx + s*s*vy + extra, c*s*(vx-vy),
                     c*s*(vx-vy), s*s*vx + c*c*vy + extra), axis=-1).reshape(-1, 2, 2)


def support_points(boxes: np.ndarray, shape: tuple[int, int], sigma: float, factor: int,
                   covariance_scale: float = .5, ambiguity_threshold: float = .2):
    """Mirror the assigner's [6,1] support / responsibility calculation.

    This uses the three fixed RTMDet FPN strides and image-coordinate GTs.  It
    is deliberately independent of detector scores, so it cannot constitute
    an example-selection or post-processing step.
    """
    height, width = shape[:2]
    if not len(boxes):
        return np.empty((0, 2)), np.empty((0,)), np.empty((0,), dtype=int)
    points, strides = [], []
    for stride in (8, 16, 32):
        yy, xx = np.mgrid[stride / 2:height:stride, stride / 2:width:stride]
        points.append(np.stack((xx.ravel(), yy.ravel()), axis=1))
        strides.append(np.full(xx.size, stride))
    points, strides = np.concatenate(points), np.concatenate(strides)
    extra = covariance_scale * (sigma*sigma + max(factor*factor - 1., 0.) / 12.)
    gt_cov = covariance(boxes, extra)
    rf = np.zeros((len(points), 2, 2))
    rf[:, 0, 0] = rf[:, 1, 1] = strides ** 2
    inv_rf = np.linalg.inv(rf)
    delta = points[:, None, :] - boxes[None, :, :2]
    trace = np.einsum('pij,gji->pg', inv_rf, gt_cov)
    mahal = np.einsum('pgi,pij,pgj->pg', delta, inv_rf, delta)
    _, log_gt = np.linalg.slogdet(gt_cov); _, log_rf = np.linalg.slogdet(rf)
    costs = .5 * (trace + mahal - 2. + log_rf[:, None] - log_gt[None, :])
    first = np.zeros_like(costs, dtype=bool)
    first[np.argpartition(costs, min(5, len(points) - 1), axis=0)[:6], np.arange(len(boxes))] = True
    shrunk = rf * (.9 ** 2); inv_shrunk = np.linalg.inv(shrunk)
    costs2 = .5 * (np.einsum('pij,gji->pg', inv_shrunk, gt_cov)
                   + np.einsum('pgi,pij,pgj->pg', delta, inv_shrunk, delta) - 2.
                   + np.linalg.slogdet(shrunk)[1][:, None] - log_gt[None, :])
    second = np.zeros_like(first)
    second[np.argmin(costs2, axis=0), np.arange(len(boxes))] = True
    empty = ~first.any(axis=1)
    first[empty, np.argmin(np.where(second[empty], costs2[empty], np.inf), axis=1)] = True
    exp = np.exp(-(costs - costs.min(axis=1, keepdims=True)))
    responsibility = exp / exp.sum(axis=1, keepdims=True)
    owner = np.argmax(np.where(first, responsibility, -np.inf), axis=1)
    positive = np.isfinite(np.take_along_axis(np.where(first, costs, np.inf), owner[:, None], axis=1)[:, 0])
    top = np.partition(responsibility, -2, axis=1)[:, -2:]
    margin = top[:, 1] - top[:, 0]
    weights = np.ones(len(points), dtype=float)
    ambiguous = positive & (margin < ambiguity_threshold)
    weights[ambiguous] = np.clip(margin[ambiguous] / ambiguity_threshold, 0., 1.)
    return points[positive], weights[positive], owner[positive]


def draw_panel(image: np.ndarray, boxes: np.ndarray, prediction: dict, support) -> Image.Image:
    base = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).convert('RGBA')
    overlay = Image.new('RGBA', base.size, (0, 0, 0, 0)); marker = ImageDraw.Draw(overlay)
    points, weights, _ = support
    for (x, y), weight in zip(points, weights):
        color = (30, int(100 + 155 * weight), 255, 170)
        marker.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color)
    base = Image.alpha_composite(base, overlay); draw = ImageDraw.Draw(base)
    for box in boxes:
        polygon = corners(box); draw.line(polygon + [polygon[0]], fill=(0, 255, 90, 255), width=3)
    order = np.argsort(-np.asarray(prediction['scores']))[:100]
    for index in order:
        box = np.asarray(prediction['bboxes'])[index]
        polygon = corners(box); draw.line(polygon + [polygon[0]], fill=(255, 60, 40, 230), width=2)
    return base.convert('RGB')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples', type=Path, default=Path('configs/r005_visual_samples.json'))
    parser.add_argument('--per-image', type=Path, required=True)
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    specification = json.loads(args.samples.read_text())
    payload = pickle.loads(args.per_image.read_bytes())
    record = {str(pred['img_id']): (gt, pred) for gt, pred in payload['results']}
    args.output.mkdir(parents=True, exist_ok=True)
    sigma, factor = specification['selection']['degradation']['sigma'], specification['selection']['degradation']['factor']
    rendered = []
    for image_id in specification['image_ids']:
        if image_id not in record:
            raise KeyError(f'fixed image absent from per-image record: {image_id}')
        gt, prediction = record[image_id]
        image = cv2.imread(str(args.dataset_root / 'images' / f'{image_id}.bmp'))
        if image is None:
            raise FileNotFoundError(image_id)
        image = degraded(image, sigma, factor)
        boxes = np.asarray(gt['bboxes'])
        panel = draw_panel(image, boxes, prediction, support_points(boxes, image.shape, sigma, factor))
        panel.thumbnail((800, 600)); panel.save(args.output / f'{image_id}.jpg', quality=92)
        rendered.append(image_id)
    (args.output / 'manifest.json').write_text(json.dumps({'samples': rendered, 'specification': specification,
        'legend': {'green': 'original GT OBB', 'red': 'M1 original-OBB predictions (fixed top-100 endpoint)',
                   'blue-yellow': 'M1 positive support, colour encodes frozen ambiguity weight'}}, indent=2) + '\n')
    print(json.dumps({'rendered': len(rendered), 'output': str(args.output)}, indent=2))


if __name__ == '__main__':
    main()
