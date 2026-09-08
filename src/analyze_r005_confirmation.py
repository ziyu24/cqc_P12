"""Read-only descriptive analysis for the frozen r005 confirmation records.

The evaluator dumps GT/prediction pairs for every original-OBB grid cell.  This
script intentionally derives diagnostics from those records rather than from
training logs: per-class AP, scale and nearest-neighbour strata, matched-TP IoU
quantiles, and recall at fixed per-image false-positive budgets.  It makes no
model or post-processing choices.
"""
from __future__ import annotations

import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from mmcv.ops import box_iou_rotated
from mmdet.evaluation.functional import average_precision
from mmrotate.evaluation.functional.mean_ap import tpfp_default

GRID = [(sigma, factor) for sigma in (0., .8, 1.6, 3.2) for factor in (1, 2, 4, 8)]
FP_BUDGETS = (1, 5, 10)
SCALE_BINS = (("small", 0., 32.), ("medium", 32., 96.), ("large", 96., float("inf")))
DISTANCE_BINS = (("near", 0., 1.), ("mid", 1., 2.), ("far", 2., float("inf")))


def load(path: Path):
    with path.open("rb") as stream:
        return pickle.load(stream)


def ap(records, classes, threshold):
    values = {}
    for label, name in enumerate(classes):
        scores, tps, fps, count = [], [], [], 0
        for gt, prediction in records:
            det = np.asarray(prediction["pred_bbox_scores"][label])
            boxes, labels = np.asarray(gt["bboxes"]), np.asarray(gt["labels"])
            ignored, ignored_labels = np.asarray(gt["bboxes_ignore"]), np.asarray(gt["labels_ignore"])
            selected, selected_ignored = boxes[labels == label], ignored[ignored_labels == label]
            tp, fp = tpfp_default(det, selected, selected_ignored, iou_thr=threshold, box_type="rbox")
            scores.append(det[:, -1]); tps.append(tp[0]); fps.append(fp[0]); count += len(selected)
        if not count:
            values[name] = None
            continue
        order = np.argsort(-np.concatenate(scores))
        tp, fp = np.cumsum(np.concatenate(tps)[order]), np.cumsum(np.concatenate(fps)[order])
        values[name] = float(average_precision(tp / count, tp / np.maximum(tp + fp, 1e-12), "area"))
    finite = [value for value in values.values() if value is not None]
    return {"classes": values, "mean": float(np.mean(finite)) if finite else 0.}


def image_matches(gt, prediction, threshold=.75):
    """Score-ordered one-to-one matches, preserving every GT's diagnostics."""
    boxes, labels = np.asarray(gt["bboxes"]), np.asarray(gt["labels"])
    predicted, scores, predicted_labels = (np.asarray(prediction["bboxes"]), np.asarray(prediction["scores"]),
                                           np.asarray(prediction["labels"]))
    matched, ious, fp_count = np.zeros(len(boxes), dtype=bool), np.zeros(len(boxes)), 0
    for index in np.argsort(-scores):
        candidates = np.flatnonzero((labels == predicted_labels[index]) & ~matched)
        if not len(candidates):
            fp_count += 1; continue
        overlap = box_iou_rotated(torch.as_tensor(predicted[index:index + 1], dtype=torch.float32),
                                  torch.as_tensor(boxes[candidates], dtype=torch.float32)).cpu().numpy()[0]
        best = int(np.argmax(overlap))
        if overlap[best] >= threshold:
            target = candidates[best]; matched[target] = True; ious[target] = overlap[best]
        else:
            fp_count += 1
    # One recall value per fixed FP budget, evaluated independently per image.
    recalls = {}
    for budget in FP_BUDGETS:
        used, local = 0, np.zeros(len(boxes), dtype=bool)
        for index in np.argsort(-scores):
            candidates = np.flatnonzero((labels == predicted_labels[index]) & ~local)
            if len(candidates):
                overlap = box_iou_rotated(torch.as_tensor(predicted[index:index + 1], dtype=torch.float32),
                                          torch.as_tensor(boxes[candidates], dtype=torch.float32)).cpu().numpy()[0]
                best = int(np.argmax(overlap))
                if overlap[best] >= threshold:
                    local[candidates[best]] = True; continue
            used += 1
            if used > budget: break
        recalls[budget] = local
    return matched, ious, recalls


def strata(gt):
    boxes = np.asarray(gt["bboxes"])
    side = np.sqrt(np.maximum(boxes[:, 2] * boxes[:, 3], 1e-12))
    centres = boxes[:, :2]
    if len(boxes) < 2:
        distance = np.full(len(boxes), np.inf)
    else:
        delta = centres[:, None] - centres[None, :]
        pair = np.sqrt(np.square(delta).sum(-1)); np.fill_diagonal(pair, np.inf)
        distance = pair.min(1) / side
    return side, distance


def bucket(value, definitions):
    for name, lower, upper in definitions:
        if lower <= value and (np.isinf(upper) or value < upper): return name
    raise AssertionError(value)


def diagnostics(records):
    count = defaultdict(int); hits = defaultdict(int); ious = []; fp_hits = {budget: [0, 0] for budget in FP_BUDGETS}
    for gt, prediction in records:
        matched, overlap, recalls = image_matches(gt, prediction)
        size, distance = strata(gt)
        for index, label in enumerate(np.asarray(gt["labels"])):
            for prefix, value, definitions in (("scale", size[index], SCALE_BINS), ("neighbor", distance[index], DISTANCE_BINS)):
                key = f"{prefix}/{bucket(float(value), definitions)}"
                count[key] += 1; hits[key] += int(matched[index])
            if matched[index]: ious.append(float(overlap[index]))
        for budget, local in recalls.items():
            fp_hits[budget][0] += int(local.sum()); fp_hits[budget][1] += len(local)
    recall = {key: {"matched": hits[key], "gt": count[key], "recall": hits[key] / count[key] if count[key] else None}
              for key in sorted(count)}
    return {"stratified_recall_iou75": recall,
            "tp_iou75": {"count": len(ious), "quantiles": [float(x) for x in np.quantile(ious, (.05, .25, .5, .75, .95))] if ious else []},
            "fixed_fp_per_image_recall_iou75": {str(budget): {"matched": hits_, "gt": total,
                "recall": hits_ / total if total else None} for budget, (hits_, total) in fp_hits.items()}}


def record(root, dataset, arm, seed, sigma, factor):
    return root / dataset / f"{arm}_seed{seed}" / "per_image" / f"sigma{sigma:g}_factor{factor}_original.pkl"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True, help="runs/r005/confirmation")
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--entry", action="append", default=[], help="optional dataset/arm/seedN filter")
    args = parser.parse_args()
    confirmed = json.loads(args.summary.read_text())
    selected = set(args.entry)
    report = {"definitions": {"iou": .75, "fp_budgets_per_image": FP_BUDGETS,
              "scale_sqrt_area_pixels": SCALE_BINS, "nearest_center_distance_over_sqrt_area": DISTANCE_BINS}, "entries": {}}
    for name, item in confirmed.items():
        if selected and name not in selected: continue
        dataset, arm, seed_text = name.split("/"); seed = int(seed_text.removeprefix("seed"))
        cells = {}
        for sigma, factor in GRID:
            path = record(args.root, dataset, arm, seed, sigma, factor)
            if not path.is_file(): raise FileNotFoundError(path)
            payload = load(path); rows = payload["results"]
            cells[f"{sigma:g}/{factor}"] = {"ap50": ap(rows, payload["classes"], .5),
                                               "ap75": ap(rows, payload["classes"], .75),
                                               **diagnostics(rows)}
        report["entries"][name] = cells
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"entries": len(report["entries"]), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
