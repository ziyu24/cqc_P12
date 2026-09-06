"""r005 的训练期退化、旋转高斯证据分配与可审计扩张框对照。

本模块只在训练时改变像素和正样本归属；预测头、回归真值和推理
路径均沿用 MMRotate 的 RotatedRTMDetSepBNHead。`EvidenceAssigner` 的
zero-degradation 分支与 B2 相同，因而 M1 在零退化严格退化为 B2。
"""
from __future__ import annotations

import math
from typing import Optional

import cv2
import numpy as np
import torch
from mmcv.transforms import BaseTransform
from mmengine.structures import InstanceData
from mmdet.models.task_modules.assigners.assign_result import AssignResult
from mmdet.models.task_modules.assigners.base_assigner import BaseAssigner
from mmdet.models.task_modules import anchor_inside_flags
from mmdet.models.utils import unmap
from mmdet.structures.bbox import BaseBoxes
from mmdet.utils import ConfigType
from mmrotate.models.dense_heads.rotated_rtmdet_head import RotatedRTMDetSepBNHead
from mmrotate.registry import MODELS, TASK_UTILS, TRANSFORMS


@TRANSFORMS.register_module()
class SharedGaussianDownsample(BaseTransform):
    """Deterministic per-image degradation with metadata in image coordinates.

    A stable hash makes every non-B0 arm receive the same image-level draw for
    a fixed seed.  The blur covariance is transformed with later geometric
    transforms by the custom head only after PackDetInputs has retained it.
    r005 keeps rotation/resize after this transform fixed across all arms.
    """
    def __init__(self, prob=0.75, sigma_range=(0.0, 3.2), scale_choices=(1,2,4,8), seed=5):
        self.prob = float(prob)
        self.sigma_range = tuple(map(float, sigma_range))
        self.scale_choices = tuple(map(int, scale_choices))
        self.seed = int(seed)

    def transform(self, results):
        image_id = str(results.get('img_id', results.get('img_path', '')))
        # Python hash is process-randomized; use a tiny explicit rolling hash.
        value = self.seed
        for char in image_id:
            value = (value * 131 + ord(char)) & 0xffffffff
        rng = np.random.default_rng(value)
        if rng.random() >= self.prob:
            results['r005_degradation'] = np.array([0., 1.], dtype=np.float32)
            return results
        sigma = float(rng.uniform(*self.sigma_range))
        factor = int(rng.choice(self.scale_choices))
        image = results['img']
        if sigma > 0:
            kernel = int(2 * math.ceil(3 * sigma) + 1)
            image = cv2.GaussianBlur(image, (kernel, kernel), sigmaX=sigma, sigmaY=sigma)
        if factor > 1:
            height, width = image.shape[:2]
            reduced = cv2.resize(image, (max(1, width // factor), max(1, height // factor)), interpolation=cv2.INTER_AREA)
            image = cv2.resize(reduced, (width, height), interpolation=cv2.INTER_LINEAR)
        results['img'] = image
        results['img_shape'] = image.shape[:2]
        results['r005_degradation'] = np.array([sigma, factor], dtype=np.float32)
        return results


@TRANSFORMS.register_module()
class FixedGaussianDownsample(BaseTransform):
    """Evaluation-only fixed grid point; preserves the same metadata schema."""
    def __init__(self, sigma=0.0, factor=1):
        self.sigma, self.factor = float(sigma), int(factor)

    def transform(self, results):
        image = results['img']
        if self.sigma > 0:
            kernel = int(2 * math.ceil(3 * self.sigma) + 1)
            image = cv2.GaussianBlur(image, (kernel, kernel), sigmaX=self.sigma, sigmaY=self.sigma)
        if self.factor > 1:
            height, width = image.shape[:2]
            reduced = cv2.resize(image, (max(1, width // self.factor), max(1, height // self.factor)), interpolation=cv2.INTER_AREA)
            image = cv2.resize(reduced, (width, height), interpolation=cv2.INTER_LINEAR)
        results['img'] = image
        results['img_shape'] = image.shape[:2]
        results['r005_degradation'] = np.array([self.sigma, self.factor], dtype=np.float32)
        return results


@TRANSFORMS.register_module()
class BlurBoxExpansion(BaseTransform):
    """B3 only: expand rotated training boxes by effective isotropic support.

    The original boxes are kept in `r005_original_bboxes` for the companion
    original-OBB evaluation; the training target is the explicitly expanded
    box, never used by M0/M1.
    """
    def __init__(self, support_scale=1.0):
        self.support_scale = float(support_scale)

    def transform(self, results):
        boxes = results.get('gt_bboxes')
        degrade = results.get('r005_degradation', np.array([0., 1.], np.float32))
        sigma, factor = map(float, degrade)
        if boxes is None or (sigma == 0 and factor == 1):
            return results
        raw = boxes.tensor.clone()
        results['r005_original_bboxes'] = raw
        # Variance of Gaussian blur plus box-equivalent sampling footprint.
        radius = self.support_scale * math.sqrt(sigma * sigma + (factor * factor - 1.) / 12.)
        expanded = raw.clone()
        expanded[:, 2:4] += 2.0 * radius
        boxes.tensor[:] = expanded
        return results


def _box_covariance(boxes: torch.Tensor) -> torch.Tensor:
    """Return each le90 rbox's centred Gaussian covariance in image pixels."""
    angle = boxes[:, 4]
    c, s = angle.cos(), angle.sin()
    # A uniform rectangle's covariance; it preserves orientation and scale.
    vx, vy = (boxes[:, 2] / 2).square(), (boxes[:, 3] / 2).square()
    return torch.stack((c.square()*vx + s.square()*vy,
                        c*s*(vx-vy),
                        c*s*(vx-vy),
                        s.square()*vx + c.square()*vy), dim=-1).reshape(-1,2,2)


@TASK_UTILS.register_module()
class GaussianRFLEvidenceAssigner(BaseAssigner):
    """RFLA Gaussian-distance, hierarchical assignment with M1 extensions.

    `conditioned=False` is B2.  `conditioned=True` adds a known degradation
    covariance and ambiguity discount.  The assigner reports one owner per
    prior by argmin cost; its extra `r005_weights` are consumed by the head.
    """
    def __init__(self, topk=(6, 1), rf_scale=1.0, conditioned=False,
                 covariance_scale=1.0, ambiguity_threshold=0.1,
                 fixed_support=0.0, iou_calculator: ConfigType=dict(type='RBboxOverlaps2D')):
        # RFLA's published HieAssigner uses the two stages [6, 1].  Keep the
        # argument flexible solely for unit-sized counterexamples.
        self.topk = (int(topk), 1) if isinstance(topk, int) else tuple(map(int, topk))
        self.rf_scale = float(rf_scale)
        self.conditioned = bool(conditioned)
        self.covariance_scale = float(covariance_scale)
        self.ambiguity_threshold = float(ambiguity_threshold)
        self.fixed_support = float(fixed_support)
        self.iou_calculator = TASK_UTILS.build(iou_calculator)

    @staticmethod
    def _kl_distance(gt_cov: torch.Tensor, rf_cov: torch.Tensor,
                     delta: torch.Tensor) -> torch.Tensor:
        """KL(N_gt || N_rf), the Gaussian receptive-field distance in RFLA.

        The two covariance arguments are [G,2,2] and [P,2,2]; delta is
        [P,G,2].  This is the oriented-box adaptation: the GT covariance is
        rotated with its OBB, while every feature prior is an isotropic
        receptive-field Gaussian in the same augmented-image coordinate frame.
        """
        inverse_rf = torch.linalg.inv(rf_cov)  # [P,2,2]
        trace = torch.einsum('pij,gji->pg', inverse_rf, gt_cov)
        mahal = torch.einsum('pgi,pij,pgj->pg', delta, inverse_rf, delta)
        _, log_gt = torch.linalg.slogdet(gt_cov)
        _, log_rf = torch.linalg.slogdet(rf_cov)
        return .5 * (trace + mahal - 2. + log_rf[:, None] - log_gt[None, :])

    def _hierarchical_candidates(self, costs: torch.Tensor,
                                 rf_cov: torch.Tensor,
                                 gt_cov: torch.Tensor,
                                 delta: torch.Tensor) -> torch.Tensor:
        """Two-stage RFLA HieAssigner candidate set, adapted to rbox Gaussians.

        Stage one selects the published top-6 receptive fields per object.
        Stage two repeats the ranking with the receptive field shrunk by 0.9
        and adds one candidate only where stage one had no ownership.  This
        preserves the paper's hierarchical-label-assignment structure rather
        than substituting ordinary centre sampling.
        """
        num_priors = costs.shape[0]
        first_k, second_k = self.topk
        candidates = torch.zeros_like(costs, dtype=torch.bool)
        candidates.scatter_(0, costs.topk(min(first_k, num_priors), dim=0,
                                          largest=False).indices, True)
        # As in HieAssigner.anchor_rescale(..., ratio=.9), the second stage
        # uses a smaller receptive field.  It is only an extra proposal stage.
        second_costs = self._kl_distance(gt_cov, rf_cov * (.9 ** 2), delta)
        second = torch.zeros_like(candidates)
        second.scatter_(0, second_costs.topk(min(second_k, num_priors), dim=0,
                                               largest=False).indices, True)
        has_initial = candidates.any(dim=1)
        second_owner = second_costs.masked_fill(~second, float('inf')).argmin(dim=1)
        add = ~has_initial
        candidates[add, second_owner[add]] = True
        return candidates

    def assign(self, pred_instances: InstanceData, gt_instances: InstanceData,
               gt_instances_ignore: Optional[InstanceData] = None, **kwargs):
        boxes = gt_instances.bboxes.tensor if isinstance(gt_instances.bboxes, BaseBoxes) else gt_instances.bboxes
        num_gt, num_priors = boxes.shape[0], pred_instances.priors.shape[0]
        device = pred_instances.priors.device
        assigned = torch.zeros(num_priors, dtype=torch.long, device=device)
        labels = torch.full((num_priors,), -1, dtype=torch.long, device=device)
        overlaps = torch.zeros(num_priors, dtype=torch.float32, device=device)
        weights = torch.zeros(num_priors, dtype=torch.float32, device=device)
        if num_gt == 0 or num_priors == 0:
            result = AssignResult(num_gt, assigned, overlaps, labels=labels)
            result.set_extra_property('r005_weights', weights)
            return result
        priors = pred_instances.priors
        points = priors[:, :2]
        strides = priors[:, 2].clamp_min(1.)
        centres = boxes[:, :2]
        eye = torch.eye(2, device=device, dtype=boxes.dtype)
        covariance = _box_covariance(boxes) + 1e-5 * eye
        # The FPN location is represented by its Gaussian receptive field.
        rf_cov = (self.rf_scale * strides).square()[:, None, None] * eye
        degradation = kwargs.get('degradation', None)
        extra = 0.0
        if self.conditioned and degradation is not None:
            sigma, factor = float(degradation[0]), float(degradation[1])
            extra = self.covariance_scale * (sigma*sigma + max(factor*factor-1., 0.)/12.)
            covariance = covariance + extra * eye
        elif self.fixed_support > 0:
            covariance = covariance + self.fixed_support**2 * eye
        delta = points[:, None, :] - centres[None, :, :]
        costs = self._kl_distance(covariance, rf_cov, delta)
        responsibility = torch.softmax(-costs, dim=1)
        candidate = self._hierarchical_candidates(costs, rf_cov, covariance, delta)
        # Lower Gaussian distance is better.  A point owns only the instance
        # with the greatest normalized responsibility among its candidates.
        masked = costs.masked_fill(~candidate, float('inf'))
        best_cost, owner = masked.min(dim=1)
        positive = torch.isfinite(best_cost)
        assigned[positive] = owner[positive] + 1
        labels[positive] = gt_instances.labels[owner[positive]].long()
        # Quality target remains prediction/ORIGINAL-box IoU, never expanded.
        if positive.any():
            ious = self.iou_calculator(pred_instances.bboxes[positive], gt_instances.bboxes)
            overlaps[positive] = ious[torch.arange(positive.sum(), device=device), owner[positive]]
            weights[positive] = 1.
            # Ambiguity discount is an M1-only degradation effect.  Thus at
            # sigma=0/factor=1 both its covariance and its extra weights are
            # identically B2, including when a nonzero threshold was selected.
            if (self.conditioned and extra > 0 and num_gt > 1
                    and self.ambiguity_threshold > 0):
                # Gaussian responsibilities are normalized across all GT;
                # use their top-two separation, not a raw centre-distance.
                two = responsibility.topk(2, dim=1, largest=True).values
                margin = two[:, 0] - two[:, 1]
                ambiguous = positive & (margin < self.ambiguity_threshold)
                weights[ambiguous] = (margin[ambiguous] / self.ambiguity_threshold).clamp(0., 1.)
        result = AssignResult(num_gt, assigned, overlaps, labels=labels)
        result.set_extra_property('r005_weights', weights)
        result.set_extra_property('r005_responsibility', responsibility)
        return result


@MODELS.register_module()
class EvidenceRotatedRTMDetSepBNHead(RotatedRTMDetSepBNHead):
    """Production head glue: preserves raw OBB regression and consumes M1 weights."""
    def _get_targets_single(self, cls_scores, bbox_preds, flat_anchors,
                            valid_flags, gt_instances, img_meta,
                            gt_instances_ignore=None, unmap_outputs=True):
        inside = anchor_inside_flags(flat_anchors, valid_flags,
                                    img_meta['img_shape'][:2],
                                    self.train_cfg['allowed_border'])
        if not inside.any():
            return (None,) * 7
        anchors = flat_anchors[inside, :]
        pred = InstanceData(scores=cls_scores[inside, :],
                            bboxes=bbox_preds[inside, :], priors=anchors)
        result = self.assigner.assign(
            pred, gt_instances, gt_instances_ignore,
            degradation=img_meta.get('r005_degradation', (0., 1.)))
        sample = self.sampler.sample(result, pred, gt_instances)
        total = anchors.shape[0]
        targets = anchors.new_zeros((total, 5))
        labels = anchors.new_full((total,), self.num_classes, dtype=torch.long)
        label_weights = anchors.new_zeros(total, dtype=torch.float)
        metrics = anchors.new_zeros(total, dtype=torch.float)
        pos, neg = sample.pos_inds, sample.neg_inds
        if len(pos):
            # This is intentionally the original GT output by the dataset,
            # rather than any evidence/expanded support used for assignment.
            targets[pos] = sample.pos_gt_bboxes.regularize_boxes(self.angle_version)
            labels[pos] = sample.pos_gt_labels
            extra = result.get_extra_property('r005_weights')
            label_weights[pos] = extra[pos] if extra is not None else 1.
            for gt_index in torch.unique(sample.pos_assigned_gt_inds):
                chosen = pos[sample.pos_assigned_gt_inds == gt_index]
                metrics[chosen] = result.max_overlaps[chosen]
        if len(neg):
            label_weights[neg] = 1.
        if unmap_outputs:
            n = flat_anchors.size(0)
            anchors = unmap(anchors, n, inside)
            labels = unmap(labels, n, inside, fill=self.num_classes)
            label_weights = unmap(label_weights, n, inside)
            targets = unmap(targets, n, inside)
            metrics = unmap(metrics, n, inside)
        return anchors, labels, label_weights, targets, metrics, sample
