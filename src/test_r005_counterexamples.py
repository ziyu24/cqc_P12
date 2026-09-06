"""Executable r005 counterexamples against the actual production assigner."""
import sys
sys.path.insert(0, '.')
import torch
from mmengine.structures import InstanceData
from mmrotate.structures import RotatedBoxes
from src.r005_components import (GaussianRFLEvidenceAssigner,
                                 SharedGaussianDownsample, _box_covariance)

def sample():
    priors = torch.tensor([[0.,0.,8.,8.],[2.5,0.,8.,8.],[20.,0.,8.,8.]])
    pred = InstanceData(priors=priors, scores=torch.zeros(3,1),
                        bboxes=RotatedBoxes(torch.tensor([[0.,0.,8.,4.,0.],[2.5,0.,8.,4.,0.],[20.,0.,8.,4.,0.]])))
    gt = InstanceData(labels=torch.zeros(2,dtype=torch.long),
                      bboxes=RotatedBoxes(torch.tensor([[0.,0.,8.,4.,0.],[5.,0.,8.,4.,0.]])))
    return pred, gt

def main():
    pred, gt = sample()
    b2 = GaussianRFLEvidenceAssigner(conditioned=False, topk=3)
    m1 = GaussianRFLEvidenceAssigner(conditioned=True, topk=3, covariance_scale=1., ambiguity_threshold=.1)
    zero_b2 = b2.assign(pred, gt, degradation=(0.,1.))
    zero_m1 = m1.assign(pred, gt, degradation=(0.,1.))
    assert torch.equal(zero_b2.gt_inds, zero_m1.gt_inds), 'zero degradation must equal B2'
    assert torch.allclose(zero_b2.max_overlaps, zero_m1.max_overlaps), 'zero degradation quality differs'
    assert torch.equal(zero_b2.get_extra_property('r005_weights'), zero_m1.get_extra_property('r005_weights')), 'zero degradation weights differ'
    nonzero = m1.assign(pred, gt, degradation=(3.2,8.))
    # One prior has at most one owner; assignment representation makes this a production invariant.
    assert nonzero.gt_inds.ndim == 1 and (nonzero.gt_inds >= 0).all()
    weights = nonzero.get_extra_property('r005_weights')
    assert (weights >= 0).all() and (weights <= 1).all()
    responsibility = nonzero.get_extra_property('r005_responsibility')
    assert torch.allclose(responsibility.sum(dim=1), torch.ones(3)), 'GT responsibilities must sum to one'
    assert weights[1] < 1, 'near-tie responsibility must be downweighted'
    # Assignment support must not alter stored/regressed GT coordinates.
    assert torch.equal(gt.bboxes.tensor, torch.tensor([[0.,0.,8.,4.,0.],[5.,0.,8.,4.,0.]]))
    # Every arm receives the same image-level draw when the protocol seed and
    # image identity are fixed; this is independent of DataLoader order.
    transform = SharedGaussianDownsample(prob=1., seed=5)
    image = torch.zeros(16, 16, 3, dtype=torch.uint8).numpy()
    first = transform.transform({'img': image.copy(), 'img_id': '100000001'})
    second = transform.transform({'img': image.copy(), 'img_id': '100000001'})
    assert torch.equal(torch.from_numpy(first['r005_degradation']), torch.from_numpy(second['r005_degradation']))
    # The covariance's principal axes rotate with the rbox (not with a stale
    # pre-rotation coordinate frame); 90 degrees swaps width/height variance.
    unrotated = _box_covariance(torch.tensor([[0., 0., 8., 4., 0.]]))[0]
    rotated = _box_covariance(torch.tensor([[0., 0., 8., 4., torch.pi / 2]]))[0]
    assert torch.allclose(rotated.diag(), unrotated.diag().flip(0), atol=1e-5)
    print('r005 counterexamples: PASS')

if __name__ == '__main__':
    main()
