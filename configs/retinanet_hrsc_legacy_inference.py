"""Inference adapter for the released MMRotate 0.x HRSC RetinaNet weight.

Only anchor parameterization changes; the box coder still uses le90.
The 1.x FakeRotatedAnchorGenerator documents this compatibility setting.
"""
_base_ = [
    '/home/rspip/zy/study/third_party/ai4rs/configs/rotated_retinanet/'
    'rotated-retinanet-rbox-le90_r50_fpn_rr-6x_hrsc.py'
]
model = dict(bbox_head=dict(anchor_generator=dict(angle_version=None)))
