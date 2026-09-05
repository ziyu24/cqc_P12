"""Frozen r001 first-observer recipe: Oriented R-CNN, HRSC train -> test.

This derives only the model architecture from the public DOTA recipe.  The
split, paths, class count and 3x optimisation schedule are stated here so the
test split cannot enter fitting by accident.
"""

_base_ = [
    '/home/rspip/zy/study/third_party/ai4rs/configs/oriented_rcnn/oriented-rcnn-le90_r50_fpn_1x_dota.py'
]

data_root = '/home/rspip/zy/data/dataset/HRSC2016/'
model = dict(roi_head=dict(bbox_head=dict(num_classes=1)))

train_dataloader = dict(dataset=dict(
    type='HRSCDataset', data_root=data_root, ann_file='splits/train.txt',
    img_subdir='images', ann_subdir='annfiles', data_prefix=dict(sub_data_root='')))
val_dataloader = dict(dataset=dict(
    type='HRSCDataset', data_root=data_root, ann_file='splits/test.txt',
    img_subdir='images', ann_subdir='annfiles', data_prefix=dict(sub_data_root='')))
test_dataloader = val_dataloader

# Same 36-epoch schedule used by the archived public HRSC baseline, now with
# the protocol-required train-only split and frozen r001 seed.
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=36, val_interval=1)
param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(type='MultiStepLR', by_epoch=True, begin=0, end=36,
         milestones=[24, 33], gamma=0.1),
]
optim_wrapper = dict(optimizer=dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001),
                     clip_grad=dict(max_norm=35, norm_type=2))
randomness = dict(seed=20260905)
default_hooks = dict(checkpoint=dict(interval=1, max_keep_ckpts=2, save_best='auto'))
