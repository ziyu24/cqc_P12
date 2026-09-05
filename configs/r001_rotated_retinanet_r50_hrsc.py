"""Frozen r001 second-observer training recipe: HRSC2016 train -> test."""

_base_ = [
    '/home/rspip/zy/study/third_party/ai4rs/configs/rotated_retinanet/rotated-retinanet-rbox-le90_r50_fpn_rr-6x_hrsc.py'
]

# The r001 protocol forbids using the test images or their degradations for
# training.  Keep the upstream 6x schedule and its single-process batch/lr.
data_root = '/home/rspip/zy/data/dataset/HRSC2016/'
train_dataloader = dict(dataset=dict(
    data_root=data_root,
    ann_file='splits/train.txt',
    data_prefix=dict(sub_data_root='', img_subdir='images', ann_subdir='annfiles')))
val_dataloader = dict(dataset=dict(
    data_root=data_root,
    ann_file='splits/test.txt',
    data_prefix=dict(sub_data_root='', img_subdir='images', ann_subdir='annfiles')))
test_dataloader = val_dataloader

randomness = dict(seed=20260905)
default_hooks = dict(checkpoint=dict(interval=1, max_keep_ckpts=2, save_best='auto'))
