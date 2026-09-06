"""Frozen r005 HRSC development protocol; select one named arm via R005_ARM.

The base is the registered Rotated RTMDet-R-S HRSC recipe, but this task uses
the official train/val identities (not its trainval/test result), one public
ImageNet initialization, identical 36-epoch optimizer semantics and seed 5.
"""
import os

_base_ = '/dev/shm/zy/study/pth_data/baseline_rotated_rtmdet_s_fpn_9x_le90/HRSC_trainval_test_taos/config.py'

arm = os.environ.get('R005_ARM', 'B1')
run_root = os.environ.get('R005_WORK_DIR', './runs/r005/work_dirs')
seed = 5
dataset_root = '/home/rspip/zy/data/dataset/HRSC2016'
custom_imports = dict(imports=['src.r005_components'], allow_failed_imports=False)

# 25% clear, otherwise deterministic per-original-image blur/downsample draw.
degradation = dict(type='SharedGaussianDownsample', prob=0.0 if arm == 'B0' else .75,
                   sigma_range=(0., 3.2), scale_choices=(1, 2, 4, 8), seed=seed)
pack = dict(type='mmdet.PackDetInputs', meta_keys=('img_id','img_path','ori_shape','img_shape','scale_factor','r005_degradation'))
pipeline = [
    dict(type='mmdet.LoadImageFromFile'),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(type='mmdet.Resize', scale=(800, 800), keep_ratio=True),
    dict(type='mmdet.RandomFlip', prob=.75, direction=['horizontal','vertical','diagonal']),
    dict(type='RandomRotate', prob=.5, angle_range=180),
    degradation,
]
if arm == 'B3':
    pipeline.append(dict(type='BlurBoxExpansion', support_scale=1.0))
pipeline += [dict(type='mmdet.Pad', size=(800,800), pad_val=dict(img=(114,114,114))), pack]

train_dataloader = dict(
    batch_size=2, num_workers=4, persistent_workers=True, pin_memory=True,
    dataset=dict(type='RepeatDataset', times=3, dataset=dict(
        type='HRSCDataset', data_root=dataset_root + '/', ann_file='splits/train.txt',
        data_prefix=dict(sub_data_root=''), img_subdir='images', ann_subdir='annfiles',
        filter_cfg=dict(filter_empty_gt=True), pipeline=pipeline)))
val_dataloader = dict(dataset=dict(data_root=dataset_root + '/', ann_file='splits/val.txt',
    data_prefix=dict(sub_data_root=''), img_subdir='images', ann_subdir='annfiles'))
test_dataloader = val_dataloader

# B1 retains native DynamicSoftLabelAssigner. B2 is an RFLA-style Gaussian
# receptive field assignment; M1 with zero degradation invokes its same path.
if arm != 'B1' and arm != 'B3' and arm != 'B0':
    conditioned = arm in ('M1', 'M1_no_competition', 'M1_expanded_regression')
    model = dict(bbox_head=dict(type='EvidenceRotatedRTMDetSepBNHead'),
        train_cfg=dict(allowed_border=-1, pos_weight=-1, debug=False,
            assigner=dict(type='GaussianRFLEvidenceAssigner', topk=13,
            rf_scale=1.0, conditioned=conditioned,
            covariance_scale=1.0 if arm != 'M1_no_condition' else 0.0,
            ambiguity_threshold=0.1 if arm != 'M1_no_competition' else 0.0,
            fixed_support=1.0 if arm == 'M0' else 0.0,
            iou_calculator=dict(type='RBboxOverlaps2D'))))

work_dir = f'{run_root}/{arm}'
randomness = dict(seed=seed, deterministic=False)
default_hooks = dict(checkpoint=dict(interval=1, max_keep_ckpts=1, save_best='dota_ap07/mAP', rule='greater'))
