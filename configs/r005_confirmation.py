"""Frozen r005 confirmation configuration for HRSC stability and DOTA.

The environment selects only the already preregistered dataset/arm/seed.  No
metric outcome participates in configuration selection.
"""
import os

dataset = os.environ.get('R005_DATASET', 'hrsc')
if dataset == 'hrsc':
    _base_ = '/dev/shm/zy/study/pth_data/baseline_rotated_rtmdet_s_fpn_9x_le90/HRSC_trainval_test_taos/config.py'
    data_root = '/home/rspip/zy/data/dataset/HRSC2016'
    dataset_type = 'HRSCDataset'
    image_size = 800
    train_ann = f"splits/{os.environ.get('R005_HRSC_TRAIN_SPLIT', 'trainval')}.txt"
    eval_ann = f"splits/{os.environ.get('R005_HRSC_EVAL_SPLIT', 'test')}.txt"
    train_prefix = dict(sub_data_root='')
    eval_prefix = dict(sub_data_root='')
    train_extra = dict(img_subdir='images', ann_subdir='annfiles')
    eval_extra = dict(img_subdir='images', ann_subdir='annfiles')
    classes = None
else:
    _base_ = '/dev/shm/zy/study/pth_data/baseline_rotated_rtmdet_s_fpn_3x_le90/DOTA10_train_val_taos/config.py'
    data_root = '/home/rspip/zy/data/dataset/dota/dota1.0/split_ss_dota10'
    dataset_type = 'DOTADataset'
    image_size = 1024
    train_ann = 'train/annfiles/'
    eval_ann = 'val/annfiles/'
    train_prefix = dict(img_path='train/images/')
    eval_prefix = dict(img_path='val/images/')
    train_extra = dict()
    eval_extra = dict()
    classes = ('plane','baseball-diamond','bridge','ground-track-field','small-vehicle',
               'large-vehicle','ship','tennis-court','basketball-court','storage-tank',
               'soccer-ball-field','roundabout','harbor','swimming-pool','helicopter')

arm = os.environ['R005_ARM']
method_arm = os.environ.get('R005_METHOD_ARM', arm)
seed = int(os.environ['R005_SEED'])
work_dir = os.environ['R005_WORK_DIR']
eval_sigma = float(os.environ.get('R005_EVAL_SIGMA', '0'))
eval_factor = int(os.environ.get('R005_EVAL_FACTOR', '1'))
eval_target = os.environ.get('R005_EVAL_TARGET', 'original')
covariance_scale = float(os.environ.get('R005_COVARIANCE_SCALE', '.5'))
ambiguity_threshold = float(os.environ.get('R005_AMBIGUITY_THRESHOLD', '.2'))
per_gpu_batch = int(os.environ.get('R005_PER_GPU_BATCH', '1'))
custom_imports = dict(imports=['src.r005_components'], allow_failed_imports=False)

degradation = dict(type='SharedGaussianDownsample', prob=.75,
                   sigma_range=(0., 3.2), scale_choices=(1, 2, 4, 8), seed=seed)
pack = dict(type='mmdet.PackDetInputs', meta_keys=('img_id','img_path','ori_shape','img_shape','scale_factor','r005_degradation'))
pipeline = [
    dict(type='mmdet.LoadImageFromFile'),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(type='mmdet.Resize', scale=(image_size, image_size), keep_ratio=True),
    dict(type='mmdet.RandomFlip', prob=.75, direction=['horizontal','vertical','diagonal']),
    dict(type='RandomRotate', prob=.5, angle_range=180, **({'rect_obj_labels':[9,11]} if dataset == 'dota' else {})),
    degradation,
]
if method_arm in ('B3', 'M1_expanded_regression'):
    pipeline.append(dict(type='BlurBoxExpansion', support_scale=1.0))
pipeline += [dict(type='mmdet.Pad', size=(image_size,image_size), pad_val=dict(img=(114,114,114))), pack]

train_dataset = dict(type=dataset_type, data_root=data_root + '/', ann_file=train_ann,
                     data_prefix=train_prefix, filter_cfg=dict(filter_empty_gt=True),
                     pipeline=pipeline, **train_extra)
train_dataloader = dict(batch_size=per_gpu_batch, num_workers=4, persistent_workers=True,
                        pin_memory=True, dataset=train_dataset)

eval_pipeline = [
    dict(type='mmdet.LoadImageFromFile'),
    dict(type='mmdet.Resize', scale=(image_size,image_size), keep_ratio=True),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(type='FixedGaussianDownsample', sigma=eval_sigma, factor=eval_factor),
    *([dict(type='BlurBoxExpansion', support_scale=1.0)] if method_arm == 'B3' and eval_target == 'expanded' else []),
    dict(type='mmdet.Pad', size=(image_size,image_size), pad_val=dict(img=(114,114,114))), pack]
eval_dataset = dict(type=dataset_type, data_root=data_root + '/', ann_file=eval_ann,
                    data_prefix=eval_prefix, pipeline=eval_pipeline, test_mode=True, **eval_extra)
val_dataloader = dict(batch_size=per_gpu_batch, num_workers=4, persistent_workers=True,
                      pin_memory=True, sampler=dict(type='DefaultSampler', shuffle=False), dataset=eval_dataset)
test_dataloader = val_dataloader
val_evaluator = [dict(type='R005DOTAMetric', metric='mAP', eval_mode='area', iou_thrs=[.5,.75], prefix='r005')]
test_evaluator = val_evaluator

if method_arm not in ('B1', 'B3'):
    conditioned = method_arm == 'M1'
    model = dict(bbox_head=dict(type='EvidenceRotatedRTMDetSepBNHead'),
        train_cfg=dict(allowed_border=-1, pos_weight=-1, debug=False,
            assigner=dict(type='GaussianRFLEvidenceAssigner', topk=(6,1), rf_scale=1.0,
                conditioned=conditioned, covariance_scale=covariance_scale,
                ambiguity_threshold=ambiguity_threshold if conditioned else 0.,
                competition=conditioned, fixed_support=0.,
                iou_calculator=dict(type='RBboxOverlaps2D'))))

# HRSC test is never queried during trainval training. DOTA's val is the
# preregistered checkpoint-selection split and keeps the original AP75 rule.
if dataset == 'hrsc':
    train_cfg = dict(max_epochs=36, type='EpochBasedTrainLoop', val_interval=999)
    # Keep the frozen development-selected epoch without querying test during
    # training.  The executor picks that exact epoch afterwards.
    default_hooks = dict(checkpoint=dict(interval=1, max_keep_ckpts=36,
                                         save_last=True))
else:
    default_hooks = dict(checkpoint=dict(interval=1, max_keep_ckpts=1,
                                         save_best='r005/AP75', rule='greater'))
randomness = dict(seed=seed, deterministic=False)
