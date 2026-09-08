"""Shared, static-base configuration values for r005 confirmation."""
import os


def build_confirmation(dataset: str) -> dict:
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
    fixed_epoch = int(os.environ.get('R005_FIXED_EPOCH', '36'))
    if dataset == 'hrsc':
        data_root, dataset_type, image_size = '/home/rspip/zy/data/dataset/HRSC2016', 'HRSCDataset', 800
        train_ann = f"splits/{os.environ.get('R005_HRSC_TRAIN_SPLIT', 'trainval')}.txt"
        eval_ann = f"splits/{os.environ.get('R005_HRSC_EVAL_SPLIT', 'test')}.txt"
        train_prefix = eval_prefix = dict(sub_data_root='')
        train_extra = eval_extra = dict(img_subdir='images', ann_subdir='annfiles')
        rotate = dict(type='RandomRotate', prob=.5, angle_range=180)
    elif dataset == 'dota':
        data_root, dataset_type, image_size = '/home/rspip/zy/data/dataset/dota/dota1.0/split_ss_dota10', 'DOTADataset', 1024
        train_ann, eval_ann = 'train/annfiles/', 'val/annfiles/'
        train_prefix, eval_prefix = dict(img_path='train/images/'), dict(img_path='val/images/')
        train_extra = eval_extra = dict()
        rotate = dict(type='RandomRotate', prob=.5, angle_range=180, rect_obj_labels=[9, 11])
    else:
        raise ValueError(f'unknown r005 dataset {dataset}')
    degradation = dict(type='SharedGaussianDownsample', prob=.75,
                       sigma_range=(0., 3.2), scale_choices=(1, 2, 4, 8), seed=seed)
    pack = dict(type='mmdet.PackDetInputs', meta_keys=('img_id','img_path','ori_shape','img_shape','scale_factor','r005_degradation'))
    pipeline = [dict(type='mmdet.LoadImageFromFile'),
                dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
                dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
                dict(type='mmdet.Resize', scale=(image_size, image_size), keep_ratio=True),
                dict(type='mmdet.RandomFlip', prob=.75, direction=['horizontal','vertical','diagonal']), rotate, degradation]
    if method_arm in ('B3', 'M1_expanded_regression'):
        pipeline.append(dict(type='BlurBoxExpansion', support_scale=1.0))
    pipeline += [dict(type='mmdet.Pad', size=(image_size,image_size), pad_val=dict(img=(114,114,114))), pack]
    eval_pipeline = [dict(type='mmdet.LoadImageFromFile'),
                     dict(type='mmdet.Resize', scale=(image_size,image_size), keep_ratio=True),
                     dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
                     dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
                     dict(type='FixedGaussianDownsample', sigma=eval_sigma, factor=eval_factor),
                     *([dict(type='BlurBoxExpansion', support_scale=1.0)] if method_arm == 'B3' and eval_target == 'expanded' else []),
                     dict(type='mmdet.Pad', size=(image_size,image_size), pad_val=dict(img=(114,114,114))), pack]
    values = dict(work_dir=work_dir, custom_imports=dict(imports=['src.r005_components'], allow_failed_imports=False),
        train_dataloader=dict(_delete_=True, batch_size=per_gpu_batch, num_workers=4, persistent_workers=True, pin_memory=True,
            # Keep the development protocol's three deterministic visits per
            # source image.  The wrapper exposes r005_draw_index, so each
            # visit receives its shared, seed-fixed image degradation.
            dataset=dict(type='R005RepeatDataset', times=3, dataset=dict(
                type=dataset_type, data_root=data_root+'/', ann_file=train_ann, data_prefix=train_prefix,
                filter_cfg=dict(filter_empty_gt=True), pipeline=pipeline, **train_extra))),
        val_dataloader=dict(_delete_=True, batch_size=per_gpu_batch, num_workers=4, persistent_workers=True, pin_memory=True,
            sampler=dict(type='DefaultSampler', shuffle=False), dataset=dict(type=dataset_type, data_root=data_root+'/',
                ann_file=eval_ann, data_prefix=eval_prefix, pipeline=eval_pipeline, test_mode=True, **eval_extra)),
        val_evaluator=[dict(type='R005DOTAMetric', metric='mAP', eval_mode='area', iou_thrs=[.5,.75], prefix='r005',
                            dump_path=os.environ.get('R005_DUMP_PATH') or None)],
        randomness=dict(seed=seed, deterministic=False))
    values['test_dataloader'] = dict(values['val_dataloader'],
                                     dataset=dict(values['val_dataloader']['dataset']))
    values['test_evaluator'] = values['val_evaluator']
    if method_arm not in ('B1', 'B3'):
        values['model'] = dict(bbox_head=dict(type='EvidenceRotatedRTMDetSepBNHead'), train_cfg=dict(
            allowed_border=-1, pos_weight=-1, debug=False, assigner=dict(type='GaussianRFLEvidenceAssigner',
            topk=(6,1), rf_scale=1.0, conditioned=method_arm == 'M1', covariance_scale=covariance_scale,
            ambiguity_threshold=ambiguity_threshold if method_arm == 'M1' else 0., competition=method_arm == 'M1',
            fixed_support=0., iou_calculator=dict(type='RBboxOverlaps2D'))))
    # Both confirmation datasets use the arm-specific epoch fixed during HRSC
    # development.  In particular, DOTA val is the independent confirmation
    # endpoint and must not also choose a best checkpoint.
    values['train_cfg'] = dict(max_epochs=fixed_epoch, type='EpochBasedTrainLoop', val_interval=999)
    values['default_hooks'] = dict(checkpoint=dict(
        _delete_=True, type='CheckpointHook', interval=1,
        max_keep_ckpts=fixed_epoch, save_last=True))
    return values
