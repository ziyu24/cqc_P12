angle_version = 'le90'
backend_args = None
base_lr = 0.00025
checkpoint = 'https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-s_imagenet_600e.pth'
custom_hooks = [
    dict(type='mmdet.NumClassCheckHook'),
    dict(
        ema_type='mmdet.ExpMomentumEMA',
        momentum=0.0002,
        priority=49,
        type='EMAHook',
        update_buffers=True),
]
custom_imports = dict(
    allow_failed_imports=False, imports=[
        'src.r005_components',
    ])
data_root = '/home/rspip/cqc/data/dataset/dota/split_ss_dota/'
dataset_type = 'DOTADataset'
default_hooks = dict(
    checkpoint=dict(
        interval=1, max_keep_ckpts=35, save_last=True, type='CheckpointHook'),
    logger=dict(interval=50, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='mmdet.DetVisualizationHook'))
default_scope = 'mmrotate'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
interval = 12
launcher = 'pytorch'
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
max_epochs = 36
model = dict(
    backbone=dict(
        act_cfg=dict(type='SiLU'),
        arch='P5',
        channel_attention=True,
        deepen_factor=0.33,
        expand_ratio=0.5,
        init_cfg=dict(
            checkpoint=
            'https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-s_imagenet_600e.pth',
            prefix='backbone.',
            type='Pretrained'),
        norm_cfg=dict(type='SyncBN'),
        type='mmdet.CSPNeXt',
        widen_factor=0.5),
    bbox_head=dict(
        act_cfg=dict(type='SiLU'),
        anchor_generator=dict(
            offset=0, strides=[
                8,
                16,
                32,
            ], type='mmdet.MlvlPointGenerator'),
        angle_version='le90',
        bbox_coder=dict(angle_version='le90', type='DistanceAnglePointCoder'),
        exp_on_reg=False,
        feat_channels=128,
        in_channels=128,
        loss_angle=None,
        loss_bbox=dict(loss_weight=2.0, mode='linear', type='RotatedIoULoss'),
        loss_cls=dict(
            beta=2.0,
            loss_weight=1.0,
            type='mmdet.QualityFocalLoss',
            use_sigmoid=True),
        norm_cfg=dict(type='SyncBN'),
        num_classes=15,
        pred_kernel_size=1,
        scale_angle=False,
        share_conv=True,
        stacked_convs=2,
        type='EvidenceRotatedRTMDetSepBNHead',
        use_hbbox_loss=False,
        with_objectness=False),
    data_preprocessor=dict(
        batch_augments=None,
        bgr_to_rgb=False,
        boxtype2tensor=False,
        mean=[
            103.53,
            116.28,
            123.675,
        ],
        std=[
            57.375,
            57.12,
            58.395,
        ],
        type='mmdet.DetDataPreprocessor'),
    neck=dict(
        act_cfg=dict(type='SiLU'),
        expand_ratio=0.5,
        in_channels=[
            128,
            256,
            512,
        ],
        norm_cfg=dict(type='SyncBN'),
        num_csp_blocks=1,
        out_channels=128,
        type='mmdet.CSPNeXtPAFPN'),
    test_cfg=dict(
        max_per_img=2000,
        min_bbox_size=0,
        nms=dict(iou_threshold=0.1, type='nms_rotated'),
        nms_pre=2000,
        score_thr=0.05),
    train_cfg=dict(
        allowed_border=-1,
        assigner=dict(
            ambiguity_threshold=0.0,
            competition=False,
            conditioned=False,
            covariance_scale=0.5,
            fixed_support=0.0,
            iou_calculator=dict(type='RBboxOverlaps2D'),
            rf_scale=1.0,
            topk=(
                6,
                1,
            ),
            type='GaussianRFLEvidenceAssigner'),
        debug=False,
        pos_weight=-1),
    type='mmdet.RTMDet')
optim_wrapper = dict(
    optimizer=dict(lr=0.00025, type='AdamW', weight_decay=0.05),
    paramwise_cfg=dict(
        bias_decay_mult=0, bypass_duplicate=True, norm_decay_mult=0),
    type='OptimWrapper')
param_scheduler = [
    dict(
        begin=0, by_epoch=False, end=1000, start_factor=1e-05,
        type='LinearLR'),
    dict(
        T_max=18,
        begin=18,
        by_epoch=True,
        convert_to_iter_based=True,
        end=36,
        eta_min=1.25e-05,
        type='CosineAnnealingLR'),
]
randomness = dict(deterministic=False, seed=17)
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='val/annfiles/',
        data_prefix=dict(img_path='val/images/'),
        data_root='/home/rspip/zy/data/dataset/dota/dota1.0/split_ss_dota10/',
        pipeline=[
            dict(type='mmdet.LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                1024,
                1024,
            ), type='mmdet.Resize'),
            dict(
                box_type='qbox', type='mmdet.LoadAnnotations', with_bbox=True),
            dict(
                box_type_mapping=dict(gt_bboxes='rbox'),
                type='ConvertBoxType'),
            dict(factor=1, sigma=0.0, type='FixedGaussianDownsample'),
            dict(
                pad_val=dict(img=(
                    114,
                    114,
                    114,
                )),
                size=(
                    1024,
                    1024,
                ),
                type='mmdet.Pad'),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                    'r005_degradation',
                ),
                type='mmdet.PackDetInputs'),
        ],
        test_mode=True,
        type='DOTADataset'),
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = [
    dict(
        dump_path=None,
        eval_mode='area',
        iou_thrs=[
            0.5,
            0.75,
        ],
        metric='mAP',
        prefix='r005',
        type='R005DOTAMetric'),
]
test_pipeline = [
    dict(backend_args=None, type='mmdet.LoadImageFromFile'),
    dict(keep_ratio=True, scale=(
        1024,
        1024,
    ), type='mmdet.Resize'),
    dict(
        pad_val=dict(img=(
            114,
            114,
            114,
        )),
        size=(
            1024,
            1024,
        ),
        type='mmdet.Pad'),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
        ),
        type='mmdet.PackDetInputs'),
]
train_cfg = dict(max_epochs=35, type='EpochBasedTrainLoop', val_interval=999)
train_dataloader = dict(
    batch_size=1,
    dataset=dict(
        dataset=dict(
            ann_file='train/annfiles/',
            data_prefix=dict(img_path='train/images/'),
            data_root=
            '/home/rspip/zy/data/dataset/dota/dota1.0/split_ss_dota10/',
            filter_cfg=dict(filter_empty_gt=True),
            pipeline=[
                dict(type='mmdet.LoadImageFromFile'),
                dict(
                    box_type='qbox',
                    type='mmdet.LoadAnnotations',
                    with_bbox=True),
                dict(
                    box_type_mapping=dict(gt_bboxes='rbox'),
                    type='ConvertBoxType'),
                dict(
                    keep_ratio=True, scale=(
                        1024,
                        1024,
                    ), type='mmdet.Resize'),
                dict(
                    direction=[
                        'horizontal',
                        'vertical',
                        'diagonal',
                    ],
                    prob=0.75,
                    type='mmdet.RandomFlip'),
                dict(
                    angle_range=180,
                    prob=0.5,
                    rect_obj_labels=[
                        9,
                        11,
                    ],
                    type='RandomRotate'),
                dict(
                    prob=0.75,
                    scale_choices=(
                        1,
                        2,
                        4,
                        8,
                    ),
                    seed=17,
                    sigma_range=(
                        0.0,
                        3.2,
                    ),
                    type='SharedGaussianDownsample'),
                dict(
                    pad_val=dict(img=(
                        114,
                        114,
                        114,
                    )),
                    size=(
                        1024,
                        1024,
                    ),
                    type='mmdet.Pad'),
                dict(
                    meta_keys=(
                        'img_id',
                        'img_path',
                        'ori_shape',
                        'img_shape',
                        'scale_factor',
                        'r005_degradation',
                    ),
                    type='mmdet.PackDetInputs'),
            ],
            type='DOTADataset'),
        times=3,
        type='R005RepeatDataset'),
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(backend_args=None, type='mmdet.LoadImageFromFile'),
    dict(box_type='qbox', type='mmdet.LoadAnnotations', with_bbox=True),
    dict(box_type_mapping=dict(gt_bboxes='rbox'), type='ConvertBoxType'),
    dict(keep_ratio=True, scale=(
        1024,
        1024,
    ), type='mmdet.Resize'),
    dict(
        direction=[
            'horizontal',
            'vertical',
            'diagonal',
        ],
        prob=0.75,
        type='mmdet.RandomFlip'),
    dict(
        angle_range=180,
        prob=0.5,
        rect_obj_labels=[
            9,
            11,
        ],
        type='RandomRotate'),
    dict(
        pad_val=dict(img=(
            114,
            114,
            114,
        )),
        size=(
            1024,
            1024,
        ),
        type='mmdet.Pad'),
    dict(type='mmdet.PackDetInputs'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='val/annfiles/',
        data_prefix=dict(img_path='val/images/'),
        data_root='/home/rspip/zy/data/dataset/dota/dota1.0/split_ss_dota10/',
        pipeline=[
            dict(type='mmdet.LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                1024,
                1024,
            ), type='mmdet.Resize'),
            dict(
                box_type='qbox', type='mmdet.LoadAnnotations', with_bbox=True),
            dict(
                box_type_mapping=dict(gt_bboxes='rbox'),
                type='ConvertBoxType'),
            dict(factor=1, sigma=0.0, type='FixedGaussianDownsample'),
            dict(
                pad_val=dict(img=(
                    114,
                    114,
                    114,
                )),
                size=(
                    1024,
                    1024,
                ),
                type='mmdet.Pad'),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                    'r005_degradation',
                ),
                type='mmdet.PackDetInputs'),
        ],
        test_mode=True,
        type='DOTADataset'),
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = [
    dict(
        dump_path=None,
        eval_mode='area',
        iou_thrs=[
            0.5,
            0.75,
        ],
        metric='mAP',
        prefix='r005',
        type='R005DOTAMetric'),
]
val_pipeline = [
    dict(backend_args=None, type='mmdet.LoadImageFromFile'),
    dict(keep_ratio=True, scale=(
        1024,
        1024,
    ), type='mmdet.Resize'),
    dict(box_type='qbox', type='mmdet.LoadAnnotations', with_bbox=True),
    dict(box_type_mapping=dict(gt_bboxes='rbox'), type='ConvertBoxType'),
    dict(
        pad_val=dict(img=(
            114,
            114,
            114,
        )),
        size=(
            1024,
            1024,
        ),
        type='mmdet.Pad'),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
        ),
        type='mmdet.PackDetInputs'),
]
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='RotLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = '/home/rspip/zy/study/P12/runs/r005/confirmation/dota/B2_seed17'
