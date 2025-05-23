# Top-level training loop config (must be named train_cfg for MMEngine)
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=12, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# Model internal train_cfg and test_cfg (for heads, etc)
model_train_cfg = dict(
    rpn=dict(
        assigner=dict(type='mmdet.MaxIoUAssigner', pos_iou_thr=0.7, neg_iou_thr=0.3, min_pos_iou=0.3, match_low_quality=True, ignore_iof_thr=-1),
        sampler=dict(type='mmdet.RandomSampler', num=256, pos_fraction=0.5, neg_pos_ub=-1, add_gt_as_proposals=True),
        allowed_border=-1,
        pos_weight=-1,
        debug=False
    ),
    rcnn=dict(
        assigner=dict(type='mmdet.MaxIoUAssigner', pos_iou_thr=0.5, neg_iou_thr=0.3, min_pos_iou=0.5, match_low_quality=True, ignore_iof_thr=-1),
        sampler=dict(type='mmdet.RandomSampler', num=512, pos_fraction=0.25, neg_pos_ub=-1, add_gt_as_proposals=True),
        pos_weight=-1,
        debug=False,
        mask_size=56
    )
)

model_test_cfg = dict(
    rpn=dict(
        score_thr=0.01,
        nms_pre=300,
        max_per_img=1000,
        nms=dict(type='nms', iou_threshold=0.7),
        min_bbox_size=0
    ),
    rcnn=dict(
        score_thr=0.3,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100,
        mask_thr_binary=0.5
    )
)

checkpoint = 'https://download.openmmlab.com/mmclassification/v0/efficientnet/efficientnet-b3_3rdparty_8xb32-aa_in1k_20220119-5b4887a0.pth'

model = dict(
    type='mmdet.MaskRCNN',
    backbone=dict(
        type='mmdet.EfficientNet',
        arch='b3',
        out_indices=(3, 4, 5),
        init_cfg=dict(type='Pretrained', prefix='backbone', checkpoint=checkpoint)
    ),
    neck=dict(
        type='mmdet.BIFPN',
        in_channels=[48, 136, 384],
        out_channels=256,
        num_outs=5,
        stack=2
        # activation=dict(type='Swish'),
    ),
    rpn_head=dict(
        type='mmdet.RepPointsRPNHead',
        in_channels=256,
        feat_channels=256,
        point_feat_channels=256,
        num_classes=1,
        stacked_convs=3,
        use_grid_points=False,
        num_points=9,
        gradient_mul=0.3,
        point_strides=[8, 16, 32, 64, 128],
        point_base_scale=4,
        transform_method='minmax',
        bbox_coder=dict(
            type='mmdet.DistancePointBBoxCoder',
            point_stride=1
        ),
        loss_cls=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0
        ),
        loss_bbox_init=dict(type='mmdet.SmoothL1Loss', beta=1.0, loss_weight=0.5),
        loss_bbox_refine=dict(type='mmdet.SmoothL1Loss', beta=1.0, loss_weight=1.0),
        loss_bbox=dict(type='mmdet.SmoothL1Loss', beta=1.0, loss_weight=1.0),
        train_cfg=dict(
            init=dict(
                assigner=dict(
                    type='mmdet.PointAssigner',
                    scale=64, #==========>> Potential issue of too small object sizes, scale up
                    pos_num=3 #==========>> Potential issue of too small object sizes, scale up (3 or 5)
                ),
                allowed_border=-1,
                pos_weight=-1
            ),
            refine=dict(
                assigner=dict(
                    type='mmdet.MaxIoUAssigner',
                    iou_calculator=dict(type='mmdet.BboxOverlaps2D'),
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.4,
                    min_pos_iou=0,
                    ignore_iof_thr=-1,
                ),
                allowed_border=0,
                pos_weight=-1
            )
        )
    ),
    roi_head=dict(
        type='mmdet.StandardRoIHead',
        bbox_roi_extractor=dict(
            type='mmdet.SingleRoIExtractor',
            roi_layer=dict(
                type='RoIAlign',
                output_size=7,
                sampling_ratio=2
            ),
            out_channels=256,
            featmap_strides=[8, 16, 32, 64, 128] #==========>> Match featmap strides to BiFPN [8, 16, 32, 64, 128]
        ),
        bbox_head=dict(
            type='mmdet.Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=19
        ),
        mask_roi_extractor=dict(
            type='mmdet.SingleRoIExtractor',
            roi_layer=dict(
                type='RoIAlign',
                output_size=28,
                sampling_ratio=2
            ),
            out_channels=256,
            featmap_strides=[8, 16, 32, 64, 128] #==========>> Match featmap strides to BiFPN [8, 16, 32, 64, 128]
        ),
        mask_head=dict(
            type='mmdet.FCNMaskHead',
            num_convs=4,
            in_channels=256,
            conv_out_channels=256,
            num_classes=19,
            mask_size=56
        )
    ),
    train_cfg=model_train_cfg,
    test_cfg=model_test_cfg
)

# Dataset root
data_root = '/content/GroupEGS-Thesis-Dataset/'

# Custom classes with depth included
classes = (
    'damage-crack', 'damage-dent', 'damage-scratch', 'damage-tear',
    'panel-apillar', 'panel-bonnet', 'panel-bpillar',
    'panel-cpillar', 'panel-dpillar', 'panel-fender',
    'panel-frontdoor', 'panel-qpanel', 'panel-reardoor',
    'panel-rockerpanel', 'panel-roof', 'panel-tailgate', 'panel-trunklid',
    'depth-deep', 'depth-shallow'
)

# Visualization colors (palette) with added entries for depth classes
metainfo = {
    'classes': classes,
    'palette': [
        (220, 20, 60), (119, 11, 32), (0, 128, 0), (0, 0, 255),         # damage types
        (255, 140, 0), (255, 215, 0), (128, 0, 128), (255, 105, 180),   # panels
        (64, 224, 208), (123, 104, 238), (255, 69, 0), (154, 205, 50),
        (70, 130, 180), (95, 158, 160), (255, 20, 147), (0, 191, 255),
        (189, 183, 107),                                               # last panel
        (0, 255, 127), (160, 82, 45)                                    # depth-deep, depth-shallow
    ]
}

# Common image normalization config
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],  # Standard ImageNet means
    std=[58.395, 57.12, 57.375],
    to_rgb=True
)

# Dataloaders
train_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='mmdet.DefaultSampler', shuffle=True),
    dataset=dict(
        type='mmdet.CocoDataset',
        metainfo=metainfo,
        ann_file=data_root + 'train/_annotations_cleaned_final.coco.json',
        data_prefix=dict(img=data_root + 'train/'),
        filter_cfg=dict(filter_empty_gt=True),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='mmdet.LoadAnnotations', with_bbox=True, with_mask=True),
            dict(type='mmdet.FilterAnnotations', min_gt_bbox_wh=(2, 2), keep_empty=False),  #==========>> Try to add 'min_gt_bbox_wh=(1, 1)' param to check bbox per image
            dict(type='mmdet.Resize', scale=(896, 896), keep_ratio=True),
            dict(type='mmdet.RandomFlip', prob=0.5),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='mmdet.Pad', size_divisor=32, pad_val=0),
            dict(type='mmdet.PackDetInputs', meta_keys=('filename', 'ori_shape', 'img_shape', 'pad_shape',
                   'scale_factor', 'flip', 'flip_direction', 'img_norm_cfg')),
        ],
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='mmdet.DefaultSampler', shuffle=False),
    dataset=dict(
        type='mmdet.CocoDataset',
        metainfo=metainfo,
        ann_file=data_root + 'valid/_annotations_cleaned_final.coco.json',
        data_prefix=dict(img=data_root + 'valid/'),
        filter_cfg=dict(filter_empty_gt=True),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='mmdet.LoadAnnotations', with_bbox=True, with_mask=True),
            dict(type='mmdet.FilterAnnotations', min_gt_bbox_wh=(1, 1)),
            dict(type='mmdet.Resize', scale=(896, 896), keep_ratio=True),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='mmdet.Pad', size_divisor=32, pad_val=0),
            dict(type='mmdet.PackDetInputs', meta_keys=('filename', 'ori_shape', 'img_shape', 'pad_shape',
                   'scale_factor', 'flip', 'flip_direction', 'img_norm_cfg', 'img_id')),
        ],
    )
)

test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='mmdet.DefaultSampler', shuffle=False),
    dataset=dict(
        type='mmdet.CocoDataset',
        metainfo=metainfo,
        ann_file=data_root + 'test/_annotations_cleaned_final.coco.json',
        data_prefix=dict(img=data_root + 'test/'),
        filter_cfg=dict(filter_empty_gt=True),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='mmdet.LoadAnnotations', with_bbox=True, with_mask=True),
            dict(type='mmdet.FilterAnnotations', min_gt_bbox_wh=(1, 1)),
            dict(type='mmdet.Resize', scale=(896, 896), keep_ratio=True),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='mmdet.Pad', size_divisor=32, pad_val=0),
            dict(type='mmdet.PackDetInputs', meta_keys=('filename', 'ori_shape', 'img_shape', 'pad_shape',
                   'scale_factor', 'flip', 'flip_direction', 'img_norm_cfg', 'img_id')),
        ],
    )
)

# Evaluators
val_evaluator = dict(
    type='mmdet.CocoMetric',
    ann_file=data_root + 'valid/_annotations_cleaned_final.coco.json',
    metric=['bbox', 'segm'],
    classwise=True
)

test_evaluator = dict(
    type='mmdet.CocoMetric',
    ann_file=data_root + 'test/_annotations_cleaned_final.coco.json',
    metric=['bbox', 'segm'],
    classwise=True
)

# Optimizer and LR schedule
optim_wrapper = dict(
    optimizer=dict(type='mmdet.SGD', lr=0.00125, momentum=0.9, weight_decay=0.0001)
)

param_scheduler = [
    dict(type='mmdet.LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(type='mmdet.MultiStepLR', by_epoch=True, milestones=[8, 11], gamma=0.1)
]

# Runtime settings
default_hooks = dict(
    logger=dict(type='mmdet.LoggerHook', interval=50),
    checkpoint=dict(type='mmdet.CheckpointHook', interval=1),
    param_scheduler=dict(type='mmdet.ParamSchedulerHook'),
    timer=dict(type='mmdet.IterTimerHook'),
    sampler_seed=dict(type='mmdet.DistSamplerSeedHook'),
    visualization=dict(type='mmdet.DetVisualizationHook')
)

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl')
)

log_level = 'INFO'
# load_from = 'checkpoints/retinanet_effb3_fpn_crop896_8x4_1x_coco_20220322_234806-615a0dda.pth'
resume = False

# Enable mixed precision training
fp16 = dict(loss_scale='dynamic')

visualizer = dict(
    type='mmdet.DetLocalVisualizer',
    vis_backends=[dict(type='mmdet.LocalVisBackend')],
    name='visualizer'
)
