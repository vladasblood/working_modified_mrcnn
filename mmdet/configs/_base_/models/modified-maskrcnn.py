# FP16 mixed precision
fp16 = dict(loss_scale='dynamic')

model = dict(
    type='MaskRCNN',
    backbone=dict(
        type='EfficientNet',
        arch='b3',
        pretrained=True,
        out_indices=(1, 2, 3),
        init_cfg=dict(type='Pretrained', checkpoint='open-mmlab://efficientnet_b3'),
    ),
    neck=dict(
        type='BiFPN',
        in_channels=[48, 136, 384],
        out_channels=256,
        num_outs=5,
        stack=2
    ),
    rpn_head=dict(
        type='RepPointsRPNHead',
        num_classes=1,
        in_channels=256,
        feat_channels=256,
        point_feat_channels=256,
        stacked_convs=3,
        num_points=9,
        gradient_mul=0.3,
        point_strides=[8, 16, 32, 64, 128],
        point_base_scale=4,
        transform_method='minmax',
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox_init=dict(type='SmoothL1Loss', beta=1.0, loss_weight=0.5),
        loss_bbox_refine=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)
    ),
    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(
                type='RoIAlign',
                output_size=7,
                sampling_ratio=2
            ),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32, 64]
        ),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=17
        ),
        mask_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(
                type='RoIAlign',
                output_size=14,
                sampling_ratio=2
            ),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32, 64]
        ),
        mask_head=dict(
            type='FCNMaskHead',
            num_convs=4,
            in_channels=256,
            conv_out_channels=256,
            num_classes=17
        )
    )
)
