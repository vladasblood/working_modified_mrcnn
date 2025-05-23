model = dict(
    type='MaskRCNN',
    backbone=dict(
        type='EfficientDet',
        model_name='efficientdet-d3',
        pretrained=True
    ),
    neck=dict(
        type='BiFPN',
        num_levels=5,
        num_channels=160
    ),
    roi_head=dict(
        bbox_head=dict(num_classes=80),
        mask_head=dict(num_classes=80)
    )
)
