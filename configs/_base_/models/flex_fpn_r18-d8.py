# model settings
norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained='open-mmlab://resnet18_v1c',
    backbone=dict(
        type='FlexResNet',
        stage_blocks=(2, 2, 2, 2),
        block='basic',
        stem_stride=2,
        max_pooling=True,
        in_channels=3,
        deep_stem=True,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1),
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch',
        contract_dilation=True),
    neck=dict(
        type='FlexFPN',
        out_inds=0,
        in_channels=[64, 128, 256, 512],
        out_channels=64,
        num_outs=4),
    decode_head=dict(
        type='FCNHead',
        in_channels=64,
        in_index=0,
        channels=32,
        num_convs=2,
        concat_input=True,
        dropout_ratio=0.1,
        num_classes=1,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='NoCoFocalLoss', use_sigmoid=True, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
