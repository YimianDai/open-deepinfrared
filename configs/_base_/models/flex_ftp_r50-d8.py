# model settings
norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    type='SigmoidEncoderDecoder',
    score_thr=0.5,
    mode='noco',
    pretrained='open-mmlab://resnet50_v1c',
    backbone=dict(
        type='FlexResNet',
        depths=(3, 4, 6, 3),
        block='bottleneck',
        stem_channels=64,
        base_channels=64,
        stem_stride=2,
        max_pooling=True,
        in_channels=1,
        deep_stem=True,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1),
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch',
        contract_dilation=True,
    ),
    neck=dict(
        type='FlexFTP',
        ###### PVTv2 parameters ######
        # sr_ratios=[4, 2, 1, 1], # R_i
        sr_ratios=[1, 1, 1, 1], # R_i
        num_heads=4,
        mlp_ratio=0.5,
        # mlp_ratio=1,
        trans_lateral=False,
        ###### FlexFPN parameters ######
        out_inds=0,
        in_channels=[256, 512, 1024, 2048],
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
            type='SoftIoULoss', loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
