# model settings
norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    type='SigmoidEncoderDecoder',
    score_thr=0.5,
    mode='noco',
    backbone=dict(
        type='FlexPVTv2B0', # embed_dims=[32, 64, 160, 256]
        # embed_dims=[16, 32, 80, 128],
        strides=(4, 2, 1, 1), # S_i
        # sr_ratios=[8, 4, 2, 1], # R_i
        # sr_ratios=[4, 2, 1, 1], # R_i
        in_channels=1),
    neck=dict(
        type='FlexFPN',
        out_inds=0,
        # in_channels=[64, 128, 320, 512],
        in_channels=[32, 64, 160, 256],
        # in_channels=[16, 32, 80, 128],
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
