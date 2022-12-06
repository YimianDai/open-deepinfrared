# dataset settings
dataset_type = 'SIRSTDet2NoCoDataset'
data_root = 'data/sirst/'
# img_norm_cfg = dict(
#     mean=[111.89, ], std=[27.62, ], to_rgb=False)
img_norm_cfg = dict(
    mean=[111.89, 111.89, 111.89], std=[27.62, 27.62, 27.62], to_rgb=True)
train_pipeline = [
    # dict(type='LoadImageFromFile', color_type='grayscale'),
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1000, 600), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    # dict(type='LoadImageFromFile', color_type='grayscale'),
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1000, 600),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type='RepeatDataset',
        times=3,
        dataset=dict(
            type=dataset_type,
            # ann_file=[data_root + 'splits/mini_trainval.txt',], # for debug
            # ann_file=[data_root + 'splits/trainval_v1.txt',], # for SIRST v1
            ann_file=[data_root + 'splits/trainval_full.txt',], # for SIRST v2
            img_prefix=[data_root,],
            pipeline=train_pipeline)),
    val=dict(
        type=dataset_type,
        # ann_file=data_root + 'splits/mini_test.txt', # for debug
        # ann_file=data_root + 'splits/test_v1.txt', # SIRST v1
        ann_file=data_root + 'splits/test_full.txt', # SIRST v2
        img_prefix=data_root,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        # ann_file=data_root + 'splits/mini_test.txt', # for debug
        # ann_file=data_root + 'splits/test_v1.txt', # SIRST v1
        ann_file=data_root + 'splits/test_full.txt', # SIRST v2
        img_prefix=data_root,
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='mNoCoAP')
