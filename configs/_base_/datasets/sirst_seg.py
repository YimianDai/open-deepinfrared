centerness_thr = [0.1, 0.3, 0.5, 0.7, 0.9]

# dataset settings for SIRST-Seg dataset
dataset_type = 'SIRSTSegDataset'
data_root = 'data/sirst'
img_norm_cfg = dict(
    mean=[111.89, 111.89, 111.89], std=[27.62, 27.62, 27.62], to_rgb=True)
    # mean=[111.89, ], std=[27.62, ], to_rgb=False)
crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadBinaryAnnotations'),
    dict(type='Resize', img_scale=(2048, 512), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
# train_pipeline = [
#     dict(type='LoadImageFromFile', color_type='grayscale'),
#     dict(type='LoadBinaryAnnotations'),
#     dict(type='Resize', img_scale=(2048, 512), ratio_range=(0.5, 2.0)),
#     dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
#     dict(type='RandomFlip', prob=0.5),
#     dict(type='RandomGammaCorrection'),
#     dict(type='Normalize', **img_norm_cfg),
#     dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
#     dict(type='DefaultFormatBundle'),
#     dict(type='Collect', keys=['img', 'gt_semantic_seg']),
# ]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    # dict(type='LoadImageFromFile', color_type='grayscale'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 512),
        # img_scale=(512, 512),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ]),
    # dict(type='PrintPipeline'),
]
data = dict(
    # samples_per_gpu=4,
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        centerness_thr=centerness_thr,
        data_root=data_root,
        img_dir='mixed',
        ann_dir='annotations/masks',
        split='splits/trainval_v1.txt',
        # split='splits/mini_trainval.txt',
        # split='splits/trainval_v1fullv2target.txt',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        score_thr=0.25,
        centerness_thr=centerness_thr,
        data_root=data_root,
        img_dir='mixed',
        ann_dir='annotations/masks',
        split='splits/test_v1.txt',
        # split='splits/mini_test.txt',
        # split='splits/trainvaltest_v2target.txt',
        # split='splits/hongshow.txt',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        score_thr=0.25,
        data_root=data_root,
        img_dir='mixed',
        ann_dir='annotations/masks',
        split='splits/test_v1.txt',
        # split='splits/mini_test.txt',
        # split='splits/trainvaltest_v2target.txt',
        # split='splits/trainvaltest_full.txt',
        # split='splits/mini-v2background.txt',
        pipeline=test_pipeline))
