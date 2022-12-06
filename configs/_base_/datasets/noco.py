# dataset settings for SIRST-NoCo dataset
dataset_type = 'SIRSTNoCoDataset'
data_root = 'data/sirst'
img_norm_cfg = dict(
    mean=[111.89, ], std=[27.62, ], to_rgb=False)
crop_size = (512, 512)

train_pipeline = [
    dict(type='LoadImageFromFile', color_type='grayscale'),
    dict(type='LoadBinaryAnnotations'),
    dict(type='Resize', img_scale=(2048, 512), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='RandomGammaCorrection'),
    dict(type='NoCoTargets'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='NoCoFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg', 'gt_noco_map']),
]

test_pipeline = [
    dict(type='NoCoLoadImageFromFile', color_type='grayscale'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img', 'ori_img']),
        ])
]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='mixed',
        ann_dir='annotations/masks',
        split='splits/mini_trainval.txt',   # for debug
        # split='splits/trainval_v1.txt',   # SIRST v1
        # split='splits/trainval_full.txt', # SIRST v2
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        score_thr=0.5,
        data_root=data_root,
        img_dir='mixed',
        ann_dir='annotations/masks',
        split='splits/mini_test.txt',   # for debug
        # split='splits/test_v1.txt',   # SIRST v1
        # split='splits/test_full.txt', # SIRST v2
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        score_thr=0.5,
        data_root=data_root,
        img_dir='mixed',
        ann_dir='annotations/masks',
        split='splits/mini_test.txt',   # for debug
        # split='splits/test_v1.txt',   # SIRST v1
        # split='splits/test_full.txt', # SIRST v2
        pipeline=test_pipeline)
)
