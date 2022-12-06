# dataset settings for traditional approaches
dataset_type = 'SIRSTNoCoDataset'
data_root = 'data/sirst'
img_norm_cfg = dict(
    mean=[111.89, ], std=[27.62, ], to_rgb=False)
crop_size = (512, 512)

train_pipeline = [
    dict(type='LoadImageFromFile', color_type='grayscale'),
    dict(type='Collect', keys=['img']),
]

test_pipeline = [
    dict(type='LoadImageFromFile', color_type='grayscale'),
    dict(type='Collect', keys=['img']),
]
data = dict(
    # samples_per_gpu=4,
    samples_per_gpu=4,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='mixed',
        ann_dir='annotations/masks',
        split='splits/trainval_v1.txt',
        # split='splits/mini_trainval.txt',
        # split='splits/trainval_v1fullv2target.txt',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        score_thr=0.5,
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
        score_thr=0.5,
        data_root=data_root,
        img_dir='mixed',
        ann_dir='annotations/masks',
        split='splits/test_v1.txt',
        # split='splits/mini_test.txt',
        # split='splits/trainvaltest_v2target.txt',
        # split='splits/trainvaltest_full.txt',
        # split='splits/mini-v2background.txt',
        pipeline=test_pipeline))
