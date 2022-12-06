"""Shell Script:
!python tools/train_det.py configs/oscar/oscar_w_noco_head_r18_caffe_fpn_p2_128_gn-head_1x_sirst_det2noco.py

python tools/train_det.py configs/oscar/sota/oscar_w_noco_head_r18_caffe_fpn_p2_128_gn-head_1x_sirst_det2noco.py --gpu-id 0 --work-dir work_dirs/oscar_w_noco_head_r18_caffe_fpn_p2_128_gn-head_1x_sirst_det2noco_gpu_0

python tools/train_det.py configs/oscar/sota/oscar_w_noco_head_r18_caffe_fpn_p2_128_gn-head_1x_sirst_det2noco.py --gpu-id 1 --work-dir work_dirs/oscar_w_noco_head_r18_caffe_fpn_p2_128_gn-head_1x_sirst_det2noco_gpu_1

python tools/train_det.py configs/oscar/sota/oscar_w_noco_head_r18_caffe_fpn_p2_128_gn-head_1x_sirst_det2noco.py --gpu-id 2 --work-dir work_dirs/oscar_w_noco_head_r18_caffe_fpn_p2_128_gn-head_1x_sirst_det2noco_gpu_2

python tools/train_det.py configs/oscar/sota/oscar_w_noco_head_r18_caffe_fpn_p2_128_gn-head_1x_sirst_det2noco.py --gpu-id 3 --work-dir work_dirs/oscar_w_noco_head_r18_caffe_fpn_p2_128_gn-head_1x_sirst_det2noco_gpu_3
"""

########################## Hyper-parameter Settings ##########################
out_inds = [0, 1] # index of FPN level
assert len(out_inds) == 2
sirst_version = 'sirstv2' # 'sirstv1' or 'sirstv2'
depth = 18
fpn_strides = [4, 8, 16, 32]
if sirst_version == 'sirstv1':
    split_cfg = {
        'train_split': 'splits/trainval_v1.txt',
        'val_split': 'splits/test_v1.txt',
        'test_split': 'splits/test_v1.txt',
    }
elif sirst_version == 'sirstv2':
    split_cfg = {
        'train_split': 'splits/trainval_full.txt',
        'val_split': 'splits/test_full.txt',
        'test_split': 'splits/test_full.txt',
    }
else:
    raise ValueError("wrong sirst_version")
backbone_cfg = {
    'stem_stride': 2,
    'max_pooling': True,
    'strides': (1, 2, 2, 2),
}
neck_cfg = {
    'out_inds': out_inds
}
head_cfg = {}
if depth == 18:
    backbone_cfg['depths'] = (2, 2, 2, 2)
    backbone_cfg['block'] = 'basic'
    backbone_cfg['pretrained'] = 'open-mmlab://resnet18_v1c'
    neck_cfg['in_channels'] = [64, 128, 256, 512]
    neck_cfg['out_channels'] = 128
else:
    raise ValueError("wrong backbone depth")
head_cfg['in_channels'] = neck_cfg['out_channels']

############################## Dataset Setting ##############################
dataset_type = 'SIRSTDet2NoCoDataset'
data_root = 'data/sirst/'
img_norm_cfg = dict(
    mean=[111.89, 111.89, 111.89], std=[27.62, 27.62, 27.62], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1000, 600), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='NoCoTargets', mode='det2noco'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='OSCARPad', size_divisor=32),
    dict(type='OSCARFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_noco_map'])]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        # img_scale=(1333, 800),
        img_scale=(1000, 600),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            # dict(type='NoCoTargets', mode='det2noco'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='OSCARPad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])])]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type='RepeatDataset',
        times=3,
        dataset=dict(
            type=dataset_type,
            ann_file=[data_root + split_cfg['train_split'],],
            img_prefix=[data_root,],
            pipeline=train_pipeline)),
    val=dict(
        type=dataset_type,
        ann_file=data_root + split_cfg['val_split'],
        img_prefix=data_root,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + split_cfg['test_split'],
        img_prefix=data_root,
        pipeline=test_pipeline))

############################### Model Setting ###############################
norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    type='OSCARNet',
    backbone=dict(
        type='FlexResNet',
        depths=backbone_cfg['depths'],
        block=backbone_cfg['block'],
        stem_stride=backbone_cfg['stem_stride'],
        max_pooling=backbone_cfg['max_pooling'],
        in_channels=3,
        deep_stem=True,
        out_indices=(0, 1, 2, 3),
        strides=backbone_cfg['strides'],
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch',
        contract_dilation=True,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='open-mmlab://detectron/resnet50_caffe')
    ),
    neck=dict(
        type='FlexFPN',
        # start_level=neck_cfg['out_inds'],
        out_inds=neck_cfg['out_inds'],
        in_channels=neck_cfg['in_channels'],
        out_channels=neck_cfg['out_channels'],
        num_outs=4),
    bbox_head=dict(
        type='OSCARNoCoHead',
        num_classes=1,
        stride_ratio=1.5,
        in_channels=head_cfg['in_channels'],
        regress_ranges=((-1, 1e8),),
        stacked_convs=4,
        feat_channels=head_cfg['in_channels']//2,
        strides=[fpn_strides[out_inds[0]], fpn_strides[out_inds[1]]],
        loss_coarse_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=3.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_refine_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=3.0,
            alpha=0.25,
            loss_weight=2.0),
        # loss_refine_noco=dict(
        #     type='CrossEntropyLoss', use_sigmoid=True, loss_weight=0.1),
        # loss_refine_noco=dict(
        #     type='FocalLoss',
        #     use_sigmoid=True,
        #     gamma=3.0,
        #     alpha=0.25,
        #     loss_weight=2.0),
        loss_refine_noco=dict(
            type='RegQualityFocalLoss',
            beta = 2.0,
            use_sigmoid=True,
            loss_weight=1000.0),
        loss_coarse_bbox=dict(type='DIoULoss', loss_weight=1.0),
        loss_refine_bbox=dict(type='DIoULoss', loss_weight=1.0),
    ),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000, # modified for SIRST
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100)
    # test_cfg=dict(
    #     nms_pre=10, # modified for SIRST
    #     min_bbox_size=0,
    #     score_thr=0.25,
    #     nms=dict(type='nms', iou_threshold=0.01),
    #     max_per_img=5)
#     test_cfg=dict(
#         nms_pre=40, # modified for SIRST
#         min_bbox_size=0,
#         score_thr=0.25,
#         nms=dict(type='nms', iou_threshold=0.01),
#         max_per_img=20)
)

############################## optimizer setting ##############################
# optimizer
# optimizer = dict(
#     lr=0.01, paramwise_cfg=dict(bias_lr_mult=2., bias_decay_mult=0.))
# optimizer_config = dict(
#     _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
# lr_config = dict(
#     policy='step',
#     warmup='constant',
#     warmup_iters=500,
#     warmup_ratio=1.0 / 3,
#     step=[8, 11])
optimizer = dict(
    # constructor='LearningRateDecayOptimizerConstructor',
    # _delete_=True,
    type='AdamW',
    lr=0.002,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg={
        'decay_rate': 0.9,
        'decay_type': 'stage_wise',
        'num_layers': 6
    })
lr_config = dict(
    # _delete_=True,
    policy='poly',
    warmup='linear',
    warmup_iters=1000,
    # step=[27, 33],
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    # by_epoch=False
    )
optimizer_config = dict(type='Fp16OptimizerHook', loss_scale='dynamic')
runner = dict(type='EpochBasedRunner', max_epochs=12)

############################## runtime setting ##############################

checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
custom_hooks = [dict(type='NumClassCheckHook')]

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]

# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = 'fork'