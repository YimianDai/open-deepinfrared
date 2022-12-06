# Copyright (c) OpenMMLab. All rights reserved.
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn import ConvModule, Scale
from mmcv.runner import force_fp32
from mmdet.core import multi_apply, reduce_mean
from mmdet.core import build_bbox_coder, multi_apply
from mmdet.core.anchor.point_generator import MlvlPointGenerator
from mmdet.core.utils import filter_scores_and_topk, select_single_mlvl
from mmdet.models.builder import HEADS, build_loss
from mmdet.models.dense_heads.base_dense_head import BaseDenseHead
from mmdet.models.dense_heads.dense_test_mixins import BBoxTestMixin

INF = 1e8


@HEADS.register_module()
class OSCARNoCoHead(BaseDenseHead, BBoxTestMixin):
    """Anchor-free head used in `FCOS <https://arxiv.org/abs/1904.01355>`_.

    The FCOS head does not use anchor boxes. Instead bounding boxes are
    predicted at each pixel is used to suppress low-quality predictions.
    Here norm_on_bbox and dcn_on_last_conv are training tricks used in official
    repo, which will bring remarkable mAP gains
    of up to 4.9. Please see https://github.com/tianzhi0549/FCOS for
    more detail.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        strides (list[int] | list[tuple[int, int]]): Strides of points
            in multiple feature levels. Default: (4, 8, 16, 32, 64).
        regress_ranges (tuple[tuple[int, int]]): Regress range of multiple
            level points.
        conv_bias (bool | str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias of conv will be set as True if `norm_cfg` is None, otherwise
            False. Default: "auto".
        loss_cls (dict): Config of classification loss.
        loss_bbox (dict): Config of localization loss.
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: norm_cfg=dict(type='GN', num_groups=32, requires_grad=True).
        init_cfg (dict or list[dict], optional): Initialization config dict.

    Example:
        >>> self = FCOSHead(11, 7)
        >>> feats = [torch.rand(1, 7, s, s) for s in [4, 8, 16, 32, 64]]
        >>> cls_score, bbox_pred = self.forward(feats)
        >>> assert len(cls_score) == len(self.scales)
    """  # noqa: E501

    def __init__(self,
                 num_classes,
                 in_channels,
                 stride_ratio=1,
                 feat_channels=256,
                 stacked_convs=4,
                 strides=(4, 8, 16, 32, 64),
                 dcn_on_last_conv=False,
                 conv_bias='auto',
                 regress_ranges=((-1, 64), (64, 128), (128, 256), (256, 512),
                                 (512, INF)),
                 norm_on_bbox=False,
                 # coarse layer
                 loss_coarse_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 loss_coarse_bbox=dict(type='DIoULoss', loss_weight=1.0),
                 bbox_coarse_coder=dict(type='DistancePointBBoxCoder'),
                 # refine layer
                 loss_refine_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 loss_refine_noco=dict(
                     type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
                 loss_refine_bbox=dict(type='DIoULoss', loss_weight=1.0),
                 bbox_refine_coder=dict(type='DeltaXYWHBBoxCoder'),
                 conv_cfg=None,
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=dict(
                     type='Normal',
                     layer='Conv2d',
                     std=0.01,
                     ),
                 **kwargs):
        super(OSCARNoCoHead, self).__init__(init_cfg)
        self.num_classes = num_classes
        self.stride_ratio = stride_ratio
        self.use_sigmoid_cls = loss_refine_cls.get('use_sigmoid', False)
        if self.use_sigmoid_cls:
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        assert len(strides) == 2
        self.refine_stride, self.coarse_stride = strides
        assert self.refine_stride <= self.coarse_stride
        self.dcn_on_last_conv = dcn_on_last_conv
        assert conv_bias == 'auto' or isinstance(conv_bias, bool)
        self.conv_bias = conv_bias
        # coarse layer
        self.loss_coarse_cls = build_loss(loss_coarse_cls)
        self.loss_coarse_bbox = build_loss(loss_coarse_bbox)
        self.bbox_coarse_coder = build_bbox_coder(bbox_coarse_coder)
        self.coarse_prior_generator = MlvlPointGenerator([self.coarse_stride])
        self.coarse_num_base_priors = self.coarse_prior_generator.num_base_priors[0]
        # refine layer
        self.loss_refine_cls = build_loss(loss_refine_cls)
        self.loss_refine_bbox = build_loss(loss_refine_bbox)
        self.loss_refine_noco = build_loss(loss_refine_noco)
        self.bbox_refine_coder = build_bbox_coder(bbox_refine_coder)
        self.refine_prior_generator = MlvlPointGenerator([self.refine_stride])
        self.refine_num_base_priors = self.refine_prior_generator.num_base_priors[0]

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.fp16_enabled = False

        self.regress_ranges = regress_ranges
        self.norm_on_bbox = norm_on_bbox

        self._init_layers()

    def _init_layers(self):
        """Initialize layers of the head."""
        self._init_cls_convs()
        self._init_reg_convs()
        self._init_noco_convs()
        self._init_predictor()
        self.scales = nn.ModuleList([Scale(1.0), Scale(1.0)])

    def _init_cls_convs(self):
        """Initialize classification conv layers of the head."""
        self.refine_cls_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            if self.dcn_on_last_conv and i == self.stacked_convs - 1:
                conv_cfg = dict(type='DCNv2')
            else:
                conv_cfg = self.conv_cfg
            self.refine_cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.conv_bias))

        self.coarse_cls_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            if self.dcn_on_last_conv and i == self.stacked_convs - 1:
                conv_cfg = dict(type='DCNv2')
            else:
                conv_cfg = self.conv_cfg
            self.coarse_cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.conv_bias))

    def _init_reg_convs(self):
        """Initialize bbox regression conv layers of the head."""
        self.refine_reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels + self.in_channels//2 if i == 0 else self.feat_channels
            if self.dcn_on_last_conv and i == self.stacked_convs - 1:
                conv_cfg = dict(type='DCNv2')
            else:
                conv_cfg = self.conv_cfg
            self.refine_reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.conv_bias))

        self.coarse_reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            if self.dcn_on_last_conv and i == self.stacked_convs - 1:
                conv_cfg = dict(type='DCNv2')
            else:
                conv_cfg = self.conv_cfg
            self.coarse_reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.conv_bias))

    def _init_noco_convs(self):
        """Initialize classification conv layers of the head."""
        self.refine_noco_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            if self.dcn_on_last_conv and i == self.stacked_convs - 1:
                conv_cfg = dict(type='DCNv2')
            else:
                conv_cfg = self.conv_cfg
            self.refine_noco_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.conv_bias))

    def _init_predictor(self):
        """Initialize predictor layers of the head."""
        # coarse layer
        self.coarse_conv_cls = nn.Conv2d(
            self.feat_channels, self.cls_out_channels, 3, padding=1)
        self.coarse_conv_reg = nn.Conv2d(self.feat_channels, 4, 3, padding=1)
        self.coarse_conv_mod = nn.Conv2d(
            self.feat_channels, self.cls_out_channels, 3, padding=1)
        # refine layer
        self.refine_conv_cls = nn.Conv2d(
            self.feat_channels, self.cls_out_channels, 3, padding=1)
        self.refine_conv_reg = nn.Conv2d(self.feat_channels, 4, 3, padding=1)
        self.refine_conv_noco = nn.Conv2d(self.feat_channels, 1, 3, padding=1)
        self.refine_conv_mod = nn.Conv2d(
            self.feat_channels, self.cls_out_channels, 3, padding=1)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        """Hack some keys of the model state dict so that can load checkpoints
        of previous version."""
        version = local_metadata.get('version', None)
        if version is None:
            # the key is different in early versions
            # for example, 'fcos_cls' become 'conv_cls' now
            bbox_head_keys = [
                k for k in state_dict.keys() if k.startswith(prefix)
            ]
            ori_predictor_keys = []
            new_predictor_keys = []
            # e.g. 'fcos_cls' or 'fcos_reg'
            for key in bbox_head_keys:
                ori_predictor_keys.append(key)
                key = key.split('.')
                conv_name = None
                if key[1].endswith('cls'):
                    conv_name = 'conv_cls'
                elif key[1].endswith('reg'):
                    conv_name = 'conv_reg'
                elif key[1].endswith('centerness'):
                    conv_name = 'conv_centerness'
                else:
                    assert NotImplementedError
                if conv_name is not None:
                    key[1] = conv_name
                    new_predictor_keys.append('.'.join(key))
                else:
                    ori_predictor_keys.pop(-1)
            for i in range(len(new_predictor_keys)):
                state_dict[new_predictor_keys[i]] = state_dict.pop(
                    ori_predictor_keys[i])
        super()._load_from_state_dict(state_dict, prefix, local_metadata,
                                      strict, missing_keys, unexpected_keys,
                                      error_msgs)

    def forward_train(self,
                      x,
                      img_metas,
                      gt_bboxes,
                      gt_labels=None,
                      gt_noco_maps=None,
                      gt_bboxes_ignore=None,
                      proposal_cfg=None,
                      **kwargs):
        """
        Args:
            x (list[Tensor]): Features from FPN.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            proposal_cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used

        Returns:
            tuple:
                losses: (dict[str, Tensor]): A dictionary of loss components.
                proposal_list (list[Tensor]): Proposals of each image.
        """
        outs = self(x)
        if gt_labels is None:
            loss_inputs = outs + (gt_bboxes, img_metas)
        else:
            loss_inputs = outs + (gt_bboxes, gt_labels, gt_noco_maps, img_metas)
        losses = self.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        if proposal_cfg is None:
            return losses
        else:
            proposal_list = self.get_bboxes(
                *outs, img_metas=img_metas, cfg=proposal_cfg)
            return losses, proposal_list

    def forward(self, feats):
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple:
                cls_scores (list[Tensor]): Box scores for each scale level, \
                    each is a 4D-tensor, the channel number is \
                    num_points * num_classes.
                bbox_preds (list[Tensor]): Box energies / deltas for each \
                    scale level, each is a 4D-tensor, the channel number is \
                    num_points * 4.
        """
        # reg_feats, cls_feats: 4D-tensor
        assert len(feats) == 2
        assert len(self.scales) == 2
        refine_feat, coarse_feat = feats

        coarse_score, coarse_bbox_pred, coarse_cls_feat,\
            coarse_reg_feat = self.coarse_forward(coarse_feat)
        refine_score, refine_bbox_pred, refine_noco_pred, refine_cls_feat = self.refine_forward(
            refine_feat, coarse_reg_feat)
        coarse_score, refine_score = self.mutual_modulate(coarse_score,
                                                          coarse_cls_feat,
                                                          refine_score,
                                                          refine_cls_feat)

        return [refine_score], [refine_bbox_pred], [refine_noco_pred], [coarse_score], \
            [coarse_bbox_pred]

    def coarse_forward(self, coarse_feat):
        # coarse_forward
        coarse_cls_feat = coarse_feat
        coarse_reg_feat = coarse_feat

        for cls_layer in self.coarse_cls_convs:
            coarse_cls_feat = cls_layer(coarse_cls_feat)
        coarse_score = self.coarse_conv_cls(coarse_cls_feat)

        for reg_layer in self.coarse_reg_convs:
            coarse_reg_feat = reg_layer(coarse_reg_feat)
        coarse_bbox_pred = self.coarse_conv_reg(coarse_reg_feat)

        coarse_bbox_pred = self.scales[1](coarse_bbox_pred).float()
        if self.norm_on_bbox:
            # bbox_pred needed for gradient computation has been modified
            # by F.relu(bbox_pred) when run with PyTorch 1.10. So replace
            # F.relu(bbox_pred) with bbox_pred.clamp(min=0)
            coarse_bbox_pred = coarse_bbox_pred.clamp(min=0)
            if not self.training:
                coarse_bbox_pred *= self.coarse_stride
        else:
            coarse_bbox_pred = coarse_bbox_pred.exp()
        return coarse_score, coarse_bbox_pred, coarse_cls_feat, coarse_reg_feat

    def refine_forward(self, refine_feat, coarse_reg_feat):
        # refine_forward
        refine_cls_feat = refine_feat
        refine_noco_feat = refine_feat
        # refine_reg_feat = refine_feat
        refine_featmap_size = refine_cls_feat.shape[-2:]
        coarse_reg_feat = F.interpolate(coarse_reg_feat,
                                        size=refine_featmap_size,
                                        mode='bilinear')
        refine_reg_feat = torch.cat((refine_feat, coarse_reg_feat), 1)

        for cls_layer in self.refine_cls_convs:
            refine_cls_feat = cls_layer(refine_cls_feat)
        refine_score = self.refine_conv_cls(refine_cls_feat)

        for reg_layer in self.refine_reg_convs:
            refine_reg_feat = reg_layer(refine_reg_feat)
        refine_bbox_pred = self.refine_conv_reg(refine_reg_feat)
        refine_bbox_pred = self.scales[0](refine_bbox_pred).float()

        for noco_layer in self.refine_noco_convs:
            refine_noco_feat = noco_layer(refine_noco_feat)
        refine_noco_pred = self.refine_conv_noco(refine_noco_feat)

        # if not self.training:
        #     refine_bbox_pred *= self.refine_stride

        return refine_score, refine_bbox_pred, refine_noco_pred, refine_cls_feat

    def mutual_modulate(self, coarse_score, coarse_cls_feat,
                        refine_score,  refine_cls_feat):
        refine_featmap_size = refine_cls_feat.shape[-2:]
        coarse_weight = self.coarse_conv_mod(coarse_cls_feat)
        coarse_weight = F.interpolate(coarse_weight, size=refine_featmap_size,
                                      mode='bilinear').sigmoid()

        coarse_featmap_size = coarse_cls_feat.shape[-2:]
        refine_weight = self.refine_conv_mod(refine_cls_feat)
        refine_weight = F.interpolate(refine_weight, size=coarse_featmap_size,
                                      mode='bilinear').sigmoid()

        refine_score = refine_score * coarse_weight
        coarse_score = coarse_score * refine_weight

        return coarse_score, refine_score

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'noco_preds'))
    def loss(self,
             refine_scores,
             refine_bbox_preds,
             refine_noco_preds,
             coarse_scores,
             coarse_bbox_preds,
             gt_bboxes,
             gt_labels,
             gt_noco_maps,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute loss of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level,
                each is a 4D-tensor, the channel number is
                num_points * num_classes.
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level, each is a 4D-tensor, the channel number is
                num_points * 4.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert len(coarse_scores) == len(refine_scores)
        assert len(refine_scores) == len(refine_bbox_preds) == 1
        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        batch_size = coarse_scores[0].size(0)
        coarse_featmap_size = coarse_scores[0].size()[-2:]
        coarse_hei, coarse_wid = coarse_featmap_size
        refine_featmap_size = refine_bbox_preds[0].size()[-2:]
        # gt_noco_maps
        # refine_hei, refine_wid = refine_featmap_size
        gt_noco_maps = F.interpolate(
            gt_noco_maps, size=refine_featmap_size, mode='nearest-exact')

        ############################ 1 coarse layer ############################
        ###### 1.1 prepare learning targets
        all_level_coarse_points = self.coarse_prior_generator.grid_priors(
            [coarse_featmap_size],
            dtype=coarse_scores[0].dtype,
            device=coarse_scores[0].device)
        # labels: list[Tensor]
        # each Tensor represents a level with shape (B*H_i*W_i,)
        coarse_labels, coarse_bbox_targets= self.get_coarse_targets(
            all_level_coarse_points, gt_bboxes, gt_labels)
        num_imgs = coarse_scores[0].size(0)
        flatten_coarse_labels = torch.cat(coarse_labels)
        flatten_coarse_bbox_targets = torch.cat(coarse_bbox_targets)

        ###### 1.2 prepare predictions
        flatten_coarse_scores = [
            coarse_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
            for coarse_score in coarse_scores
        ]
        flatten_coarse_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
            for bbox_pred in coarse_bbox_preds
        ]
        flatten_coarse_scores = torch.cat(flatten_coarse_scores)
        flatten_coarse_bbox_preds = torch.cat(flatten_coarse_bbox_preds)
        # repeat points to align with bbox_preds
        flatten_coarse_points = torch.cat(
            [points.repeat(num_imgs, 1) for points in all_level_coarse_points])
        ###### 1.3 compute loss
        coarse_pos_inds = ((flatten_coarse_labels >= 0)
                    & (flatten_coarse_labels < bg_class_ind)).nonzero().reshape(-1)
        coarse_num_pos = torch.tensor(
            len(coarse_pos_inds), dtype=torch.float,
            device=refine_bbox_preds[0].device)
        coarse_num_pos = max(reduce_mean(coarse_num_pos), 1.0)
        ### 1.3.1 cls loss
        loss_coarse_cls = self.loss_coarse_cls(
            flatten_coarse_scores, flatten_coarse_labels, avg_factor=coarse_num_pos)
        ### 1.3.2 bbox loss
        pos_coarse_bbox_preds = flatten_coarse_bbox_preds[coarse_pos_inds]
        pos_coarse_bbox_targets = flatten_coarse_bbox_targets[coarse_pos_inds]
        if len(coarse_pos_inds) > 0:
            pos_coarse_points = flatten_coarse_points[coarse_pos_inds]
            # print("pos_coarse_points:", pos_coarse_points)
            pos_decoded_bbox_preds = self.bbox_coarse_coder.decode(
                pos_coarse_points, pos_coarse_bbox_preds)
            # print("pos_decoded_bbox_preds:", pos_decoded_bbox_preds)
            pos_decoded_bbox_targets = self.bbox_coarse_coder.decode(
                pos_coarse_points, pos_coarse_bbox_targets)
            # print("pos_decoded_bbox_targets:", pos_decoded_bbox_targets)
            loss_coarse_bbox = self.loss_coarse_bbox(
                pos_decoded_bbox_preds,
                pos_decoded_bbox_targets)
        else:
            loss_coarse_bbox = pos_coarse_bbox_preds.sum()

        ############################ 2 pred align ############################
        ### 2.1 先解码得到 bbox 坐标
        # flatten_coarse_bbox_preds *= self.coarse_stride
        flatten_decoded_coarse_bbox_preds = self.bbox_coarse_coder.decode(
            flatten_coarse_points, flatten_coarse_bbox_preds)
        ### 2.2 对 bbox 恢复成 (B, 4, H, W) 再插值
        decoded_coarse_bbox_preds = flatten_decoded_coarse_bbox_preds.reshape(
            batch_size, coarse_hei, coarse_wid, 4).permute(0, 3, 1, 2)
        interp_decoded_coarse_bbox_preds = F.interpolate(
            decoded_coarse_bbox_preds, size=refine_featmap_size,
            mode='nearest-exact')
        ### 2.3 为接下来的输入做准备
        interp_decoded_coarse_bbox_pred_list = torch.chunk(
            interp_decoded_coarse_bbox_preds, batch_size, dim=0)
        anchors = [
            bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
            for bbox_pred in interp_decoded_coarse_bbox_pred_list
        ]
        flatten_anchors = torch.cat(anchors)
        # for _ in range(4):
        #     print("flatten_anchors:", flatten_anchors)

        ############################ 3 refine layer ############################
        ###### 3.1 prepare learning targets
        all_level_refine_points = self.refine_prior_generator.grid_priors(
            [refine_featmap_size],
            dtype=refine_bbox_preds[0].dtype,
            device=refine_bbox_preds[0].device)
        # labels: list[Tensor]
        # each Tensor represents a level with shape (B*H_i*W_i,)
        refine_labels, refine_bbox_targets, refine_noco_targets = self.get_refine_targets(
            anchors, all_level_refine_points, gt_bboxes, gt_labels, gt_noco_maps)
        flatten_refine_labels = torch.cat(refine_labels)
        flatten_refine_bbox_targets = torch.cat(refine_bbox_targets)
        flatten_refine_noco_targets = torch.cat(refine_noco_targets)

        ###### 3.2 prepare predictions
        flatten_refine_scores = [
            refine_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
            for refine_score in refine_scores
        ]
        flatten_refine_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
            for bbox_pred in refine_bbox_preds
        ]
        # flatten_refine_noco_preds = [
        #     noco_pred.permute(0, 2, 3, 1).reshape(-1)
        #     for noco_pred in refine_noco_preds
        # ]
        flatten_refine_scores = torch.cat(flatten_refine_scores)
        flatten_refine_bbox_preds = torch.cat(flatten_refine_bbox_preds)
        # flatten_refine_noco_preds = torch.cat(flatten_refine_noco_preds)
        refine_noco_preds = torch.cat(refine_noco_preds)

        ###### 2.3 compute loss
        refine_pos_inds = ((flatten_refine_labels >= 0)
                    & (flatten_refine_labels < bg_class_ind)).nonzero().reshape(-1)
        refine_num_pos = torch.tensor(
            len(refine_pos_inds), dtype=torch.float,
            device=refine_bbox_preds[0].device)
        refine_num_pos = max(reduce_mean(refine_num_pos), 1.0)
        ### 2.3.1 cls loss
        loss_refine_cls = self.loss_refine_cls(
            flatten_refine_scores, flatten_refine_labels, avg_factor=refine_num_pos)
        loss_refine_noco = self.loss_refine_noco(
            refine_noco_preds, gt_noco_maps)
        ### 2.3.2 bbox loss
        pos_refine_bbox_preds = flatten_refine_bbox_preds[refine_pos_inds]
        # pos_refine_noco_preds = flatten_refine_noco_preds[refine_pos_inds]
        pos_refine_bbox_targets = flatten_refine_bbox_targets[refine_pos_inds]
        # pos_refine_noco_targets = flatten_refine_noco_targets[refine_pos_inds]
        if len(refine_pos_inds) > 0:
            pos_anchors = flatten_anchors[refine_pos_inds]
            pos_decoded_bbox_preds = pos_refine_bbox_preds * self.refine_stride + pos_anchors
            pos_decoded_bbox_targets = pos_refine_bbox_targets * self.refine_stride + pos_anchors
            loss_refine_bbox = self.loss_refine_bbox(
                pos_decoded_bbox_preds,
                pos_decoded_bbox_targets)
            # loss_refine_noco = self.loss_refine_noco(
            #     pos_refine_noco_preds, pos_refine_noco_targets, avg_factor=refine_num_pos)
        else:
            loss_refine_bbox = pos_refine_bbox_preds.sum()
            # loss_refine_noco = pos_refine_noco_preds.sum()

        return dict(
            # coarse layer
            loss_coarse_cls=loss_coarse_cls,
            loss_coarse_bbox=loss_coarse_bbox,
            # refine layer
            loss_refine_cls=loss_refine_cls,
            loss_refine_bbox=loss_refine_bbox,
            loss_refine_noco=loss_refine_noco)

    def get_coarse_targets(self, coarse_points, gt_bboxes_list, gt_labels_list):
        """Compute regression and classification targets for points
        in multiple images.

        Args:
            # points (list[Tensor]): Points of each fpn level, each has shape
            #     (num_points, 2).
            cls_point (Tensor): Points of cls fpn level with shape
                (num_points, 2).
            gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image,
                each has shape (num_gt, 4).
            gt_labels_list (list[Tensor]): Ground truth labels of each box,
                each has shape (num_gt,).

        Returns:
            tuple:
                concat_lvl_labels (list[Tensor]): Labels of each level. \
                concat_lvl_bbox_targets (list[Tensor]): BBox targets of each \
                    level.
        """
        num_levels = 1
        concat_points = torch.cat(coarse_points, dim=0)
        # get labels and bbox_targets of each image
        labels_list, bbox_targets_list = multi_apply(
            self._get_coarse_target_single,
            gt_bboxes_list,
            gt_labels_list,
            points=concat_points)

        # split to per img, per level
        labels_list = [[labels] for labels in labels_list]
        bbox_targets_list = [
            [bbox_targets] for bbox_targets in bbox_targets_list]

        # concat per level image
        concat_lvl_labels = []
        concat_lvl_bbox_targets = []
        for i in range(num_levels):
            concat_lvl_labels.append(
                torch.cat([labels[i] for labels in labels_list]))
            bbox_targets = torch.cat(
                [bbox_targets[i] for bbox_targets in bbox_targets_list])
            if self.norm_on_bbox:
                bbox_targets = bbox_targets / self.coarse_stride
            concat_lvl_bbox_targets.append(bbox_targets)

        return concat_lvl_labels, concat_lvl_bbox_targets

    def _get_coarse_target_single(self, gt_bboxes, gt_labels, points):
        """Compute classification targets for a single image."""

        num_points = points.size(0)
        num_gts = gt_labels.size(0)

        if num_gts == 0:
            return gt_labels.new_full((num_points,), self.num_classes), \
                gt_bboxes.new_zeros((num_points, 4))

        areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * (
            gt_bboxes[:, 3] - gt_bboxes[:, 1])

        # generate pseudo_gt_bboxes
        pseudo_gt_bboxes = gt_bboxes.clone()
        pseudo_center_xs = (gt_bboxes[:, 0] + gt_bboxes[:, 2]) / 2
        pseudo_center_ys = (gt_bboxes[:, 1] + gt_bboxes[:, 3]) / 2
        pseudo_gt_bboxes[:, 0] = pseudo_center_xs - self.coarse_stride/2
        pseudo_gt_bboxes[:, 2] = pseudo_center_xs + self.coarse_stride/2
        pseudo_gt_bboxes[:, 1] = pseudo_center_ys - self.coarse_stride/2
        pseudo_gt_bboxes[:, 3] = pseudo_center_ys + self.coarse_stride/2
        area_thresh = self.coarse_stride**2
        area_mask = areas > area_thresh
        # 面积不足的, 用 pseduo bbox; 面积够大的, 保留原 bbox
        pseudo_gt_bboxes[area_mask] = gt_bboxes[area_mask]

        # generate bbox_targets
        areas = areas[None].repeat(num_points, 1)
        gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 4)
        pseudo_gt_bboxes = pseudo_gt_bboxes[None].expand(num_points, num_gts, 4)
        xs, ys = points[:, 0], points[:, 1]
        xs = xs[:, None].expand(num_points, num_gts)
        ys = ys[:, None].expand(num_points, num_gts)

        pseudo_left = xs - pseudo_gt_bboxes[..., 0]
        pseudo_right = pseudo_gt_bboxes[..., 2] - xs
        pseudo_top = ys - pseudo_gt_bboxes[..., 1]
        pseudo_bottom = pseudo_gt_bboxes[..., 3] - ys
        pseudo_bbox_targets = torch.stack((
            pseudo_left, pseudo_top, pseudo_right, pseudo_bottom), -1)

        left = xs - gt_bboxes[..., 0]
        right = gt_bboxes[..., 2] - xs
        top = ys - gt_bboxes[..., 1]
        bottom = gt_bboxes[..., 3] - ys
        bbox_targets = torch.stack((left, top, right, bottom), -1)

        # condition: inside a gt bbox
        inside_gt_bbox_mask = (pseudo_bbox_targets.min(-1)[0] > 0).float()

        # if there are still more than one objects for a location,
        # we choose the one with minimal area
        areas[inside_gt_bbox_mask == 0] = INF
        min_area, min_area_inds = areas.min(dim=1)

        labels = gt_labels[min_area_inds]
        labels[min_area == INF] = self.num_classes  # set as BG
        # bbox_targets = bbox_targets[range(num_points), min_area_inds]
        bbox_targets = pseudo_bbox_targets[range(num_points), min_area_inds]

        return labels, bbox_targets

    def get_refine_targets(self, anchors, refine_points, gt_bboxes_list,
                           gt_labels_list, gt_noco_maps):
        """Compute regression and classification targets for points
        in multiple images.

        Args:
            points (list[Tensor]): Points of each fpn level, each has shape
                (num_points, 2).
            gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image,
                each has shape (num_gt, 4).
            gt_labels_list (list[Tensor]): Ground truth labels of each box,
                each has shape (num_gt,).

        Returns:
            tuple:
                concat_lvl_labels (list[Tensor]): Labels of each level. \
                concat_lvl_bbox_targets (list[Tensor]): BBox targets of each \
                    level.
        """
        num_levels = 1
        concat_points = torch.cat(refine_points, dim=0)

        gt_noco_map_list = [gt_noco_maps[i].reshape(-1) for i in range(len(gt_noco_maps))]
        # for _ in range(10):
        #     print("len(gt_noco_map_list):", len(gt_noco_map_list))
        #     print('gt_noco_map_list[0].shape:', gt_noco_map_list[0].shape)

        # get labels and bbox_targets of each image
        refine_labels_list, refine_bbox_targets_list, refine_noco_targets_list = multi_apply(
            self._get_refine_target_single,
            anchors,
            gt_bboxes_list,
            gt_labels_list,
            gt_noco_map_list,
            points=concat_points)

        # split to per img, per level
        refine_labels_list = [[labels] for labels in refine_labels_list]
        refine_bbox_targets_list = [
            [bbox_targets]
            for bbox_targets in refine_bbox_targets_list
        ]
        refine_noco_targets_list = [
            [noco_targets]
            for noco_targets in refine_noco_targets_list
        ]

        # concat per level image
        concat_lvl_labels = []
        concat_lvl_bbox_targets = []
        concat_lvl_noco_targets = []
        for i in range(num_levels):
            concat_lvl_labels.append(
                torch.cat([labels[i] for labels in refine_labels_list]))
            bbox_targets = torch.cat(
                [bbox_targets[i] for bbox_targets in refine_bbox_targets_list])
            noco_targets = torch.cat(
                [noco_targets[i] for noco_targets in refine_noco_targets_list])
            # if self.norm_on_bbox:
            #     bbox_targets = bbox_targets / self.refine_stride
            concat_lvl_bbox_targets.append(bbox_targets)
            concat_lvl_noco_targets.append(noco_targets)
        return concat_lvl_labels, concat_lvl_bbox_targets, concat_lvl_noco_targets

    def _get_refine_target_single(self, anchors, gt_bboxes, gt_labels, gt_noco_maps, points):
        """Compute regression and classification targets for a single image."""
        num_points = points.size(0)
        num_gts = gt_labels.size(0)
        if num_gts == 0:
            return gt_labels.new_full((num_points,), self.num_classes), \
                   gt_bboxes.new_zeros((num_points, 4)), \
                   gt_bboxes.new_zeros((num_points,)),

        ############################ 1 cls label ############################
        areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * (
            gt_bboxes[:, 3] - gt_bboxes[:, 1])

        pseudo_gt_bboxes = gt_bboxes.clone()
        pseudo_center_xs = (gt_bboxes[:, 0] + gt_bboxes[:, 2]) / 2
        pseudo_center_ys = (gt_bboxes[:, 1] + gt_bboxes[:, 3]) / 2
        pseudo_gt_bboxes[:, 0] = pseudo_center_xs - self.refine_stride/2 * self.stride_ratio
        pseudo_gt_bboxes[:, 2] = pseudo_center_xs + self.refine_stride/2 * self.stride_ratio
        pseudo_gt_bboxes[:, 1] = pseudo_center_ys - self.refine_stride/2 * self.stride_ratio
        pseudo_gt_bboxes[:, 3] = pseudo_center_ys + self.refine_stride/2 * self.stride_ratio
        area_thresh = (self.refine_stride*self.stride_ratio)**2
        area_mask = areas > area_thresh
        # 面积不足的, 用 pseduo bbox; 面积够大的, 保留原 bbox
        pseudo_gt_bboxes[area_mask] = gt_bboxes[area_mask]

        # TODO: figure out why these two are different
        # areas = areas[None].expand(num_points, num_gts)
        areas = areas[None].repeat(num_points, 1)
        pseudo_gt_bboxes = pseudo_gt_bboxes[None].expand(num_points, num_gts, 4)
        xs, ys = points[:, 0], points[:, 1]
        xs = xs[:, None].expand(num_points, num_gts)
        ys = ys[:, None].expand(num_points, num_gts)

        pseudo_left = xs - pseudo_gt_bboxes[..., 0]
        pseudo_right = pseudo_gt_bboxes[..., 2] - xs
        pseudo_top = ys - pseudo_gt_bboxes[..., 1]
        pseudo_bottom = pseudo_gt_bboxes[..., 3] - ys
        pseudo_bbox_targets = torch.stack((
            pseudo_left, pseudo_top, pseudo_right, pseudo_bottom), -1)

        # condition: inside a gt bbox
        inside_gt_bbox_mask = (pseudo_bbox_targets.min(-1)[0] > 0).float()
        # if there are still more than one objects for a location,
        # we choose the one with minimal area
        areas[inside_gt_bbox_mask == 0] = INF
        min_area, min_area_inds = areas.min(dim=1)
        labels = gt_labels[min_area_inds]
        labels[min_area == INF] = self.num_classes  # set as BG

        ############################ 2 bbox target ############################
        # gt_bboxes shape: (num_points, num_gts, 4)
        gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 4)
        x1s = anchors[:, 0]
        x1s = x1s[:, None].expand(num_points, num_gts) # shape: (num_points, num_gts)
        y1s = anchors[:, 1]
        y1s = y1s[:, None].expand(num_points, num_gts)
        x2s = anchors[:, 2]
        x2s = x2s[:, None].expand(num_points, num_gts)
        y2s = anchors[:, 3]
        y2s = y2s[:, None].expand(num_points, num_gts)

        ##################### encode bbox begins #####################
        # dx1 = gt_bboxes[..., 0] - x1s
        # dy1 = gt_bboxes[..., 1] - y1s
        # dx2 = gt_bboxes[..., 2] - x2s
        # dy2 = gt_bboxes[..., 3] - y2s
        dx1 = (gt_bboxes[..., 0] - x1s) / self.refine_stride
        dy1 = (gt_bboxes[..., 1] - y1s) / self.refine_stride
        dx2 = (gt_bboxes[..., 2] - x2s) / self.refine_stride
        dy2 = (gt_bboxes[..., 3] - y2s) / self.refine_stride
        bbox_targets = torch.stack((dx1, dy1, dx2, dy2), -1)

        # px = (x1s + x2s) * 0.5
        # py = (y1s + y2s) * 0.5
        # pw = x2s - x1s
        # ph = y2s - y1s

        # gx = (gt_bboxes[..., 0] + gt_bboxes[..., 2]) * 0.5
        # gy = (gt_bboxes[..., 1] + gt_bboxes[..., 3]) * 0.5
        # gw = gt_bboxes[..., 2] - gt_bboxes[..., 0]
        # gh = gt_bboxes[..., 3] - gt_bboxes[..., 1]

        # dx = (gx - px) / pw
        # dy = (gy - py) / ph
        # dw = torch.log(gw / pw)
        # dh = torch.log(gh / ph)

        # bbox_targets = torch.stack((dx, dy, dw, dh), -1)
        ##################### encode bbox ends #####################
        bbox_targets = bbox_targets[range(num_points), min_area_inds]

        noco_targets = gt_noco_maps
        return labels, bbox_targets, noco_targets

    @force_fp32(apply_to=('refine_scores', 'refine_bbox_preds', 'refine_noco_preds',
                          'coarse_scores', 'coarse_bbox_preds'))
    def get_bboxes(self,
                   refine_scores,
                   refine_bbox_preds,
                   refine_noco_preds,
                   coarse_scores,
                   coarse_bbox_preds,
                #    score_factors=None,
                   img_metas=None,
                   cfg=None,
                   rescale=False,
                   with_nms=True,
                   **kwargs):
        """Transform network outputs of a batch into bbox results.

        Note: When score_factors is not None, the cls_scores are
        usually multiplied by it then obtain the real score used in NMS,
        such as CenterNess in FCOS, IoU branch in ATSS.

        Args:
            cls_scores (list[Tensor]): Classification scores for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * 4, H, W).
            score_factors (list[Tensor], Optional): Score factor for
                all scale level, each is a 4D-tensor, has shape
                (batch_size, num_priors * 1, H, W). Default None.
            img_metas (list[dict], Optional): Image meta info. Default None.
            cfg (mmcv.Config, Optional): Test / postprocessing configuration,
                if None, test_cfg would be used.  Default None.
            rescale (bool): If True, return boxes in original image space.
                Default False.
            with_nms (bool): If True, do nms before return boxes.
                Default True.

        Returns:
            list[list[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (n, 5) tensor, where the first 4 columns
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the
                5-th column is a score between 0 and 1. The second item is a
                (n,) tensor where each item is the predicted class label of
                the corresponding box.
        """
        # assert len(cls_scores) == len(bbox_preds)
        assert len(refine_scores) == len(refine_bbox_preds) ==\
            len(coarse_scores) == len(coarse_bbox_preds)

        with_score_factors = True

        num_levels = len(refine_scores)

        ############################ 2 pred align ############################
        batch_size = coarse_bbox_preds[0].size(0)
        num_imgs = coarse_bbox_preds[0].size(0)
        refine_featmap_size = refine_bbox_preds[0].size()[-2:]
        ### 2.1 将 (B, 4, H, W) 的 bbox pred 转化为 (B*H*W, 4)
        # repeat points to align with bbox_preds
        coarse_featmap_size = coarse_scores[0].size()[-2:]
        coarse_hei, coarse_wid = coarse_featmap_size
        all_level_coarse_points = self.coarse_prior_generator.grid_priors(
            [coarse_featmap_size],
            dtype=coarse_scores[0].dtype,
            device=coarse_scores[0].device)
        flatten_coarse_points = torch.cat(
            [points.repeat(num_imgs, 1) for points in all_level_coarse_points])
        flatten_coarse_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
            for bbox_pred in coarse_bbox_preds
        ]
        flatten_coarse_bbox_preds = torch.cat(flatten_coarse_bbox_preds)
        ### 2.2 先解码得到 bbox 在原图上的坐标
        # flatten_coarse_bbox_preds *= self.coarse_stride
        flatten_decoded_coarse_bbox_preds = self.bbox_coarse_coder.decode(
            flatten_coarse_points, flatten_coarse_bbox_preds)
        ### 2.3 对 bbox 恢复成 (B, 4, H, W) 再插值
        decoded_coarse_bbox_preds = flatten_decoded_coarse_bbox_preds.reshape(
            batch_size, coarse_hei, coarse_wid, 4).permute(0, 3, 1, 2)
        anchors = [F.interpolate(decoded_coarse_bbox_preds,
                                 size=refine_featmap_size,
                                 mode='nearest-exact')]

        refine_featmap_sizes = [refine_bbox_preds[i].shape[-2:] for i in range(num_levels)]
        # score_factors = [
        #     F.interpolate(coarse_scores[i], size=refine_featmap_sizes[i],
        #                   mode='nearest-exact') for i in range(num_levels)]
        # score_factors = refine_noco_preds
        # score_factors = [
        #     F.interpolate(coarse_scores[i], size=refine_featmap_sizes[i],
        #                   mode='nearest-exact') * refine_noco_preds[i] for i in range(num_levels)]
        # score_factors = [
        #     F.interpolate(coarse_scores[i], size=refine_featmap_sizes[i],
        #                   mode='nearest-exact').sigmoid() for i in range(num_levels)]
        score_factors_1 = [
            F.interpolate(coarse_scores[i], size=refine_featmap_sizes[i],
                          mode='nearest-exact') for i in range(num_levels)]
        score_factors_2 = refine_noco_preds
        score_factors = [
            score_factors_1[i].sigmoid() * score_factors_2[i].sigmoid()
            for i in range(num_levels)]


        result_list = []

        for img_id in range(len(img_metas)):
            img_meta = img_metas[img_id]
            cls_score_list = select_single_mlvl(refine_scores, img_id)
            bbox_pred_list = select_single_mlvl(refine_bbox_preds, img_id)
            # bbox_pred_list = select_single_mlvl(anchors, img_id)
            anchor_list = select_single_mlvl(anchors, img_id)
            if with_score_factors:
                score_factor_list = select_single_mlvl(score_factors, img_id)
            else:
                score_factor_list = [None for _ in range(num_levels)]

            results = self._get_bboxes_single(cls_score_list, bbox_pred_list,
                                              score_factor_list, anchor_list,
                                              img_meta, cfg, rescale, with_nms,
                                              **kwargs)
            result_list.append(results)
        return result_list

    def _get_bboxes_single(self,
                           cls_score_list,
                           bbox_pred_list,
                           score_factor_list,
                           anchor_list,
                           img_meta,
                           cfg,
                           rescale=False,
                           with_nms=True,
                           **kwargs):
        """Transform outputs of a single image into bbox predictions.

        Args:
            cls_score_list (list[Tensor]): Box scores from all scale
                levels of a single image, each item has shape
                (num_priors * num_classes, H, W).
            bbox_pred_list (list[Tensor]): Box energies / deltas from
                all scale levels of a single image, each item has shape
                (num_priors * 4, H, W).
            score_factor_list (list[Tensor]): Score factor from all scale
                levels of a single image, each item has shape
                (num_priors * 1, H, W).
            mlvl_priors (list[Tensor]): Each element in the list is
                the priors of a single level in feature pyramid. In all
                anchor-based methods, it has shape (num_priors, 4). In
                all anchor-free methods, it has shape (num_priors, 2)
                when `with_stride=True`, otherwise it still has shape
                (num_priors, 4).
            img_meta (dict): Image meta info.
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.

        Returns:
            tuple[Tensor]: Results of detected bboxes and labels. If with_nms
                is False and mlvl_score_factor is None, return mlvl_bboxes and
                mlvl_scores, else return mlvl_bboxes, mlvl_scores and
                mlvl_score_factor. Usually with_nms is False is used for aug
                test. If with_nms is True, then return the following format

                - det_bboxes (Tensor): Predicted bboxes with shape \
                    [num_bboxes, 5], where the first 4 columns are bounding \
                    box positions (tl_x, tl_y, br_x, br_y) and the 5-th \
                    column are scores between 0 and 1.
                - det_labels (Tensor): Predicted labels of the corresponding \
                    box with shape [num_bboxes].
        """
        if score_factor_list[0] is None:
            # e.g. Retina, FreeAnchor, etc.
            with_score_factors = False
        else:
            # e.g. FCOS, PAA, ATSS, etc.
            with_score_factors = True

        cfg = self.test_cfg if cfg is None else cfg
        img_shape = img_meta['img_shape'] # (600, 700, 3)
        # for _ in range(5):
        #     print("img_shape:", img_shape)
        nms_pre = cfg.get('nms_pre', -1)

        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_labels = []
        if with_score_factors:
            mlvl_score_factors = []
        else:
            mlvl_score_factors = None
        for level_idx, (cls_score, bbox_pred, score_factor, anchor) in \
                enumerate(zip(cls_score_list, bbox_pred_list,
                              score_factor_list, anchor_list)):

            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]

            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            anchor = anchor.permute(1, 2, 0).reshape(-1, 4)
            if with_score_factors:
                # score_factor = score_factor.permute(1, 2,
                #                                     0).reshape(-1).sigmoid()
                score_factor = score_factor.permute(1, 2, 0).reshape(-1)
            cls_score = cls_score.permute(1, 2,
                                          0).reshape(-1, self.cls_out_channels)
            if self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                # remind that we set FG labels to [0, num_class-1]
                # since mmdet v2.0
                # BG cat_id: num_class
                scores = cls_score.softmax(-1)[:, :-1]

            # After https://github.com/open-mmlab/mmdetection/pull/6268/,
            # this operation keeps fewer bboxes under the same `nms_pre`.
            # There is no difference in performance for most models. If you
            # find a slight drop in performance, you can set a larger
            # `nms_pre` than before.
            results = filter_scores_and_topk(
                scores, cfg.score_thr, nms_pre,
                dict(bbox_pred=bbox_pred, priors=anchor))
            scores, labels, keep_idxs, filtered_results = results

            bbox_pred = filtered_results['bbox_pred']
            anchor = filtered_results['priors']

            if with_score_factors:
                score_factor = score_factor[keep_idxs]

            ################# coarse layer output begins ##################
            # bboxes = bbox_pred
            ################# coarse layer output ends ##################

            ################# coarse layer output begins ##################
            bboxes = anchor + bbox_pred * self.refine_stride
            bboxes[:, 0::2].clamp_(min=0, max=img_shape[1])
            bboxes[:, 1::2].clamp_(min=0, max=img_shape[0])
            ################# coarse layer output ends ##################

            ################# DeltaXYWHBBoxCoder begins ##################
            # bboxes = self.bbox_refine_coder.decode(anchor, bbox_pred,
            #                                        max_shape=img_shape)
            # print("\nanchor:", anchor)
            # print("\nbboxes:", bboxes)
            ################# DeltaXYWHBBoxCoder ends ##################

            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_labels.append(labels)
            if with_score_factors:
                mlvl_score_factors.append(score_factor)

        return self._bbox_post_process(mlvl_scores, mlvl_labels, mlvl_bboxes,
                                       img_meta['scale_factor'], cfg, rescale,
                                       with_nms, mlvl_score_factors, **kwargs)

    def get_targets(self, points, gt_bboxes_list, gt_labels_list):
        """Compute regression, classification and centerness targets for points
        in multiple images.

        Args:
            points (list[Tensor]): Points of each fpn level, each has shape
                (num_points, 2).
            gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image,
                each has shape (num_gt, 4).
            gt_labels_list (list[Tensor]): Ground truth labels of each box,
                each has shape (num_gt,).
        """
        raise NotImplementedError