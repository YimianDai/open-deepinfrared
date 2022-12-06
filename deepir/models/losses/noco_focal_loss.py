import torch
import torch.nn as nn
import torch.nn.functional as F

import mmcv
from mmseg.models.builder import LOSSES as SEGLOSSES
from mmdet.models.builder import LOSSES as DETLOSSES
from mmseg.models.losses import weighted_loss


@mmcv.jit(derivate=True, coderize=True)
@weighted_loss
def noco_focal_loss(pred, target, alpha=2.0, beta=2.0):
    """NoCo Focal Loss (QFL) is from TODO: paper title and url.

    Args:
        pred (torch.Tensor): Predicted normalized contrast maps with shape
            (N, C, H, W), C is the number of classes.
        target (tuple([torch.Tensor])): Ground truth normalized contrast
            maps with shape (N, H, W).
        beta (float): The beta parameter for calculating the modulating factor.
            Defaults to 2.0.

    Returns:
        torch.Tensor: Loss tensor with shape (N,).
    """

    # negatives are supervised by 0 quality score
    pred_sigmoid = pred.sigmoid()
    noco_weight = torch.exp(alpha * target)
    focal_weight = (pred_sigmoid - target).abs().pow(beta)
    loss = F.binary_cross_entropy_with_logits(
        pred, target, reduction='none') * noco_weight * focal_weight
    loss = loss.sum(dim=1, keepdim=False)
    return loss

@mmcv.jit(derivate=True, coderize=True)
@weighted_loss
def reg_quality_focal_loss(pred, target, beta=2.0):
    """NoCo Focal Loss (QFL) is from TODO: paper title and url.

    Args:
        pred (torch.Tensor): Predicted normalized contrast maps with shape
            (N, C, H, W), C is the number of classes.
        target (tuple([torch.Tensor])): Ground truth normalized contrast
            maps with shape (N, H, W).
        beta (float): The beta parameter for calculating the modulating factor.
            Defaults to 2.0.

    Returns:
        torch.Tensor: Loss tensor with shape (N,).
    """

    # negatives are supervised by 0 quality score
    pred_sigmoid = pred.sigmoid()
    focal_weight = (pred_sigmoid - target).abs().pow(beta)
    loss = F.binary_cross_entropy_with_logits(
        pred, target, reduction='none') * focal_weight
    loss = loss.sum(dim=1, keepdim=False)
    return loss

@SEGLOSSES.register_module()
@DETLOSSES.register_module()
class NoCoFocalLoss(nn.Module):
    """NoCo Focal Loss (NoCoFL) is a variant of TODO: paper title and url.

    Args:
        use_sigmoid (bool): Whether sigmoid operation is conducted in QFL.
            Defaults to True.
        beta (float): The beta parameter for calculating the modulating factor.
            Defaults to 2.0.
        reduction (str): Options are "none", "mean" and "sum".
        loss_weight (float): Loss weight of current loss.
    """

    def __init__(self,
                 use_sigmoid=True,
                 alpha=5.0,
                 beta=5.0,
                 reduction='mean',
                 loss_weight=1.0,
                 loss_name='loss_noco_focal'):
        super(NoCoFocalLoss, self).__init__()
        assert use_sigmoid is True, 'Only sigmoid in NoCoFL supported now.'
        self.use_sigmoid = use_sigmoid
        self.alpha = alpha
        self.beta = beta
        self.reduction = reduction
        self.loss_weight = loss_weight
        self._loss_name = loss_name

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                ignore_index=255,
                **kwargs):
        """Forward function.

        Args:
            pred (torch.Tensor): Predicted normalized contrast maps with shape
                (N, C, H, W), C is the number of classes.
            target (tuple([torch.Tensor])): Ground truth normalized contrast
                maps with shape (N, H, W).
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        target = target.float()
        assert isinstance(ignore_index, int), \
            'ignore_index must be of type int'
        assert reduction_override in (None, 'none', 'mean', 'sum'), \
            "AssertionError: reduction should be 'none', 'mean' or " \
            "'sum'"
        assert pred.shape == target.shape or \
               (pred.size(0) == target.size(0) and
                pred.shape[2:] == target.shape[1:]), \
               "The shape of pred doesn't match the shape of target"

        reduction = (
            reduction_override if reduction_override else self.reduction)
        if target.ndim == 3:
            target = torch.unsqueeze(target, 1) # (N, H, W) -> (N, 1, H, W)
        if self.use_sigmoid:
            loss_noco = self.loss_weight * noco_focal_loss(
                pred,
                target,
                weight,
                alpha=self.alpha,
                beta=self.beta,
                reduction=reduction,
                avg_factor=avg_factor)
        else:
            raise NotImplementedError
        return loss_noco

    @property
    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.
        Returns:
            str: The name of this loss item.
        """
        return self._loss_name

@SEGLOSSES.register_module()
@DETLOSSES.register_module()
class RegQualityFocalLoss(nn.Module):
    """NoCo Focal Loss (NoCoFL) is a variant of TODO: paper title and url.

    Args:
        use_sigmoid (bool): Whether sigmoid operation is conducted in QFL.
            Defaults to True.
        beta (float): The beta parameter for calculating the modulating factor.
            Defaults to 2.0.
        reduction (str): Options are "none", "mean" and "sum".
        loss_weight (float): Loss weight of current loss.
    """

    def __init__(self,
                 use_sigmoid=True,
                 beta=5.0,
                 reduction='mean',
                 loss_weight=1.0,
                 loss_name='loss_reg_quality_focal'):
        super(RegQualityFocalLoss, self).__init__()
        assert use_sigmoid is True, 'Only sigmoid in NoCoFL supported now.'
        self.use_sigmoid = use_sigmoid
        self.beta = beta
        self.reduction = reduction
        self.loss_weight = loss_weight
        self._loss_name = loss_name

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                ignore_index=255,
                **kwargs):
        """Forward function.

        Args:
            pred (torch.Tensor): Predicted normalized contrast maps with shape
                (N, C, H, W), C is the number of classes.
            target (tuple([torch.Tensor])): Ground truth normalized contrast
                maps with shape (N, H, W).
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        target = target.float()
        assert isinstance(ignore_index, int), \
            'ignore_index must be of type int'
        assert reduction_override in (None, 'none', 'mean', 'sum'), \
            "AssertionError: reduction should be 'none', 'mean' or " \
            "'sum'"
        assert pred.shape == target.shape or \
               (pred.size(0) == target.size(0) and
                pred.shape[2:] == target.shape[1:]), \
               "The shape of pred doesn't match the shape of target"

        reduction = (
            reduction_override if reduction_override else self.reduction)
        if target.ndim == 3:
            target = torch.unsqueeze(target, 1) # (N, H, W) -> (N, 1, H, W)
        if self.use_sigmoid:
            loss_noco = self.loss_weight * reg_quality_focal_loss(
                pred,
                target,
                weight,
                beta=self.beta,
                reduction=reduction,
                avg_factor=avg_factor)
        else:
            raise NotImplementedError
        return loss_noco

    @property
    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.
        Returns:
            str: The name of this loss item.
        """
        return self._loss_name