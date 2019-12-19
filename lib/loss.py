# pylint: disable=arguments-differ
"""Custom losses.
Losses are subclasses of gluon.loss.Loss which is a HybridBlock actually.
"""
import os
import sys
sys.path.insert(0, os.path.expanduser('~/incubator-mxnet/python'))
from mxnet import gluon
from mxnet import nd
from mxnet.gluon.loss import Loss, _apply_weighting, _reshape_like

__all__ = ['FocalLoss', 'SSDMultiBoxLoss', 'HuberLoss']


def _as_list(arr):
    """Make sure input is a list of mxnet NDArray"""
    if not isinstance(arr, (list, tuple)):
        return [arr]
    return arr


class SSDMultiBoxLoss(gluon.Block):
    r"""Single-Shot Multibox Object Detection Loss.

    .. note::

        Since cross device synchronization is required to compute batch-wise statistics,
        it is slightly sub-optimal compared with non-sync version. However, we find this
        is better for converged model performance.

    Parameters
    ----------
    negative_mining_ratio : float, default is 3
        Ratio of negative vs. positive samples.
    rho : float, default is 1.0
        Threshold for trimmed mean estimator. This is the smooth parameter for the
        L1-L2 transition.
    lambd : float, default is 1.0
        Relative weight between classification and box regression loss.
        The overall loss is computed as :math:`L = loss_{class} + \lambda \times loss_{loc}`.
    min_hard_negatives : int, default is 0
        Minimum number of negatives samples.

    """
    def __init__(self, negative_mining_ratio=3, rho=1.0, lambd=1.0,
                 min_hard_negatives=0, **kwargs):
        super(SSDMultiBoxLoss, self).__init__(**kwargs)
        self._negative_mining_ratio = max(0, negative_mining_ratio)
        self._rho = rho
        self._lambd = lambd
        self._min_hard_negatives = max(0, min_hard_negatives)

    def forward(self, cls_pred, box_pred, cls_target, box_target):
        """Compute loss in entire batch across devices."""
        # require results across different devices at this time
        cls_pred, box_pred, cls_target, box_target = [_as_list(x) \
            for x in (cls_pred, box_pred, cls_target, box_target)]
        # cross device reduction to obtain positive samples in entire batch
        pos_ct = [ct > 0 for ct in cls_target]
        num_pos = [ct.sum() for ct in pos_ct]
        num_pos_all = sum([p.asscalar() for p in num_pos])
        # print ('num_pos_all: {}'.format(num_pos_all))
        if num_pos_all < 1 and self._min_hard_negatives < 1:
            # no positive samples and no hard negatives, return dummy losses
            cls_losses = [nd.sum(cp * 0) for cp in cls_pred]
            box_losses = [nd.sum(bp * 0) for bp in box_pred]
            sum_losses = [nd.sum(cp * 0) + nd.sum(bp * 0) for cp, bp in zip(cls_pred, box_pred)]
            return sum_losses, cls_losses, box_losses

        # compute element-wise cross entropy loss and sort, then perform negative mining
        cls_losses = []
        box_losses = []
        sum_losses = []
        for cp, bp, ct, bt in zip(*[cls_pred, box_pred, cls_target, box_target]):
            # print ('cp shape: {}'.format(cp.shape))
            # print ('bp shape: {}'.format(bp.shape))
            # print ('ct shape: {}'.format(ct.shape))
            # print ('bt shape: {}'.format(bt.shape))
            pred = nd.log_softmax(cp, axis=-1)
            pos = ct > 0
            cls_loss = -nd.pick(pred, ct, axis=-1, keepdims=False)
            rank = (cls_loss * (pos - 1)).argsort(axis=1).argsort(axis=1)
            hard_negative = rank < nd.maximum(self._min_hard_negatives, pos.sum(axis=1)
                                              * self._negative_mining_ratio).expand_dims(-1)
            # mask out if not positive or negative
            cls_loss = nd.where((pos + hard_negative) > 0, cls_loss, nd.zeros_like(cls_loss))
            cls_losses.append(nd.sum(cls_loss, axis=0, exclude=True) / max(1., num_pos_all))

            bp = _reshape_like(nd, bp, bt)
            box_loss = nd.abs(bp - bt)
            box_loss = nd.where(box_loss > self._rho, box_loss - 0.5 * self._rho,
                                (0.5 / self._rho) * nd.square(box_loss))
            # box loss only apply to positive samples
            box_loss = box_loss * pos.expand_dims(axis=-1)
            box_losses.append(nd.sum(box_loss, axis=0, exclude=True) / max(1., num_pos_all))
            sum_losses.append(cls_losses[-1] + self._lambd * box_losses[-1])

        return sum_losses, cls_losses, box_losses


class FocalLoss(Loss):
    def __init__(self, num_class, alpha=0.25, gamma=2.0,
                 weight=None, batch_axis=0, **kwargs):
        super(FocalLoss, self).__init__(weight, batch_axis, **kwargs)
        self._num_class = num_class
        self._alpha = alpha
        self._gamma = gamma

    def hybrid_forward(self, F, pred, label, mask):
        pred = F.clip(pred, 1e-4, 1-1e-4)
        
        cls_mask = F.expand_dims(mask, axis=-1)
        cls_mask = F.tile(cls_mask, reps=(1, 1, self._num_class))
        cls_mask = F.where(cls_mask != -1, F.ones_like(cls_mask), F.zeros_like(cls_mask))
        
        # num of positive samples
        num_pos = F.sum(mask>0, axis=0, exclude=True)
        num_pos = F.where(num_pos>0, num_pos, F.ones_like(num_pos))
        
        # convert 0 to -1 for one_hot encoding
        label = F.where(label == 0, F.ones_like(label)*-1, label)
        # encode coco class to be [0, 79] instead of [1, 80] for one-hot
        label = F.where(label != -1, label-1, label)
        label = F.one_hot(label, self._num_class)

        alpha_factor = F.ones_like(label) * self._alpha
        alpha_factor = F.where(label, alpha_factor, 1.-alpha_factor)
            
        focal_weight = F.where(label, 1.-pred, pred)
        focal_weight = alpha_factor * F.power(focal_weight, self._gamma)

        bce = -(label * F.log(pred) + (1.0-label) * F.log(1.0-pred))

        loss = focal_weight * bce
        loss = loss*cls_mask

        loss = F.sum(loss, axis=0, exclude=True)/num_pos

        return loss


class HuberLoss(Loss):
    def __init__(self, rho=1.0/9.0, weight=None, batch_axis=0, **kwargs):
        super(HuberLoss, self).__init__(weight, batch_axis, **kwargs)
        self._rho = rho

    def hybrid_forward(self, F, pred, label, mask):
        # num of positive samples
        num_pos = F.sum(mask>0, axis=0, exclude=True)
        num_pos = F.where(num_pos>0, num_pos, F.ones_like(num_pos))
        
        box_mask = F.expand_dims(mask, axis=-1)
        box_mask = F.tile(box_mask, reps=(1, 1, 4))
        box_mask = F.where(box_mask==1, box_mask, F.zeros_like(box_mask))
        # print ('nd sum box_mask: {}'.format(nd.sum(box_mask)))
        
        loss = F.abs(label - pred)
        loss = F.where(loss > self._rho, loss - 0.5 * self._rho, (0.5 / self._rho) * F.square(loss))
        loss = loss*box_mask
        
        loss = F.sum(loss, axis=0, exclude=True)/num_pos

        return loss


class HeatmapFocalLoss(Loss):
    """Focal loss for heatmaps.

    Parameters
    ----------
    from_logits : bool
        Whether predictions are after sigmoid or softmax.
    batch_axis : int
        Batch axis.
    weight : float
        Loss weight.

    """
    def __init__(self, from_logits=False, batch_axis=0, weight=None, **kwargs):
        super(HeatmapFocalLoss, self).__init__(weight, batch_axis, **kwargs)
        self._from_logits = from_logits

    def hybrid_forward(self, F, pred, label):
        """Loss forward"""
        if not self._from_logits:
            pred = F.sigmoid(pred)
        pos_inds = label == 1
        neg_inds = label < 1
        neg_weights = F.power(1 - label, 4)
        pos_loss = F.log(pred) * F.power(1 - pred, 2) * pos_inds
        neg_loss = F.log(1 - pred) * F.power(pred, 2) * neg_weights * neg_inds

        # normalize
        num_pos = F.clip(F.sum(pos_inds), a_min=1, a_max=1e30)
        pos_loss = F.sum(pos_loss)
        neg_loss = F.sum(neg_loss)
        return -(pos_loss + neg_loss) / num_pos


class MaskedL1Loss(Loss):
    r"""Calculates the mean absolute error between `label` and `pred` with `mask`.

    .. math:: L = \sum_i \vert ({label}_i - {pred}_i) * {mask}_i \vert / \sum_i {mask}_i.

    `label`, `pred` and `mask` can have arbitrary shape as long as they have the same
    number of elements. The final loss is normalized by the number of non-zero elements in mask.

    Parameters
    ----------
    weight : float or None
        Global scalar weight for loss.
    batch_axis : int, default 0
        The axis that represents mini-batch.


    Inputs:
        - **pred**: prediction tensor with arbitrary shape
        - **label**: target tensor with the same size as pred.
        - **sample_weight**: element-wise weighting tensor. Must be broadcastable
          to the same shape as pred. For example, if pred has shape (64, 10)
          and you want to weigh each sample in the batch separately,
          sample_weight should have shape (64, 1).

    Outputs:
        - **loss**: loss tensor with shape (batch_size,). Dimenions other than
          batch_axis are averaged out.
    """

    def __init__(self, weight=None, batch_axis=0, **kwargs):
        super(MaskedL1Loss, self).__init__(weight, batch_axis, **kwargs)

    def hybrid_forward(self, F, pred, label, mask, sample_weight=None):
        label = _reshape_like(F, label, pred)
        loss = F.abs(label * mask - pred * mask)
        loss = _apply_weighting(F, loss, self._weight, sample_weight)
        norm = F.sum(mask).clip(1, 1e30)
        return F.sum(loss) / norm


if __name__ == '__main__':
    pred_origin = nd.random.uniform(-1, 1, (1, 5, 10))
    pred = nd.sigmoid(pred_origin)
    print ('pred: {}'.format(pred))
    label = nd.array((0, 1, 10, 0, 6)).reshape((1, 5))
    criterion = FocalLoss(num_class=10, debug=True)
    loss = criterion(pred, label)

    import torch
    label = nd.where(label == 0, nd.ones_like(label)*-1, label)
    label = nd.where(label != -1, label-1, label)
    label = nd.one_hot(label, 10)
    label = torch.from_numpy(label.asnumpy()).float()
    pred = torch.from_numpy(pred_origin.asnumpy()).float()
    sys.path.insert(0, os.path.expanduser('~/retinanet-examples/retinanet'))
    from loss import FocalLoss
    py_criterion = FocalLoss()
    loss = py_criterion(pred, label)
    print ('pytorch loss: {}'.format(loss))




