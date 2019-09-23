from __future__ import absolute_import
from mxnet import gluon
from mxnet import nd
from mxnet.gluon.loss import Loss

__all__ = ['FocalLoss', 'HuberLoss']


class FocalLoss(Loss):
    def __init__(self, num_class, alpha=0.25, gamma=2, huber_rho=0.11,
                 eps=1e-12, weight=None, batch_axis=0, **kwargs):
        super(FocalLoss, self).__init__(weight, batch_axis, **kwargs)
        self._num_class = num_class
        self._alpha = alpha
        self._gamma = gamma
        self._huber_rho = huber_rho
        self._eps = eps

    def hybrid_forward(self, F, pred, label):
        # cls loss
        # print ('pred: {}'.format(pred))
        print ('max pred: {}'.format(F.max(pred)))
        print ('min pred: {}'.format(F.min(pred)))
        print ('mean pred: {}'.format(F.mean(pred)))
        # TODO
        # pred = F.clip(pred, 1e-9, 1-1e-9)

        # fake_label = F.where(label>0, F.zeros_like(label), F.ones_like(label)*-1)
        # label = F.one_hot(label, self._num_class)

        # num of positive samples
        # num_pos = F.sum(label>0, axis=-1, keepdims=True)
        # num_pos = F.where(num_pos<=0, F.ones_like(num_pos), num_pos) 
        
        # convert 0 to -1 for one_hot encoding
        # encode coco class to be [0, 79] instead of [1, 80]
        label = F.where(label == 0, F.ones_like(label)*-1, label)
        label = F.where(label != -1, label-1, label)
        label = F.one_hot(label, self._num_class)

        # mask = F.where(label != -1, F.ones_like(label), F.zeros_like(label))

        alpha_factor = F.ones_like(label) * self._alpha
        alpha_factor = F.where(label, alpha_factor, 1.-alpha_factor)
            
        focal_weight = F.where(label, 1.-pred, pred)
        focal_weight = alpha_factor * F.power(focal_weight, self._gamma)

        bce = -(label * F.log(pred+self._eps) + (1.0-label) * F.log(1.0-pred))

        # loss = focal_weight * bce
        loss = bce

        loss = F.mean(loss, axis=self._batch_axis, exclude=True)  # .reshape((0, 1))
        # loss = loss/num_pos

        return loss


class HuberLoss(Loss):
    def __init__(self, rho=1, weight=None, batch_axis=0, **kwargs):
        super(HuberLoss, self).__init__(weight, batch_axis, **kwargs)
        self._rho = rho

    def hybrid_forward(self, F, pred, label):
        # num of positive samples
        # num_pos = F.sum(label != 0, axis=(1, 2), keepdims=True).reshape((0, -1))/4
        # num_pos = F.where(num_pos<=0, F.ones_like(num_pos), num_pos) 
        # box loss
        # mask = F.where(label==0, F.zeros_like(label), F.ones_like(label))
        loss = F.abs(label - pred)
        loss = F.where(loss > self._rho, loss - 0.5 * self._rho, (0.5 / self._rho) * F.square(loss))
        # loss = loss * mask
        loss = F.mean(loss, axis=self._batch_axis, exclude=True)  # .reshape((0, 1))
        # loss = loss/num_pos

        return loss





