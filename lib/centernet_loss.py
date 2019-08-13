from __future__ import absolute_import
from mxnet import gluon
from mxnet import nd
from mxnet.gluon.loss import Loss, _apply_weighting, _reshape_like

__all__ = ['FocalLoss']


class FocalLoss(Loss):
    def __init__(self, alpha=2, gamma=4, from_logits=True, eps=1e-12,
                 weight=None, batch_axis=0, **kwargs):
        super(FocalLoss, self).__init__(weight, batch_axis, **kwargs)
        self._alpha = alpha
        self._gamma = gamma
        self._from_logits = from_logits
        self._eps = eps

    def hybrid_forward(self, F, pred, label):
        if not self._from_logits:
            pred = F.sigmoid(pred)
        pred = F.clip(pred, 1e-4, 1-1e-4)

        pos_idx = (label == 1)
        neg_idx = (label < 1)

        neg_weight = F.power(1-label, self._gamma)
        print ('sum neg_weight: {}'.format(F.sum(neg_weight)))

        loss = 0

        pos_loss = F.log(pred)*F.power(1-pred, self._alpha)*pos_idx
        neg_loss = F.log(1-pred)*F.power(pred, self._alpha)*neg_weight*neg_idx

        num_pos = F.sum(pos_idx)
        print ('num_pos: {}'.format(F.sum(label == 1, axis=(1, 2, 3))))

        pos_loss = F.sum(pos_loss)
        neg_loss = F.sum(neg_loss)
        print ('pos_loss: {}, neg_loss: {}'.format(pos_loss, neg_loss))

        if num_pos == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss+neg_loss)/num_pos
        
        return loss

