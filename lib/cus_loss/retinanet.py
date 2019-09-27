import os
import sys
sys.path.insert(0, os.path.expanduser('~/incubator-mxnet/python'))
from mxnet import gluon
from mxnet import nd
from mxnet.gluon.loss import Loss

__all__ = ['FocalLoss', 'HuberLoss']


class FocalLoss(Loss):
    def __init__(self, num_class, alpha=0.25, gamma=2, huber_rho=0.11,
                 debug=False, eps=1e-12, weight=None, batch_axis=0, **kwargs):
        super(FocalLoss, self).__init__(weight, batch_axis, **kwargs)
        self._num_class = num_class
        self._alpha = alpha
        self._gamma = gamma
        self._huber_rho = huber_rho
        self._eps = eps
        self._debug = debug

    def hybrid_forward(self, F, pred, label):
        pred = F.clip(pred, 1e-4, 1-1e-4)
        
        mask = F.where(label == -1, F.zeros_like(label), F.ones_like(label))
        mask = F.expand_dims(mask, axis=-1)
        mask = F.tile(mask, reps=(1, 1, self._num_class))

        # num of positive samples
        num_pos = F.where(label>0, F.ones_like(label), F.zeros_like(label))
        num_pos = F.sum(label>0, axis=0, exclude=True)
        num_pos = F.where(num_pos>0, num_pos, F.ones_like(num_pos))
        # print ('cls num_pos: {}'.format(num_pos))
        
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
        loss = loss*mask

        loss = F.sum(loss, axis=0, exclude=True)/num_pos

        return loss


class HuberLoss(Loss):
    def __init__(self, rho=1, weight=None, batch_axis=0, **kwargs):
        super(HuberLoss, self).__init__(weight, batch_axis, **kwargs)
        self._rho = rho

    def hybrid_forward(self, F, pred, label):
        # print ('box max pred: {}'.format(F.max(pred)))
        # print ('box sum pred: {}'.format(F.sum(pred)))
        DEBUG = False
        # num of positive samples
        pos = F.where(label != 0, F.ones_like(label), F.zeros_like(label))
        num_pos = F.sum(pos, axis=-1)
        num_pos = F.where(num_pos != 0, F.ones_like(num_pos), F.zeros_like(num_pos)) 
        num_pos = F.sum(num_pos, axis=0, exclude=True)
        num_pos = F.where(num_pos>0, num_pos, F.ones_like(num_pos))
        # print ('box num_pos: {}'.format(num_pos))
        if DEBUG:
            label_np = label[0, :, :].asnumpy()
            label_np_sum = label_np.sum(axis=-1)
            pos_idx = (label_np_sum != 0)
            pos_label = label_np[pos_idx, :]
            # print ('pos_label: {}'.format(pos_label))
            print ('pos: {}'.format(pos[0, :, :].asnumpy()[pos_idx, :]))

        # print ('pred mean: {}'.format(F.mean(pred)))
        loss = F.abs(label - pred)
        loss = F.where(loss > self._rho, loss - 0.5 * self._rho, (0.5 / self._rho) * F.square(loss))
        loss = loss*pos
        loss = F.sum(loss, axis=0, exclude=True)/num_pos

        return loss


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




