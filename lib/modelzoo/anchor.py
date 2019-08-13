import os
import sys
import numpy as np
sys.path.insert(0, os.path.expanduser('~/incubator-mxnet/python'))
import mxnet as mx
import mxnet as mx
from mxnet import nd
from mxnet.gluon import nn

def generate_anchors(base_size=16, ratios=None, scales=None):
    """
    Generate anchor (reference) windows by enumerating aspect ratios X
    scales w.r.t. a reference window.
    """

    if ratios is None:
        ratios = np.array([0.5, 1, 2])

    if scales is None:
        scales = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])

    num_anchors = len(ratios) * len(scales)

    # initialize output anchors
    anchors = np.zeros((num_anchors, 4))

    # scale base_size
    anchors[:, 2:] = base_size * np.tile(scales, (2, len(ratios))).T

    # compute areas of anchors
    areas = anchors[:, 2] * anchors[:, 3]

    # correct for ratios
    anchors[:, 2] = np.sqrt(areas / np.repeat(ratios, len(scales)))
    anchors[:, 3] = anchors[:, 2] * np.repeat(ratios, len(scales))

    # transform from (x_ctr, y_ctr, w, h) -> (x1, y1, x2, y2)
    anchors[:, 0::2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T
    anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T

    return anchors

def shift(shape, stride, anchors):
    shift_x = (np.arange(0, shape[1]) + 0.5) * stride
    shift_y = (np.arange(0, shape[0]) + 0.5) * stride

    shift_x, shift_y = np.meshgrid(shift_x, shift_y)

    shifts = np.vstack((
        shift_x.ravel(), shift_y.ravel(),
        shift_x.ravel(), shift_y.ravel()
    )).transpose()

    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (K*A, 4) shifted anchors
    A = anchors.shape[0]
    K = shifts.shape[0]
    all_anchors = (anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
    all_anchors = all_anchors.reshape((K * A, 4))

    return all_anchors


def generate_level_anchors(stage_index, image_shape, size=None, stride=None, ratios=None,
                           scales=None, **kwargs):
    # assert (stage_index in [3, 4, 5, 6, 7]) ('Expected stage_index in range(3, 8), but got {}'.format(stage_index))

    if size is None:
        size = 2 ** (stage_index + 2)
    if stride is None:
        stride = 2 ** stage_index
    if ratios is None:
        ratios = np.array([0.5, 1, 2])
    if scales is None:
        scales = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])

    if isinstance(image_shape, int):
        image_shape = np.array((image_shape, image_shape))
    elif isinstance(image_shape, (tuple, list)):
        image_shape = np.array(image_shape)
    image_shape = (image_shape + 2 ** stage_index - 1) // (2 ** stage_index)
    
    base_anchor = generate_anchors(size, ratios, scales)
    anchors = shift(image_shape, stride, base_anchor)
    # print ('size: {}'.format(size))
    # print ('stride: {}'.format(stride))
    # print ('base_anchor: {}'.format(base_anchor))
    # print ('anchors: {}'.format(anchors))

    return anchors


if __name__ == '__main__':
    image_shape = (224, 224)
    level_anchors_list = []
    for i in range(3, 8):
        level_anchors = generate_level_anchors(i, image_shape)
        level_anchors_list.append(level_anchors)
        # print (level_anchors)
    anchors = np.concatenate(level_anchors_list, axis=0)
    print (anchors.shape)

