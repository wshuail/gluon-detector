import os
import sys
import numpy as np
sys.path.insert(0, os.path.expanduser('~/incubator-mxnet/python'))
import mxnet as mx


def get_scales(min_scale=0.2, max_scale=0.9, num_layers=6):
    # this code follows the original implementation of wei liu
    # for more, look at ssd/score_ssd_pascal.py:310 in the original caffe implementation
    min_ratio = int(min_scale * 100)
    max_ratio = int(max_scale * 100)
    step = int(np.floor((max_ratio - min_ratio) / (num_layers - 2)))
    
    min_sizes = []
    max_sizes = []
    for ratio in range(min_ratio, max_ratio + 1, step):
        min_sizes.append(ratio / 100.)
        max_sizes.append((ratio + step) / 100.)
    min_sizes = [int(100*min_scale / 2.0) / 100.0] + min_sizes
    max_sizes = [min_scale] + max_sizes
    
    # convert it back to this implementation's notation:
    scales = []
    for layer_idx in range(num_layers):
        scales.append([min_sizes[layer_idx], np.single(np.sqrt(min_sizes[layer_idx] * max_sizes[layer_idx]))])
    return scales


def generate_base_anchors(sizes, ratios):
    assert ratios[0] == 1, 'Expected 1st of ratios to be 1, but got {}'.format(ratios[0])
    ratios = [ratios[0] for _ in range(len(sizes)-1)] + ratios
    sizes = sizes + [sizes[0] for _ in range(len(ratios)-len(sizes))]
    assert len(ratios) == len(sizes), 'Expected same length of ratios and sizes,\
        but got {} and {}'.format(len(ratios), len(sizes))
    ratios = np.array(ratios).reshape(-1, 1)
    sizes = np.array(sizes).reshape(-1, 1)
    r = np.sqrt(ratios)
    w = sizes*r
    h = sizes/r

    # anchor format (x_ctr, y_ctr, w, h)
    anchors = np.tile(np.zeros_like(ratios), reps=4)
    anchors[:, 2] = w.flatten()
    anchors[:, 3] = h.flatten()
    
    return anchors


def shift_anchors(base_anchors, feat_shape, stride):
    shift_x = (np.arange(0, feat_shape[1]) + 0.5) * stride
    shift_y = (np.arange(0, feat_shape[0]) + 0.5) * stride
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shifts = np.vstack((
        shift_x.ravel(), shift_y.ravel(),
        np.zeros_like(shift_x).ravel(), np.zeros_like(shift_y).ravel()
    )).transpose()

    A = base_anchors.shape[0]
    K = shifts.shape[0]
    lvl_anchors = (base_anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
    lvl_anchors = lvl_anchors.reshape((K * A, 4))

    return lvl_anchors


def generate_ssd_anchors(input_size, anchor_ratios, pyramid_levels=(4, 5, 6, 7, 8, 9)):
    assert isinstance(input_size, int), 'Expected input_size to be int but got {}'.format(type(input_size))
    input_shape = np.array((input_size, input_size))
    
    scales = get_scales()
    anchor_sizes = np.array(scales)*input_size
    anchor_sizes = anchor_sizes.tolist()
    
    pyramid_levels = (4, 5, 6, 7, 8, 9)
    feat_shapes = [(input_shape + 2 ** x - 1) // (2 ** x) for x in pyramid_levels]
    strides = [2**x for x in pyramid_levels]
    
    anchors = []
    for sizes, ratios, feat_shape, stride in zip(anchor_sizes, anchor_ratios, feat_shapes, strides):
        base_anchors = generate_base_anchors(sizes, ratios)
        lvl_anchors = shift_anchors(base_anchors, feat_shape, stride)
        anchors.append(lvl_anchors)
    anchors = np.concatenate(anchors, axis=0)
    anchors = mx.nd.array(anchors)
    return anchors



if __name__ == '__main__':
    anchor_ratios = [[1, 2, 0.5], [1, 2, 0.5, 3, 0.3333], [1, 2, 0.5, 3, 0.3333], [1, 2, 0.5, 3, 0.3333], [1, 2, 0.5], [1, 2, 0.5]]
    input_size = 512
    anchors = generate_ssd_anchors(input_size, anchor_ratios)
    print ('anchors: {}'.format(anchors))
    print ('anchors shape: {}'.format(anchors.shape))





