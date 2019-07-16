import numpy as np
from mxnet import nd
from mxnet import autograd


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


def get_anchors(net, input_shape):
    h, w = input_shape
    x = nd.random.uniform(0, 1, shape=(1, 3, h, w))
    with autograd.record():
        _, _, anchors = net(x)
    return anchors

def generate_ssd_anchors(net, input_shape, anchor_scales, anchor_ratios, steps, offsets=(0.5, 0.5)):
    h, w = input_shape
    x = nd.random.uniform(0, 1, shape=(1, 3, h, w))
    cls_preds, loc_preds = net(x)
    # anchor_scales = [(51.2, 102.4), (102.4, 189.4), (189.4, 276.4), (276.4, 363.52), (363.52, 450.6), (450.6, 492)]

    anchors = []
    for cls_pred, loc_pred, sizes, ratios, step in zip(cls_preds, loc_preds, anchor_scales, anchor_ratios, steps):
        _, _, fh, fw = loc_pred.shape
        alloc_size = (fh, fw)
        sizes = (sizes[0], np.sqrt(sizes[0] * sizes[1]))
        feat_anchors = generate_feat_anchors(sizes, ratios, step, alloc_size, offsets)
        anchors.append(feat_anchors)

    anchors = np.concatenate(anchors, axis=0)
    # anchors = np.reshape(anchors, (1, -1, 4))
    anchors = nd.array(anchors)

    return anchors

def generate_feat_anchors(sizes, ratios, step, alloc_size, offsets):
    """Generate anchors for once. Anchors are stored with (center_x, center_y, w, h) format."""
    assert len(sizes) == 2, "SSD requires sizes to be (size_min, size_max)"
    anchors = []
    for i in range(alloc_size):
        for j in range(alloc_size):
            cy = (i + offsets[0]) * step
            cx = (j + offsets[1]) * step
            # ratio = ratios[0], size = size_min or sqrt(size_min * size_max)
            r = ratios[0]
            anchors.append([cx, cy, sizes[0], sizes[0]])
            anchors.append([cx, cy, sizes[1], sizes[1]])
            # size = sizes[0], ratio = ...
            for r in ratios[1:]:
                sr = np.sqrt(r)
                w = sizes[0] * sr
                h = sizes[0] / sr
                anchors.append([cx, cy, w, h])
    return np.array(anchors)
