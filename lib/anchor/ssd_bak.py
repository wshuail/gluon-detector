import os
import sys
import numpy as np
sys.path.insert(0, os.path.expanduser('~/incubator-mxnet/python'))
import mxnet as mx
from mxnet.gluon import nn

def generate_level_anchors(feat_shape, sizes, ratios, step, offsets):
    assert len(sizes) == 2, "SSD requires sizes to be (size_min, size_max)"
    if isinstance(feat_shape, int):
        feat_shape = (feat_shape, feat_shape)
        
    sizes = (sizes[0], np.sqrt(sizes[0] * sizes[1]))
    print ('sizes: {}'.format(sizes))

    anchors = []
    for i in range(feat_shape[0]):
        for j in range(feat_shape[1]):
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


def generate_ssd_anchors(input_shape, sizes, ratios, steps, pyramid_levels=(4, 5, 6, 7, 8, 9)):
    input_shape = np.array(input_shape)
    feat_shapes = [(input_shape + 2 ** x - 1) // (2 ** x) for x in pyramid_levels]
    
    anchors = []
    for feat_shape, size, ratio, step in zip(feat_shapes, sizes, ratios, steps):
        level_anchor = generate_level_anchors(feat_shape, size, ratio, step, offsets=(0.5, 0.5))
        print ('level_anchor: {}'.format(level_anchor))
        anchors.append(level_anchor)
    anchors = np.concatenate(anchors, axis=0)
    anchors = mx.nd.array(anchors)
    return anchors
 

class SSDAnchorGenerator(nn.HybridBlock):
    def __init__(self, index, sizes, ratios, step, alloc_size, offsets=(0.5, 0.5), **kwargs):
        super(SSDAnchorGenerator, self).__init__(**kwargs)

        sizes = (sizes[0], np.sqrt(sizes[0] * sizes[1]))
        anchors = self.generate_feat_anchors(sizes, ratios, step, alloc_size, offsets)
        self.anchors = self.params.get_constant('anchors_{}'.format(index), anchors)

    def generate_feat_anchors(self, sizes, ratios, step, alloc_size, offsets):
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
        return np.array(anchors).reshape((1, 1, alloc_size, alloc_size, -1))

    def hybrid_forward(self, F, x, anchors):
        # x: (B, C, H, W)
        # anchor: (1, 1, MAX_SIZE, MAX_SIZE, M)
        # ==> (1, 1, H, W, M)
        anchors = F.slice_like(anchors, x, axes=(2, 3))
        anchors = F.reshape(anchors, (1, -1, 4))
        return anchors


if __name__ == '__main__':
    anchor_ratios = [[1, 2, 0.5], [1, 2, 0.5, 3, 0.3333], [1, 2, 0.5, 3, 0.3333], [1, 2, 0.5, 3, 0.3333], [1, 2, 0.5], [1, 2, 0.5]]
    anchor_sizes = [51.2, 102.4, 189.4, 276.4, 363.52, 450.6, 492]
    anchor_sizes = list(zip(anchor_sizes[:-1], anchor_sizes[1:]))
    print ('anchor_sizes: {}'.format(anchor_sizes))
    steps = [16, 32, 64, 128, 256, 512]
    input_shape = (512, 512)
    
    anchors = generate_ssd_anchors(input_shape, anchor_sizes, anchor_ratios, steps)
    print (anchors.shape)

    """
    anchors = []
    asz = 128
    input_shapes = (32, 16, 8, 4, 2, 1)
    for index in range(6):
        size = anchor_sizes[index]
        ratio = anchor_ratios[index]
        step = steps[index]
        input_shape = input_shapes[index]
        anchor_generator = SSDAnchorGenerator(index, size, ratio, step, asz)

        anchor_generator.initialize()
        x = mx.nd.random.uniform(0, 1, (1, 3, input_shape, input_shape))
        anchor = anchor_generator(x)
        print (anchor.shape)
        anchors.append(anchor)
    """
