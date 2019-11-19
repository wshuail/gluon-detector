"""You Only Look Once Object Detection v3"""
# pylint: disable=arguments-differ
from __future__ import absolute_import
from __future__ import division

import os
import sys
import warnings
import numpy as np
sys.path.insert(0, os.path.expanduser('~/incubator-mxnet/python'))
import mxnet as mx
from mxnet import nd
from mxnet import gluon
from mxnet import autograd
from mxnet.gluon import nn
from mxnet.gluon.nn import BatchNorm
sys.path.insert(0, os.path.expanduser('~/det'))
from gluoncv.model_zoo import get_mobilenet

def _conv2d(channel, kernel, padding, stride, norm_layer=BatchNorm, norm_kwargs=None):
    """A common conv-bn-leakyrelu cell"""
    cell = nn.HybridSequential(prefix='')
    cell.add(nn.Conv2D(channel, kernel_size=kernel,
                       strides=stride, padding=padding, use_bias=False))
    cell.add(norm_layer(epsilon=1e-5, momentum=0.9, **({} if norm_kwargs is None else norm_kwargs)))
    cell.add(nn.LeakyReLU(0.1))
    return cell

def _upsample(x, stride=2):
    return x.repeat(axis=-1, repeats=stride).repeat(axis=-2, repeats=stride)


class YOLOOutputV3(nn.HybridBlock):
    def __init__(self, index, num_class, anchors, strides, alloc_size=(128, 128), **kwargs):
        super(YOLOOutputV3, self).__init__()
        anchors = np.array(anchors).astype('float32')
        self._num_pred = 1 + 4 + num_class
        self._num_anchors = anchors.size//2
        self._strides = strides

        with self.name_scope():
            all_pred = self._num_pred * self._num_anchors
            self.prediction = nn.Conv2D(all_pred, kernel_size=1, padding=0, strides=1) 
            
            anchors = anchors.reshape((1, 1, -1, 2))
            self.anchors = self.params.get_constant('{}_anchors'.format(index), anchors)
            
            grid_x = np.arange(alloc_size[0])
            grid_y = np.arange(alloc_size[1])
            grid_x, grid_y = np.meshgrid(grid_x, grid_y)
            # stack to (n, n, 2)
            offsets = np.concatenate((grid_x[:, :, np.newaxis], grid_y[:, :, np.newaxis]), axis=-1)
            # expand dims to (1, 1, n, n, 2) so it's easier for broadcasting
            offsets = np.expand_dims(np.expand_dims(offsets, axis=0), axis=0)
            self.offsets = self.params.get_constant('offset_%d'%(index), offsets)

    def hybrid_forward(self, F, x, anchors, offsets):
        # x ==> (B, pred per pixel, height*width)
        pred = self.prediction(x).reshape((0, self._num_anchors*self._num_pred, -1))
        pred = F.transpose(pred, (0, 2, 1)).reshape((0, -1, self._num_anchors, self._num_pred))
        # components
        raw_box_centers = pred.slice_axis(axis=-1, begin=0, end=2)
        raw_box_scales = pred.slice_axis(axis=-1, begin=2, end=4)
        objness = pred.slice_axis(axis=-1, begin=4, end=5)
        class_pred = pred.slice_axis(axis=-1, begin=5, end=None)

        # get offsets
        # (1, 1, n, n, 2) ==> (1, 1, height, width, 2)
        offsets = F.slice_like(offsets, x*0, axes=(2, 3))
        # (1, 1, height, width, 2) ==> (1, height*width, 1, 2)
        offsets = F.reshape(offsets, (1, -1, 1, 2))

        box_centers = F.broadcast_add(F.sigmoid(raw_box_centers), offsets)*self._strides
        box_scales = F.broadcast_mul(F.exp(raw_box_scales), anchors)
        confidence = F.sigmoid(objness)
        class_score = F.broadcast_mul(confidence, F.sigmoid(class_pred))
        wh = box_scales/2
        bbox = F.concat(box_centers - wh, box_centers + wh, dim=-1)
        bbox = F.reshape(bbox, (0, -1, 4))

        if autograd.is_training():
            return bbox, raw_box_centers, raw_box_scales, objness, class_pred, anchors, offsets


class YOLODetectionBlockV3(nn.HybridBlock):
    def __init__(self, channel, **kwargs):
        super(YOLODetectionBlockV3, self).__init__()
        with self.name_scope():
            self.body = nn.HybridSequential()
            for i in range(2):
                self.body.add(_conv2d(channel, 1, 0, 1))
                self.body.add(_conv2d(channel*2, 3, 1, 1)) 
            self.body.add(_conv2d(channel, 1, 0, 1))

            self.tip = _conv2d(channel*2, 3, 1, 1)

    def hybrid_forward(self, F, x):
        route = self.body(x)
        tip = self.tip(route)
        return route, tip


class YOLO(nn.HybridBlock):
    def __init__(self, stages, channels, anchors, strides, num_class):
        super(YOLO, self).__init__()

        with self.name_scope():
            self.stages = nn.HybridSequential()
            self.blocks = nn.HybridSequential()
            self.transitions = nn.HybridSequential()
            self.outputs = nn.HybridSequential()
            
            for i, stage, channel, anchor, stride in zip(range(len(stages)), stages,\
                                                         channels, anchors[::-1], strides[::-1]):
                self.stages.add(stage)
                block = YOLODetectionBlockV3(channel=channel)
                self.blocks.add(block)
                output = YOLOOutputV3(i, num_class, anchor, stride)
                self.outputs.add(output)
                if i > 0:
                    self.transitions.add(_conv2d(channel, 1, 0, 1))

    def hybrid_forward(self, F, x):
        routes = []
        for stage in self.stages:
            x = stage(x)
            print ('x shape: {}'.format(x.shape))
            routes.append(x)
        for i in reversed(range(len(routes))):
            x = routes[i]
            print ('x 3 shape: {}'.format(x.shape))
            
        for i, block, output in zip(range(len(routes)), self.blocks, self.outputs):
            print ('x 2 shape: {}'.format(x.shape))
            x, tip = block(x)
            print ('tip shape: {}'.format(tip.shape))
            if i >= len(routes) - 1:
                break
            x = self.transitions[i](x)
            upsample = _upsample(x, stride=2)
            route_now = routes[::-1][i + 1]
            x = F.concat(F.slice_like(upsample, route_now * 0, axes=(2, 3)), route_now, dim=1)

    def hybrid_forward_2(self, F, x):
        routes = []
        for stage in self.stages:
            x = stage(x)
            print ('x shape: {}'.format(x.shape))
            routes.append(x)
            
        all_dets = []
        all_box_centers = []
        all_box_scales = []
        all_objectness = []
        all_class_pred = []
        all_anchors = []
        all_offsets = []
        for i, block, output in zip(range(len(routes)), self.blocks, self.outputs):
            print ('x 2 shape: {}'.format(x.shape))
            x, tip = block(x)
            print ('tip shape: {}'.format(tip.shape))
            if autograd.is_training():
                dets, box_centers, box_scales, objness, class_pred, anchors, offsets = output(tip)
                all_dets.append(dets)
                all_box_centers.append(box_centers)
                all_box_scales.append(box_scales)
                all_objectness.append(objness)
                all_class_pred.append(class_pred)
                all_anchors.append(anchors)
                all_offsets.append(offsets)
            if i >= len(routes) - 1:
                break
            x = self.transitions[i](x)
            upsample = _upsample(x, stride=2)
            route_now = routes[::-1][i + 1]
            x = F.concat(F.slice_like(upsample, route_now * 0, axes=(2, 3)), route_now, dim=1)

        return all_dets, all_box_centers, all_box_scales, all_objectness, all_class_pred, all_anchors, all_offsets


def yolo3_mobilenet1_0_coco(pretrained_base=True, pretrained=False, norm_layer=BatchNorm,
                            norm_kwargs=None, **kwargs):
    pretrained_base = False if pretrained else pretrained_base
    base_net = get_mobilenet(
        multiplier=1,
        pretrained=pretrained_base,
        norm_layer=norm_layer, norm_kwargs=norm_kwargs,
        **kwargs)
    stages = [base_net.features[:33],
              base_net.features[33:69],
              base_net.features[69:-2]]

    anchors = [[10, 13, 16, 30, 33, 23],
               [30, 61, 62, 45, 59, 119],
               [116, 90, 156, 198, 373, 326]]
    strides = [8, 16, 32]
    classes = COCODetection.CLASSES
    return get_yolov3(
        'mobilenet1.0', stages, [512, 256, 128], anchors, strides, classes, 'coco',
        pretrained=pretrained, norm_layer=norm_layer, norm_kwargs=norm_kwargs, **kwargs)

if __name__ == '__main__':
    pretrained_base=False
    norm_layer=BatchNorm
    norm_kwargs=None
    base_net = get_mobilenet(multiplier=1, pretrained=pretrained_base,
                             norm_layer=norm_layer, norm_kwargs=norm_kwargs)
    stages = [base_net.features[:33], base_net.features[33:69], base_net.features[69:-2]]
    channels = [512, 256, 128]
    anchors = [[10, 13, 16, 30, 33, 23],
               [30, 61, 62, 45, 59, 119],
               [116, 90, 156, 198, 373, 326]]
    strides = [8, 16, 32]
    num_class = 80

    yolo = YOLO(stages, channels, anchors, strides, num_class)
    yolo.initialize()

    x = nd.random.uniform(0, 1, (1, 3, 320, 320))
    with autograd.record():
        yolo(x)
    """
        all_dets, all_box_centers, all_box_scales, all_objectness, all_class_pred, all_anchors, all_offsets = yolo(x)
    print ('det shapes: {}'.format([det.shape for det in all_dets]))
    print ('box_centers shapes: {}'.format([box_centers.shape for box_centers in all_box_centers]))
    print ('box_scales shapes: {}'.format([box_scales.shape for box_scales in all_box_scales]))
    print ('all_objectness shapes: {}'.format([objectness.shape for objectness in all_objectness]))
    print ('all_class_pred shapes: {}'.format([class_pred.shape for class_pred in all_class_pred]))
    print ('all_anchors shapes: {}'.format([anchors.shape for anchors in all_anchors]))
    print ('all_offsets shapes: {}'.format([offsets.shape for offsets in all_offsets]))
    """




