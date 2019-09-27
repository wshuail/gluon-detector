import os
import sys
import warnings
import numpy as np
sys.path.insert(0, os.path.expanduser('~/incubator-mxnet/python'))
import mxnet as mx
from mxnet import autograd
from mxnet import nd
from mxnet import gluon
from mxnet.gluon import nn
sys.path.insert(0, os.path.expanduser('~/gluon_detector'))
from lib.modelzoo.feature import network_extractor
from lib.modelzoo.target import BoxDecoder


def _upsample(x, stride=2):
    return x.repeat(axis=-1, repeats=stride).repeat(axis=-2, repeats=stride)


class RetinaNet(nn.HybridBlock):
    def __init__(self, network, layers, num_class, anchors,
                 nms_thresh=0.45, nms_topk=400, post_nms=100,
                 pyramid_filters=256):
        super(RetinaNet, self).__init__()
        self.num_class = num_class
        self.nms_thresh = nms_thresh
        self.nms_topk = nms_topk
        self.post_nms = post_nms
        
        self.num_anchor = 9

        with self.name_scope():
            
            if not isinstance(anchors, nd.NDArray):
                anchors = nd.array(anchors)
            anchors = nd.reshape(anchors, (1, -1, 4))
            self.anchors = self.params.get_constant('anchors', anchors)
            
            self.features = network_extractor(network, layers)

            self.p5_1 = nn.Conv2D(channels=pyramid_filters, kernel_size=(1, 1), strides=(1, 1), padding=(0, 0))
            self.p5_2 = nn.Conv2D(channels=pyramid_filters, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1))

            self.p4_1 = nn.Conv2D(channels=pyramid_filters, kernel_size=(1, 1), strides=(1, 1), padding=(0, 0))
            self.p4_2 = nn.Conv2D(channels=pyramid_filters, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1))
            
            self.p3_1 = nn.Conv2D(channels=pyramid_filters, kernel_size=(1, 1), strides=(1, 1), padding=(0, 0))
            self.p3_2 = nn.Conv2D(channels=pyramid_filters, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1))
            
            self.p6 = nn.Conv2D(channels=256, kernel_size=(3, 3), strides=(2, 2), padding=(1, 1)) 

            self.p7 = nn.HybridSequential()
            self.p7.add(
                nn.Activation(activation='relu'),
                nn.Conv2D(channels=256, kernel_size=(3, 3), strides=(2, 2), padding=(1, 1)))

            self.cls_subnet = nn.HybridSequential()
            pi = 0.01
            bias = -np.log((1-pi)/pi)
            self.cls_subnet.add(
                nn.Conv2D(channels=pyramid_filters, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1),
                          weight_initializer=mx.init.Normal(sigma=0.01), bias_initializer='zeros'),
                nn.Activation(activation='relu'),
                nn.Conv2D(channels=pyramid_filters, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1),
                          weight_initializer=mx.init.Normal(sigma=0.01), bias_initializer='zeros'),
                nn.Activation(activation='relu'),
                nn.Conv2D(channels=pyramid_filters, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1),
                          weight_initializer=mx.init.Normal(sigma=0.01), bias_initializer='zeros'),
                nn.Activation(activation='relu'),
                nn.Conv2D(channels=pyramid_filters, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1),
                          weight_initializer=mx.init.Normal(sigma=0.01), bias_initializer='zeros'),
                nn.Activation(activation='relu'),
                nn.Conv2D(channels=num_class*self.num_anchor, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1),
                          weight_initializer=mx.init.Normal(sigma=0.01),
                          bias_initializer=mx.init.Constant(bias)),
                nn.Activation(activation='sigmoid')
            )

            self.loc_subnet = nn.HybridSequential()
            self.loc_subnet.add(
                nn.Conv2D(channels=pyramid_filters, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1),
                          weight_initializer=mx.init.Normal(sigma=0.01), bias_initializer='zeros'),
                nn.Activation(activation='relu'),
                nn.Conv2D(channels=pyramid_filters, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1),
                          weight_initializer=mx.init.Normal(sigma=0.01), bias_initializer='zeros'),
                nn.Activation(activation='relu'),
                nn.Conv2D(channels=pyramid_filters, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1),
                          weight_initializer=mx.init.Normal(sigma=0.01), bias_initializer='zeros'),
                nn.Activation(activation='relu'),
                nn.Conv2D(channels=pyramid_filters, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1),
                          weight_initializer=mx.init.Normal(sigma=0.01), bias_initializer='zeros'),
                nn.Activation(activation='relu'),
                nn.Conv2D(channels=4*self.num_anchor, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1),
                          weight_initializer=mx.init.Normal(sigma=0.01), bias_initializer='zeros'),
            )
            
            self.bbox_decoder = BoxDecoder()

    def hybrid_forward(self, F, x, anchors):
        # avoid bug in https://github.com/apache/incubator-mxnet/issues/13967
        anchors = F.identity(anchors)
        
        p3, p4, p5 = self.features(x)
        p6 = self.p6(p5)
        p7 = self.p7(p6)

        p5_1 = self.p5_1(p5)
        p5_2 = self.p5_2(p5_1)
        
        p5_up = F.UpSampling(p5_1, scale=2, sample_type='nearest')
        # p5_up = _upsample(p5_1, stride=2)
        p5_up = F.slice_like(p5_up, p4* 0, axes=(2, 3))
        p4_1 = self.p4_1(p4) + p5_up
        p4_2 = self.p4_2(p4_1)
        
        p4_up = F.UpSampling(p4_1, scale=2, sample_type='nearest')
        # p4_up = _upsample(p4_1, stride=2)
        p4_up = F.slice_like(p4_up, p3* 0, axes=(2, 3))
        p3_1 = self.p3_1(p3) + p4_up
        p3_2 = self.p3_2(p3_1)

        heads = [p3_2, p4_2, p5_2, p6, p7]

        cls_heads = [self.cls_subnet(x) for x in heads]
        cls_heads = [F.transpose(cls_head, (0, 2, 3, 1)) for cls_head in cls_heads]
        cls_heads = [F.reshape(cls_head, (0, 0, 0, self.num_anchor, self.num_class)) for cls_head in cls_heads]
        cls_heads = [F.reshape(cls_head, (0, -1, self.num_class)) for cls_head in cls_heads]
        cls_heads = F.concat(*cls_heads, dim=1)
        
        loc_heads = [self.loc_subnet(x) for x in heads]
        loc_heads = [F.transpose(loc_head, (0, 2, 3, 1)) for loc_head in loc_heads]
        loc_heads = [F.reshape(loc_head, (0, -1, 4)) for loc_head in loc_heads]
        loc_heads = F.concat(*loc_heads, dim=1)

        if autograd.is_training():
            return cls_heads, loc_heads, anchors
        
        # bboxes: (B, N, 4)
        bboxes = self.bbox_decoder(loc_heads, anchors)
        # cls_preds: (B, N, num_classes)
        cls_preds = cls_heads
        # scores: (B, N, 1) cls_ids: (B, N, 1)
        cls_ids = F.argmax(cls_preds, axis=-1, keepdims=True)
        scores = F.max(cls_preds, axis=-1, keepdims=True)
        scores_mask = (scores > 0.05)
        cls_ids = F.where(scores_mask, cls_ids, F.ones_like(cls_ids)*-1)
        scores = F.where(scores_mask, scores, F.zeros_like(scores))

        # (B, N, 6)
        result = F.concat(cls_ids, scores, bboxes, dim=-1)

        if self.nms_thresh > 0 and self.nms_thresh < 1:
            result = F.contrib.box_nms(
                result, overlap_thresh=self.nms_thresh, topk=self.nms_topk, valid_thresh=0.01,
                id_index=0, score_index=1, coord_start=2, force_suppress=False)
            if self.post_nms > 0:
                result = result.slice_axis(axis=1, begin=0, end=self.post_nms)
        
        cls_ids = F.slice_axis(result, axis=2, begin=0, end=1)
        scores = F.slice_axis(result, axis=2, begin=1, end=2)
        bboxes = F.slice_axis(result, axis=2, begin=2, end=6)

        return cls_ids, scores, bboxes


if __name__ == '__main__':
    from lib.anchor.retinanet import generate_retinanet_anchors
    input_size = 512
    anchors = generate_retinanet_anchors(input_size)
    x = nd.random.uniform(0, 1, (1, 3, input_size, input_size))
    network = 'resnet50_v1'
    layers = ['stage2_activation3', 'stage3_activation5', 'stage4_activation2']
    num_class = 80
    net = RetinaNet(network, layers, num_class, anchors)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        net.initialize()
    with autograd.record():
        cls_heads, loc_heads, _ = net(x)
        print (cls_heads.shape)
        print (loc_heads.shape)


