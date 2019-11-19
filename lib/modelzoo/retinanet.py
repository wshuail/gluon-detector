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
from pathlib import Path
d = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(d))
from lib.modelzoo.feature import network_extractor


class RetinaNet(nn.HybridBlock):
    def __init__(self, network, layers, num_class, pyramid_filters=256):
        super(RetinaNet, self).__init__()
        self.num_class = num_class
        self.num_anchor = 9

        with self.name_scope():
            
            self.features = network_extractor(network, layers, fix_bn=True)

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
                          weight_initializer=mx.init.Normal(sigma=0.01),
                          bias_initializer='zeros'),
            )
            
    def hybrid_forward(self, F, x):
        p3, p4, p5 = self.features(x)
        p6 = self.p6(p5)
        p7 = self.p7(p6)

        p5_1 = self.p5_1(p5)
        p5_2 = self.p5_2(p5_1)
        
        p5_up = F.UpSampling(p5_1, scale=2, sample_type='nearest')
        p4_1 = self.p4_1(p4) + p5_up
        p4_2 = self.p4_2(p4_1)
        
        p4_up = F.UpSampling(p4_1, scale=2, sample_type='nearest')
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

        return cls_heads, loc_heads
    

def retinanet_resnet50_v1_coco():
    network = 'resnet50_v1'
    layers = ['stage2_activation3', 'stage3_activation5', 'stage4_activation2']
    freeze_layers = ['resnetv11_conv0_weight']
    num_class = 80
    net = RetinaNet(network, layers, num_class)
    params = net.collect_params()
    # fix some params
    stage0_conv_weight = list(params.values())[0]
    stage0_conv_weight.grad_req = 'null'
    for k, v in params.items():
        if 'stage1' in k:
            v.grad_req = 'null'
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        net.initialize()
    return net


if __name__ == '__main__':
    x = nd.random.uniform(0, 1, (1, 3, 1024, 640))
    net = retinanet_resnet50_v1_coco()
    with autograd.record():
        cls_heads, loc_heads = net(x)
        print (cls_heads.shape)
        print (loc_heads.shape)


