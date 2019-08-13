import os
import sys
sys.path.insert(0, os.path.expanduser('~/incubator-mxnet/python'))
import mxnet as mx
from mxnet import nd
from mxnet import gluon
from mxnet.gluon import nn
sys.path.insert(0, os.path.expanduser('~/gluon_detector'))
from lib.modelzoo.feature import network_extractor


def _upsample(x, stride=2):
    return x.repeat(axis=-1, repeats=stride).repeat(axis=-2, repeats=stride)


class RetinaNet(nn.HybridBlock):
    def __init__(self, network, layers, num_class, pyramid_filters=256):
        super(RetinaNet, self).__init__()
        
        num_anchor = 9

        with self.name_scope():
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
            self.cls_subnet.add(
                nn.Conv2D(channels=pyramid_filters, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1)),
                nn.Activation(activation='relu'),
                nn.Conv2D(channels=pyramid_filters, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1)),
                nn.Activation(activation='relu'),
                nn.Conv2D(channels=pyramid_filters, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1)),
                nn.Activation(activation='relu'),
                nn.Conv2D(channels=pyramid_filters, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1)),
                nn.Activation(activation='relu'),
                nn.Conv2D(channels=num_class*num_anchor, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1))
            )

            self.loc_subnet = nn.HybridSequential()
            self.loc_subnet.add(
                nn.Conv2D(channels=pyramid_filters, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1)),
                nn.Activation(activation='relu'),
                nn.Conv2D(channels=pyramid_filters, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1)),
                nn.Activation(activation='relu'),
                nn.Conv2D(channels=pyramid_filters, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1)),
                nn.Activation(activation='relu'),
                nn.Conv2D(channels=pyramid_filters, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1)),
                nn.Activation(activation='relu'),
                nn.Conv2D(channels=4*num_anchor, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1))
            )

    def hybrid_forward(self, F, x):
        p3, p4, p5 = self.features(x)
        p6 = self.p6(p5)
        p7 = self.p7(p6)

        p5_1 = self.p5_1(p5)
        p5_2 = self.p5_2(p5_1)
        
        p5_up = _upsample(p5_1, stride=2)
        p5_up = F.slice_like(p5_up, p4* 0, axes=(2, 3))
        p4_1 = self.p4_1(p4) + p5_up
        p4_2 = self.p4_2(p4_1)
        
        p4_up = _upsample(p4_1, stride=2)
        p4_up = F.slice_like(p4_up, p3* 0, axes=(2, 3))
        p3_1 = self.p3_1(p3) + p4_up
        p3_2 = self.p3_2(p3_1)


        heads = [p3_2, p4_2, p5_2, p6, p7]
        for head in heads:
            print (head.shape)

        cls_heads = [self.cls_subnet(x) for x in heads]
        cls_heads = [F.transpose(cls_head, (0, 2, 3, 1)) for cls_head in cls_heads]
        cls_heads = [F.reshape(cls_head, (0, -1, num_class)) for cls_head in cls_heads]
        cls_heads = F.concat(*cls_heads, dim=1)
        
        loc_heads = [self.loc_subnet(x) for x in heads]
        loc_heads = [F.transpose(loc_head, (0, 2, 3, 1)) for loc_head in loc_heads]
        loc_heads = [F.reshape(loc_head, (0, -1, 4)) for loc_head in loc_heads]
        loc_heads = F.concat(*loc_heads, dim=1)

        return cls_heads, loc_heads



if __name__ == '__main__':
    x = nd.random.uniform(0, 1, (1, 3, 224, 224))
    network = 'resnet50_v1'
    layers = ['stage2_activation3', 'stage3_activation5', 'stage4_activation2']
    num_class = 80
    net = RetinaNet(network, layers, num_class)
    net.initialize()
    cls_heads, loc_heads = net(x)
    print (cls_heads.shape)
    print (loc_heads.shape)
    """
    for cls_head in cls_heads:
        print (cls_head.shape)
    for loc_head in loc_heads:
        print (loc_head.shape)
    """


