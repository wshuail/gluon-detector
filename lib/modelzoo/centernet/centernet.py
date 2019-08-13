import os
import sys
sys.path.insert(0, os.path.expanduser('~/incubator-mxnet/python'))
import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn

sys.path.insert(0, os.path.expanduser('~/lib/gluon-cv'))
from gluoncv.model_zoo import get_model

from ..feature import centernet_extractor


class CenterNet(nn.HybridBlock):
    def __init__(self, network, layers, heads, head_conv, **kwargs):
        super(CenterNet, self).__init__(**kwargs)
        self.heads = heads
        self.head_conv = head_conv

        self.feature = centernet_extractor(network, layers)

        self.deconv_layers = self._make_deconv_layers()
        
        # hm, offset, wh
        self.head_layers = nn.HybridSequential()
        for head, head_class in self.heads.items():
            head_layers = self._build_heads(head, head_class)
            self.head_layers.add(head_layers)

    def _make_deconv_layers(self):
        
        num_layers = 3
        num_filters = [256, 128, 64]
        num_kernels = [4, 4, 4]
        
        layers = nn.HybridSequential()

        for i in range(num_layers):
            num_filter = num_filters[i]
            kernel_size = num_kernels[i]
            # kernel, padding, output_padding = \
            #     self._get_deconv_cfg(num_kernels[i], i)

            layers.add(nn.Conv2D(channels=num_filter, kernel_size=3, strides=1,
                                 padding=1, dilation=1, use_bias=False))
            layers.add(nn.BatchNorm())
            layers.add(nn.Activation(activation='relu'))
            layers.add(nn.Conv2DTranspose(channels=num_filter, kernel_size=kernel_size,
                                          strides=(2, 2), padding=(1, 1),
                                          output_padding=(0, 0)))
            layers.add(nn.BatchNorm())
            layers.add(nn.Activation(activation='relu'))
        
        return layers

    def _build_heads(self, head, head_class):
        layers = nn.HybridSequential()
        layers.add(nn.Conv2D(channels=self.head_conv, kernel_size=3,
                             padding=(1, 1), use_bias=True))
        layers.add(nn.Activation(activation='relu'))
        layers.add(nn.Conv2D(channels=head_class, kernel_size=(1, 1),
                             strides=(1, 1), padding=(0, 0), use_bias=True))
        if head == 'hm':
            layers.add(nn.Activation(activation='sigmoid'))
        return layers

    def hybrid_forward(self, F, x):
        x = self.feature(x)
        x = self.deconv_layers(x)
        outputs = [layer(x) for layer in self.head_layers]
        
        return outputs


"""
if __name__ == '__main__':

    network = 'resnet18_v1'
    layers = ['stage4_activation1']
    heads = {'hm': 80}
    head_conv = 64
    net = CenterNet(network, layers, heads, head_conv)

    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        net.initialize()

    x = mx.nd.random.uniform(0, 1, (1, 3, 512, 512))
    out = net(x)[0]
    print (out.shape)

"""



