import sys
import os
import warnings
sys.path.insert(0, os.path.expanduser('/home/wangshuailong/incubator-mxnet/python/'))
import mxnet as mx

sys.path.insert(0, os.path.expanduser('~/det/'))
from mxnet import nd
from mxnet.gluon import nn
from mxnet.gluon import HybridBlock, Block, SymbolBlock


def conv_act_layer(data, num_filter, kernel, stride, pad, name, use_bn=False):
    data = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=kernel,
                               stride=stride, pad=pad, name='{}_conv'.format(name))
    if use_bn:
        data = mx.sym.BatchNorm(data=data, name='{}_bn'.format(name))
    data = mx.sym.Activation(data=data, act_type='relu', name='{}_act'.format(name))
    return data


def parse_network(network, layers, inputs='data'):

    from gluoncv.model_zoo import get_model
    network = get_model(network, pretrained=True)

    params = network.collect_params()
    prefix = network._prefix

    inputs = mx.sym.Variable('data')
    network = network(inputs)

    internals = network.get_internals()

    output_layers = []
    for layer in layers:
        layer_name = '{}{}_output'.format(prefix, layer)
        output_layer = internals[layer_name]
        output_layers.append(output_layer)

    return output_layers, inputs, params

def expand_network(network, layers, num_filters, min_num_filter=128):
    output_layers, inputs, params = parse_network(network, layers)
    
    layer = output_layers[-1]
    for idx, num_filter in enumerate(num_filters):
        num_filter_1x1 = max(min_num_filter, num_filter//2)
        layer = conv_act_layer(data=layer, num_filter=num_filter_1x1, kernel=(1, 1),
                               stride=(1, 1), pad=(0, 0), name='multi_feat_{}_conv_1x1'.format(idx))
        layer = conv_act_layer(data=layer, num_filter=num_filter, kernel=(3, 3),
                               stride=(2, 2), pad=(1, 1), name='multi_feat_{}_conv_3x3'.format(idx))
        output_layers.append(layer)
    
    output_blocks = SymbolBlock(outputs=output_layers, inputs=inputs, params=params)

    return output_blocks

def centernet_extractor(network, layers):
    output_layers, inputs, params = parse_network(network, layers)
    output_blocks = SymbolBlock(outputs=output_layers, inputs=inputs, params=params)
    return output_blocks

