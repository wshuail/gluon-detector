import sys
import os
import yaml
sys.path.insert(0, os.path.expanduser('~/incubator-mxnet/python'))
import mxnet as mx

from .resnet import get_symbol as get_backbone

def print_shape(net, input_shape=(1, 3, 300, 300)):
    arg_shape, output_shape, aux_shape = net.infer_shape(data=input_shape)
    print(output_shape)

def conv_act_layer(data, num_filter, kernel, stride, pad, name, use_bn=False):
    data = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=kernel,
                               stride=stride, pad=pad, name='{}_conv'.format(name))
    if use_bn:
        data = mx.sym.BatchNorm(data=data, name='{}_bn'.format(name))
    data = mx.sym.Activation(data=data, act_type='relu', name='{}_act'.format(name))
    return data

def multi_layer_feature(body, from_layers, num_filters, strides, pads, min_num_filter=128):

    internals = body.get_internals()

    layers = []
    for idx, params in enumerate(zip(from_layers, num_filters, strides, pads)):
        from_layer, num_filter, s, p = params
        
        if from_layer:
            layer_name = from_layer + '_output'
            layer = internals[layer_name]
            print_shape(layer)
            layers.append(layer)
        else:
            layer = layers[-1]
            num_filter_1x1 = max(min_num_filter, num_filter//2)
            layer = conv_act_layer(data=layer, num_filter=num_filter_1x1, kernel=(1, 1),
                                   stride=(1, 1), pad=(0, 0), name='multi_feat_{}_conv_1x1'.format(idx))
            layer = conv_act_layer(data=layer, num_filter=num_filter, kernel=(3, 3),
                                   stride=(s, s), pad=(p, p), name='multi_feat_{}_conv_3x3'.format(idx))
            print_shape(layer)
            layers.append(layer)

    return layers

def multibox_layer(layers, num_classes, sizes, ratios,
                   normalization, num_channels, steps,
                   clip=False, interm_layer=0):

    loc_preds = []
    cls_preds = []
    all_anchors = []

    for idx, from_layer in enumerate(layers):
        from_name = from_layer.name
        size = sizes[idx]
        ratio = ratios[idx]
        num_anchors = len(size) + len(ratio) - 1 

        size_str = '(' + ','.join([str(s) for s in size]) + ')'
        ratio_str = '(' + ','.join([str(r) for r in ratio]) + ')'
        
        bias = mx.symbol.Variable(name="{}_loc_pred_conv_bias".format(from_name),
            init=mx.init.Constant(0.0), attr={'__lr_mult__': '2.0'})
        loc_pred = mx.sym.Convolution(data=from_layer, num_filter=4*num_anchors,
                                      kernel=(3, 3), stride=(1, 1), pad=(1, 1), bias=bias,
                                      name='multi_layer_output_{}_bbox_pred'.format(idx))
        loc_pred = mx.sym.transpose(data=loc_pred, axes=(0, 2, 3, 1))
        loc_pred = mx.sym.Flatten(data=loc_pred)
        loc_preds.append(loc_pred)

        
        num_cls_pred = num_anchors * (num_classes+1)
        bias = mx.symbol.Variable(name="{}_cls_pred_conv_bias".format(from_name),
            init=mx.init.Constant(0.0), attr={'__lr_mult__': '2.0'})
        cls_pred = mx.sym.Convolution(data=from_layer, num_filter=num_cls_pred,
                                      kernel=(3, 3), stride=(1, 1), pad=(1, 1), bias=bias,
                                      name='multi_layer_output_{}_cls_pred'.format(idx))
        cls_pred = mx.sym.transpose(data=cls_pred, axes=(0, 2, 3, 1))
        cls_pred = mx.sym.Flatten(data=cls_pred)
        cls_preds.append(cls_pred)

        anchors = mx.sym.contrib.MultiBoxPrior(data=from_layer, sizes=size_str, ratios=ratio_str)
        anchors = mx.sym.Flatten(data=anchors)
        all_anchors.append(anchors)

    loc_preds = mx.sym.concat(*loc_preds, dim=1)
    
    cls_preds = mx.sym.concat(*cls_preds, dim=1)
    cls_preds = mx.sym.reshape(data=cls_preds, shape=(0, -1, (num_classes+1)))
    cls_preds = mx.sym.transpose(data=cls_preds, axes=(0, 2, 1))
    
    anchors = mx.sym.concat(*all_anchors, dim=1)
    anchors = mx.sym.reshape(data=anchors, shape=(0, -1, 4))

    return loc_preds, cls_preds, anchors


def get_symbol(config, num_classes, num_layers, image_shape,
               nms_thresh=0.5, force_suppress=False, nms_topk=400):

    label = mx.sym.Variable('label')
    body = get_backbone(num_classes, num_layers, image_shape='3, 224, 224')
    
    from_layers = config['from_layers']
    num_filters = config['num_filters']
    strides = config['strides']
    pads = config['pads']
    layers = multi_layer_feature(body, from_layers, num_filters, strides, pads)

    sizes = config['sizes']
    ratios = config['ratios']
    normalization = config['normalizations']
    num_filters = config['num_filters']
    steps = config['steps']
    loc_preds, cls_preds, anchor_boxes = multibox_layer(layers, num_classes, sizes=sizes, ratios=ratios,
                   normalization=normalization, num_channels=num_filters,
                   clip=False, interm_layer=0, steps=steps)

    # loc_preds, cls_preds, anchors = multibox_layer(from_layers=layers, num_classes=num_classes, config=config)
    print ('multi_layer_output')
    print_shape(loc_preds)
    print_shape(cls_preds)
    print_shape(anchor_boxes)

    tmp = mx.sym.contrib.MultiBoxTarget(anchor=anchor_boxes,
                                        label=label,
                                        cls_pred=cls_preds,
                                        overlap_threshold=0.5,
                                        ignore_label=-1,
                                        negative_mining_ratio=3,
                                        minimum_negative_samples=0,
                                        negative_mining_thresh=0.5,
                                        variances=(0.1, 0.1, 0.2, 0.2),
                                        name='multibox_target')
    loc_target, loc_target_mask, cls_target = tmp

    cls_prob = mx.sym.SoftmaxOutput(data=cls_preds, 
                                    label=cls_target,
                                    ignore_label=-1,
                                    use_ignore=True,
                                    grad_scale=1.0,
                                    multi_output=True,
                                    normalization='valid',
                                    name='cls_prob')

    loc_loss_ = mx.sym.smooth_l1(data=loc_target_mask*(loc_target-loc_preds),
                                 scalar=1.0,
                                 name='loc_loss_')
    loc_loss = mx.sym.MakeLoss(data=loc_loss_,
                               grad_scale=1.0,
                               normalization='valid',
                               name='loc_loss')

    cls_label = mx.sym.MakeLoss(data=cls_target,
                                grad_scale=0,
                                name='cls_label')

    det = mx.sym.contrib.MultiBoxDetection(cls_prob=cls_prob,
                                           loc_pred=loc_preds,
                                           anchor=anchor_boxes,
                                           name="detection",
                                           nms_threshold=nms_thresh,
                                           force_suppress=force_suppress,
                                           variances=(0.1, 0.1, 0.2, 0.2),
                                           nms_topk=nms_topk)
    det = mx.symbol.MakeLoss(data=det, grad_scale=0, name="det_out")

    # group output
    out = mx.symbol.Group([cls_prob, loc_loss, cls_label, det])

    return out


