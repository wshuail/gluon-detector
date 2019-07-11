import mxnet as mx
from mxnet import nd
from mxnet import autograd
from mxnet.gluon import nn
from ..feature import expand_network
from ..target import BoxDecoder, MultiPerClassDecoder
from ..target import SSDAnchorGenerator


class SSD(nn.HybridBlock):
    def __init__(self, network, layers, num_filters, num_classes, anchor_sizes, anchor_ratios,
                 steps, max_anchor_size=128, nms_thresh=0.45, nms_topk=400, post_nms=100,
                 anchor_mode=False, **kwargs):
        super(SSD, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.nms_thresh = nms_thresh
        self.nms_topk = nms_topk
        self.post_nms = post_nms
        
        anchor_sizes = list(zip(anchor_sizes[:-1], anchor_sizes[1:]))
        
        self.features = expand_network(network, layers, num_filters)
        num_layers = len(layers) + len(num_filters)

        self.anchor_generators = nn.HybridSequential()
        self.cls_predictors = nn.HybridSequential()
        self.loc_predictors = nn.HybridSequential()
        asz = max_anchor_size
        for index, size, ratio, step in zip(range(num_layers), anchor_sizes, anchor_ratios, steps):
            anchor_generator = SSDAnchorGenerator(index, size, ratio, step, asz)
            self.anchor_generators.add(anchor_generator)
            asz = max(asz // 2, 16)  # pre-compute larger than 16x16 anchor map
            
            num_anchor = len(size) + len(ratio) - 1
            self.cls_predictors.add(nn.Conv2D(num_anchor*(self.num_classes+1), kernel_size=(3, 3),
                                              strides=(1, 1), padding=(1, 1),
                                              weight_initializer=mx.init.Xavier(magnitude=2),
                                              bias_initializer='zeros'))
            self.loc_predictors.add(nn.Conv2D(num_anchor*4, kernel_size=(3, 3), strides=(1, 1),
                                              padding=(1, 1),
                                              weight_initializer=mx.init.Xavier(magnitude=2),
                                              bias_initializer='zeros'))

        self.bbox_decoder = BoxDecoder()
        self.cls_decoder = MultiPerClassDecoder(num_class=self.num_classes+1)

    def hybrid_forward(self, F, x):
        features = self.features(x)
        
        cls_preds, loc_preds, anchors = [], [], []
        for feature, cls_predictor, loc_predictor, anchor_generator in \
                zip(features, self.cls_predictors, self.loc_predictors, self.anchor_generators):
            cls_pred = cls_predictor(feature)
            cls_preds.append(cls_pred)
            loc_pred = loc_predictor(feature)
            loc_preds.append(loc_pred)
            anchor = anchor_generator(feature)
            anchors.append(anchor)

        cls_preds = [F.flatten(F.transpose(cp, (0, 2, 3, 1))) for cp in cls_preds]
        cls_preds = F.concat(*cls_preds, dim=1).reshape((0, -1, self.num_classes+1))
        
        loc_preds = [F.flatten(F.transpose(lp, (0, 2, 3, 1))) for lp in loc_preds]
        loc_preds = F.concat(*loc_preds, dim=1).reshape((0, -1, 4))

        anchors = F.concat(*anchors, dim=1).reshape((0, -1, 4))
        
        if autograd.is_training():
            return cls_preds, loc_preds, anchors

        # bboxes: (B, N, 4)
        bboxes = self.bbox_decoder(loc_preds, anchors)
        # cls_ids: (B, N, num_classes), scores: (B, N, num_classes)
        cls_ids, scores = self.cls_decoder(F.softmax(cls_preds, axis=-1))

        result = []
        for i in range(self.num_classes):
            cls_id = F.slice_axis(cls_ids, axis=-1, begin=i, end=i+1)  # (B, N ,1)
            score = F.slice_axis(scores, axis=-1, begin=i, end=i+1)  # (B, N, 1)
            cls_result = F.concat(*[cls_id, score, bboxes], dim=-1)  # (B, N, 6)
            result.append(cls_result)
        result = F.concat(*result, dim=1)
       
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


