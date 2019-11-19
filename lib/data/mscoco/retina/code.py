import os
import sys
import logging
logging.basicConfig(level=logging.INFO)
sys.path.insert(0, os.path.expanduser('~/incubator-mxnet/python'))
import mxnet as mx
from mxnet import nd


def encode_box_target2(box_targets, anchors, means=(0., 0., 0., 0.),
                      stds=(0.1, 0.1, 0.2, 0.2)):
    g = nd.split(box_targets, num_outputs=4, axis=-1)
    a = nd.split(anchors, num_outputs=4, axis=-1)
    t0 = ((g[0] - a[0]) / a[2] - means[0]) / stds[0]
    t1 = ((g[1] - a[1]) / a[3] - means[1]) / stds[1]
    t2 = (nd.log(g[2] / a[2]) - means[2]) / stds[2]
    t3 = (nd.log(g[3] / a[3]) - means[3]) / stds[3]
    box_targets = nd.concat(t0, t1, t2, t3, dim=-1)
    return box_targets
        

def decode_box_target2(box_preds, anchors, means=(0., 0., 0., 0.),
                      stds=(0.1, 0.1, 0.2, 0.2)):
    a = nd.split(anchors, axis=-1, num_outputs=4)
    p = nd.split(box_preds, axis=-1, num_outputs=4)
    ox = nd.broadcast_add(nd.broadcast_mul(p[0] * stds[0] + means[0], a[2]), a[0])
    oy = nd.broadcast_add(nd.broadcast_mul(p[1] * stds[1] + means[1], a[3]), a[1])
    tw = nd.broadcast_mul(nd.exp(p[2] * stds[2] + means[2]), a[2])
    th = nd.broadcast_mul(nd.exp(p[3] * stds[3] + means[3]), a[3])

    xmin = ox - tw/2
    ymin = oy - th/2
    xmax = ox + tw/2
    ymax = oy + th/2

    bboxes = nd.concat(xmin, ymin, xmax, ymax, dim=-1)

    return bboxes

def encode_box_target(box_targets, anchors):
    g = nd.split(box_targets, num_outputs=4, axis=-1)
    a = nd.split(anchors, num_outputs=4, axis=-1)
    t0 = (g[0] - a[0]) / a[2]
    t1 = (g[1] - a[1]) / a[3]
    t2 = nd.log(g[2] / a[2])
    t3 = nd.log(g[3] / a[3])
    box_targets = nd.concat(t0, t1, t2, t3, dim=-1)
    return box_targets
        

def decode_box_target(box_preds, anchors):
    a = nd.split(anchors, axis=-1, num_outputs=4)
    p = nd.split(box_preds, axis=-1, num_outputs=4)
    ox = nd.broadcast_add(nd.broadcast_mul(p[0], a[2]), a[0])
    oy = nd.broadcast_add(nd.broadcast_mul(p[1], a[3]), a[1])
    tw = nd.broadcast_mul(nd.exp(p[2]), a[2])
    th = nd.broadcast_mul(nd.exp(p[3]), a[3])

    xmin = ox - tw/2
    ymin = oy - th/2
    xmax = ox + tw/2
    ymax = oy + th/2

    bboxes = nd.concat(xmin, ymin, xmax, ymax, dim=-1)

    return bboxes

def decode_retinanet_result(box_preds, cls_preds, anchors, nms_thresh=0.5,
                            nms_topk=400, post_nms=100):
    
    bboxes = decode_box_target(box_preds, anchors)
       
    cls_ids = nd.argmax(cls_preds, axis=-1, keepdims=True)
    scores = nd.max(cls_preds, axis=-1, keepdims=True)
    scores_mask = (scores > 0.05)
    cls_ids = nd.where(scores_mask, cls_ids, nd.ones_like(cls_ids)*-1)
    scores = nd.where(scores_mask, scores, nd.zeros_like(scores))

    # (B, N, 6)
    result = nd.concat(cls_ids, scores, bboxes, dim=-1)
    # print ('result shape: {}'.format(result.shape))

    if nms_thresh > 0 and nms_thresh < 1:
        result = nd.contrib.box_nms(
            result, overlap_thresh=nms_thresh, topk=nms_topk, valid_thresh=0.01,
            id_index=0, score_index=1, coord_start=2, force_suppress=False)
        if post_nms > 0:
            result = result.slice_axis(axis=1, begin=0, end=post_nms)
    
    ids = nd.slice_axis(result, axis=2, begin=0, end=1)
    scores = nd.slice_axis(result, axis=2, begin=1, end=2)
    bboxes = nd.slice_axis(result, axis=2, begin=2, end=6)

    return ids, scores, bboxes

