import numpy as np
import mxnet as mx
from mxnet import nd
from mxnet.gluon import nn

class Center2Corner(nn.HybridBlock):
    def __init__(self, axis=-1, **kwargs):
        super(Center2Corner, self).__init__(**kwargs)
        self.axis = -1

    def hybrid_forward(self, F, x):
        # (B, N, 4) ==> 4*(B, N, 1)
        cx, cy, w, h = F.split(x, num_outputs=4, axis=self.axis)
        hw, hh = w/2, h/2
        xmin = cx - hw
        xmax = cx + hw
        ymin = cy - hh
        ymax = cy + hh

        return F.concat(xmin, ymin, xmax, ymax, dim=self.axis)


class Corner2Center(nn.HybridBlock):
    def __init__(self, axis=-1, split=False, **kwargs):
        super(Corner2Center, self).__init__(**kwargs)
        self.axis = -1
        self.split = split

    def hybrid_forward(self, F, x):
        # (B, N, 4) ==> (B, N, 1)
        xmin, ymin, xmax, ymax = F.split(x, num_outputs=4, axis=self.axis)
        cx, cy = (xmax + xmin)/2, (ymax + ymin)/2
        w, h = (xmax - xmin), (ymax - ymin)

        if self.split:
            return cx, cy, w, h
        else:
            return F.concat(cx, cy, w, h, dim=self.axis)


class BipartiteMatcher(nn.HybridBlock):
    def __init__(self, threshold=1e-12, is_ascend=False, **kwargs):
        super(BipartiteMatcher, self).__init__(**kwargs)
        self.threshold = threshold
        self.is_ascend = is_ascend
        self.eps = 1e-12

    def hybrid_forward(self, F, x):
        # (B, N, M) ==> [(B, N), (B, M)]
        match = F.contrib.bipartite_matching(x, threshold=self.threshold,
                                             is_ascend=self.is_ascend)
        # anchor may have same iou with more than 1 gtbox
        # avoid case like: 
        # [[x1,  x2],
        #  [max, x3],
        #  [max, x3]]
        anchor_argmax = F.argmax(x, axis=-1, keepdims=True)  # (B, N, 1)
        anchor_max = F.pick(x, anchor_argmax, keepdims=True)  # (B, N, 1)
        maxs = F.max(x, axis=-2, keepdims=True)  # (B, 1, M)
        mask = F.broadcast_greater_equal(anchor_max + self.eps, maxs)  # (B, N, M)
        mask = F.pick(mask, anchor_argmax, keepdims=True)  # (B, N, 1)
        new_mask = F.where(mask>0, anchor_argmax, F.ones_like(anchor_argmax)*-1)  # (B, N, 1)
        result = F.where(match[0] < 0, new_mask.squeeze(axis=-1), match[0])  # (B, N)
        return result


class MaximumMatcher(nn.HybridBlock):
    def __init__(self, threshold=0.5, **kwargs):
        super(MaximumMatcher, self).__init__(**kwargs)
        self.threshold = threshold

    def hybrid_forward(self, F, x):
        anchor_argmax = F.argmax(x, axis=-1, keepdims=True)  # (B, N, 1)
        anchor_max = F.pick(x, anchor_argmax, axis=-1, keepdims=True)  # (B, N, 1)
        match = F.where(anchor_max >= self.threshold, anchor_argmax, F.ones_like(anchor_argmax)*-1)  # (B, N, 1)
        match = F.squeeze(match, axis=-1)  # (B, N)
        return match


class NaiveSampler(nn.HybridBlock):
    def __init__(self, **kwargs):
        super(NaiveSampler, self).__init__(**kwargs)

    def hybrid_forward(self, F, x):
        marker = F.ones_like(x)
        samples = F.where(x >= 0, marker, marker*-1)
        return samples


class ClassEncoder(nn.HybridBlock):
    def __init__(self, ignore_label=-1, **kwargs):
        super(ClassEncoder, self).__init__()
        self.ignore_label = ignore_label

    def hybrid_forward(self, F, samples, matches, gt_ids):
        # samples [B, N], 1: pos, -1: neg, 0: ignore
        # matches [B, N]
        # gt_ids: [B, M]
        gt_ids = F.reshape(gt_ids, (0, 1, -1))  # (B, 1, M)
        gt_ids = F.repeat(gt_ids, repeats=matches.shape[1], axis=1)  # (B, N, M) 
        target_ids = F.pick(gt_ids, matches, axis=-1) + 1  # wrong when match==-1, fixed by sample
        # filter invalid target_id
        # for 0 sample, ignore
        target_ids = F.where(samples > 0.5, target_ids, F.ones_like(target_ids)*self.ignore_label)
        # for -1 negative sample, set as background
        target_ids = F.where(samples < -0.5, F.zeros_like(target_ids), target_ids)

        return target_ids


class BoxEncoder(nn.Block):
    def __init__(self, stds=(0.1, 0.1, 0.2, 0.2), means=(0., 0., 0., 0.), **kwargs):
        super(BoxEncoder, self).__init__()
        self._stds = stds
        self._means = means
        self.corner_to_center = Corner2Center(split=True)

    def forward(self, samples, matches, anchors, gt_boxes):
        # samples [B, N], 1: pos, -1: neg, 0: ignore
        # matches [B, N]
        # anchors [N, 4]
        # gt_boxes [B, M, 4]
        F = nd
        ref_boxes = F.reshape(gt_boxes, (0, 1, -1, 4))  # (B, 1, M, 4)
        ref_boxes = F.repeat(ref_boxes, repeats=matches.shape[1], axis=1)  # (B, N, M, 4)
        ref_boxes = F.split(ref_boxes, num_outputs=4, axis=-1, squeeze_axis=True)  # 4*(B, N, M)
        # (B, N, 1) ==> (B, N, 4)
        ref_boxes = F.concat(*[F.pick(ref_boxes[i], matches, axis=2, keepdims=True) for i in range(4)], dim=-1)

        g = self.corner_to_center(ref_boxes)
        a = self.corner_to_center(anchors.expand_dims(axis=0))
        
        t0 = ((g[0] - a[0]) / a[2] - self._means[0]) / self._stds[0]
        t1 = ((g[1] - a[1]) / a[3] - self._means[1]) / self._stds[1]
        t2 = (F.log(g[2] / a[2]) - self._means[2]) / self._stds[2]
        t3 = (F.log(g[3] / a[3]) - self._means[3]) / self._stds[3]
        
        codecs = F.concat(t0, t1, t2, t3, dim=2)  # (B, N, 4)
        
        # samples [B, N] -> [B, N, 1] -> [B, N, 4] -> boolean
        temp = F.tile(samples.reshape((0, -1, 1)), reps=(1, 1, 4)) > 0.5
        # fill targets and masks [B, N, 4]
        targets = F.where(temp, codecs, F.zeros_like(codecs))
        masks = F.where(temp, F.ones_like(temp), F.zeros_like(temp))
        
        return targets, masks


class BoxDecoder(nn.HybridBlock):
    def __init__(self, stds=(0.1, 0.1, 0.2, 0.2), means=(0., 0., 0., 0.), **kwargs):
        super(BoxDecoder, self).__init__(**kwargs)
        self.stds = stds
        self.means = means

    def hybrid_forward(self, F, x, anchors):
        a = F.split(anchors, axis=-1, num_outputs=4)
        p = F.split(x, axis=-1, num_outputs=4)
        ox = F.broadcast_add(F.broadcast_mul(p[0] * self.stds[0] + self.means[0], a[2]), a[0])
        oy = F.broadcast_add(F.broadcast_mul(p[1] * self.stds[1] + self.means[1], a[3]), a[1])
        tw = F.broadcast_mul(F.exp(p[2] * self.stds[2] + self.means[2]), a[2])
        th = F.broadcast_mul(F.exp(p[3] * self.stds[3] + self.means[3]), a[3])

        xmin = ox - tw/2
        ymin = oy - th/2
        xmax = ox + tw/2
        ymax = oy + th/2

        return F.concat(xmin, ymin, xmax, ymax, dim=-1)

class ClassDecoder(nn.HybridBlock):
    def __init__(self, thresh=0.01, **kwargs):
        super(ClassDecoder, self).__init__(**kwargs)
        self.thresh = thresh

    def hybrid_forward(self, F, x):
        pos_x = x.slice_axis(axis=-1, begin=1, end=None)
        cls_id = F.argmax(pos_x, axis=-1)
        scores = F.pick(pos_x, cls_id)
        mask = scores > self.thresh
        cls_id = F.where(mask, cls_id, F.ones_like(cls_id)*-1)
        scores = F.where(mask, scores, F.zeros_like(scores))
        return cls_id, scores

class MultiPerClassDecoder(nn.HybridBlock):
    def __init__(self, num_class, axis=-1, thresh=0.01):
        super(MultiPerClassDecoder, self).__init__()
        self._fg_class = num_class - 1
        self._axis = axis
        self._thresh = thresh

    def hybrid_forward(self, F, x):
        scores = x.slice_axis(axis=self._axis, begin=1, end=None)  # b x N x fg_class
        template = F.zeros_like(x.slice_axis(axis=-1, begin=0, end=1))
        cls_ids = []
        for i in range(self._fg_class):
            cls_ids.append(template + i)  # b x N x 1
        cls_id = F.concat(*cls_ids, dim=-1)  # b x N x fg_class
        mask = scores > self._thresh
        cls_id = F.where(mask, cls_id, F.ones_like(cls_id) * -1)
        scores = F.where(mask, scores, F.zeros_like(scores))
        return cls_id, scores

class TargetGenerator(nn.Block):
    def __init__(self, iou_threshold=0.5, **kwargs):
        super(TargetGenerator, self).__init__(**kwargs)
        self.center_to_corner = Center2Corner()
        self.bip_matcher = BipartiteMatcher()
        self.max_matcher = MaximumMatcher(threshold=iou_threshold)
        self.sampler = NaiveSampler()

        self.class_encoder = ClassEncoder()
        self.box_encoder = BoxEncoder()

    def forward(self, anchors, gt_boxes, gt_ids):
        # anchors: (N, 4)
        # gt_boxes: (B, M, 4)
        # gt_ids: (B, M, 1)
        assert len(gt_boxes.shape) == len(gt_ids.shape) == 3,\
            'Expected dims of input gt_boxes and gt_ids to be 3.'
        anchors = self.center_to_corner(anchors)  # (N, 4)
        # (N, 4) + (B, M, 4) ==> (N, B, M)
        box_ious = nd.contrib.box_iou(anchors, gt_boxes)
        box_ious = nd.transpose(box_ious, (1, 0, 2)) # => (B, N, M)

        bip_match = self.bip_matcher(box_ious)  # (B, N)
        max_match = self.max_matcher(box_ious)  # (B, N)

        matches = nd.where(bip_match >= 0, bip_match, max_match)  # (B, N)

        samples = self.sampler(matches)  # (B, N)

        class_targets = self.class_encoder(samples, matches, gt_ids)
        box_targets, box_masks = self.box_encoder(samples, matches, anchors, gt_boxes)

        return class_targets, box_targets, box_masks


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

