import sys
import mxnet as mx
import numpy as np
from .utils import bbox as tbbox
from .utils import image as timage
from ..target import TargetGenerator

class SSDDefaultTrainTransform(object):
    def __init__(self, width, height, anchors,
                 rgb_mean=(0.485, 0.456, 0.406),
                 rgb_std=(0.229, 0.224, 0.225)):
        self.width = width
        self.height = height
        self.anchors = anchors
        # self.target_shape = target_shape
        self.rgb_mean = rgb_mean
        self.rgb_std = rgb_std

        self._target_generator = TargetGenerator()

    def __call__(self, img, label):
        if isinstance(img, mx.nd.NDArray):
            img = img.asnumpy()

        target_width, target_height = self.width, self.height
        # random color jittering
        # img = experimental.image.random_color_distort(src)

        # random expansion with prob 0.5
        if np.random.uniform(0, 1) > 0.5:
            img, expand = timage.random_expand(img, fill=[m * 255 for m in self.rgb_mean])
            bbox = tbbox.translate(label, x_offset=expand[0], y_offset=expand[1])
        else:
            img, bbox = img, label

        # random cropping
        h, w, _ = img.shape
        bbox, crop = tbbox.random_crop_with_constraints(bbox, (w, h))
        x0, y0, w, h = crop
        img = img[y0: y0+h, x0: x0+w]

        # resize with random interpolation
        h, w, _ = img.shape
        interp = np.random.randint(0, 5)
        img = timage.imresize(img, target_width, target_height, interp=interp)
        bbox = tbbox.resize(bbox, (w, h), (target_width, target_height))

        # random horizontal flip
        h, w, _ = img.shape
        img, flips = timage.random_flip(img, px=0.5)
        bbox = tbbox.flip(bbox, (w, h), flip_x=flips[0])

        # normalize
        img = timage.normalize(img, self.rgb_mean, self.rgb_std)
        
        img = np.transpose(img, axes=(2, 0, 1))
        img = mx.nd.array(img)
        # print ('img shape: {}'.format(img.shape))

        gt_bboxes = mx.nd.array(bbox[np.newaxis, :, :4])
        gt_ids = mx.nd.array(bbox[np.newaxis, :, 4:5])
        cls_targets, box_targets, _ = self._target_generator(
            self.anchors, gt_bboxes, gt_ids)
        return img, cls_targets[0], box_targets[0]



