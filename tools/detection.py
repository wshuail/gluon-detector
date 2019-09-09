from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time
import logging
logging.basicConfig(level=logging.DEBUG)
import numpy as np

import mxnet as mx
from mxnet import gluon
from mxnet.gluon.data.vision import transforms
from mxnet.gluon.block import SymbolBlock


__all__ = ['CenterNetDetector']


class CenterNetDetector(object):
    def __init__(self, params_file, input_size=320,
                 gpu_id=0, nms_thresh=None, nms_topk=400,
                 force_suppress=False):
        if isinstance(input_size, int):
            self.width, self.height = input_size, input_size
        elif isinstance(input_size, (list, tuple)):
            self.width, self.height = input_size
        else:
            raise ValueError('Expected int or tuple for input size')
        self.ctx = mx.gpu(gpu_id)

        self.transform_fn = transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
        ])

        symbol_file = params_file[:params_file.rfind('-')] + "-symbol.json"
        # self.net = gluon.nn.SymbolBlock.imports(symbol_file, ['data'], params_file, ctx=self.ctx)
        self.net = SymbolBlock.imports(symbol_file, ['data'], params_file, ctx=self.ctx)
        self.net.hybridize()

    def detect(self, imgs, conf_thresh=0.5, batch_size=4):

        # self.net.set_nms(nms_thresh=nms_thresh, nms_topk=400)

        num_example = len(imgs)

        all_detections = []

        t0 = time.time()
        for i in range(0, num_example, batch_size):
            batch_raw_imgs = imgs[i: min(i+batch_size, num_example)]
            orig_sizes = []
            batch_img_lst = []
            for img in batch_raw_imgs:
                orig_sizes.append(img.shape)
                if not isinstance(img, mx.nd.NDArray):
                    img = mx.nd.array(img)
                img = self.transform_fn(img)
                batch_img_lst.append(img)
            batch_img = mx.nd.stack(*batch_img_lst)
            batch_img = batch_img.as_in_context(self.ctx)
            mx.nd.waitall()
            t1 = time.time()

            outputs = self.net(batch_img)
            all_detections.append(outputs)
        
        return all_detections




