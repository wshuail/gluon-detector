import os
import sys
import time
import logging
logging.basicConfig(level=logging.INFO)
import numpy as np
sys.path.insert(0, os.path.expanduser('~/incubator-mxnet/python'))
import mxnet as mx
from mxnet import nd
from mxnet import autograd
from mxnet import gluon
from mxnet.gluon.data.vision import transforms
from mxnet.gluon.block import SymbolBlock
sys.path.insert(0, os.path.expanduser('~/gluon_detector'))
from lib.data.mscoco.retina.code import decode_retinanet_result
from lib.anchor.retinanet import generate_retinanet_anchors


__all__ = ['Detector']


class Detector(object):
    def __init__(self, params_file, input_size=320, gpu_id=0,
                 batch_size=4, nms_thresh=None, nms_topk=400,
                 force_suppress=False):
        if isinstance(input_size, int):
            self.width, self.height = input_size, input_size
        elif isinstance(input_size, (list, tuple)):
            self.width, self.height = input_size
        else:
            raise ValueError('Expected int or tuple for input size')
        self.ctx = mx.gpu(gpu_id)
        self.batch_size = batch_size

        self.transform_fn = transforms.Compose([
            # transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
        ])

        symbol_file = params_file[:params_file.rfind('-')] + "-symbol.json"
        # self.net = gluon.nn.SymbolBlock.imports(symbol_file, ['data'], params_file, ctx=self.ctx)
        self.net = SymbolBlock.imports(symbol_file, ['data'], params_file, ctx=self.ctx)
        self.net.hybridize()

    def detect(self, imgs):

        num_example = len(imgs)

        all_detections = []

        t0 = time.time()
        for i in range(0, num_example, self.batch_size):
            batch_raw_imgs = imgs[i: min(i+self.batch_size, num_example)]
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

            print ('batch_img shape: {}'.format(batch_img.shape))
            print ('batch_img sum: {}'.format(batch_img.sum()))
            print ('batch_img mean: {}'.format(batch_img.mean()))
            outputs = self.net(batch_img)
            all_detections.append(outputs)
        
        return all_detections

def resize_short_within(src, short, max_size, mult_base=1, interp=2):
    from mxnet.image.image import _get_interp_method as get_interp
    h, w, _ = src.shape
    im_size_min, im_size_max = (h, w) if w > h else (w, h)
    scale = float(short) / float(im_size_min)
    if np.round(scale * im_size_max / mult_base) * mult_base > max_size:
        # fit in max_size
        scale = float(np.floor(max_size / mult_base) * mult_base) / float(im_size_max)
    new_w, new_h = (int(np.round(w * scale / mult_base) * mult_base),
                    int(np.round(h * scale / mult_base) * mult_base))
    return mx.image.imresize(src, new_w, new_h, interp=get_interp(interp, (h, w, new_h, new_w)))


class RetinaNetDetector(object):
    def __init__(self, params_file, max_size=1024, resize_shorter=640, gpu_id=0,
                 fix_shape=False, batch_size=4, nms_thresh=None, nms_topk=400,
                 force_suppress=False):
        self.max_size = max_size
        self.resize_shorter = resize_shorter
        self.ctx = mx.gpu(gpu_id)
        self.fix_shape = fix_shape
        self.batch_size = batch_size
        self.mean = nd.array([0.485 * 255, 0.456 * 255, 0.406 * 255]).reshape(1, 1, 3)
        self.std = nd.array([0.229 * 255, 0.224 * 255, 0.225 * 255]).reshape(1, 1, 3)

        self.transform_fn = transforms.Compose([
            # transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
        ])

        symbol_file = params_file[:params_file.rfind('-')] + "-symbol.json"
        self.net = SymbolBlock.imports(symbol_file, ['data'], params_file, ctx=self.ctx)
        self.net.hybridize()

    def detect(self, image, conf_thresh=0.5):
        if not isinstance(image, mx.nd.NDArray):
            image = mx.nd.array(image)
        orig_h, orig_w, _ = image.shape

        image = resize_short_within(image, self.resize_shorter, self.max_size)
        resized_h, resized_w, _ = image.shape
        if resized_h >= resized_w:
            image_h, image_w = self.max_size, self.resize_shorter
        else:
            image_h, image_w = self.resize_shorter, self.max_size
        patten = nd.ones((image_h, image_w, 3))*-1
        patten[0: resized_h, 0: resized_w, :] = image
        image = patten
        image = ((image-self.mean)/self.std)
        image = image.as_in_context(self.ctx)
        image = nd.transpose(image, (2, 0, 1))
        image = nd.expand_dims(image, axis=0)
        cls_preds, box_preds = self.net(image)
        anchors = generate_retinanet_anchors((image_h, image_w))
        anchors = nd.reshape(anchors, (1, -1, 4))
        anchors = anchors.as_in_context(cls_preds.context)
        ids, scores, bboxes = decode_retinanet_result(box_preds, cls_preds, anchors)

        cls_ids = ids.asnumpy().squeeze()
        scores = scores.asnumpy().squeeze()
        bboxes = bboxes.asnumpy().squeeze()

        height_scale = float(orig_h/resized_h)
        width_scale = float(orig_w/resized_w)

        bboxes[:, 0] = bboxes[:, 0]*width_scale
        bboxes[:, 1] = bboxes[:, 1]*height_scale
        bboxes[:, 2] = bboxes[:, 2]*width_scale
        bboxes[:, 3] = bboxes[:, 3]*height_scale

        pos_idx = (scores > conf_thresh)
        pos_cls_ids = cls_ids[pos_idx].flatten()
        pos_scores = scores[pos_idx].flatten()
        pos_bboxes = bboxes[pos_idx, :]

        dets = []
        for i in range(pos_cls_ids.shape[0]):
            bbox = pos_bboxes[i, :].tolist()
            cls_id = int(pos_cls_ids[i])
            score = pos_scores[i]
            degree = 0
            
            box_info={}
            box_info['bbox'] = bbox
            box_info['class'] = int(cls_id)
            box_info['score'] = score
            box_info['degree'] = degree
            dets.append(box_info)
        return dets

if __name__ == '__main__':
    import cv2
    params_file = '/home/wangshuailong/gluon_detector/output/retinanet_coco_resnet50_v1_512x512-deploy-0000.params'
    detector = RetinaNetDetector(params_file, fix_shape=False)

    img_path = '/home/wangshuailong/.mxnet/datasets/coco/images/val2017/000000000785.jpg'
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    dets = detector.detect(img)
    for det in dets:
        print (det)







