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
from mxnet.gluon.block import SymbolBlock
sys.path.insert(0, os.path.expanduser('~/gluon_detector'))
from lib.modelzoo import get_model
from lib.anchor.retinanet import generate_retinanet_anchors
from lib.data.mscoco.retina.val import RetinaNetValLoader
from lib.data.mscoco.retina.code import decode_retinanet_result
from lib.metrics.coco_detection import RetinaNetCOCODetectionMetric


class Evaluator(object):
    def __init__(self, params_file, split='val2017', max_size=1024, resize_shorter=640,
                 gpus='0,1,2,3', thread_batch_size=2, save_prefix='~/gluon_detector/output',
                 nms_thresh=0.45, nms_topk=400, post_nms=100, **kwargs):
        self.nms_thresh = nms_thresh
        self.nms_topk = nms_topk
        self.post_nms = post_nms
        gpu_ids = [int(gpu_id) for gpu_id in gpus.split(',')]
        self.ctx = [mx.gpu(gpu_id) for gpu_id in gpu_ids]
        num_devices = len(self.ctx)
        # symbol_file = params_file[:params_file.rfind('-')] + "-symbol.json"
        # self.net = SymbolBlock.imports(symbol_file, ['data'], params_file, ctx=self.ctx)
        model_name = 'retinanet_resnet50_v1_coco'
        self.net = get_model(model_name)
        self.net.load_parameters(params_file)
        self.net.collect_params().reset_ctx(self.ctx)
        logging.info('network initilized.')
        
        self.val_loader = RetinaNetValLoader(split=split,
                                    thread_batch_size=thread_batch_size,
                                    max_size=max_size,
                                    resize_shorter=resize_shorter,
                                    num_devices=num_devices,
                                    fix_shape=False)
        logging.info('dataloader initilized.')

        log_file = 'retinanet_coco_eval'
        log_path = os.path.expanduser(os.path.join(save_prefix, log_file))
        self.eval_metric = RetinaNetCOCODetectionMetric(dataset=split, save_prefix=log_path, score_thresh=0.3)
        logging.info('eval_metric initilized.')
 
    def validation(self):
        self.eval_metric.reset()
        self.net.hybridize(static_alloc=True, static_shape=True)
        
        image_count = 0
        for i, data_batch in enumerate(self.val_loader):
            batch_data, batch_labels, batch_resize_attrs, batch_img_ids = data_batch
            
            det_bboxes = []
            det_ids = []
            det_scores = []
            gt_bboxes = []
            gt_ids = []
            
            for thread_data, thread_labels, thread_img_ids in \
                    zip(batch_data, batch_labels, batch_img_ids):
                for image, labels, image_id in zip(thread_data, thread_labels, thread_img_ids):
                    image_count += image.shape[0]
                    _, c, h, w = image.shape
                    cls_preds, box_preds = self.net(image)
                    
                    anchors = generate_retinanet_anchors((h, w))
                    anchors = nd.reshape(anchors, (1, -1, 4))
                    anchors = anchors.as_in_context(image.context)
                    ids, scores, bboxes = decode_retinanet_result(box_preds, cls_preds, anchors, nms_thresh=0.5)
                    
                    det_ids.append(ids)
                    det_scores.append(scores)
                    # clip to image size
                    det_bboxes.append(bboxes)
                    # det_bboxes.append(bboxes.clip(0, image.shape[2]))
                    # split ground truths
                    gt_ids.append(labels.slice_axis(axis=-1, begin=4, end=5))
                    gt_bboxes.append(labels.slice_axis(axis=-1, begin=0, end=4))

                    if image_count % 1000 == 0:
                        logging.info('{} images processed'.format(image_count))

            self.eval_metric.update(det_bboxes, det_ids, det_scores, batch_resize_attrs, batch_img_ids, gt_bboxes, gt_ids)

        logging.info('image_count: {}'.format(image_count))
        map_name, mean_ap = self.eval_metric.get()
        val_msg = '\n'.join(['{}={}'.format(k, v) for k, v in zip(map_name, mean_ap)])
        logging.info('Validation: \n{}'.format(val_msg))


if __name__ == '__main__':
    params_file = '/home/wangshuailong/gluon_detector/output/retinanet_coco_resnet50_v1_1024x640-9999.params'
    evaluator = Evaluator(params_file, gpus='0')
    evaluator.validation()


