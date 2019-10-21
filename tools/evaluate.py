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
from lib.modelzoo.retinanet import RetinaNet
from lib.anchor.retinanet import generate_retinanet_anchors
from lib.data.mscoco.retina.val import ValLoader as ValLoader2
from lib.data.mscoco.retina.val import decode_retinanet_result
from lib.data.mscoco.detection import ValPipeline, ValLoader
from lib.metrics.coco_detection import COCODetectionMetric


class Evaluator(object):
    def __init__(self, params_file, split='val2017', max_size=800, resize_shorter=640,
                 gpus='0,1,2,3', thread_batch_size=2, save_prefix='~/gluon_detector/output',
                 nms_thresh=0.45, nms_topk=400, post_nms=100,
                 stds=(0.1, 0.1, 0.2, 0.2), means=(0., 0., 0., 0.), **kwargs):
        self.nms_thresh = nms_thresh
        self.nms_topk = nms_topk
        self.post_nms = post_nms
        self.stds = stds
        self.means = means
        gpu_ids = [int(gpu_id) for gpu_id in gpus.split(',')]
        self.ctx = [mx.gpu(gpu_id) for gpu_id in gpu_ids]
        num_devices = len(self.ctx)
        """
        symbol_file = params_file[:params_file.rfind('-')] + "-symbol.json"
        self.net = SymbolBlock.imports(symbol_file, ['data'], params_file, ctx=self.ctx)
        """
        
        layers = ['stage2_activation3', 'stage3_activation5', 'stage4_activation2']
        anchors = generate_retinanet_anchors((512, 512))
        self.net = RetinaNet('resnet50_v1', layers, num_class=80, anchors=anchors)
        self.net.load_parameters(params_file)
        self.net.collect_params().reset_ctx(self.ctx)
        
        # self.net.hybridize()
        logging.info('network initilized.')
        
        self.val_loader = ValLoader2(split=split,
                                    thread_batch_size=thread_batch_size,
                                    max_size=max_size,
                                    resize_shorter=resize_shorter,
                                    num_devices=num_devices,
                                    fix_shape=True)
        logging.info('dataloader initilized.')

        log_file = 'retinanet_coco_eval'
        log_path = os.path.expanduser(os.path.join(save_prefix, log_file))
        val_metric = COCODetectionMetric(dataset=split,
                                         save_prefix=log_path,
                                         use_time=True,
                                         cleanup=False,
                                         data_shape=(512, 512))
        self.eval_metric = val_metric
        logging.info('val_metric initilized.')
 
    def validation(self):
        self.eval_metric.reset()
        self.net.hybridize(static_alloc=True, static_shape=True)
        
        image_count = 0
        
        for i, data_batch in enumerate(self.val_loader):
            batch_data, batch_labels, batch_img_ids = data_batch
            
            det_bboxes = []
            det_ids = []
            det_scores = []
            gt_bboxes = []
            gt_ids = []
            gt_difficults = []
            
            for thread_data, thread_labels, thread_img_ids in zip(batch_data, batch_labels, batch_img_ids):
                for image, labels, image_id in zip(thread_data, thread_labels, thread_img_ids):
                    # with autograd.record():
                    image_count += image.shape[0]
                    _, c, h, w = image.shape
                    cls_preds, box_preds, _ = self.net(image)
                    
                    anchors = generate_retinanet_anchors((h, w))
                    anchors = nd.reshape(anchors, (1, -1, 4))
                    anchors = anchors.as_in_context(image.context)
        
                    ids, scores, bboxes = decode_retinanet_result(box_preds, cls_preds, anchors)
                    det_ids.append(ids)
                    det_scores.append(scores)
                    # clip to image size
                    det_bboxes.append(bboxes)
                    # det_bboxes.append(bboxes.clip(0, image.shape[2]))
                    # split ground truths
                    gt_ids.append(labels.slice_axis(axis=-1, begin=4, end=5))
                    gt_bboxes.append(labels.slice_axis(axis=-1, begin=0, end=4))
                    gt_difficults.append(labels.slice_axis(axis=-1, begin=5, end=6) if labels.shape[-1] > 5 else None)
            
            self.eval_metric.update(det_bboxes, det_ids, det_scores, batch_img_ids, gt_bboxes, gt_ids, gt_difficults)

        logging.info('image_count: {}'.format(image_count))
        map_name, mean_ap = self.eval_metric.get()
        val_msg = '\n'.join(['{}={}'.format(k, v) for k, v in zip(map_name, mean_ap)])
        logging.info('Validation: \n{}'.format(val_msg))


if __name__ == '__main__':
    # params_file = '/home/wangshuailong/gluon_detector/output/retinanet_coco_resnet50_v1_512x512-deploy-0000.params'
    params_file = '/home/wangshuailong/gluon_detector/output/retinanet_coco_resnet50_v1_512x512-1000.params'
    evaluator = Evaluator(params_file)
    evaluator.validation()


