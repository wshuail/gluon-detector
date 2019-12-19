import argparse
import os
import sys
import yaml
import logging
import warnings
import time
import numpy as np
sys.path.insert(0, os.path.expanduser('~/lib/incubator-mxnet/python'))
import mxnet as mx
from mxnet import nd
from mxnet import gluon
from mxnet import autograd
from mxnet.contrib import amp
sys.path.insert(0, os.path.expanduser('~/gluon_detector'))
from lib.utils.lr_scheduler import LRSequential, LRScheduler
from lib.loss import HeatmapFocalLoss, MaskedL1Loss
from lib.modelzoo import get_model
from lib.data.mscoco.centernet import CenterNetTrainPipeline
from lib.data.mscoco.centernet import CenterNetTrainLoader
from lib.data.mscoco.detection import ValPipeline
from lib.data.mscoco.detection import ValLoader
from lib.metrics.coco_detection import COCODetectionMetric
from lib.utils.export_helper import export_block


class CenterNetSolver(object):
    def __init__(self, backbone, dataset, input_size, gpu_batch_size, optimizer,
                 lr, wd, resume_epoch=0, train_split='train2017', val_split='val2017',
                 use_amp=False, gpus='0,1,2,3', save_frequent=5, save_dir='~/gluon_detector/output'):
        self.backbone = backbone
        self.dataset = dataset
        self.input_size = input_size
        self.thread_batch_size = gpu_batch_size
        self.train_split = train_split
        self.val_split = val_split
        self.optimizer = optimizer
        self.lr = lr
        self.wd = wd
        self.resume_epoch = resume_epoch
        self.use_amp = use_amp
        self.ctx = [mx.gpu(int(i)) for i in gpus.split(',') if i.strip()]
        self.num_devices = len(self.ctx)
        self.batch_size = gpu_batch_size * self.num_devices
        self.save_frequent = save_frequent
        self.save_dir = os.path.expanduser(save_dir)
        
        prefix = 'retinanet_{}_{}_{}x{}'.format(dataset, backbone, input_size, input_size)
        self.save_prefix = os.path.join(self.save_dir, prefix)

        self.net = self.build_net()
        
        self.train_data, self.val_data = self.get_dataloader()
        
        num_example = self.train_data.size()
        logging.info('number of example in training set is {}'.format(num_example))
        self.lr_scheduler, self.epoch = self.get_lr_scheduler(self.lr, num_example,
                                                              self.batch_size,
                                                              lr_schd='1x')
        
        eval_file = 'centernet_{}_{}_{}x{}_eval'.format(dataset, backbone, input_size, input_size) 
        eval_path = os.path.join(self.save_dir, eval_file)
        self.eval_metric = COCODetectionMetric(dataset=self.val_split, save_prefix=eval_path,
                                               use_time=False, cleanup=True,
                                               data_shape=(self.input_size, self.input_size))

        logging.info('RetinaNetSolver initialized')
    
    def build_net(self):
        model_name = 'centernet_{}_{}'.format(self.backbone, self.dataset)
        net = get_model(model_name)
        if self.resume_epoch>0:
            resume_params_file = '{}-{:04d}.params'.format(self.save_prefix, self.resume_epoch)
            net.load_parameters(resume_params_file)
            logging.info('Resume training from epoch {}'.format(self.resume_epoch))
        else:
            logging.info('Start training from scratch...')

        return net

    def get_dataloader(self):
        logging.info('getting data loader.')
        num_classes = 80
        train_loader = CenterNetTrainLoader(split=self.train_split,
                                            batch_size=self.thread_batch_size,
                                            num_classes=num_classes,
                                            data_shape=self.input_size,
                                            num_devices=self.num_devices)

        print ("val dataloder")
        val_loader = None
        print ('load dataloder done')

        return train_loader, val_loader
    
    def get_lr_scheduler(self, lr, num_example, batch_size, lr_schd='1x'):
        # lr = 0.005 / 8 * len(self.ctx) * self.thread_batch_size
        iters_per_epoch = num_example // batch_size
        if lr_schd == '1x':
            step_iter = (60000, 80000)
            total_iter = 90000
            epoch = int(np.ceil(total_iter/iters_per_epoch))
        else:
            raise NotImplementedError
        logging.info('Total training epoch {}'.format(epoch))
        lr_scheduler = LRSequential([
            LRScheduler('linear', base_lr=0, target_lr=lr, niters=500),
            LRScheduler('step', base_lr=self.lr, niters=total_iter, step_iter=step_iter)
        ])

        return lr_scheduler, epoch

    def train(self):
        
        self.net.collect_params().reset_ctx(self.ctx)
        
        trainer = gluon.Trainer(
            params=self.net.collect_params(),
            optimizer='sgd',
            optimizer_params={'lr_scheduler': self.lr_scheduler,
                              'wd': self.wd,
                              'momentum': 0.9},
            update_on_kvstore=(False if self.use_amp else None)
        )

        heatmap_loss = HeatmapFocalLoss(from_logits=True)
        wh_loss = MaskedL1Loss(weight=0.1)
        center_reg_loss = MaskedL1Loss(weight=1.0)
        heatmap_loss_metric = mx.metric.Loss('HeatmapFocal')
        wh_metric = mx.metric.Loss('WHL1')
        center_reg_metric = mx.metric.Loss('CenterRegL1')

        logging.info('Start training from scratch...')
        
        for epoch in range(self.epoch):
           
            heatmap_loss_metric.reset()
            wh_metric.reset()
            center_reg_metric.reset()
            
            tic = time.time()
            btic = time.time()
            self.net.collect_params().reset_ctx(self.ctx)
            self.net.hybridize(static_alloc=True, static_shape=True)
            
            for i, data_batch in enumerate(self.train_data):
                print ('i: {}'.format(i))
                
                all_data, all_hm, all_wh_target, all_wh_mask, all_center_reg,\
                    all_center_reg_mask, all_img_ids, origin_gtbox = data_batch

                with autograd.record():
                    sum_losses = []
                    heatmap_losses = []
                    wh_losses = []
                    center_reg_losses = []
                    wh_preds = []
                    for data, heatmap_target, wh_target, wh_mask, center_reg_target, center_reg_mask in \
                            zip(all_data, all_hm, all_wh_target, all_wh_mask, all_center_reg, all_center_reg_mask):
                        outputs = self.net(data)
                        heatmap_pred, wh_pred, center_reg_pred = outputs
                        print ('hm_target: {}'.format(heatmap_target.shape))
                        print ('hm_pred: {}'.format(heatmap_pred.shape))
                        
                        print ('wh_target: {}'.format(wh_target.shape))
                        print ('wh_mask: {}'.format(wh_mask.shape))
                        print ('wh_pred: {}'.format(wh_pred.shape))

                        print ('center_reg_target: {}'.format(center_reg_target.shape))
                        print ('center_reg_mask: {}'.format(center_reg_mask.shape))
                        print ('center_reg_pred: {}'.format(center_reg_pred.shape))
                        heatmap_losses.append(heatmap_loss(heatmap_pred, heatmap_target))
        """
                        wh_losses.append(wh_loss(wh_pred, wh_target, wh_mask))
                        center_reg_losses.append(center_reg_loss(center_reg_pred, center_reg_target, center_reg_mask))
                        curr_loss = heatmap_losses[-1]+ wh_losses[-1] + center_reg_losses[-1]
                        sum_losses.append(curr_loss)
                    autograd.backward(sum_losses)
                trainer.step(len(sum_losses))  # step with # gpus
                heatmap_loss_metric.update(0, heatmap_losses)
                wh_metric.update(0, wh_losses)
                center_reg_metric.update(0, center_reg_losses)
                if i > 0 and (i + 1) % 50 == 0:
                    name2, loss2 = wh_metric.get()
                    name3, loss3 = center_reg_metric.get()
                    name4, loss4 = heatmap_loss_metric.get()
                    logging.info('[Epoch {}][Batch {}], Speed: {:.3f} samples/sec, LR={}, {}={:.3f}, {}={:.3f}, {}={:.3f}'.format(epoch, i, batch_size/(time.time()-btic), trainer.learning_rate, name2, loss2, name3, loss3, name4, loss4))
                btic = time.time()

            name2, loss2 = wh_metric.get()
            name3, loss3 = center_reg_metric.get()
            name4, loss4 = heatmap_loss_metric.get()
            logging.info('[Epoch {}] Training cost: {:.3f}, {}={:.3f}, {}={:.3f}, {}={:.3f}'.format(
                epoch, (time.time()-tic), name2, loss2, name3, loss3, name4, loss4))
        """


    def validation(self):
        self.eval_metric.reset()
        # set nms threshold and topk constraint
        # net.set_nms(nms_thresh=0.45, nms_topk=400)
        self.net.hybridize(static_alloc=True, static_shape=True)
        for (batch, img_ids) in self.val_data:
            data = [d.data[0] for d in batch]
            label = [d.label[0] for d in batch]
            
            det_bboxes = []
            det_ids = []
            det_scores = []
            gt_bboxes = []
            gt_ids = []
            gt_difficults = []
            for x, y in zip(data, label):
                # get prediction results
                ids, scores, bboxes = self.net(x)
                det_ids.append(ids)
                det_scores.append(scores)
                # clip to image size
                det_bboxes.append(bboxes.clip(0, x.shape[2]))
                # split ground truths
                gt_ids.append(y.slice_axis(axis=-1, begin=4, end=5))
                gt_bboxes.append(y.slice_axis(axis=-1, begin=0, end=4))
                gt_difficults.append(y.slice_axis(axis=-1, begin=5, end=6) if y.shape[-1] > 5 else None)

            # update metric
            self.eval_metric.update(det_bboxes, det_ids, det_scores, img_ids, gt_bboxes, gt_ids, gt_difficults)
        return self.eval_metric.get()



