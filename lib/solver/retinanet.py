import argparse
import os
import sys
import yaml
import logging
import warnings
import time
import datetime
import numpy as np
sys.path.insert(0, os.path.expanduser('~/lib/incubator-mxnet/python'))
import mxnet as mx
from mxnet import nd
from mxnet import gluon
from mxnet import autograd
from mxnet.contrib import amp
sys.path.insert(0, os.path.expanduser('~/gluon_detector'))
from lib.utils.logger import build_logger
from lib.utils.export_helper import export_block
from lib.utils.lr_scheduler import LRSequential, LRScheduler
from lib.loss import FocalLoss, HuberLoss
from lib.modelzoo import get_model
from lib.anchor.retinanet import generate_retinanet_anchors
from lib.metrics.coco_detection import RetinaNetCOCODetectionMetric
from lib.data.mscoco.retina.train import RetinaNetTrainLoader
from lib.data.mscoco.retina.val import RetinaNetValLoader
from lib.data.mscoco.retina.code import decode_retinanet_result


class RetinaNetSolver(object):
    def __init__(self, backbone, dataset, max_size, resize_shorter, gpu_batch_size, optimizer,
                 lr, wd, resume_epoch=0, train_split='train2017', val_split='val2017',
                 use_amp=False, gpus='0,1,2,3', save_frequent=5, save_dir='~/gluon_detector/output'):
        self.backbone = backbone
        self.dataset = dataset
        self.max_size = max_size
        self.resize_shorter = resize_shorter
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
        
        prefix = 'retinanet_{}_{}_{}x{}'.format(dataset, backbone, max_size, resize_shorter)
        self.save_prefix = os.path.join(self.save_dir, prefix)

        self.net = self.build_net()
        
        self.train_data, self.val_data = self.get_dataloader()
        
        num_example = self.train_data.size()
        logging.info('number of example in training set is {}'.format(num_example))
        self.lr_scheduler, self.epoch = self.get_lr_scheduler(self.lr, num_example,
                                                              self.batch_size,
                                                              lr_schd='1x')
        
        eval_file = 'retinanet_{}_{}_{}x{}_eval'.format(dataset, backbone, max_size, resize_shorter) 
        eval_path = os.path.join(self.save_dir, eval_file)
        self.eval_metric = RetinaNetCOCODetectionMetric(dataset=val_split, save_prefix=eval_path)

        if self.use_amp:
            amp.init()

        logging.info('RetinaNetSolver initialized')


    def build_net(self):
        model_name = 'retinanet_{}_{}'.format(self.backbone, self.dataset)
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
        train_loader = RetinaNetTrainLoader(split=self.train_split,
                                            thread_batch_size=self.thread_batch_size,
                                            max_size=self.max_size,
                                            resize_shorter=self.resize_shorter,
                                            num_devices=self.num_devices,
                                            fix_shape=False)
        print ("train dataloder done")
        
        val_loader = RetinaNetValLoader(split=self.val_split,
                                        thread_batch_size=self.thread_batch_size,
                                        max_size=self.max_size,
                                        resize_shorter=self.resize_shorter,
                                        num_devices=self.num_devices,
                                        fix_shape=False)
        print ("val dataloder done")

        return train_loader, val_loader
    
    def get_lr_scheduler(self, lr, num_example, batch_size, lr_schd='1x'):
        # lr = 0.005 / 8 * len(self.ctx) * self.thread_batch_size
        iters_per_epoch = num_example // batch_size
        if lr_schd == '1x':
            step_factor = 16 // (len(self.ctx) * self.thread_batch_size)
            step_iter = (60000*step_factor, 80000*step_factor)
            total_iter = 90000*step_factor
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

        if self.use_amp:
            amp.init_trainer(trainer)
        
        cls_criterion = FocalLoss(num_class=80)
        box_criterion = HuberLoss(rho=0.11)
        cls_metric = mx.metric.Loss('FocalLoss')
        box_metric = mx.metric.Loss('SmoothL1')

        
        for epoch in range(self.epoch):
            cls_metric.reset()
            box_metric.reset()
            tic = time.time()
            btic = time.time()
            # reset cause save params may change
            self.net.collect_params().reset_ctx(self.ctx)
            self.net.hybridize(static_alloc=True, static_shape=True)
            for i, batch in enumerate(iter(self.train_data)):
                data, box_targets, cls_targets, masks, _, _, _ = batch
                with autograd.record():
                    cls_preds = []
                    box_preds = []
                    for x in data:
                        cls_pred, box_pred = self.net(x)
                        cls_preds.append(cls_pred)
                        box_preds.append(box_pred)
                    cls_loss = [cls_criterion(cls_pred, cls_target, mask) for cls_pred, cls_target, mask in
                                zip(cls_preds, cls_targets, masks)]
                    box_loss = [box_criterion(box_pred, box_target, mask) for box_pred, box_target, mask in
                                zip(box_preds, box_targets, masks)]
                    sum_loss = [(cl+bl) for cl, bl in zip(cls_loss, box_loss)]
                    
                    if self.use_amp:
                        with amp.scale_loss(sum_loss, trainer) as scaled_loss:
                            autograd.backward(scaled_loss)
                    else:
                        autograd.backward(sum_loss)
                # since we have already normalized the loss, we don't want to normalize
                # by batch-size anymore
                trainer.step(1)
                cls_metric.update(0, [l * self.batch_size for l in cls_loss])
                box_metric.update(0, [l * self.batch_size for l in box_loss])
                if i > 0 and i % 50 == 0:
                    name1, loss1 = cls_metric.get()
                    name2, loss2 = box_metric.get()
                    speed = self.batch_size/(time.time()-btic)
                    logging.info('Epoch {} Batch {} Speed: {:.3f} samples/s, {}={:.5f}, {}={:.5f}, lr={:.7f}'.\
                                 format(epoch+1, i, speed, name1, loss1, name2, loss2, trainer.learning_rate))
            
                btic = time.time()
            logging.info('[Epoch {}] time cost {:.3f}s'.format(epoch+1, time.time() - tic))
            self.save_params(epoch+1)
            self.validation()


    def validation(self):
        logging.info('Starting Validation.')
        self.eval_metric.reset()
        self.net.collect_params().reset_ctx(self.ctx)
        self.net.hybridize(static_alloc=True, static_shape=True)
        
        count = 0
        for i, data_batch in enumerate(self.val_data):
            batch_data, batch_labels, batch_resize_attrs, batch_img_ids = data_batch
            
            det_bboxes = []
            det_ids = []
            det_scores = []
            gt_bboxes = []
            gt_ids = []
            
            for thread_data, thread_labels, thread_img_ids in zip(batch_data, batch_labels, batch_img_ids):
                for image, labels, image_id in zip(thread_data, thread_labels, thread_img_ids):
                    count += 1
                    _, c, h, w = image.shape
                    cls_preds, box_preds = self.net(image)
                    
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

                    if count % 1000 == 0:
                        logging.info('Validation {} images processed.'.format(count))
            
            self.eval_metric.update(det_bboxes, det_ids, det_scores, batch_resize_attrs, batch_img_ids, gt_bboxes, gt_ids)

        map_name, mean_ap = self.eval_metric.get()
        val_msg = '\n'.join(['{}={}'.format(k, v) for k, v in zip(map_name, mean_ap)])
        logging.info('Validation Result: \n{}'.format(val_msg))
        
    def get_logger(self, save_prefix):
        timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        log_path = '{}_train_{}.log'.format(save_prefix, timestamp) 
        build_logger(log_path)

    def save_params(self, epoch):
        if epoch % self.save_frequent == 0:
            epoch += self.resume_epoch
            # save parameters
            filename = '{}-{:04d}.params'.format(self.save_prefix, epoch)
            self.net.save_parameters(filename=filename)
            logging.info('[Epoch {}] save checkpoint to {}'.format(epoch, filename))

            # export model
            deploy_prefix = self.save_prefix + '-deploy'
            export_block(path=deploy_prefix,
                         block=self.net,
                         data_shape=None,
                         epoch=epoch,
                         preprocess=False,
                         layout='CHW',
                         ctx=self.ctx[0])
            logging.info('[Epoch {}] export model to {}-{:04d}.params'.format(epoch, deploy_prefix, epoch))


