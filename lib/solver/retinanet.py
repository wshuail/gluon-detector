import argparse
import os
import sys
import yaml
import logging
import warnings
import time
sys.path.insert(0, os.path.expanduser('~/lib/incubator-mxnet/python'))
import mxnet as mx
from mxnet import nd
from mxnet import gluon
from mxnet import autograd
from mxnet.contrib import amp
from nvidia.dali.plugin.mxnet import DALIGenericIterator
sys.path.insert(0, os.path.expanduser('~/gluon_detector'))
from lib.loss import SSDMultiBoxLoss
from lib.modelzoo.ssd import SSD
from lib.modelzoo.anchor import generate_level_anchors
from lib.nn.anchor import get_anchors
from lib.data.mscoco.ssd import SSDTrainPipeline
from lib.data.mscoco.detection import ValPipeline
from lib.data.mscoco.detection import ValLoader
from lib.metrics.coco_detection import COCODetectionMetric
from .base import BaseSolver


class RetinaNetSolver(BaseSolver):
    def __init__(self, config):
        self.config = config

        self.ctx = [mx.gpu(int(i)) for i in config['gpus'].split(',') if i.strip()]

        self.net = self.build_net()
        self.anchors = self.get_anchors()

        self.train_data, self.val_data = self.get_dataloader()
        
        self.eval_metric = self.get_eval_metric()
        
        self.width, self.height = config['input_shape']
        prefix = '{}_{}_{}_{}x{}'.format(config['model'], config['dataset'],
                                              config['network'], config['input_shape'][0],
                                              config['input_shape'][1]) 
        self.save_prefix = os.path.expanduser(os.path.join(config['save_prefix'], prefix))

        self.get_logger()

        if config['amp']:
            amp.init()

        logging.info('SSDSolver initialized')

    def get_anchors(self):
        image_shape = self.config['input_shape']
        level_anchors_list = []
        for i in range(3, 8):
            level_anchors = generate_level_anchors(i, image_shape)
            level_anchors_list.append(level_anchors)
        anchors = np.concatenate(level_anchors_list, axis=0)

    def build_net(self):
        config = self.config
        network = config['network']
        layers = config['layers']
        num_filters = config['num_filters']
        anchor_sizes = config['anchor_sizes']
        anchor_ratios = config['anchor_ratios']
        steps = config['steps']
        net = SSD(network, layers, num_filters, 80, anchor_sizes, anchor_ratios, steps)
    
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            net.initialize()

        return net

    def get_dataloader(self):
        logging.info('getting data loader.')
        config = self.config
        num_devices = len(self.ctx)
        train_split = config['train_split']
        batch_size = config['batch_size']
        input_shape = config['input_shape']
        thread_batch_size = batch_size // num_devices
        print ("train dataloder")
        train_pipelines = [SSDTrainPipeline(split=train_split,
                                            batch_size=thread_batch_size,
                                            data_shape=input_shape[0],
                                            num_shards=num_devices,
                                            device_id=i,
                                            anchors=self.anchors,
                                            num_workers=16) for i in range(num_devices)]
        epoch_size = train_pipelines[0].size()
        train_loader = DALIGenericIterator(train_pipelines, [('data', DALIGenericIterator.DATA_TAG),
                                                             ('bboxes', DALIGenericIterator.LABEL_TAG),
                                                             ('label', DALIGenericIterator.LABEL_TAG)],
                                           epoch_size, auto_reset=True)

        print ("val dataloder")
        val_split = self.config['val_split']
        val_pipelines = [ValPipeline(split=val_split, batch_size=thread_batch_size,
                                     data_shape=input_shape[0], num_shards=num_devices,
                                     device_id=i, num_workers=16) for i in range(num_devices)]
        epoch_size = val_pipelines[0].size()
        val_loader = ValLoader(val_pipelines, epoch_size, thread_batch_size, input_shape)
        print ('load dataloder done')

        return train_loader, val_loader
    
    def get_eval_metric(self):
        config = self.config
        log_file = '{}_{}_{}_{}x{}_eval'.format(config['model'], config['dataset'], config['network'],
                                                config['input_shape'][0], config['input_shape'][1]) 
        log_path = os.path.expanduser(os.path.join(config['save_prefix'], log_file))

        val_split = self.config['val_split']
        
        val_metric = COCODetectionMetric(dataset=val_split,
                                         save_prefix=log_path,
                                         use_time=False,
                                         cleanup=True,
                                         data_shape=config['input_shape'])
        return val_metric
    
    def train(self):
        config = self.config

        batch_size = config['batch_size']
        
        self.net.collect_params().reset_ctx(self.ctx)
        
        trainer = gluon.Trainer(
            params=self.net.collect_params(),
            optimizer='sgd',
            optimizer_params={'learning_rate': config['lr'],
                              'wd': config['wd'],
                              'momentum': config['momentum']},
            update_on_kvstore=(False if config['amp'] else None)
        )

        if config['amp']:
            amp.init_trainer(trainer)
        
        lr_decay = config.get('lr_decay', 0.1)
        lr_steps = sorted([float(ls) for ls in config['lr_decay_epoch'].split(',') if ls.strip()])

        mbox_loss = SSDMultiBoxLoss()
        ce_metric = mx.metric.Loss('CrossEntropy')
        smoothl1_metric = mx.metric.Loss('SmoothL1')

        logging.info('Start training from scratch...')
        
        for epoch in range(config['epoch']):
            while lr_steps and epoch > lr_steps[0]:
                new_lr = trainer.learning_rate*lr_decay
                lr_steps.pop(0)
                trainer.set_learning_rate(new_lr)
                logging.info("Epoch {} Set learning rate to {}".format(epoch, new_lr))
            ce_metric.reset()
            smoothl1_metric.reset()
            tic = time.time()
            btic = time.time()
            # reset cause save params may change
            self.net.collect_params().reset_ctx(self.ctx)
            self.net.hybridize(static_alloc=True, static_shape=True)
            for i, batch in enumerate(self.train_data):
                data = [d.data[0] for d in batch]
                box_targets = [d.label[0] for d in batch]
                cls_targets = [nd.cast(d.label[1], dtype='float32') for d in batch]
                
                with autograd.record():
                    cls_preds = []
                    box_preds = []
                    for x in data:
                        cls_pred, box_pred, _ = self.net(x)
                        cls_preds.append(cls_pred)
                        box_preds.append(box_pred)
                    sum_loss, cls_loss, box_loss = mbox_loss(
                        cls_preds, box_preds, cls_targets, box_targets)
                    if config['amp']:
                        with amp.scale_loss(sum_loss, trainer) as scaled_loss:
                            autograd.backward(scaled_loss)
                    else:
                        autograd.backward(sum_loss)
                # since we have already normalized the loss, we don't want to normalize
                # by batch-size anymore
                trainer.step(1)
                ce_metric.update(0, [l * batch_size for l in cls_loss])
                smoothl1_metric.update(0, [l * batch_size for l in box_loss])
                if i > 0 and i % 50 == 0:
                    name1, loss1 = ce_metric.get()
                    name2, loss2 = smoothl1_metric.get()
                    logging.info('Epoch {} Batch {} Speed: {:.3f} samples/s, {}={:.3f}, {}={:.3f}'.\
                           format(epoch, i, batch_size/(time.time()-btic), name1, loss1, name2, loss2))
            
                btic = time.time()
            map_name, mean_ap = self.validation()
            val_msg = '\n'.join(['{}={}'.format(k, v) for k, v in zip(map_name, mean_ap)])
            logging.info('[Epoch {}] Validation: \n{}'.format(epoch, val_msg))
            self.save_params(epoch)


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



