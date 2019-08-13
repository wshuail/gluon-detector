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
from lib.centernet_loss import FocalLoss
from lib.modelzoo.centernet import CenterNet
from lib.data.mscoco.centernet import CenterNetTrainPipeline
from lib.data.mscoco.centernet import CenterNetTrainLoader
from lib.data.mscoco.detection import ValPipeline
from lib.data.mscoco.detection import ValLoader
from lib.metrics.coco_detection import COCODetectionMetric
from lib.utils.export_helper import export_block
from .base import BaseSolver


class CenterNetSolver(BaseSolver):
    def __init__(self, config):
        super(CenterNetSolver, self).__init__(config=config)
    
    def build_net(self):
        config = self.config
        network = config['network']
        layers = config['layers']
        heads = config['heads']
        head_conv = config['head_conv']
        net = CenterNet(network, layers, heads, head_conv)
    
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            net.initialize()
        logging.info('network initialized done.')

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
        train_pipelines = [CenterNetTrainPipeline(split=train_split,
                                                  batch_size=thread_batch_size,
                                                  data_shape=input_shape[0],
                                                  num_shards=num_devices,
                                                  device_id=i,
                                                  num_workers=16) for i in range(num_devices)]
        epoch_size = train_pipelines[0].size()
        num_classes = config['num_classes']
        train_loader = CenterNetTrainLoader(train_pipelines, epoch_size, thread_batch_size, num_classes, input_shape)

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
            optimizer=config['optimizer'],
            optimizer_params={'learning_rate': config['lr'],
                              'wd': config['wd'],
                              'momentum': config['momentum']},
            update_on_kvstore=(False if config['amp'] else None)
        )

        lr_decay = config.get('lr_decay', 0.1)
        lr_steps = sorted([float(ls) for ls in config['lr_decay_epoch'].split(',') if ls.strip()])

        hm_creteria = FocalLoss()
        offset_creteria = gluon.loss.L1Loss()
        wh_creteria = gluon.loss.L1Loss()
        
        hm_metric = mx.metric.Loss('FocalLoss')
        offset_metric = mx.metric.Loss('Offset_L1')
        wh_metric = mx.metric.Loss('WH_L1')

        logging.info('Start training from scratch...')
        
        for epoch in range(config['epoch']):
            while lr_steps and epoch > lr_steps[0]:
                new_lr = trainer.learning_rate*lr_decay
                lr_steps.pop(0)
                trainer.set_learning_rate(new_lr)
                logging.info("Epoch {} Set learning rate to {}".format(epoch, new_lr))
            
            hm_metric.reset()
            offset_metric.reset()
            wh_metric.reset()
            
            tic = time.time()
            btic = time.time()
            self.net.collect_params().reset_ctx(self.ctx)
            # self.net.hybridize(static_alloc=True, static_shape=True)
            for i, (batch, _, _) in enumerate(self.train_data):
                batch_data = [d.data[0] for d in batch]
                batch_hm = [d.label[0] for d in batch]
                batch_offset = [d.label[1] for d in batch]
                batch_wh = [d.label[2] for d in batch]
                
                with autograd.record():
                    hm_losses, offset_losses, wh_losses, sum_losses = [], [], [], []
                    for data, hm, offset, wh in zip(batch_data, batch_hm, batch_offset, batch_wh):
                        # print ('input data sum: {}'.format(nd.sum(data)))
                        outputs = self.net(data)
                        # hm_pred, offset_pred, wh_pred = outputs
                        hm_pred = outputs[0]
                        # print ('sum hm_pred: {}'.format(nd.sum(hm_pred)))
                        
                        # hm_loss = hm_creteria(hm_pred, hm)
                        hm_loss = offset_creteria(hm_pred, hm)
                        # offset_loss = offset_creteria(offset_pred, offset)
                        # wh_loss = wh_creteria(wh_pred, wh)

                        # sum_loss = hm_loss  #  + offset_loss + 0.1*wh_loss
                        
                        hm_losses.append(hm_loss)
                        # offset_losses.append(offset_loss)
                        # wh_losses.append(wh_loss)
                        # sum_losses.append(sum_loss)

                        autograd.backward(hm_loss)
                    
                    # for sum_loss in sum_losses:
                    #     autograd.backward(sum_loss)
                
                # since we have already normalized the loss, we don't want to normalize
                # by batch-size anymore
                trainer.step(batch_size)

                hm_metric.update(0, [l * batch_size for l in hm_losses])
                # offset_metric.update(0, [l * batch_size for l in offset_losses])
                # wh_metric.update(0, [l * batch_size for l in wh_losses])
                if i > 0 and i % 50 == 0:
                    name1, loss1 = hm_metric.get()
                    # name2, loss2 = offset_metric.get()
                    # name3, loss3 = wh_metric.get()
                    # logging.info('Epoch {} Batch {} Speed: {:.3f} samples/s, {}={:.3f}, {}={:.3f}, {}={:.3f}'.\
                    #        format(epoch, i, batch_size/(time.time()-btic), name1, loss1, name2, loss2, name3, loss3))
                    logging.info('Epoch {} Batch {} Speed: {:.3f} samples/s, {}={:.3f}'.\
                           format(epoch, i, batch_size/(time.time()-btic), name1, loss1))
            
                btic = time.time()
            # map_name, mean_ap = self.validation()
            # val_msg = '\n'.join(['{}={}'.format(k, v) for k, v in zip(map_name, mean_ap)])
            # logging.info('[Epoch {}] Validation: \n{}'.format(epoch, val_msg))
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



