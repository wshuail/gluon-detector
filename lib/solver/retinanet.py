import argparse
import os
import sys
import yaml
import logging
import warnings
import time
import datetime
sys.path.insert(0, os.path.expanduser('~/lib/incubator-mxnet/python'))
import mxnet as mx
from mxnet import nd
from mxnet import gluon
from mxnet import autograd
from mxnet.contrib import amp
from nvidia.dali.plugin.mxnet import DALIGenericIterator
sys.path.insert(0, os.path.expanduser('~/gluon_detector'))
from lib.utils.logger import build_logger
from lib.utils.export_helper import export_block
from lib.cus_loss.retinanet import FocalLoss, HuberLoss
from lib.modelzoo.retinanet import RetinaNet
from lib.anchor.retinanet import generate_retinanet_anchors
from lib.data.mscoco.retinanet import RetinaNetTrainPipeline
from lib.data.mscoco.detection import ValPipeline
from lib.data.mscoco.detection import ValLoader
from lib.metrics.coco_detection import COCODetectionMetric


class RetinaNetSolver(object):
    def __init__(self, network, layers, dataset, input_shape, batch_size, optimizer,
                 lr, wd, momentum, epoch, lr_decay, train_split='train2017', val_split='val2017',
                 use_amp=False, gpus='0,1,2,3', save_prefix='~/gluon_detector/output'):
        self.network = network
        self.layers = layers
        self.dataset = dataset

        if isinstance(input_shape, int):
            self.input_shape = (input_shape, input_shape)
        elif isinstance(input_shape, (tuple, list)):
            self.input_shape = input_shape
        else:
            raise TypeError ('Expected input_shape to be either int or tuple, \
                but got {}'.format(type(input_shape)))
        self.width, self.height = self.input_shape

        self.batch_size = batch_size
        self.train_split = train_split
        self.val_split = val_split
        self.optimizer = optimizer
        self.lr = lr
        self.wd = wd
        self.momentum = momentum
        self.epoch = epoch
        self.lr_decay = lr_decay
        self.lr_decay_epoch = ','.join([str(l*epoch) for l in [0.6, 0.8]])

        self.use_amp = use_amp
        
        self.ctx = [mx.gpu(int(i)) for i in gpus.split(',') if i.strip()]

        self.save_prefix = save_prefix

        self.anchors = self.get_anchors()
        self.net = self.build_net()

        self.train_data, self.val_data = self.get_dataloader()
        
        self.eval_metric = self.get_eval_metric()
        
        prefix = 'retinanet_{}_{}_{}x{}'.format(self.dataset, self.network,
                                                self.input_shape[0], self.input_shape[1])
        self.save_prefix = os.path.expanduser(os.path.join(save_prefix, prefix))

        self.get_logger()

        if self.use_amp:
            amp.init()

        self.save_frequent = 10

        logging.info('RetinaNetSolver initialized')

    def build_net(self):
        net = RetinaNet(self.network, self.layers, num_class=80, anchors=self.anchors)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            net.initialize()
        return net

    def get_anchors(self):
        anchors = generate_retinanet_anchors(self.input_shape)
        return anchors

    def get_dataloader(self):
        logging.info('getting data loader.')
        num_devices = len(self.ctx)
        thread_batch_size = self.batch_size // num_devices
        print ("train dataloder")
        train_pipelines = [RetinaNetTrainPipeline(split=self.train_split,
                                            batch_size=thread_batch_size,
                                            data_shape=self.input_shape[0],
                                            num_shards=num_devices,
                                            device_id=i,
                                            anchors=self.anchors,
                                            num_workers=16) for i in range(num_devices)]
        epoch_size = train_pipelines[0].size()
        train_loader = DALIGenericIterator(train_pipelines, [('data', DALIGenericIterator.DATA_TAG),
                                                             ('bboxes', DALIGenericIterator.LABEL_TAG),
                                                             ('label', DALIGenericIterator.LABEL_TAG),
                                                             ('img_ids', DALIGenericIterator.LABEL_TAG)],
                                           epoch_size, auto_reset=True)

        print ("val dataloder")
        val_pipelines = [ValPipeline(split=self.val_split, batch_size=thread_batch_size,
                                     data_shape=self.input_shape[0], num_shards=num_devices,
                                     device_id=i, num_workers=16) for i in range(num_devices)]
        epoch_size = val_pipelines[0].size()
        val_loader = ValLoader(val_pipelines, epoch_size, thread_batch_size, self.input_shape)
        print ('load dataloder done')

        return train_loader, val_loader
    
    def get_eval_metric(self):
        log_file = 'retinanet_{}_{}_{}x{}_eval'.format(self.dataset, self.network,
                                                 self.input_shape[0], self.input_shape[1]) 
        log_path = os.path.expanduser(os.path.join(self.save_prefix, log_file))
        val_metric = COCODetectionMetric(dataset=self.val_split,
                                         save_prefix=log_path,
                                         use_time=False,
                                         cleanup=True,
                                         data_shape=self.input_shape)
        return val_metric
    
    def train(self):

        self.net.collect_params().reset_ctx(self.ctx)
        
        trainer = gluon.Trainer(
            params=self.net.collect_params(),
            optimizer='sgd',
            optimizer_params={'learning_rate': self.lr,
                              'wd': self.wd,
                              'momentum': self.momentum},
            update_on_kvstore=(False if self.use_amp else None)
        )

        if self.use_amp:
            amp.init_trainer(trainer)
        
        lr_decay = self.lr_decay
        lr_steps = sorted([float(ls) for ls in self.lr_decay_epoch.split(',') if ls.strip()])

        cls_criterion = FocalLoss(num_class=80)
        box_criterion = HuberLoss(rho=0.11)
        cls_metric = mx.metric.Loss('FocalLoss')
        box_metric = mx.metric.Loss('SmoothL1')

        logging.info('Start training from scratch...')
        
        for epoch in range(self.epoch):
            while lr_steps and epoch > lr_steps[0]:
                new_lr = trainer.learning_rate*lr_decay
                lr_steps.pop(0)
                trainer.set_learning_rate(new_lr)
                logging.info("Epoch {} Set learning rate to {}".format(epoch, new_lr))
            cls_metric.reset()
            box_metric.reset()
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
                    cls_loss = [cls_criterion(cls_pred, cls_target) for cls_pred, cls_target in
                                zip(cls_preds, cls_targets)]
                    box_loss = [box_criterion(box_pred, box_target) for box_pred, box_target in
                                zip(box_preds, box_targets)]
                    sum_loss = [(cl+bl) for cl, bl in zip(cls_loss, box_loss)]
                    
            """
                    if self.use_amp:
                        with amp.scale_loss(sum_loss, trainer) as scaled_loss:
                            autograd.backward(scaled_loss)
                    else:
                        autograd.backward(sum_loss)
                # since we have already normalized the loss, we don't want to normalize
                # by batch-size anymore
                # trainer.step(self.batch_size)
                trainer.step(1)
                cls_metric.update(0, [l * self.batch_size for l in cls_loss])
                box_metric.update(0, [l * self.batch_size for l in box_loss])
                if i > 0 and i % 50 == 0:
                    name1, loss1 = cls_metric.get()
                    name2, loss2 = box_metric.get()
                    logging.info('Epoch {} Batch {} Speed: {:.3f} samples/s, {}={:.5f}, {}={:.5f}'.\
                           format(epoch, i, self.batch_size/(time.time()-btic), name1, loss1, name2, loss2))
            
                btic = time.time()
            
            logging.info('[Epoch {}] Starting Validation.'.format(epoch))
            map_name, mean_ap = self.validation()
            val_msg = '\n'.join(['{}={}'.format(k, v) for k, v in zip(map_name, mean_ap)])
            logging.info('[Epoch {}] Validation: \n{}'.format(epoch, val_msg))
            self.save_params(epoch)
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
    
    def get_logger(self):
        timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        log_path = '{}_train_{}.log'.format(self.save_prefix, timestamp) 
        # log_path = os.path.expanduser(log_file)
        build_logger(log_path)

    def save_params(self, epoch):
        if epoch % self.save_frequent == 0:
            # save parameters
            # filename = '{}-{:04d}.params'.format(self.output_prefix, model_epoch)
            # self.net.save_parameters(filename=filename)
            # logging.info('[Epoch {}] save checkpoint to {}'.format(epoch, filename))

            # export model
            data_shape = (self.height, self.width, 3)
            deploy_prefix = self.save_prefix + '-deploy'
            export_block(path=deploy_prefix,
                         block=self.net,
                         data_shape=data_shape,
                         epoch=epoch,
                         preprocess=False,
                         layout='CHW',
                         ctx=self.ctx[0])
            logging.info('[Epoch {}] export model to {}-{:04d}.params'.format(epoch, deploy_prefix, epoch))


