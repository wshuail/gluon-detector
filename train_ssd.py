"""Train SSD"""
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
from nvidia.dali.plugin.mxnet import DALIGenericIterator
sys.path.insert(0, os.path.expanduser('~/ssd'))
from lib.loss import SSDMultiBoxLoss
from lib.metrics.coco_detection import COCODetectionMetric
from lib.util import get_scales
from lib.util import get_logger
from lib.ssd import SSD
from lib.anchor import get_anchors
from lib.data.mscoco.detection import DALICOCODetection
from lib.data.mscoco.detection import SSDTrainPipeline
from lib.data.mscoco.detection import SSDValPipeline
from lib.data.mscoco.detection import ValLoader


def parse_args():
    parser = argparse.ArgumentParser(description='Train SSD networks.')
    parser.add_argument('--cfg', type=str, default='./config.yaml')
    args = parser.parse_args()
    return args

def get_dataset(config, ctx):
    num_devices = len(ctx)
    
    train_split = config['train_split']
    train_dataset = [DALICOCODetection(train_split, num_devices, shard_id) for shard_id in range(num_devices)] 
    
    val_split = config['val_split']
    val_dataset = [DALICOCODetection(val_split, num_devices, shard_id) for shard_id in range(num_devices)] 
    
    val_metric = COCODetectionMetric(dataset=val_split,
                                     save_prefix=config['save_prefix'] + '_eval',
                                     cleanup=True,
                                     data_shape=config['input_shape'])
    return train_dataset, val_dataset, val_metric

def get_dataloader(config, train_dataset, val_dataset, anchors, input_shape, batch_size, ctx, num_workers=32):
    width, height = input_shape
    num_devices = len(ctx)
    thread_batch_size = batch_size // num_devices
    train_pipelines = [SSDTrainPipeline(device_id=i, batch_size=thread_batch_size,
                                        data_shape=input_shape[0], anchors=anchors,
                                        num_workers=16, dataset_reader=train_dataset[i]) for i in range(num_devices)]
    epoch_size = train_dataset[0].size()
    train_loader = DALIGenericIterator(train_pipelines, [('data', DALIGenericIterator.DATA_TAG),
                                                         ('bboxes', DALIGenericIterator.LABEL_TAG),
                                                         ('label', DALIGenericIterator.LABEL_TAG)],
                                       epoch_size, auto_reset=True)
        
    val_pipelines = [SSDValPipeline(device_id=i, batch_size=thread_batch_size,
                                 data_shape=input_shape[0],
                                 num_workers=16,
                                 dataset_reader=val_dataset[i]) for i in range(num_devices)]
    epoch_size = val_dataset[0].size()
    val_loader = ValLoader(val_pipelines, epoch_size, thread_batch_size, input_shape)
    
    return train_loader, val_loader


def save_params(net, best_map, current_map, epoch, save_interval, prefix):
    current_map = float(current_map)
    if current_map > best_map[0]:
        best_map[0] = current_map
        net.save_params('{:s}_best.params'.format(prefix, epoch, current_map))
        with open(prefix+'_best_map.log', 'a') as f:
            f.write('{:04d}:\t{:.4f}\n'.format(epoch, current_map))
    if save_interval and epoch % save_interval == 0:
        net.save_params('{:s}_{:04d}_{:.4f}.params'.format(prefix, epoch, current_map))

def validate(net, val_data, ctx, eval_metric, config):
    """Test on validation dataset."""
    eval_metric.reset()
    # set nms threshold and topk constraint
    # net.set_nms(nms_thresh=0.45, nms_topk=400)
    net.hybridize(static_alloc=True, static_shape=True)
    for (batch, img_ids) in val_data:
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
            ids, scores, bboxes = net(x)
            det_ids.append(ids)
            det_scores.append(scores)
            # clip to image size
            det_bboxes.append(bboxes.clip(0, x.shape[2]))
            # split ground truths
            gt_ids.append(y.slice_axis(axis=-1, begin=4, end=5))
            gt_bboxes.append(y.slice_axis(axis=-1, begin=0, end=4))
            gt_difficults.append(y.slice_axis(axis=-1, begin=5, end=6) if y.shape[-1] > 5 else None)

        # update metric
        eval_metric.update(det_bboxes, det_ids, det_scores, img_ids, gt_bboxes, gt_ids, gt_difficults)
    return eval_metric.get()


def train(net, train_data, val_data, eval_metric, ctx, config):
    batch_size = config['batch_size']
    
    net.collect_params().reset_ctx(ctx)
    trainer = gluon.Trainer(
        params=net.collect_params(),
        optimizer='sgd',
        optimizer_params={'learning_rate': config['lr'],
                          'wd': config['wd'],
                          'momentum': config['momentum']},
        update_on_kvstore=(False if config['amp'] else None)
    )

    if config['amp']:
        amp.init_trainer(trainer)

    mbox_loss = SSDMultiBoxLoss()
    ce_metric = mx.metric.Loss('CrossEntropy')
    smoothl1_metric = mx.metric.Loss('SmoothL1')

    logging.info('Start training from scratch...')
    
    for epoch in range(config['epoch']):
        ce_metric.reset()
        smoothl1_metric.reset()
        tic = time.time()
        btic = time.time()
        net.hybridize(static_alloc=True, static_shape=True)
        for i, batch in enumerate(train_data):
            if i >= 1000:
                break
            
            data = [d.data[0] for d in batch]
            box_targets = [d.label[0] for d in batch]
            cls_targets = [nd.cast(d.label[1], dtype='float32') for d in batch]
            
            with autograd.record():
                cls_preds = []
                box_preds = []
                for x in data:
                    cls_pred, box_pred, _ = net(x)
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
                print ('Epoch {} Batch {} Speed: {:.3f} samples/s, {}={:.3f}, {}={:.3f}'.\
                       format(epoch, i, batch_size/(time.time()-btic), name1, loss1, name2, loss2))
        
            btic = time.time()
        map_name, mean_ap = validate(net, val_data, ctx, eval_metric, config)
        val_msg = '\n'.join(['{}={}'.format(k, v) for k, v in zip(map_name, mean_ap)])
        print('[Epoch {}] Validation: \n{}'.format(epoch, val_msg))



def build_net(config):

    if config['model'] == 'ssd':
        network = config['network']
        layers = config['layers']
        num_filters = config['num_filters']
        anchor_sizes = config['anchor_sizes']
        anchor_ratios = config['anchor_ratios']
        steps = config['steps']
        net = SSD(network, layers, num_filters, 80, anchor_sizes, anchor_ratios, steps)
    else:
        raise NotImplementedError
    
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        net.initialize()

    return net


if __name__ == '__main__':
    args = parse_args()
    with open(args.cfg, 'r') as f:
        config = yaml.full_load(f)
    print (config)
    
    if config['amp']:
        amp.init()

    net = build_net(config)
    input_shape = config['input_shape']
    anchors = get_anchors(net, input_shape)
    
    ctx = [mx.gpu(int(i)) for i in config['gpus'].split(',') if i.strip()]
    
    train_dataset, val_dataset, val_metric = get_dataset(config, ctx)
    
    batch_size = config['batch_size']
    
    train_loader, val_loader = get_dataloader(config, train_dataset, val_dataset, anchors, input_shape, batch_size, ctx)
    train(net, train_loader, val_loader, val_metric, ctx, config)


