"""Train SSD"""
import argparse
import os
import sys
import yaml
import logging
import warnings
import time
import datetime
from pprint import pprint
import numpy as np
sys.path.insert(0, os.path.expanduser('~/lib/incubator-mxnet/python'))
import mxnet as mx
from mxnet import nd
from mxnet import gluon
from mxnet import autograd
from mxnet.contrib import amp
sys.path.insert(0, os.path.expanduser('~/gluon_detector'))
from lib.loss import FocalLoss
from lib.metrics.coco_detection import COCODetectionMetric
from lib.utils.logger import build_logger
from lib.utils.centernet import get_pred_result
from lib.modelzoo.centernet import CenterNet
from lib.data.mscoco.detection import DALICOCODetection
from lib.data.mscoco.detection import SSDValPipeline, SSDValLoader
from lib.data.mscoco.centernet import CenterNetTrainPipeline
from lib.data.mscoco.centernet import CenterNetTrainLoader


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
    
    log_file = '{}_{}_{}_{}x{}_eval'.format(config['model'], config['dataset'], config['network'],
                                            config['input_shape'][0], config['input_shape'][1]) 
    log_path = os.path.expanduser(os.path.join(config['save_prefix'], log_file))
    
    val_metric = COCODetectionMetric(dataset=val_split,
                                     save_prefix=log_path,
                                     use_time=False,
                                     cleanup=True,
                                     data_shape=config['input_shape'])
    return train_dataset, val_dataset, val_metric

def get_dataloader(config, train_dataset, val_dataset, anchors, input_shape, batch_size, ctx, num_workers=32):
    width, height = input_shape
    num_devices = len(ctx)
    thread_batch_size = batch_size // num_devices
    train_pipelines = [CenterNetTrainPipeline(device_id=i, batch_size=thread_batch_size,
                                        data_shape=input_shape[0],
                                        num_workers=16, dataset_reader=train_dataset[i]) for i in range(num_devices)]
    epoch_size = train_dataset[0].size()
    print ('train dataset epoch size: {}'.format(epoch_size))
    num_classes = config['num_classes']
    train_loader = CenterNetTrainLoader(train_pipelines, epoch_size, thread_batch_size, num_classes, input_shape)
    
    val_pipelines = [SSDValPipeline(device_id=i, batch_size=thread_batch_size,
                                 data_shape=input_shape[0],
                                 num_workers=16,
                                 dataset_reader=val_dataset[i]) for i in range(num_devices)]
    epoch_size = val_dataset[0].size()
    print ('val dataset epoch size: {}'.format(epoch_size))
    val_loader = SSDValLoader(val_pipelines, epoch_size, thread_batch_size, input_shape)
        
    return train_loader, val_loader

def validate(net, val_data, ctx, eval_metric, config):
    """Test on validation dataset."""
    eval_metric.reset()
    # set nms threshold and topk constraint
    # net.set_nms(nms_thresh=0.45, nms_topk=400)
    net.hybridize(static_alloc=True, static_shape=True)
    for idx, (batch, img_ids) in enumerate(val_data):
        # print ('val idx: {}'.format(idx))
        # print ('img_ids: {}'.format(img_ids))
        data = [d.data[0] for d in batch]
        label = [d.label[0] for d in batch]
        
        det_bboxes = []
        det_ids = []
        det_scores = []
        gt_bboxes = []
        gt_ids = []
        gt_difficults = []
        
        for (x, y) in zip(data, label):
            outputs = net(x)
            hm_pred, offset_pred, wh_pred = outputs
            results = get_pred_result(hm_pred, offset_pred, wh_pred)
            continue
            nd.waitall()
            bboxes = results.slice_axis(axis=-1, begin=0, end=4)
            ids = results.slice_axis(axis=-1, begin=4, end=5)
            scores = results.slice_axis(axis=-1, begin=5, end=6)
            # print ('bboxes: {}'.format(bboxes.shape))
            # print ('ids: {}'.format(ids.shape))
            # print ('scores: {}'.format(scores.shape))
            nd.waitall()

            det_bboxes.append(bboxes.clip(0, x.shape[2]))
            det_ids.append(ids)
            det_scores.append(scores)
            # split ground truths
            gt_bboxes.append(y.slice_axis(axis=-1, begin=0, end=4))
            gt_ids.append(y.slice_axis(axis=-1, begin=4, end=5))
            gt_difficults.append(y.slice_axis(axis=-1, begin=5, end=6) if y.shape[-1] > 5 else None)

        # update metric
        # eval_metric.update(det_bboxes, det_ids, det_scores, img_ids, gt_bboxes, gt_ids, gt_difficults)
    # return eval_metric.get()
    return 'map', 100.0


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

    hm_creteria = FocalLoss(sparse_label=False)
    offset_creteria = gluon.loss.L1Loss()
    wh_creteria = gluon.loss.L1Loss()
    
    hm_metric = mx.metric.Loss('FocalLoss')
    offset_metric = mx.metric.Loss('Offset_L1')
    wh_metric = mx.metric.Loss('WH_L1')

    logging.info('Start training from scratch...')
    
    for epoch in range(config['epoch']):
        hm_metric.reset()
        offset_metric.reset()
        wh_metric.reset()
        tic = time.time()
        btic = time.time()
        net.hybridize(static_alloc=True, static_shape=True)
        for i, (batch, img_ids, _) in enumerate(train_data):
            # if i >= 100:
            #     print ('continue')
            #     break
            batch_data = [d.data[0] for d in batch]
            batch_hm = [d.label[0] for d in batch]
            batch_offset = [d.label[1] for d in batch]
            batch_wh = [d.label[2] for d in batch]
            
            with autograd.record():
                hm_losses, offset_losses, wh_losses, sum_losses = [], [], [], []
                for data, hm, offset, wh in zip(batch_data, batch_hm, batch_offset, batch_wh):
                    outputs = net(data)
                    hm_pred, offset_pred, wh_pred = outputs
                    
                    hm_loss = hm_creteria(hm_pred, hm)
                    offset_loss = offset_creteria(offset_pred, offset)
                    wh_loss = wh_creteria(wh_pred, wh)

                    sum_loss = hm_loss + offset_loss + 0.1*wh_loss
                    
                    hm_losses.append(hm_loss)
                    offset_losses.append(offset_loss)
                    wh_losses.append(wh_loss)

                    sum_losses.append(sum_loss)

                for sum_loss in sum_losses:
                    autograd.backward(sum_loss)

            # since we have already normalized the loss, we don't want to normalize
            # by batch-size anymore
            trainer.step(batch_size)

            hm_metric.update(0, [l * batch_size for l in hm_losses])
            offset_metric.update(0, [l * batch_size for l in offset_losses])
            wh_metric.update(0, [l * batch_size for l in wh_losses])
            if i > 0 and i % 50 == 0:
                name1, loss1 = hm_metric.get()
                name2, loss2 = offset_metric.get()
                name3, loss3 = wh_metric.get()
                print('Epoch {} Batch {} Speed: {:.3f} samples/s, {}={:.3f}, {}={:.3f}, {}={:.3f}'.\
                      format(epoch, i, batch_size/(time.time()-btic), name1, loss1, name2, loss2, name3, loss3))
            btic = time.time()
        print ('starting validation')
        map_name, mean_ap = validate(net, val_data, ctx, eval_metric, config)
        # val_msg = '\n'.join(['{}={}'.format(k, v) for k, v in zip(map_name, mean_ap)])
        # print('[Epoch {}] Validation: \n{}'.format(epoch, val_msg))
        

def build_net(config):

    if config['model'] == 'ssd':
        network = config['network']
        layers = config['layers']
        num_filters = config['num_filters']
        anchor_sizes = config['anchor_sizes']
        anchor_ratios = config['anchor_ratios']
        steps = config['steps']
        net = SSD(network, layers, num_filters, 80, anchor_sizes, anchor_ratios, steps)
    elif config['model'] == 'centernet':
        network = config['network']
        layers = config['layers']
        heads = config['heads']
        head_conv = config['head_conv']
        net = CenterNet(network, layers, heads, head_conv)
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

    """
    save_prefix = config['save_prefix']
    timestamp = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    log_file = '{}_{}_{}_{}x{}_train_{}.log'.format(config['model'], config['dataset'], config['network'],
                                                    config['input_shape'][0], config['input_shape'][1],
                                                    timestamp) 
    log_path = os.path.expanduser(os.path.join(save_prefix, log_file))
    build_logger(log_path)
    """
    
    pprint (config)
    print (config['heads'])

    # logging.info(config)
    
    if config['amp']:
        amp.init()

    net = build_net(config)
    
    ctx = [mx.gpu(int(i)) for i in config['gpus'].split(',') if i.strip()]
    
    train_dataset, val_dataset, val_metric = get_dataset(config, ctx)
    
    input_shape = config['input_shape']
    batch_size = config['batch_size']
    anchors = None
    
    train_loader, val_loader = get_dataloader(config, train_dataset, val_dataset, anchors, input_shape, batch_size, ctx)
    train(net, train_loader, val_loader, val_metric, ctx, config)


