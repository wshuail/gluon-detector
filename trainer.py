import sys
import os
import argparse
import yaml
import pickle
import logging
from pprint import pprint
sys.path.insert(0, os.path.expanduser('~/incubator-mxnet/python'))
import mxnet as mx

from lib.symbol.symbol_builder import get_symbol
from lib.data.loader import SSDLoader
from lib.metrics.metric import MultiBoxMetric
from lib.config import get_config

def parse_args():
    parser = argparse.ArgumentParser(description='Train Detection.')
    parser.add_argument('--cfg', type=str, required=True, help='configure file name')
    args = parser.parse_args()
    return args

def get_roidb(roidb_cache_path):
    if os.path.exists(roidb_cache_path):
        with open(roidb_cache_path, 'rb') as f:
            roidb = pickle.load(f)
        print('loaded roidb from {}'.format(roidb_cache_path))
    else:
        coco_dataset = coco(image_set='train2017')
        roidb = coco_dataset.load_gt_roidb()
        with open(roidb_cache_path, "wb") as f:
            pickle.dump(roidb, f)
        print('wring roidb into {}'.format(roidb_cache_path))
    return roidb

def get_lr_scheduler(num_example, batch_size, epoch, lr_decay_ratio=0.1):
    epoch_size = num_example//batch_size
    lr_decay_epoch = [epoch*l for l in (0.5, 0.8)]
    steps = [epoch_size*epoch for epoch in lr_decay_epoch]
    lr_scheduler = mx.lr_scheduler.MultiFactorScheduler(step=steps, factor=lr_decay_ratio)
    return lr_scheduler

def get_logger(save_prefix):
    fmt = '%(asctime)s;%(message)s'
    formater = logging.Formatter(fmt=fmt)
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    log_path = '{}_{}'.format(save_prefix, 'train.log')
    fh = logging.FileHandler(filename=log_path)
    fh.setFormatter(formater)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formater)
    logger.addHandler(sh)
    
    return logger


if __name__ == '__main__':
    args = parse_args()
    with open(args.cfg, 'r') as f:
        config = yaml.full_load(f)
    config['roidb_cache_path'] = os.path.join(os.path.dirname(__file__), config['roidb_cache_path'])
    config['save_prefix'] = os.path.join(os.path.dirname(__file__), config['save_prefix'])
    
    logger = get_logger(config['save_prefix'])

    logging.info(config)
    pprint (config)

    network = config['network']
    num_layers = config['num_layers']
    net_config = get_config(name='{}{}'.format(network, num_layers))

    num_classes = config['num_classes']

    symbol = get_symbol(net_config, num_classes, num_layers)

    roidb_cache_path = config['roidb_cache_path']
    roidb = get_roidb(roidb_cache_path)

    batch_size = config['batch_size']
    target_shape = config['target_shape']
    data_iter = SSDLoader(roidb, batch_size, target_shape)

    ctx = [mx.gpu(int(i)) for i in config['gpus'].split(',')]

    mod = mx.mod.Module(symbol, data_names=('data', ), label_names=('label', ), logger=logger, context=ctx)

    num_example = data_iter.num_example
    lr_scheduler = get_lr_scheduler(num_example, batch_size, config['epoch'])
    optimizer_params = {'learning_rate': config['lr'],
                        'momentum': config['momentum'],
                        'wd': config['wd'],
                        'lr_scheduler': lr_scheduler,
                        'rescale_grad': 1.0 / len(ctx) if len(ctx) > 0 else 1.0}
    batch_end_callback = mx.callback.Speedometer(batch_size, frequent=50)
    epoch_end_callback = mx.callback.do_checkpoint(config['save_prefix'])
    kv = mx.kvstore.create('device')


    print ('start training from scratch...')
    mod.fit(data_iter,
            eval_metric=MultiBoxMetric(),
            batch_end_callback=batch_end_callback,
            epoch_end_callback=epoch_end_callback,
            optimizer='sgd',
            optimizer_params=optimizer_params,
            num_epoch=config['epoch'],
            initializer=mx.init.Xavier(),
            kvstore=kv)
