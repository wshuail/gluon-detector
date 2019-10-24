import argparse
import os
import sys
import yaml
import logging
sys.path.insert(0, os.path.expanduser('~/incubator-mxnet/python'))
sys.path.insert(0, os.path.expanduser('~/gluon_detector'))
from lib.solver import SSDSolver
from lib.solver import CenterNetSolver
from lib.solver import RetinaNetSolver


def parse_args():
    parser = argparse.ArgumentParser(description='Train networks.')
    parser.add_argument('--cfg', type=str, default='./config.yaml')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    with open(args.cfg, 'r') as f:
        config = yaml.full_load(f)
    
    logging.info(config)

    model = config['model'].lower()
    if model == 'ssd':
        network = config['network']
        layers = config['layers']
        num_filters = config['num_filters']
        anchor_sizes = config['anchor_sizes']
        anchor_ratios = config['anchor_ratios']
        steps = config['steps']
        dataset = config['dataset']
        input_shape = config['input_shape']
        train_split = config['train_split']
        batch_size = config['batch_size']
        optimizer = config['optimizer']
        lr = config['lr']
        wd = config['wd']
        momentum = config['momentum']
        epoch = config['epoch']
        lr_decay = config.get('lr_decay', 0.1)
        train_split = config['train_split']
        val_split = config['val_split']
        use_amp = config['use_amp']
        gpus = config['gpus']
        save_prefix = config['save_prefix']
        solver = SSDSolver(network, layers, num_filters, anchor_sizes,
                           anchor_ratios, steps, dataset, input_shape,
                           batch_size, optimizer, lr, wd, momentum, epoch,
                           lr_decay, train_split,
                           val_split, use_amp, gpus, save_prefix)
    elif model == 'retinanet':
        backbone = config['backbone']
        dataset = config['dataset']
        train_split = config['train_split']
        val_split = config['val_split']
        max_size = config['max_size']
        resize_shorter = config['resize_shorter']
        gpu_batch_size = config['gpu_batch_size']
        optimizer = config['optimizer']
        lr = config['lr']
        wd = config['wd']
        momentum = config['momentum']
        gpus = config['gpus']
        epoch = config['epoch']
        lr_decay = config.get('lr_decay', 0.1)
        use_amp = config['use_amp']
        save_frequent = config['save_frequent']
        save_prefix = config['save_prefix']
        solver = RetinaNetSolver(backbone, dataset, max_size, resize_shorter,
                                 gpu_batch_size, optimizer, lr, wd, momentum, epoch,
                                 lr_decay, train_split, val_split, use_amp, gpus,
                                 save_frequent, save_prefix)
    else:
        raise NotImplementedError

    solver.train()



