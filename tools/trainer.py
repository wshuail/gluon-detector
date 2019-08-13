import argparse
import os
import sys
import yaml
import logging
sys.path.insert(0, os.path.expanduser('~/incubator-mxnet/python'))
sys.path.insert(0, os.path.expanduser('~/gluon_detector'))
from lib.solver import SSDSolver
from lib.solver import CenterNetSolver


def parse_args():
    parser = argparse.ArgumentParser(description='Train networks.')
    parser.add_argument('--cfg', type=str, default='./config.yaml')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    with open(args.cfg, 'r') as f:
        config = yaml.full_load(f)
    epoch = config['epoch']
    config['lr_decay_epoch'] = ','.join([str(l*epoch) for l in [0.6, 0.8]])

    solver_dict = {
        'ssd': SSDSolver,
        'centernet': CenterNetSolver
    }

    model = config['model'].lower()
    solver = solver_dict[model](config)

    logging.info(config)

    solver.train()



