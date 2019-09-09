import argparse
import os
import sys
import yaml
import logging
import warnings
import time
import datetime
import numpy as np
sys.path.insert(0, os.path.expanduser('~/incubator-mxnet/python'))
import mxnet as mx
from mxnet import nd
from mxnet import gluon
from mxnet import autograd
sys.path.insert(0, os.path.expanduser('~/gluon_detector'))
from lib.utils.logger import build_logger
from lib.utils.export_helper import export_block


class BaseSolver(object):
    def __init__(self, config):
        self.config = config

        self.ctx = [mx.gpu(int(i)) for i in config['gpus'].split(',') if i.strip()]

        self.net = self.build_net()

        self.train_data, self.val_data = self.get_dataloader()
        self.eval_metric = self.get_eval_metric()
        self.width, self.height = config['input_shape']

        prefix = '{}_{}_{}_{}x{}'.format(config['model'], config['dataset'],
                                              config['network'], config['input_shape'][0],
                                              config['input_shape'][1]) 
        self.save_prefix = os.path.expanduser(os.path.join(config['save_prefix'], prefix))
        
        self.get_logger()

        logging.info('CenterNetSolver initialized')


    def build_net(self):
        return NotImplementedError

    def get_dataloader(self):
        return NotImplementedError
    
    def get_eval_metric(self):
        return NotImplementedError

    def train(self):
        return NotImplementedError

    def validation(self):
        return NotImplementedError

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

