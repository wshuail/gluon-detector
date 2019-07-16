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
sys.path.insert(0, os.path.expanduser('~/gluon_detector'))
from lib.utils.logger import build_logger
from lib.utils.export_helper import export_block


class BaseSolver(object):

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
        save_frequent = self.config['save_frequent']
        if epoch % save_frequent == 0:
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

