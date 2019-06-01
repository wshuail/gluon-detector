import sys
import os
import yaml
import pickle
import logging
sys.path.insert(0, os.path.expanduser('~/incubator-mxnet/python'))
import mxnet as mx
from mxnet import gluon

from symbol.symbol_builder import get_symbol
from data.mscoco.detection import COCODetection
from data.coco import coco
from data.loader import SSDLoader
from metric import MultiBoxMetric

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

image_shape = '3, 224, 224'
num_classes = 80
num_layers = 50

with open('config.yaml', 'r') as f:
    config = yaml.full_load(f)

resnet50_config = config['RESNET50']

symbol = get_symbol(resnet50_config, num_classes, num_layers, image_shape)
# net = 'resnet50'
# symbol = get_symbol_train(net, num_classes, 50, **resnet50_config)

roidb_cache_path = config['roidb_cache_path']
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

batch_size = 16
target_shape = (300, 300)
data_iter = SSDLoader(roidb, batch_size, target_shape)

ctx = [mx.gpu(i) for i in range(4)]

mod = mx.mod.Module(symbol, data_names=('data', ), label_names=('label', ), logger=logger, context=ctx)

optimizer_params = {'learning_rate': 0.002}
batch_end_callback = mx.callback.Speedometer(batch_size, frequent=50)


print ('start training from scratch...')
mod.fit(data_iter,
        eval_metric=MultiBoxMetric(),
        batch_end_callback=batch_end_callback,
        optimizer='sgd',
        optimizer_params=optimizer_params,
        num_epoch=1,
        initializer=mx.init.Xavier())
