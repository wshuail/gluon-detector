from mxnet import nd
import logging
import numpy as np


def get_scales(min_scale=0.2, max_scale=0.9, num_layers=6):
    # this code follows the original implementation of wei liu
    # for more, look at ssd/score_ssd_pascal.py:310 in the original caffe implementation
    min_ratio = int(min_scale * 100)
    max_ratio = int(max_scale * 100)
    step = int(np.floor((max_ratio - min_ratio) / (num_layers - 2)))
    
    min_sizes = []
    max_sizes = []
    for ratio in range(min_ratio, max_ratio + 1, step):
        min_sizes.append(ratio / 100.)
        max_sizes.append((ratio + step) / 100.)
    min_sizes = [int(100*min_scale / 2.0) / 100.0] + min_sizes
    max_sizes = [min_scale] + max_sizes
    
    # convert it back to this implementation's notation:
    scales = []
    for layer_idx in range(num_layers):
        scales.append([min_sizes[layer_idx], np.single(np.sqrt(min_sizes[layer_idx] * max_sizes[layer_idx]))])
    return scales

def build_logger(log_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    fmt = '%(asctime)s;%(message)s'
    formater = logging.Formatter(fmt=fmt)

    fh = logging.FileHandler(filename=log_path)
    fh.setFormatter(formater)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formater)
    logger.addHandler(sh)

