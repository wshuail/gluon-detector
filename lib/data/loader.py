import os
import sys
import random
import multiprocessing
import numpy as np
import cv2
sys.path.insert(0, os.path.expanduser('~/incubator-mxnet/python'))
import mxnet as mx
from mxnet import nd
from mxnet.io import DataBatch
from mxnet.gluon.data.dataloader import Queue
from mxnet.gluon.data.dataloader import SimpleQueue
from .transform import ssd_transform


class SSDLoader(mx.io.DataIter):
    def __init__(self, roidb, batch_size, target_shape=(512, 512),
                 num_workers=4, num_collectors=2, shuffle=True, max_num_label=100):
        self._roidb = roidb
        self.batch_size = batch_size
        self.target_shape = target_shape
        self._num_workers = num_workers
        self._num_collectors = num_collectors
        self.shuffle = shuffle
        self.max_num_label = max_num_label

        self.max_size = len(roidb)
        self.max_batch = int(np.ceil(self.max_size/self.batch_size))
        self.batch_idx = 0

        self._data_names = ['data']
        self._label_names = ['label']
        self._data = None
        self._label = None

        self.key_queue = Queue()
        self.data_queue = Queue()
        self.result_queue = Queue()

        self.workers = None
        self.collectors = None
        self.thread_start()
        
        self.load_first_batch()
        self.reset()

    @property
    def provide_data(self):
        return [(k, v.shape) for k, v in zip(self._data_names, self._data)]

    @property
    def provide_label(self):
        return [(k, v.shape) for k, v in zip(self._label_names, self._label)]

    @property
    def num_example(self):
        return self.max_size

    def thread_start(self):

        self.workers = []
        for _ in range(self._num_workers):
            worker = multiprocessing.Process(
                target=self.work_loop)
            self.workers.append(worker)
        for worker in self.workers:
            worker.daemon = True
            worker.start()

        self.collectors = []
        for _ in range(self._num_collectors):
            collector = multiprocessing.Process(
                target=self.collect_loop)
            self.collectors.append(collector)
        for collector in self.collectors:
            collector.daemon = True
            collector.start()
    
    def reset(self):
        self.batch_idx = 0
        if self.shuffle:
            random.shuffle(self._roidb)

        for _ in range(self._num_workers):
            self.push_next()

    def load_first_batch(self):
        self.key_queue.put(0)
        self.next()

    def iter_next(self):
        return self.batch_idx <= self.max_batch

    def next(self):
        if self.iter_next():
            data_batch = self.result_queue.get()
            if self._data is None and self._label is None:
                self._data = data_batch.data
                self._label = data_batch.label
            self.push_next()
            return data_batch
        else:
            raise StopIteration
    
    def push_next(self):
        if self.batch_idx < self.max_batch:
            self.key_queue.put(self.batch_idx)
            self.batch_idx += 1

    def work_loop(self):
        while True:
            idx = self.key_queue.get()
            if idx is None:
                continue
            batch_imgs, batch_label = self.get_batch(idx)
            self.data_queue.put((batch_imgs, batch_label))
    
    def collect_loop(self):
        while True:
            batch_imgs, batch_label = self.data_queue.get()
            
            pad_labels = []
            for label in batch_label:
                num_label = label.shape[0]
                if num_label > self.max_num_label:
                    label = label[0: self.max_num_label, :]
                else:
                    pad_width = self.max_num_label - num_label
                    label = np.pad(label, ((0, pad_width), (0, 0)), mode='constant', constant_values=-1)
                label = label[np.newaxis, :, :]
                pad_labels.append(label)
            batch_label = pad_labels

            batch_imgs = np.concatenate(batch_imgs, axis=0)
            batch_label = np.concatenate(batch_label, axis=0)
            
            data = [nd.array(batch_imgs)]
            label = [nd.array(batch_label)]
            
            if self._data is None and self._label is None:
                provide_data = [(k, v.shape) for k, v in zip(self._data_names, data)]
                provide_label = [(k, v.shape) for k, v in zip(self._label_names, label)]
            else:
                provide_data = self.provide_data
                provide_label = self.provide_label

            data_batch = DataBatch(data=data,
                                   label=label,
                                   provide_data=provide_data,
                                   provide_label=provide_label)
            self.result_queue.put(data_batch)

    def get_batch(self, idx):
        batch_imgs = []
        batch_label = []
        # max_num_label = 0
        for i in range(idx*self.batch_size, min((idx+1)*self.batch_size, self.max_size)):
            rec_roidb = self._roidb[i]
            img_path = rec_roidb['image']
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w, _ = img.shape
            
            gtboxes = rec_roidb['boxes']
            gt_classes = rec_roidb['gt_classes']
            gt_classes = gt_classes[:, np.newaxis]
            label = np.hstack([gtboxes, gt_classes])
            img, label = ssd_transform(img, label, target_shape=self.target_shape)
            # scale label
            label[:, 0: 2] /= w
            label[:, 2: 4] /= h
            label = np.hstack((np.expand_dims(label[:, 4], axis=1), label[:, 0: 4]))
            
            img = np.transpose(img, axes=(2, 0, 1))
            img = img[np.newaxis, :, :, :]
            
            batch_imgs.append(img)
            batch_label.append(label)
        
        return (batch_imgs, batch_label)

