import os
import sys
import logging
logging.basicConfig(level=logging.INFO)
import numpy as np
sys.path.insert(0, os.path.expanduser('~/incubator-mxnet/python'))
import mxnet as mx
from mxnet import nd
from nvidia import dali
from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin.mxnet import feed_ndarray


class ValPipeline(Pipeline):
    def __init__(self, split, batch_size, max_size, resize_shorter, num_shards, device_id, num_workers,
                 fix_shape=False, root_dir='~/.mxnet/datasets/coco'):
        super(ValPipeline, self).__init__(
            batch_size=batch_size,
            device_id=device_id,
            num_threads=num_workers)
        
        file_root = os.path.expanduser(os.path.join(root_dir, split))
        anno_file_name = 'instances_{}.json'.format(split)
        annotations_file = os.path.expanduser(os.path.join(root_dir, 'annotations', anno_file_name))
        self.input = dali.ops.COCOReader(
            file_root=file_root,
            annotations_file=annotations_file,
            pad_last_batch=False,
            skip_empty=False,
            shard_id=device_id,
            num_shards=num_shards,
            ratio=True,
            ltrb=True,
            shuffle_after_epoch=False,
            save_img_ids=True)

        self.decode = dali.ops.ImageDecoder(device="mixed", output_type=dali.types.RGB)

        if fix_shape:
            data_shape = 512
            self.resize = dali.ops.Resize(
                device="gpu",
                resize_x=data_shape,
                resize_y=data_shape,
                min_filter=dali.types.DALIInterpType.INTERP_TRIANGULAR)
        else:
            self.resize = dali.ops.Resize(
                device="gpu",
                max_size=max_size,
                resize_shorter=resize_shorter,
                min_filter=dali.types.DALIInterpType.INTERP_TRIANGULAR)

        self.normalize = dali.ops.CropMirrorNormalize(
            device="gpu",
            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
            std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
            mirror=0,
            output_dtype=dali.types.FLOAT,
            output_layout=dali.types.NCHW,
            pad_output=False)

        # We need to build the COCOReader ops to parse the annotations
        # and have acces to the dataset size.
        # TODO(spanev): Replace by DALI standalone ops when available
        class DummyMicroPipe(Pipeline):
            """ Dummy pipeline which sole purpose is to build COCOReader
            and get the epoch size. To be replaced by DALI standalone op, when available.
            """
            def __init__(self):
                super(DummyMicroPipe, self).__init__(batch_size=1,
                                                     device_id=0,
                                                     num_threads=1)
                self.input = dali.ops.COCOReader(
                    file_root=file_root,
                    annotations_file=annotations_file)
            def define_graph(self):
                inputs, bboxes, labels = self.input(name="Reader")
                return (inputs, bboxes, labels)

        micro_pipe = DummyMicroPipe()
        micro_pipe.build()
        self._size = micro_pipe.epoch_size(name="Reader")
        del micro_pipe

    def define_graph(self):
        inputs, bboxes, labels, img_ids = self.input(name="Reader")
        images = self.decode(inputs)
        images = self.resize(images)
        # images = self.normalize(images)

        return (images, bboxes.gpu(), labels.gpu(), img_ids.gpu())

    def size(self):
        """Returns size of COCO dataset
        """
        return self._size


class RetinaNetValLoader(object):
    def __init__(self, split, thread_batch_size, max_size, resize_shorter, num_devices, fix_shape=False):
        self.max_size = max_size
        self.resize_shorter = resize_shorter
        pipelines = [ValPipeline(split=split,
                                 batch_size=thread_batch_size,
                                 max_size=max_size,
                                 resize_shorter=resize_shorter,
                                 num_shards=4,
                                 device_id=i,
                                 num_workers=16,
                                 fix_shape=fix_shape) for i in range(num_devices)]
        self.pipelines = pipelines
        self.batch_size = pipelines[0].batch_size
        self.size = pipelines[0].size()
        print ('total size {}'.format(self.size))
        self.means = nd.array([0.485 * 255, 0.456 * 255, 0.406 * 255]).reshape(-1, 1, 1)
        self.stds = nd.array([0.229 * 255, 0.224 * 255, 0.225 * 255]).reshape(-1, 1, 1)
        
        for pipeline in self.pipelines:
            pipeline.build()
        
        self.count = 0

        logging.info('RetinaNetValLoader Initilized.')
    
    def __next__(self):
        
        if self.count >= self.size:
            self.reset()
            raise StopIteration
        
        all_data = []
        all_labels = []
        all_resized_attrs = []
        all_img_ids = []
        for idx, pipe in enumerate(self.pipelines):
            data, bboxes, labels, img_ids = pipe.run()
            data, labels, resized_attrs = self.format_data(data, bboxes, labels, img_ids, idx)
            img_ids = [int(img_ids.as_cpu().at(idx)) for idx in range(self.batch_size)]
            
            self.count += len(data)
            
            all_data.append(data)
            all_labels.append(labels)
            all_resized_attrs.append(resized_attrs)
            all_img_ids.append(img_ids)
        
        return all_data, all_labels, all_resized_attrs, all_img_ids

    def format_data(self, batch_data, batch_bboxes, batch_labels, batch_image_ids, idx):
        ctx = mx.gpu(idx)
        # batch_image_ids = batch_image_ids.as_cpu()
        all_images, all_labels = [], []
        resized_attrs = []
        for i in range(self.batch_size):
            # image_id = int(batch_image_ids.at(i))
            image = batch_data.at(i)
            image = self.feed_tensor_into_mx(image, ctx)
            image = nd.transpose(image, (2, 0, 1))
            image = image.astype(np.float32)
            _, height, width = image.shape
            resized_attrs.append((height, width))

            hw_ratio = (height/width)
            if hw_ratio >= 1:
                image_h, image_w = self.max_size, self.resize_shorter
            else:
                image_h, image_w = self.resize_shorter, self.max_size
            pad_h = image_h - height
            pad_w = image_w - width
            image = nd.expand_dims(image, axis=0)
            # nd.pad only support 4/5-dimentional data so expand then squeeze
            image = nd.pad(image, mode='constant', constant_value=-1.0,
                           pad_width=(0, 0, 0, 0, 0, pad_h, 0, pad_w))
            image = nd.squeeze(image)
            image = ((image - self.means.as_in_context(image.context))/self.stds.as_in_context(image.context))
            image = nd.expand_dims(image, axis=0)

            bboxes = batch_bboxes.at(i)
            labels = batch_labels.at(i)
            assert isinstance(bboxes, dali.backend_impl.TensorGPU) and \
                isinstance(bboxes, dali.backend_impl.TensorGPU), \
                'Expected bboxes and labels are dali.backend_impl.TensorGPU, \
                but got {} and {}'.format(type(bboxes), type(labels))
            assert bboxes.shape()[0] == labels.shape()[0]
            assert bboxes.shape()[1] == 4
            assert labels.shape()[1] == 1
            num_gtboxes = bboxes.shape()[0]
            if num_gtboxes > 0:
                bboxes = self.feed_tensor_into_mx(bboxes, ctx)
                bboxes[:, 0] *= width
                bboxes[:, 1] *= height
                bboxes[:, 2] *= width
                bboxes[:, 3] *= height
                labels = self.feed_tensor_into_mx(labels, ctx)
            else:
                bboxes = mx.nd.zeros((1, 4), ctx=ctx)
                labels = mx.nd.ones((1, 1), ctx=ctx)*-1
            labels = labels.astype(bboxes.dtype)
            labels = mx.nd.concat(bboxes, labels, dim=-1)
            labels = nd.expand_dims(labels, axis=0)
            all_images.append(image)
            all_labels.append(labels)

        return all_images, all_labels, resized_attrs

    def feed_tensor_into_mx(self, pipe_out, ctx):
        if isinstance(pipe_out, dali.backend_impl.TensorGPU):
            dtype = np.dtype(pipe_out.dtype())
            batch_data = mx.nd.zeros(pipe_out.shape(), ctx=ctx, dtype=dtype)
            feed_ndarray(pipe_out, batch_data)
        else:
            raise NotImplementedError ('pipe out should be TensorGPU.')
        return batch_data

    def next(self):
        return self.__next__()
    
    def __iter__(self):
        return self

    def reset(self):
        for pipe in self.pipelines:
            pipe.reset()
        self.count = 0

if __name__ == '__main__':
    val_split = 'val2017'
    thread_batch_size = 2
    max_size = 1024
    resize_shorter = 640
    num_devices = 4
    val_loader = RetinaNetValLoader(split=val_split, thread_batch_size=thread_batch_size,
                                    max_size=max_size, resize_shorter=resize_shorter,
                                    num_devices=num_devices, fix_shape=False)
    n = 0
    for data_batch in val_loader:
        batch_data, batch_labels, batch_attrs, batch_img_ids = data_batch
        for thread_image, thread_labels, thread_attrs in zip(batch_data, batch_labels, batch_attrs):
            for image, labels, attrs in zip(thread_image, thread_labels, thread_attrs):
                print ('image shape: {}'.format(image.shape))
                n += image.shape[0]
    print ('final n: {}'.format(n))




