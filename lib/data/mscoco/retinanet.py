from __future__ import absolute_import
from __future__ import division
import os
import sys
import numpy as np
sys.path.insert(0, os.path.expanduser('~/incubator-mxnet/python'))
import mxnet as mx
from nvidia import dali
from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin.mxnet import feed_ndarray


class RetinaNetTrainPipeline(Pipeline):
    def __init__(self, split, batch_size, data_shape, num_shards, device_id, anchors, 
                 num_workers, root_dir='~/.mxnet/datasets/coco'):
        super(RetinaNetTrainPipeline, self).__init__(
            batch_size=batch_size,
            device_id=device_id,
            num_threads=num_workers)
        file_root = os.path.expanduser(os.path.join(root_dir, split))
        anno_file_name = 'instances_{}.json'.format(split)
        annotations_file = os.path.expanduser(os.path.join(root_dir, 'annotations', anno_file_name))
        self.input = dali.ops.COCOReader(
            file_root=file_root,
            annotations_file=annotations_file,
            skip_empty=True,
            shard_id=device_id,
            num_shards=num_shards,
            ratio=True,
            ltrb=True,
            shuffle_after_epoch=True,
            save_img_ids=True)

        self.decode = dali.ops.ImageDecoder(device="cpu", output_type=dali.types.RGB)

        # Augumentation techniques
        self.crop = dali.ops.RandomBBoxCrop(
            device="cpu",
            aspect_ratio=[0.5, 2.0],
            thresholds=[0, 0.1, 0.3, 0.5, 0.7, 0.9],
            scaling=[0.3, 1.0],
            ltrb=True,
            allow_no_crop=True,
            num_attempts=1)
        self.slice = dali.ops.Slice(device="cpu")
        self.twist = dali.ops.ColorTwist(device="gpu")
        self.resize = dali.ops.Resize(
            device="cpu",
            resize_x=data_shape,
            resize_y=data_shape,
            min_filter=dali.types.DALIInterpType.INTERP_TRIANGULAR)

        # output_dtype = types.FLOAT16 if args.fp16 else types.FLOAT
        output_dtype = dali.types.FLOAT

        self.normalize = dali.ops.CropMirrorNormalize(
            device="gpu",
            crop=(data_shape, data_shape),
            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
            std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
            mirror=0,
            output_dtype=output_dtype,
            output_layout=dali.types.NCHW,
            pad_output=False)

        # Random variables
        self.rng1 = dali.ops.Uniform(range=[0.5, 1.5])
        self.rng2 = dali.ops.Uniform(range=[0.875, 1.125])
        self.rng3 = dali.ops.Uniform(range=[-0.5, 0.5])

        self.flip = dali.ops.Flip(device="cpu")
        self.bbflip = dali.ops.BbFlip(device="cpu", ltrb=True)
        self.flip_coin = dali.ops.CoinFlip(probability=0.5)

        self.box_encoder_0_5 = dali.ops.BoxEncoder(
            device="cpu",
            criteria=0.5,
            anchors=self._to_normalized_ltrb_list(anchors, data_shape),
            offset=True,
            stds=[0.1, 0.1, 0.2, 0.2],
            scale=data_shape)
        
        self.box_encoder_0_4 = dali.ops.BoxEncoder(
            device="cpu",
            criteria=0.4,
            anchors=self._to_normalized_ltrb_list(anchors, data_shape),
            offset=True,
            stds=[0.1, 0.1, 0.2, 0.2],
            scale=data_shape)

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
        print ('train dataset size {} for split {}'.format(self._size, split))
        del micro_pipe
    
    def _to_normalized_ltrb_list(self, anchors, size):
        """Prepare anchors into ltrb (normalized DALI anchors format list)"""
        if isinstance(anchors, list):
            return anchors
        anchors_np = anchors.squeeze().asnumpy()
        anchors_np_ltrb = anchors_np.copy()
        anchors_np_ltrb[:, 0] = anchors_np[:, 0] - 0.5 * anchors_np[:, 2]
        anchors_np_ltrb[:, 1] = anchors_np[:, 1] - 0.5 * anchors_np[:, 3]
        anchors_np_ltrb[:, 2] = anchors_np[:, 0] + 0.5 * anchors_np[:, 2]
        anchors_np_ltrb[:, 3] = anchors_np[:, 1] + 0.5 * anchors_np[:, 3]
        anchors_np_ltrb /= size
        return anchors_np_ltrb.flatten().tolist()

    def define_graph(self):
        saturation = self.rng1()
        contrast = self.rng1()
        brightness = self.rng2()
        hue = self.rng3()
        coin_rnd = self.flip_coin()

        inputs, bboxes, labels, img_ids = self.input(name="Reader")
        images = self.decode(inputs)

        crop_begin, crop_size, bboxes, labels = self.crop(bboxes, labels)
        images = self.slice(images, crop_begin, crop_size)

        images = self.flip(images, horizontal=coin_rnd)
        bboxes = self.bbflip(bboxes, horizontal=coin_rnd)
        images = self.resize(images)
        images = images.gpu()
        images = self.twist(
            images,
            saturation=saturation,
            contrast=contrast,
            brightness=brightness,
            hue=hue)
        images = self.normalize(images)
        bboxes_1, labels_1 = self.box_encoder_0_5(bboxes, labels)
        bboxes_2, labels_2 = self.box_encoder_0_4(bboxes, labels)

        return (images, bboxes_1.gpu(), labels_1.gpu(), bboxes_2.gpu(), labels_2.gpu(), img_ids.gpu())

    def size(self):
        """Returns size of COCO dataset
        """
        return self._size


class RetinaNetTrainLoader(object):
    def __init__(self, pipelines):
        self.pipelines = pipelines
        self.num_worker = len(pipelines)
        self.size = pipelines[0].size()
        self.batch_size = pipelines[0].batch_size
        for pipeline in self.pipelines:
            pipeline.build()
        
        self.count = 0
    
    def __next__(self):
        
        if self.count >= self.size:
            self.reset()
            raise StopIteration
        
        batch_data = []
        batch_img_ids = []
        batch_origin_gtbox = []
        for idx, pipe in enumerate(self.pipelines):
            ctx = mx.gpu(idx)
            batch_images, batch_bboxes, batch_labels_1, _, batch_labels_2, batch_img_ids = pipe.run()
            for i in range(self.batch_size):
                image_tensor = batch_images.at(i)
                image = mx.nd.zeros(image_tensor.shape(), ctx=ctx, dtype=np.dtype(image_tensor.dtype()))
                feed_ndarray(image_tensor, image)
                
                bboxes_tensor = batch_bboxes.at(i)
                bboxes = mx.nd.zeros(bboxes_tensor.shape(), ctx=ctx, dtype=np.dtype(bboxes_tensor.dtype()))
                feed_ndarray(bboxes_tensor, bboxes)
                
                labels_1_tensor = batch_labels_1.at(i)
                labels_1 = mx.nd.zeros(labels_1_tensor.shape(), ctx=ctx, dtype=np.dtype(labels_1_tensor.dtype()))
                feed_ndarray(labels_1_tensor, labels_1)
                
                labels_2_tensor = batch_labels_2.at(i)
                labels_2 = mx.nd.zeros(labels_2_tensor.shape(), ctx=ctx, dtype=np.dtype(labels_2_tensor.dtype()))
                feed_ndarray(labels_2_tensor, labels_2)
                
                print ('image shape: {}'.format(image.shape))
                print ('bboxes shape: {}'.format(bboxes.shape))
                print ('labels_1 shape: {}'.format(labels_1.shape))
                print ('labels_2 shape: {}'.format(labels_2.shape))
            
        
        
        self.count += self.num_worker * self.batch_size
        
        return batch_data, batch_img_ids, batch_origin_gtbox
        
    def next(self):
        return self.__next__()
    
    def __iter__(self):
        return self

    def reset(self):
        for pipe in self.pipelines:
            pipe.reset()
        self.count = 0


if __name__ == '__main__':
    sys.path.insert(0, os.path.expanduser('~/gluon_detector'))
    from lib.anchor.retinanet import generate_retinanet_anchors
    train_split = 'train2017'
    thread_batch_size = 4
    input_size = 512
    num_devices = 4
    anchors = generate_retinanet_anchors(input_size)
    train_pipelines = [RetinaNetTrainPipeline(split=train_split,
                                        batch_size=thread_batch_size,
                                        data_shape=input_size,
                                        num_shards=num_devices,
                                        device_id=i,
                                        anchors=anchors,
                                        num_workers=16) for i in range(num_devices)]
    data_loader = RetinaNetTrainLoader(train_pipelines)
    for data_batch in iter(data_loader):
        print ('xx')

 
