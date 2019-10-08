from __future__ import absolute_import
from __future__ import division
import os
import sys
import numpy as np
sys.path.insert(0, os.path.expanduser('~/incubator-mxnet/python'))
import mxnet as mx
from mxnet import nd
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

        self.decode = dali.ops.ImageDecoder(device="mixed", output_type=dali.types.RGB)

        # Augumentation techniques
        self.crop = dali.ops.RandomBBoxCrop(
            device="cpu",
            aspect_ratio=[0.5, 2.0],
            thresholds=[0, 0.1, 0.3, 0.5, 0.7, 0.9],
            scaling=[0.3, 1.0],
            ltrb=True,
            allow_no_crop=True,
            num_attempts=1)
        self.slice = dali.ops.Slice(device="gpu")
        self.twist = dali.ops.ColorTwist(device="gpu")
        self.resize = dali.ops.Resize(
            device="gpu",
            resize_x=data_shape,
            resize_y=data_shape,
            min_filter=dali.types.DALIInterpType.INTERP_TRIANGULAR)

        self.normalize = dali.ops.CropMirrorNormalize(
            device="gpu",
            crop=(data_shape, data_shape),
            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
            std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
            mirror=0,
            output_dtype=dali.types.FLOAT,
            output_layout=dali.types.NCHW,
            pad_output=False)

        # Random variables
        self.rng1 = dali.ops.Uniform(range=[0.5, 1.5])
        self.rng2 = dali.ops.Uniform(range=[0.875, 1.125])
        self.rng3 = dali.ops.Uniform(range=[-0.5, 0.5])

        self.flip = dali.ops.Flip(device="gpu")
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
        
        self.cast = dali.ops.Cast(
            dtype = dali.types.FLOAT)

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
        images = self.twist(
            images,
            saturation=saturation,
            contrast=contrast,
            brightness=brightness,
            hue=hue)
        images = self.normalize(images)
        bboxes_1, labels_1 = self.box_encoder_0_5(bboxes, labels)
        bboxes_2, labels_2 = self.box_encoder_0_4(bboxes, labels)
        labels_1 = self.cast(labels_1)
        labels_2 = self.cast(labels_2)

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
        
        images = []
        box_targets = []
        cls_targets_1 = []
        cls_targets_2 = []
        for idx, pipe in enumerate(self.pipelines):
            ctx = mx.gpu(idx)
            batch_images, batch_box_targets, batch_cls_taregts_1, _, batch_cls_targets_2, batch_img_ids = pipe.run()
            
            batch_images = self.feed_tensor_into_mx(batch_images, ctx)
            batch_box_targets = self.feed_tensor_into_mx(batch_box_targets, ctx)
            batch_cls_taregts_1 = self.feed_tensor_into_mx(batch_cls_taregts_1, ctx)
            batch_cls_targets_2 = self.feed_tensor_into_mx(batch_cls_targets_2, ctx)

            images.append(batch_images)
            box_targets.append(batch_box_targets)
            cls_targets_1.append(batch_cls_taregts_1)
            cls_targets_2.append(batch_cls_targets_2)

            # print ('batch_images: {}'.format(batch_images.shape))
            # print ('batch_box_targets: {}'.format(batch_box_targets.shape))
            # print ('batch_cls_taregts_1: {}'.format(batch_cls_taregts_1.shape))
            # print ('batch_cls_targets_2: {}'.format(batch_cls_targets_2.shape))
        
        cls_targets = self.get_cls_targets(cls_targets_1, cls_targets_2)
        
        self.count += self.num_worker * self.batch_size
        
        return images, box_targets, cls_targets
    
    def feed_tensor_into_mx(self, pipe_out, ctx):
        if pipe_out.is_dense_tensor():
            pipe_out_tensor = pipe_out.as_tensor()
            dtype = np.dtype(pipe_out_tensor.dtype())
            batch_data = mx.nd.zeros(pipe_out_tensor.shape(), ctx=ctx, dtype=dtype)
            feed_ndarray(pipe_out_tensor, batch_data)
            # batch_data = [batch_data[i, :, :, :] for i in range(batch_data.shape[0])]
        else:
            raise NotImplementedError ('pipe out should be dense_tensor now.')
            batch_data = []
            for i in range(self.batch_size):
                data_tensor = pipe_out.at(i)
                dtype = np.dtype(data_tensor.dtype())
                img = mx.nd.zeros(data_tensor.shape(), ctx=ctx, dtype=dtype)
                feed_ndarray(data_tensor, img)
                batch_data.append(img)
        return batch_data

    def next(self):
        return self.__next__()
    
    def __iter__(self):
        return self

    def reset(self):
        for pipe in self.pipelines:
            pipe.reset()
        self.count = 0
    
    @staticmethod
    def get_cls_targets(cls_targets_1, cls_targets_2):
        cls_targets = []
        for (cls_target_1, cls_target_2) in zip(cls_targets_1, cls_targets_2):
            cls_target_1_idx = nd.where(cls_target_1 > 0, nd.ones_like(cls_target_1), nd.zeros_like(cls_target_1))
            cls_target_2_idx = nd.where(cls_target_2 > 0, nd.ones_like(cls_target_2), nd.zeros_like(cls_target_2))
            cls_target_idx = nd.where(cls_target_1_idx == cls_target_2_idx, nd.ones_like(cls_target_1_idx),\
                                      nd.zeros_like(cls_target_1_idx))
            cls_target = nd.where(cls_target_idx, cls_target_1, nd.ones_like(cls_target_1)*-1)
            cls_targets.append(cls_target)
        return cls_targets



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
        images, box_targets, cls_targets = data_batch
        for x, box_target, cls_target in zip(images, box_targets, cls_targets):
            print ('x shape: {}'.format(x.shape))
            print ('x dtype: {}'.format(x.dtype))
            print ('box_target shape: {}'.format(box_target.shape))
            print ('box_target dtype: {}'.format(box_target.dtype))
            print ('cls_target shape: {}'.format(cls_target.shape))
            print ('cls_target dtype: {}'.format(cls_target.dtype))

 
