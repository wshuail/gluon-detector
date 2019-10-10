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
        labels = self.cast(labels)

        return (images, bboxes.gpu(), labels.gpu(), bboxes_1.gpu(), labels_1.gpu(), bboxes_2.gpu(), labels_2.gpu(), img_ids.gpu())
        # return images, bboxes.gpu(), labels.gpu(), img_ids.gpu()

    def size(self):
        """Returns size of COCO dataset
        """
        return self._size


class RetinaNetTrainLoader(object):
    def __init__(self, pipelines, anchors, stds=(0.1, 0.1, 0.2, 0.2),
                 means=(0., 0., 0., 0.), **kwargs):
        self.pipelines = pipelines
        # anchors = self._xywh_to_normalized_ltrb(anchors, size=512)
        self._stds = stds
        self._means = means
        self.num_worker = len(pipelines)
        self.anchors_list = [anchors.as_in_context(mx.gpu(i)) for i in range(self.num_worker)]
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
        cls_targets = []
        for idx, pipe in enumerate(self.pipelines):
            ctx = mx.gpu(idx)
            anchors = self.anchors_list[idx]
            batch_images, batch_bboxes, batch_labels, batch_bboxes_1, batch_labels_1, bboxes_2, labels_2, batch_img_ids = pipe.run()
            batch_images = self.feed_tensor_into_mx(batch_images, ctx)
            batch_box_targets, batch_cls_targets = [], []
            for i in range(self.batch_size):
                bboxes = batch_bboxes.at(i)
                bboxes = self.feed_tensor_into_mx(bboxes, ctx)
                bboxes = self._normalized_ltrb_to_xywh(bboxes, size=512)
                # print ('bboxes: {}'.format(bboxes*512))
                labels = batch_labels.at(i)
                labels = self.feed_tensor_into_mx(labels, ctx)
                
                box_ious = nd.contrib.box_iou(anchors, bboxes, format='center')
                ious, indices = nd.topk(box_ious, axis=-1, ret_typ='both', k=1)
                # print ('max ious: {}'.format(nd.max(ious)))

                box_target = nd.take(bboxes, indices).reshape((-1, 4))
                box_target = self.encode_box_target(box_target, anchors)
                if False:
                    box_target_np = box_target.asnumpy()
                    index = (ious.asnumpy()>0.5)
                    pos_box_target_np = box_target_np[index.flatten(), :]
                    print ('pos_box_target_np: {}'.format(pos_box_target_np))

                    box_target_np = box_target.asnumpy()
                    pos_box_target_np = box_target_np[index.flatten(), :]
                    # print ('pos_box_target_1_np: {}'.format(pos_box_target_np))

                    anchors_np = anchors.asnumpy()
                    pos_anchors_np = anchors_np[index.flatten(), :]
                    # print ('pos_anchors_np: {}'.format(pos_anchors_np))

                cls_target = nd.take(labels, indices).reshape((-1, 1))
     
                mask = nd.ones_like(ious)*-1
                mask = nd.where(ious<0.4, nd.zeros_like(ious), mask)
                mask = nd.where(ious>0.5, nd.ones_like(ious), mask)

                box_mask = nd.tile(mask, reps=(1, 4))
                box_target = nd.where(box_mask, box_target, nd.zeros_like(box_target))
                batch_box_targets.append(box_target)

                cls_target = nd.where(mask == 1.0, cls_target, mask)
                batch_cls_targets.append(cls_target)

                """
                bboxes_1 = batch_bboxes_1.at(i)
                bboxes_1 = self.feed_tensor_into_mx(bboxes_1, ctx)
                if True:
                    bboxes_1 = bboxes_1.asnumpy()
                    index = (np.sum(bboxes_1, axis=-1) != 0)
                    pos_bboxes_1 = bboxes_1[index, :]
                    print ('mean bboxes_1: {}'.format(np.mean(pos_bboxes_1)))
                    print ('pos_bboxes_1: {}'.format(pos_bboxes_1))
                labels_1 = batch_labels_1.at(i)
                labels_1 = self.feed_tensor_into_mx(labels_1, ctx)
                if True:
                    labels_1_np = labels_1.asnumpy()
                    index = (labels_1_np > 0)
                    pos_labels_1 = labels_1_np[index]
                    print ('pos_labels_1: {}'.format(pos_labels_1))

                    cls_target_np = cls_target.asnumpy()
                    index = (cls_target_np>0)
                    pos_cls_target = cls_target_np[index]
                    print ('pos_cls_target: {}'.format(pos_cls_target))
                """

            batch_box_targets = [nd.expand_dims(box_target, axis=0) for box_target in batch_box_targets]
            batch_cls_targets = [nd.expand_dims(cls_target, axis=0) for cls_target in batch_cls_targets]

            batch_box_targets = nd.concat(*batch_box_targets, dim=0)
            batch_cls_targets = nd.concat(*batch_cls_targets, dim=0).squeeze()
            # print ('batch_box_targets shape: {}'.format(batch_box_targets.shape))
            # print ('batch_cls_targets shape: {}'.format(batch_cls_targets.shape))

            images.append(batch_images)
            box_targets.append(batch_box_targets)
            cls_targets.append(batch_cls_targets)

        self.count += self.num_worker * self.batch_size
        
        return images, box_targets, cls_targets

    def encode_box_target(self, box_targets, anchors):
        g = nd.split(box_targets, num_outputs=4, axis=-1)
        a = nd.split(anchors, num_outputs=4, axis=-1)
        t0 = ((g[0] - a[0]) / a[2] - self._means[0]) / self._stds[0]
        t1 = ((g[1] - a[1]) / a[3] - self._means[1]) / self._stds[1]
        t2 = (nd.log(g[2] / a[2]) - self._means[2]) / self._stds[2]
        t3 = (nd.log(g[3] / a[3]) - self._means[3]) / self._stds[3]
        box_targets = nd.concat(t0, t1, t2, t3, dim=-1)
        return box_targets
    
    def feed_tensor_into_mx(self, pipe_out, ctx):
        if isinstance(pipe_out, dali.backend_impl.TensorListGPU):
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
        elif isinstance(pipe_out, dali.backend_impl.TensorGPU):
            dtype = np.dtype(pipe_out.dtype())
            batch_data = mx.nd.zeros(pipe_out.shape(), ctx=ctx, dtype=dtype)
            feed_ndarray(pipe_out, batch_data)
        else:
            raise NotImplementedError ('pipe out should be TensorListGPU or TensorGPU.')
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
    
    @staticmethod
    def _xywh_to_normalized_ltrb(anchors, size):
        anchors_ltrb = nd.zeros_like(anchors)
        anchors_ltrb[:, 0] = anchors[:, 0] - 0.5 * anchors[:, 2]
        anchors_ltrb[:, 1] = anchors[:, 1] - 0.5 * anchors[:, 3]
        anchors_ltrb[:, 2] = anchors[:, 0] + 0.5 * anchors[:, 2]
        anchors_ltrb[:, 3] = anchors[:, 1] + 0.5 * anchors[:, 3]
        anchors_ltrb /= size
        return anchors_ltrb

    @staticmethod
    def _normalized_ltrb_to_xywh(matrix, size):
        matrix_xywh = nd.zeros_like(matrix)
        matrix_xywh[:, 0] = (matrix[:, 0] + matrix[:, 2])/2
        matrix_xywh[:, 1] = (matrix[:, 1] + matrix[:, 3])/2
        matrix_xywh[:, 2] = matrix[:, 2] - matrix[:, 0]
        matrix_xywh[:, 3] = matrix[:, 3] - matrix[:, 1]
        matrix_xywh *= size
        return matrix_xywh




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
    data_loader = RetinaNetTrainLoader(train_pipelines, anchors)
    for data_batch in iter(data_loader):
        images, box_targets, cls_targets = data_batch
        for x, box_target, cls_target in zip(images, box_targets, cls_targets):
            print ('x shape: {}'.format(x.shape))
            print ('box_target shape: {}'.format(box_target.shape))
            print ('cls_target shape: {}'.format(cls_target.shape))
        assert False

 
