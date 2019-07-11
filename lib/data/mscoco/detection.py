"""MS COCO object detection dataset."""
from __future__ import absolute_import
from __future__ import division
import os
from nvidia import dali
from nvidia.dali.pipeline import Pipeline
import numpy as np
import mxnet as mx


class DALICOCODetection(object):
    """DALI partial pipeline with COCO Reader and loader. To be passed as
    a parameter of a DALI transform pipeline.

    Parameters
    ----------
    num_shards: int
         DALI pipeline arg - Number of pipelines used, indicating to the reader
         how to split/shard the dataset.
    shard_id: int
         DALI pipeline arg - Shard id of the pipeline must be in [0, num_shards).
    file_root
        Directory containing the COCO dataset.
    annotations_file
        The COCO annotation file to read from.
    """
    def __init__(self, split, num_shards, shard_id, root_dir='~/.mxnet/datasets/coco'):
        file_root = os.path.expanduser(os.path.join(root_dir, split))
        anno_file_name = 'instances_{}.json'.format(split)
        annotations_file = os.path.expanduser(os.path.join(root_dir, 'annotations', anno_file_name))
        self.input = dali.ops.COCOReader(
            file_root=file_root,
            annotations_file=annotations_file,
            skip_empty=True,
            shard_id=shard_id,
            num_shards=num_shards,
            ratio=True,
            ltrb=True,
            shuffle_after_epoch=True,
            save_img_ids=True)

        self.decode = dali.ops.HostDecoder(device="cpu", output_type=dali.types.RGB)

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

    def __call__(self):
        """Returns three DALI graph nodes: inputs, bboxes, labels.
        To be called in `define_graph`.
        """
        inputs, bboxes, labels, img_ids = self.input(name="Reader")
        images = self.decode(inputs)
        return (images, bboxes, labels, img_ids)
    
    def size(self):
        """Returns size of COCO dataset
        """
        return self._size


class SSDTrainPipeline(Pipeline):
    """DALI Pipeline with SSD training transform.

    Parameters
    ----------
    device_id: int
         DALI pipeline arg - Device id.
    num_workers:
        DALI pipeline arg - Number of CPU workers.
    batch_size:
        Batch size.
    data_shape: int
        Height and width length. (height==width in SSD)
    anchors: float list
        Normalized [ltrb] anchors generated from SSD networks.
        The shape length be ``N*4`` since it is a list of the N anchors that have
        all 4 float elements.
    dataset_reader: float
        Partial pipeline object, which __call__ function has to return
        (images, bboxes, labels) DALI EdgeReference tuple.
    """
    def __init__(self, num_workers, device_id, batch_size, data_shape,
                 anchors, dataset_reader):
        super(SSDTrainPipeline, self).__init__(
            batch_size=batch_size,
            device_id=device_id,
            num_threads=num_workers)

        self.dataset_reader = dataset_reader

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

        self.box_encoder = dali.ops.BoxEncoder(
            device="cpu",
            criteria=0.5,
            anchors=self._to_normalized_ltrb_list(anchors, data_shape),
            offset=True,
            stds=[0.1, 0.1, 0.2, 0.2],
            scale=data_shape)

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

        images, bboxes, labels, _ = self.dataset_reader()

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
        bboxes, labels = self.box_encoder(bboxes, labels)

        return (images, bboxes.gpu(), labels.gpu())


class SSDValPipeline(Pipeline):
    def __init__(self, num_workers, device_id, batch_size, data_shape,
                 dataset_reader):
        super(SSDValPipeline, self).__init__(
            batch_size=batch_size,
            device_id=device_id,
            num_threads=num_workers)

        self.dataset_reader = dataset_reader

        self.resize = dali.ops.Resize(
            device="cpu",
            resize_x=data_shape,
            resize_y=data_shape,
            min_filter=dali.types.DALIInterpType.INTERP_TRIANGULAR)

        # output_dtype = types.FLOAT16 if args.fp16 else types.FLOAT
        output_dtype = dali.types.FLOAT

        self.normalize = dali.ops.CropMirrorNormalize(
            device="gpu",
            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
            std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
            mirror=0,
            output_dtype=output_dtype,
            output_layout=dali.types.NCHW,
            pad_output=False)

    def define_graph(self):
        images, bboxes, labels, img_ids = self.dataset_reader()

        images = self.resize(images)
        images = images.gpu()
        images = self.normalize(images)

        return (images, bboxes.gpu(), labels.gpu(), img_ids.gpu())


class SSDValLoader(object):
    def __init__(self, pipelines, size, batch_size, data_shape=None):
        self.pipelines = pipelines
        self.size = size
        print ('size {}'.format(size))
        self.batch_size = batch_size
        # self.width, self.height = data_shape
        self.num_worker = len(pipelines)
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
        for idx, pipe in enumerate(self.pipelines):
            data, bboxes, labels, img_ids = pipe.run()
            data, labels = self.format_data(data, bboxes, labels, idx)
            data_batch = mx.io.DataBatch(data=[data], label=[labels])
            img_ids = [int(img_ids.as_cpu().at(idx)) for idx in range(self.batch_size)]
            img_ids = mx.nd.array(img_ids)
            batch_data.append(data_batch)
            batch_img_ids.append(img_ids)
        
        self.count += self.num_worker * self.batch_size
        if self.count > self.size:
            overflow = self.count - self.size
            overflow_per_device = overflow // self.num_worker
            last_batch_data = []
            last_img_ids = []
            for data_batch, img_ids in zip(batch_data, batch_img_ids):
                data = data_batch.data[0][0: self.batch_size-overflow_per_device, :, :, :]
                label = data_batch.label[0][0: self.batch_size-overflow_per_device, :, :]
                data_batch = mx.io.DataBatch(data=[data], label=[labels])
                img_ids = img_ids[0: self.batch_size-overflow_per_device]
                last_batch_data.append(data_batch)
                last_img_ids.append(img_ids)
            batch_data = last_batch_data
            batch_img_ids = last_img_ids
        
        return batch_data, batch_img_ids
    
    def format_data(self, data, bboxes, labels, idx):
        ctx = mx.gpu(idx)
        data = [data.as_cpu().at(idx) for idx in range(self.batch_size)]
        data = [mx.nd.array(d).expand_dims(axis=0) for d in data]
        data = mx.nd.concat(*data, dim=0)
        data = data.as_in_context(ctx)
        
        num_boxes = [bboxes.as_cpu().at(idx).shape[0] for idx in range(self.batch_size)]
        max_num_boxes = max(num_boxes)
        box_dim = bboxes.as_cpu().at(0).shape[1]
        
        format_bboxes = []
        format_labels = []
        for idx in range(self.batch_size):
            box_container = mx.nd.zeros((1, max_num_boxes, box_dim))
            label_dim = labels.as_cpu().at(0).shape[1]
            assert label_dim == 1, 'Expected label dim to be 1 but got {}.'.format(label_dim)
            label_container = mx.nd.ones((1, max_num_boxes, 1))*-1
            bbox = bboxes.as_cpu().at(idx)
            label = labels.as_cpu().at(idx)
            num_box = bbox.shape[0]
            num_label = label.shape[0]
            assert num_box == num_label, 'Expected same length of boxes and labels,\
                got {} and {}'.format(num_box, num_label)
            box_container[:, 0: num_box, :] = bbox
            label_container[:, 0: num_label, :] = label
            format_bboxes.append(box_container)
            format_labels.append(label_container)
        format_bboxes = mx.nd.concat(*format_bboxes, dim=0)
        # format_bboxes[:, :, 0] *= self.width
        # format_bboxes[:, :, 1] *= self.height
        # format_bboxes[:, :, 2] *= self.width
        # format_bboxes[:, :, 3] *= self.height
        format_labels = mx.nd.concat(*format_labels, dim=0)
        labels = mx.nd.concat(format_bboxes, format_labels, dim=-1)
        labels = labels.as_in_context(ctx)

        return data, labels

    def next(self):
        return self.__next__()
    
    def __iter__(self):
        return self

    def reset(self):
        for pipe in self.pipelines:
            pipe.reset()
        self.count = 0


