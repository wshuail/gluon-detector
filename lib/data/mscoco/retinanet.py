from __future__ import absolute_import
from __future__ import division
import os
from nvidia import dali
from nvidia.dali.pipeline import Pipeline
import numpy as np
import mxnet as mx


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

        self.decode = dali.ops.HostDecoder(device="cpu", output_type=dali.types.RGB)

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
        bboxes, labels = self.box_encoder(bboxes, labels)

        return (images, bboxes.gpu(), labels.gpu(), img_ids.gpu())

    def size(self):
        """Returns size of COCO dataset
        """
        return self._size


