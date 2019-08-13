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
        anchors_np = anchors.squeeze()  # .asnumpy()
        anchors_np_ltrb = anchors_np.copy()
        anchors_np_ltrb /= size
        return anchors_np_ltrb.flatten().tolist()

    def define_graph(self):
        saturation = self.rng1()
        contrast = self.rng1()
        brightness = self.rng2()
        hue = self.rng3()
        coin_rnd = self.flip_coin()

        inputs, bboxes, labels, _ = self.input(name="Reader")
        images = self.decode(inputs)

        # crop_begin, crop_size, bboxes, labels = self.crop(bboxes, labels)
        # images = self.slice(images, crop_begin, crop_size)

        # images = self.flip(images, horizontal=coin_rnd)
        # bboxes = self.bbflip(bboxes, horizontal=coin_rnd)
        images = self.resize(images)
        images = images.gpu()
        images = self.twist(
            images,
            saturation=saturation,
            contrast=contrast,
            brightness=brightness,
            hue=hue)
        # images = self.normalize(images)
        bboxes, labels = self.box_encoder(bboxes, labels)

        return (images, bboxes.gpu(), labels.gpu())

    def size(self):
        """Returns size of COCO dataset
        """
        return self._size

if __name__ == '__main__':
    sys.path.insert(0, os.path.expanduser('~/gluon_detector'))
    from lib.modelzoo.anchor import generate_level_anchors
    from nvidia.dali.plugin.mxnet import DALIGenericIterator
    split = 'train2017'
    batch_size = 1
    data_shape = 512
    num_shards = 1
    device_id = 0
    num_workers = 4
        
    image_shape = data_shape
    level_anchors_list = []
    for i in range(3, 8):
        level_anchors = generate_level_anchors(i, image_shape)
        level_anchors_list.append(level_anchors)
    anchors = np.concatenate(level_anchors_list, axis=0)
    
    pipe = RetinaNetTrainPipeline(split, batch_size, data_shape,
                                  num_shards, device_id, anchors,
                                  num_workers)
    pipe.build()
    outputs = pipe.run()
    for output in outputs:
        if output.is_dense_tensor():
            output= output.as_tensor()
            print (output.shape())
    epoch_size = pipe.size()
    data_loader = DALIGenericIterator([pipe], [('data', DALIGenericIterator.DATA_TAG),
                                                         ('bboxes', DALIGenericIterator.LABEL_TAG),
                                                         ('label', DALIGenericIterator.LABEL_TAG)],
                                       epoch_size, auto_reset=True)
    for i, batch in enumerate(data_loader):
        data = [d.data[0] for d in batch]
        box_targets = [d.label[0] for d in batch]
        cls_targets = [nd.cast(d.label[1], dtype='float32') for d in batch]
        for x, box_target, cls_target in zip(data, box_targets, cls_targets):
            print (x.shape, box_target.shape, cls_target.shape)
        if i >= 5:
            assert False





