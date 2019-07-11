import os
import sys
import math
sys.path.insert(0, os.path.expanduser('~/incubator-mxnet/python'))
import mxnet as mx
import numpy as np
from nvidia import dali
from nvidia.dali.pipeline import Pipeline
from .image import gaussian_radius
from .image import draw_umich_gaussian

class CenterNetTrainPipeline(Pipeline):
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
    def __init__(self, device_id, batch_size, data_shape, num_workers,
                 dataset_reader):
        super(CenterNetTrainPipeline, self).__init__(
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

    def define_graph(self):
        saturation = self.rng1()
        contrast = self.rng1()
        brightness = self.rng2()
        hue = self.rng3()
        coin_rnd = self.flip_coin()

        images, bboxes, labels, img_ids = self.dataset_reader()

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

        return (images, bboxes.gpu(), labels.gpu(), img_ids.gpu())


class CenterNetTrainLoader(object):
    def __init__(self, pipelines, size, batch_size, num_classes, data_shape):
        self.pipelines = pipelines
        self.size = size
        print ('size {}'.format(size))
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.width, self.height = data_shape
        self.output_w, self.output_h = self.width//4, self.height//4
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
        batch_origin_gtbox = []
        for idx, pipe in enumerate(self.pipelines):
            data, bboxes, labels, img_ids = pipe.run()
            img, hp, offset, wh, origin_gtbox = self.format_data(data, bboxes, labels, idx)
            data_batch = mx.io.DataBatch(data=[img], label=[hp, offset, wh])
            img_ids = [int(img_ids.as_cpu().at(idx)) for idx in range(self.batch_size)]
            img_ids = np.array(img_ids)
            batch_data.append(data_batch)
            batch_img_ids.append(img_ids)
            batch_origin_gtbox.append(origin_gtbox)
        
        """
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
        """
        
        return batch_data, batch_img_ids, batch_origin_gtbox
    
    def format_data(self, data, bboxes, labels, idx):
        ctx = mx.gpu(idx)
        batch_img = []
        batch_hp = []
        batch_offset = []
        batch_wh = [] 
        batch_origin_gtbox = []
        for i in range(self.batch_size):
            img = data.as_cpu().at(i)
            
            img_bbox = bboxes.as_cpu().at(i)
            img_label = labels.as_cpu().at(i)
            # print ('img_box shape: {}'.format(img_bbox.shape))
            # print ('img_label shape: {}'.format(img_label.shape))
        
            img_bbox[:, 0] *= self.width
            img_bbox[:, 1] *= self.height
            img_bbox[:, 2] *= self.width
            img_bbox[:, 3] *= self.height
            
            origin_gtbox = np.concatenate((img_bbox, img_label), axis=-1)
            batch_origin_gtbox.append(origin_gtbox)
            
            img_label -= 1  # for map to heatmap as index

            num_box = img_bbox.shape[0]
            num_label = img_label.shape[0]
            assert num_box == num_label, 'Expected same length of boxes and labels,\
                got {} and {}'.format(num_box, num_label)
            
            heatmap = np.zeros((self.num_classes, self.output_h, self.output_w), dtype=np.float32)
            offset = np.zeros((2, self.output_h, self.output_w), dtype=np.float32)
            wh = np.zeros((2, self.output_h, self.output_w), dtype=np.float32)
            for k in range(num_box):
                bbox = img_bbox[k, :]
                xmin, ymin, xmax, ymax = bbox
                cls_id = img_label[k]
                h, w = (ymax-ymin)//4, (xmax-xmin)//4
                radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                radius = max(0, int(radius))
                ct = np.array([(bbox[0] + bbox[2]) / (2*4), (bbox[1] + bbox[3]) / (2*4)], dtype=np.float32)
                ct_int = ct.astype(np.int32)
                heatmap = draw_umich_gaussian(heatmap, cls_id, ct_int, radius)
                ct_offset = ct - ct_int

                offset[:, ct_int[1], ct_int[0]] = ct_offset
                wh[:, ct_int[1], ct_int[1]] = np.array(h, w)
            batch_img.append(np.expand_dims(img, axis=0))
            batch_hp.append(np.expand_dims(heatmap, axis=0))
            batch_offset.append(np.expand_dims(offset, axis=0))
            batch_wh.append(np.expand_dims(wh, axis=0))
        batch_img = np.concatenate(batch_img, axis=0)
        batch_hp = np.concatenate(batch_hp, axis=0)
        batch_offset = np.concatenate(batch_offset, axis=0)
        batch_wh = np.concatenate(batch_wh, axis=0)
        batch_img = mx.nd.array(batch_img, ctx=ctx)
        batch_hp = mx.nd.array(batch_hp, ctx=ctx)
        batch_offset = mx.nd.array(batch_offset, ctx=ctx)
        batch_wh = mx.nd.array(batch_wh, ctx=ctx)

        return batch_img, batch_hp, batch_offset, batch_wh, batch_origin_gtbox
    
    def next(self):
        return self.__next__()
    
    def __iter__(self):
        return self

    def reset(self):
        for pipe in self.pipelines:
            pipe.reset()
        self.count = 0


