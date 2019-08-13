import os
import sys
import math
sys.path.insert(0, os.path.expanduser('~/incubator-mxnet/python'))
import mxnet as mx
import numpy as np
from nvidia import dali
from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin.mxnet import feed_ndarray
from .image import gaussian_radius
from .image import draw_umich_gaussian

class CenterNetTrainPipeline(Pipeline):
    def __init__(self, split, batch_size, data_shape, num_shards, device_id,
                 num_workers, root_dir='~/.mxnet/datasets/coco'):
        super(CenterNetTrainPipeline, self).__init__(
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
            shuffle_after_epoch=False,
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

        return (images, bboxes.gpu(), labels.gpu(), img_ids.gpu())

    def size(self):
        """Returns size of COCO dataset
        """
        return self._size


class CenterNetTrainLoader(object):
    def __init__(self, pipelines, size, batch_size, num_classes, data_shape,
                 max_objs=128, dense_mode=True):
        self.pipelines = pipelines
        self.size = size
        print ('size {}'.format(size))
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.width, self.height = data_shape
        self.max_objs = max_objs
        self.dense_mode = dense_mode
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
            if self.dense_mode:
                img, hp, offset, wh, origin_gtbox = self.format_data(data, bboxes, labels, idx)
                data_batch = mx.io.DataBatch(data=[img], label=[hp, offset, wh])
            else:
                img, hp, offset, wh, ind, origin_gtbox = self.format_data(data, bboxes, labels, idx)
                data_batch = mx.io.DataBatch(data=[img], label=[hp, offset, wh, ind])
            # img, bboxes, labels = self.format_data(data, bboxes, labels, idx)
            # data_batch = mx.io.DataBatch(data=[img], label=[bboxes, labels])
            img_ids = [int(img_ids.as_cpu().at(idx)) for idx in range(self.batch_size)]
            img_ids = np.array(img_ids)
            batch_data.append(data_batch)
            batch_img_ids.append(img_ids)
            batch_origin_gtbox.append(origin_gtbox)
        
        self.count += self.num_worker * self.batch_size
        
        return batch_data, batch_img_ids, batch_origin_gtbox
    
    def format_data(self, data, bboxes, labels, idx):
        ctx = mx.gpu(idx)
        # Copy data from DALI Tensors to MXNet NDArrays
        data = data.as_tensor()
        dtype = np.dtype(data.dtype())
        batch_img = mx.nd.zeros(data.shape(), ctx=ctx, dtype=dtype)
        feed_ndarray(data, batch_img)
        
        batch_hp = []
        batch_offset = []
        batch_wh = [] 
        batch_origin_gtbox = []
        if not self.dense_mode:
            batch_ind = []
        for i in range(self.batch_size):
            
            bbox_tensor = bboxes.at(i)
            dtype = np.dtype(bbox_tensor.dtype())
            img_bbox = mx.nd.zeros(bbox_tensor.shape(), ctx=ctx, dtype=dtype)
            feed_ndarray(bbox_tensor, img_bbox)
            
            label_tensor = labels.at(i)
            dtype = np.dtype(label_tensor.dtype())
            img_label = mx.nd.zeros(label_tensor.shape(), ctx=ctx, dtype=dtype)
            feed_ndarray(label_tensor, img_label)
        
            img_bbox[:, 0] *= self.width
            img_bbox[:, 1] *= self.height
            img_bbox[:, 2] *= self.width
            img_bbox[:, 3] *= self.height

            origin_gtbox = mx.nd.concat(img_bbox, img_label.astype(img_bbox), dim=-1)
            batch_origin_gtbox.append(origin_gtbox)
            
            img_label -= 1  # for map to heatmap as index

            num_box = img_bbox.shape[0]
            # print ('num_box: {}'.format(num_box))
            num_label = img_label.shape[0]
            assert num_box == num_label, 'Expected same length of boxes and labels,\
                got {} and {}'.format(num_box, num_label)
            
            heatmap = np.zeros((self.num_classes, self.output_h, self.output_w), dtype=np.float32)
            if self.dense_mode:
                offset = np.zeros((2, self.output_h, self.output_w), dtype=np.float32)
                wh = np.zeros((2, self.output_h, self.output_w), dtype=np.float32)
            else:
                wh = np.zeros((self.max_objs, 2), dtype=np.float32)
                offset = np.zeros((self.max_objs, 2), dtype=np.float32)
                ind = np.zeros((self.max_objs), dtype=np.int64)

            # rescale bbox for feature map
            img_bbox /= 4.0
            hs = img_bbox[:, 3] - img_bbox[:, 1]
            ws = img_bbox[:, 2] - img_bbox[:, 0]
            cx = (img_bbox[:, 2] + img_bbox[:, 0])/2
            cy = (img_bbox[:, 3] + img_bbox[:, 1])/2
            cx = cx.reshape((-1, 1))
            cy = cy.reshape((-1, 1))
            ct_xy = mx.nd.concat(cx, cy, dim=1)
            ct_xy_int = ct_xy.astype(np.int32)
            
            for k in range(min(num_box, self.max_objs)):
                # bbox = img_bbox[k, :].asnumpy()
                # xmin, ymin, xmax, ymax = bbox
                h = hs[k].asnumpy()
                w = ws[k].asnumpy()
                cls_id = img_label[k].asnumpy()
                # h, w = (ymax-ymin), (xmax-xmin)
                radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                radius = max(0, int(radius))
                # ct = np.array([(bbox[0] + bbox[2])/2, (bbox[1] + bbox[3])/2], dtype=np.float32)
                # ct_int = ct.astype(np.int32)
                ct = ct_xy[k, :].asnumpy()
                ct_int = ct_xy_int[k, :].asnumpy()
                heatmap = draw_umich_gaussian(heatmap, cls_id, ct_int, radius)
                # print ('hm sum: {}'.format(np.sum(heatmap)))
                if self.dense_mode:
                    offset[:, ct_int[1], ct_int[0]] = ct - ct_int  # order: w, h
                    wh[:, ct_int[1], ct_int[0]] = np.array((h, w)).reshape((1, 2))
                else:
                    ind[k] = ct_int[1] * self.output_w + ct_int[0]
                    offset[k, :] = ct - ct_int  # order: w, h
                    wh[k, :] = np.array((w, h)).reshape((1, 2))

            batch_hp.append(np.expand_dims(heatmap, axis=0))
            batch_offset.append(np.expand_dims(offset, axis=0))
            batch_wh.append(np.expand_dims(wh, axis=0))
            if not self.dense_mode:
                batch_ind.append(np.expand_dims(ind, axis=0))
        batch_hp = np.concatenate(batch_hp, axis=0)
        batch_offset = np.concatenate(batch_offset, axis=0)
        batch_wh = np.concatenate(batch_wh, axis=0)
        if not self.dense_mode:
            batch_ind = np.concatenate(batch_ind, axis=0)
        
        batch_hp = mx.nd.array(batch_hp, ctx=ctx)
        batch_offset = mx.nd.array(batch_offset, ctx=ctx)
        batch_wh = mx.nd.array(batch_wh, ctx=ctx)
        if not self.dense_mode:
            batch_idx = mx.nd.array(batch_ind, ctx=ctx)

        if self.dense_mode:
            return batch_img, batch_hp, batch_offset, batch_wh, batch_origin_gtbox
        else:
            return batch_img, batch_hp, batch_offset, batch_wh, batch_idx, batch_origin_gtbox

    
    def next(self):
        return self.__next__()
    
    def __iter__(self):
        return self

    def reset(self):
        for pipe in self.pipelines:
            pipe.reset()
        self.count = 0


