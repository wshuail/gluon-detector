import os
import sys
import math
sys.path.insert(0, os.path.expanduser('~/incubator-mxnet/python'))
import mxnet as mx
import numpy as np
from nvidia import dali
from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin.mxnet import feed_ndarray


def _gaussian_radius(det_size, min_overlap=0.7):
    """Calculate gaussian radius for foreground objects.

    Parameters
    ----------
    det_size : tuple of int
        Object size (h, w).
    min_overlap : float
        Minimal overlap between objects.

    Returns
    -------
    float
        Gaussian radius.

    """
    height, width = det_size

    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2
    return min(r1, r2, r3)


def _gaussian_2d(shape, sigma=1):
    """Generate 2d gaussian.

    Parameters
    ----------
    shape : tuple of int
        The shape of the gaussian.
    sigma : float
        Sigma for gaussian.

    Returns
    -------
    float
        2D gaussian kernel.

    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1, -n:n+1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def _draw_umich_gaussian(heatmap, center, radius, k=1):
    """Draw a 2D gaussian heatmap.

    Parameters
    ----------
    heatmap : numpy.ndarray
        Heatmap to be write inplace.
    center : tuple of int
        Center of object (h, w).
    radius : type
        The radius of gaussian.

    Returns
    -------
    numpy.ndarray
        Drawn gaussian heatmap.

    """
    diameter = 2 * radius + 1
    gaussian = _gaussian_2d((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


class CenterNetTrainPipeline(Pipeline):
    def __init__(self, split, batch_size, data_shape, device_id, num_shards=4,
                 num_workers=8, root_dir='~/.mxnet/datasets/coco'):
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
    def __init__(self, split, batch_size, num_classes, data_shape, num_devices):
        self.pipelines = [CenterNetTrainPipeline(split, batch_size, data_shape, device_id)
                           for device_id in range(num_devices)]
        self._size = self.pipelines[0].size()
        print ('size {}'.format(self._size))
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.width, self.height = data_shape, data_shape
        self.output_w, self.output_h = self.width//4, self.height//4
        self.num_worker = len(self.pipelines)
        self.batch_size = self.pipelines[0].batch_size
        for pipeline in self.pipelines:
            pipeline.build()
        
        self.count = 0
    
    def __next__(self):
        
        if self.count >= self._size:
            self.reset()
            raise StopIteration
        
        all_data = []
        all_hm = []
        all_wh_target = []
        all_wh_mask = []
        all_center_reg = []
        all_center_reg_mask = []
        all_img_ids = []
        origin_gtbox = []
        for idx, pipe in enumerate(self.pipelines):
            data, bboxes, labels, img_ids = pipe.run()
            batch_img, batch_hm, batch_wh_target, batch_wh_mask, batch_center_reg,\
                batch_center_reg_mask, batch_origin_gtbox = self.format_data(data, bboxes, labels, idx)
            
            img_ids = [int(img_ids.as_cpu().at(idx)) for idx in range(self.batch_size)]
            img_ids = np.array(img_ids)
        
            all_data.append(batch_img)
            all_hm.append(batch_hm)
            all_wh_target.append(batch_wh_target)
            all_wh_mask.append(batch_wh_mask)
            all_center_reg.append(batch_center_reg)
            all_center_reg.append(batch_center_reg_mask)
            all_center_reg_mask.append(batch_center_reg_mask)
            all_img_ids.append(img_ids)
            origin_gtbox.append(batch_origin_gtbox)
            
        self.count += self.num_worker * self.batch_size
        
        return (all_data, all_hm, all_wh_target, all_wh_mask, all_center_reg,\
            all_center_reg_mask, all_img_ids, origin_gtbox)

    def format_data(self, data, bboxes, labels, idx):
        ctx = mx.gpu(idx)
        # Copy data from DALI Tensors to MXNet NDArrays
        data = data.as_tensor()
        dtype = np.dtype(data.dtype())
        batch_img = mx.nd.zeros(data.shape(), ctx=ctx, dtype=dtype)
        feed_ndarray(data, batch_img)
        
        batch_hm = []
        batch_wh_target = [] 
        batch_wh_mask = [] 
        batch_center_reg = []
        batch_center_reg_mask = []
        batch_origin_gtbox = []
        
        for i in range(self.batch_size):
            img_bbox = bboxes.as_cpu().at(i)
            img_label = labels.as_cpu().at(i)
            
            img_bbox[:, 0] *= self.width
            img_bbox[:, 1] *= self.height
            img_bbox[:, 2] *= self.width
            img_bbox[:, 3] *= self.height

            origin_gtbox = np.concatenate((img_bbox, img_label.astype(img_bbox.dtype)), axis=-1)
            batch_origin_gtbox.append(origin_gtbox)
            
            # rescale bbox for feature map
            img_bbox /= 4.0
            img_label -= 1  # for map to heatmap as index

            num_box = img_bbox.shape[0]

            heatmap = np.zeros((self.num_classes, self.output_h, self.output_w), dtype=np.float32)
            wh_target = np.zeros((2, self.output_h, self.output_w), dtype=np.float32)
            wh_mask = np.zeros((2, self.output_h, self.output_w), dtype=np.float32)
            center_reg = np.zeros((2, self.output_h, self.output_w), dtype=np.float32)
            center_reg_mask = np.zeros((2, self.output_h, self.output_w), dtype=np.float32)

            ws = img_bbox[:, 2] - img_bbox[:, 0]
            hs = img_bbox[:, 3] - img_bbox[:, 1]
            cx = (img_bbox[:, 2] + img_bbox[:, 0])/2.0
            cy = (img_bbox[:, 3] + img_bbox[:, 1])/2.0
            cx = cx.reshape((-1, 1))
            cy = cy.reshape((-1, 1))
            ct_xy = np.concatenate((cx,cy), axis=-1)
            ct_xy_int = ct_xy.astype(np.int32)
        
            h_scale = 1.0  # already scaled in affinetransform
            w_scale = 1.0  # already scaled in affinetransform
            
            for k in range(num_box):
                cid = int(img_label[k])
                box_h, box_w = hs[k], ws[k]
                if box_h > 0 and box_w > 0:
                    radius = _gaussian_radius((np.ceil(box_h), np.ceil(box_w)))
                    radius = max(0, int(radius))
                    center = ct_xy[k]
                    center_int = ct_xy_int[k]
                    center_x, center_y = center_int
                    _draw_umich_gaussian(heatmap[cid], center_int, radius)
                    wh_target[0, center_y, center_x] = box_w * w_scale
                    wh_target[1, center_y, center_x] = box_h * h_scale
                    wh_mask[:, center_y, center_x] = 1.0
                    center_reg[:, center_y, center_x] = center - center_int
                    center_reg_mask[:, center_y, center_x] = 1.0

            batch_hm.append(np.expand_dims(heatmap, axis=0))
            batch_wh_target.append(np.expand_dims(wh_target, axis=0))
            batch_wh_mask.append(np.expand_dims(wh_mask, axis=0))
            batch_center_reg.append(np.expand_dims(center_reg, axis=0))
            batch_center_reg_mask.append(np.expand_dims(center_reg_mask, axis=0))
        
        batch_hm = np.concatenate(batch_hm, axis=0)
        batch_wh_target = np.concatenate(batch_wh_target, axis=0)
        batch_wh_mask = np.concatenate(batch_wh_mask, axis=0)
        batch_center_reg = np.concatenate(batch_center_reg, axis=0)
        batch_center_reg_mask = np.concatenate(batch_center_reg_mask, axis=0)
        
        batch_hm = mx.nd.array(batch_hm, ctx=ctx)
        batch_wh_target = mx.nd.array(batch_wh_target, ctx=ctx)
        batch_wh_mask = mx.nd.array(batch_wh_mask, ctx=ctx)
        batch_center_reg = mx.nd.array(batch_center_reg, ctx=ctx)
        batch_center_reg_mask = mx.nd.array(batch_center_reg_mask, ctx=ctx)

        return batch_img, batch_hm, batch_wh_target, batch_wh_mask,\
            batch_center_reg, batch_center_reg_mask, batch_origin_gtbox
    
    def next(self):
        return self.__next__()
    
    def __iter__(self):
        return self

    def reset(self):
        for pipe in self.pipelines:
            pipe.reset()
        self.count = 0

    def size(self):
        """Returns size of COCO dataset
        """
        return self._size

if __name__ == '__main__':
    split = 'train2017'
    batch_size = 32
    num_classes = 80
    data_shape = 512
    num_devices = 4
    data_loader = CenterNetTrainLoader(split, batch_size, num_classes, data_shape, num_devices)
    for data_batch in data_loader:
        print ('xx')
