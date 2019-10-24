import os
import sys
import random
import logging
logging.getLogger().setLevel(level=logging.INFO)
import numpy as np
from pycocotools.coco import COCO
sys.path.insert(0, os.path.expanduser('~/incubator-mxnet/python'))
import mxnet as mx
from mxnet import nd
from nvidia import dali
from nvidia.dali import ops
from nvidia.dali import types
from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin.mxnet import feed_ndarray
sys.path.insert(0, os.path.expanduser('~/gluon_detector'))
from lib.anchor.retinanet import generate_retinanet_anchors


class AspectRatioBasedSampler(object):
    def __init__(self, split, thread_batch_size=2, num_devices=4, root_dir='~/.mxnet/datasets/coco'):
        self.thread_batch_size = thread_batch_size
        self.num_devices = num_devices
        self.batch_size = thread_batch_size*num_devices
        self.image_dir = os.path.expanduser(os.path.join(root_dir, 'images', split))
        anno_file_name = 'instances_{}.json'.format(split)
        annotations_file = os.path.expanduser(os.path.join(root_dir, 'annotations', anno_file_name))
        self.coco = COCO(annotations_file)
        self.image_ids = self.coco.getImgIds()
        logging.info('len of image ids: {}'.format(len(self.image_ids)))
        self.image_ids = self.filter_image_id(self.image_ids)
        self.coco_label_to_contiguous_id = {v: (k+1) for k, v in enumerate(self.coco.getCatIds())}

        self.batch_long_image_ids, self.batch_wide_image_ids, self.annotations = self.init_annotations()
        self.num_batch = len(self.batch_long_image_ids) + len(self.batch_wide_image_ids)
        print ('num_batch: {}'.format(self.num_batch))

        self.reset()

    def filter_image_id(self, image_ids):
        logging.info('len of image ids before filter: {}'.format(len(image_ids)))
        # image_ids = [image_id for image_id in image_ids if len(self.coco.getAnnIds(imgIds=image_id, iscrowd=False))>0]
        image_ids = [image_id for image_id in image_ids if len(self.coco.getAnnIds(imgIds=image_id))>0]
        logging.info('len of image ids after filter: {}'.format(len(image_ids)))
        return image_ids

    def init_annotations(self):
        hw_ratios = {}
        annotations = {}
        for i, image_id in enumerate(self.image_ids):
            hw_ratio, image_path, bboxes, labels = self.load_image_info(image_id)
            hw_ratios[image_id] = hw_ratio
            annotations[image_id] = {'image_path': image_path, 'bboxes': bboxes, 'labels': labels}

        wide_items, long_items = {}, {}
        for k, v in hw_ratios.items():
            if v >= 1:
                long_items[k] = v
            else:
                wide_items[k] = v

        batch_long_image_ids, num_long_example = self.pad_items_list(long_items)
        logging.info('len of long image ids: {}'.format(num_long_example))
        batch_wide_image_ids, num_wide_example = self.pad_items_list(wide_items)
        logging.info('len of wide image ids: {}'.format(num_wide_example))
        self._size = num_long_example + num_wide_example
        logging.info('len of total size for COCO: {}'.format(self._size))

        return batch_long_image_ids, batch_wide_image_ids, annotations

    def pad_items_list(self, hw_ratios):
        hw_ratios = [[k, v] for k, v in hw_ratios.items()]
        if len(hw_ratios)%self.batch_size != 0:
            pad_size = self.batch_size - len(hw_ratios)%self.batch_size
        else:
            pad_size = 0
        pad_items = random.choices(hw_ratios, k=pad_size)
        hw_ratios += pad_items
        hw_ratios = sorted(hw_ratios, key=lambda kv: kv[1])

        image_ids = [hw_ratio[0] for hw_ratio in hw_ratios]
        num_example = len(image_ids)
        batch_image_ids = [image_ids[i: (i+self.batch_size)] for i in range(0, len(image_ids), self.batch_size)]
        
        return batch_image_ids, num_example
    
    def reset(self):
        random.shuffle(self.batch_long_image_ids)
        grouped_batch_long_image_ids = [[batch_ids[i: i+self.thread_batch_size] for batch_ids in \
                                         self.batch_long_image_ids] for i in \
                                        range(0, self.batch_size, self.thread_batch_size)]
        random.shuffle(self.batch_wide_image_ids)
        grouped_batch_wide_image_ids = [[batch_ids[i: i+self.thread_batch_size] for batch_ids in \
                                         self.batch_wide_image_ids] for i in \
                                        range(0, self.batch_size, self.thread_batch_size)]
        for i in range(self.num_devices):
            grouped_batch_long_image_ids[i] += grouped_batch_wide_image_ids[i]
        self.grouped_batch_image_ids = grouped_batch_long_image_ids
        
        self.counter = dict.fromkeys(list(range(self.num_devices)), 0)

    def iter_done(self):
        counters = self.counter.values()
        done = np.all(np.array(list(counters)) >= self.num_batch)
        if done:
            return True
        else:
            return False
   
    def __next__(self, device_id):
        
        if self.iter_done():
            self.reset()
            raise StopIteration
        
        thread_counter = self.counter[device_id]

        batch_image_ids = self.grouped_batch_image_ids[device_id][thread_counter]
        # batch_image_ids = [94326, 94326]
        batch_data = [self.annotations[image_id] for image_id in batch_image_ids]
        batch_images = [open(image_info['image_path'], 'rb') for image_info in batch_data]
        batch_images = [np.frombuffer(f.read(), dtype = np.uint8) for f in batch_images]
        batch_bboxes = [image_info['bboxes'] for image_info in batch_data]
        batch_labels = [image_info['labels'] for image_info in batch_data]
        self.counter[device_id] += 1
        
        return batch_image_ids, batch_images, batch_bboxes, batch_labels

    def next(self, device_id):
        return self.__next__(device_id)
    
    def __iter__(self):
        return self

    def load_image_info(self, image_id):
        image_info = self.coco.loadImgs(image_id)[0]
        image_id = image_info['id']
        height = image_info['height']
        width = image_info['width']
        hw_ratio = height/width
        image_path = os.path.join(self.image_dir, image_info['file_name'])
    
        # get ground truth annotations
        annotations_ids = self.coco.getAnnIds(imgIds=image_id)  # , iscrowd=False)
        annotations     = np.zeros((0, 5))

        # parse annotations
        coco_annotations = self.coco.loadAnns(annotations_ids)
        for idx, a in enumerate(coco_annotations):

            # some annotations have basically no width / height, skip them
            if a['bbox'][2] < 1 or a['bbox'][3] < 1:
                continue

            annotation        = np.zeros((1, 5))
            annotation[0, :4] = a['bbox']
            annotation[0, 4]  = self.coco_label_to_contiguous_id[a['category_id']]
            annotations       = np.append(annotations, annotation, axis=0)
        
        # for image without bboxes
        if annotations.shape[0] == 0:
            annotations     = np.zeros((1, 5))


        # transform from [xmin, ymin, w, h] to [xmin, ymin, xmax, ymax]
        annotations[:, 2] = annotations[:, 0] + annotations[:, 2]
        annotations[:, 3] = annotations[:, 1] + annotations[:, 3]

        bboxes = annotations[:, 0:4].reshape((-1, 4))
        labels = annotations[:, 4].reshape((-1, 1))

        bboxes[:, 0] /= width
        bboxes[:, 1] /= height
        bboxes[:, 2] /= width
        bboxes[:, 3] /= height

        return hw_ratio, image_path, bboxes, labels


class TrainPipeline(Pipeline):
    def __init__(self, iterator, batch_size, max_size, resize_shorter, thread_id,
                 num_threads, device_id, fix_shape=False):
        super(TrainPipeline, self).__init__(batch_size,
                                      num_threads,
                                      device_id,
                                      seed=12)
        self.iterator = iter(iterator)
        self.thread_id = thread_id
        self._size = iterator._size
        self._first_iter = False
        self.input_images_ids_op = ops.ExternalSource()
        self.input_images_op = ops.ExternalSource()
        self.input_bboxes_op = ops.ExternalSource()
        self.input_labels_op = ops.ExternalSource()
        self.decode = ops.ImageDecoder(device = "mixed", output_type = types.RGB)
        
        # Augumentation techniques
        self.crop = dali.ops.RandomBBoxCrop(
            device="cpu",
            aspect_ratio=[0.5, 2.0],
            thresholds=[0, 0.1, 0.3, 0.5, 0.7, 0.9],
            # scaling=[0.3, 1.0],
            ltrb=True,
            allow_no_crop=True,
            num_attempts=1)
        self.slice = dali.ops.Slice(device="gpu")
        self.twist = dali.ops.ColorTwist(device="gpu")

        if fix_shape:
            data_shape = 512
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
        
        # Random variables
        self.rng1 = dali.ops.Uniform(range=[0.5, 1.5])
        self.rng2 = dali.ops.Uniform(range=[0.875, 1.125])
        self.rng3 = dali.ops.Uniform(range=[-0.5, 0.5])

        self.flip = dali.ops.Flip(device="gpu")
        self.bbflip = dali.ops.BbFlip(device="cpu", ltrb=True)
        self.flip_coin = dali.ops.CoinFlip(probability=0.5)
        
        self.cast = dali.ops.Cast(dtype = dali.types.FLOAT)

        logging.info('Train Pipeline Initilized.')

    def define_graph(self):                                                                
        
        saturation = self.rng1()
        contrast = self.rng1()
        brightness = self.rng2()
        hue = self.rng3()
        coin_rnd = self.flip_coin()
        
        self.image_ids = self.input_images_ids_op()
        self.images = self.input_images_op()
        self.bboxes = self.input_bboxes_op()
        self.labels = self.input_labels_op()
        images = self.decode(self.images)                                                   

        crop_begin, crop_size, bboxes, labels = self.crop(self.bboxes, self.labels)
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
        # images = self.normalize(images)
        labels = self.cast(labels)

        return images, bboxes.gpu(), labels.gpu(), self.image_ids.gpu()

    def iter_setup(self):
        (image_ids, images, bboxes, labels) = self.iterator.next(self.device_id)
        image_ids = [np.array(image_id) for image_id in image_ids]
        images = [np.ascontiguousarray(image) for image in images]
        bboxes = [np.ascontiguousarray(bbox) for bbox in bboxes]
        labels = [np.ascontiguousarray(label) for label in labels]
        bboxes = [bbox.astype(np.float32) for bbox in bboxes]
        labels = [label.astype(np.int32) for label in labels]
        self.feed_input(self.image_ids, image_ids)
        self.feed_input(self.images, images)
        self.feed_input(self.bboxes, bboxes)
        self.feed_input(self.labels, labels)
    
    def size(self):
        """Returns size of COCO dataset
        """
        return self._size

    def reset_iterator(self):
        self.iterator.reset()


class RetinaNetTrainLoader(object):
    def __init__(self, split, thread_batch_size, max_size, resize_shorter,
                 num_devices=4, stds=(0.1, 0.1, 0.2, 0.2), means=(0., 0., 0., 0.),
                 fix_shape=False, **kwargs):
        sampler = AspectRatioBasedSampler(split=split, thread_batch_size=thread_batch_size)
        pipes = [TrainPipeline(sampler, batch_size=thread_batch_size, max_size=max_size,
                               resize_shorter=resize_shorter, fix_shape=fix_shape, num_threads=2,
                               thread_id=i, device_id=i) for i in range(num_devices)]
        self.pipelines = pipes
        self.max_size = max_size
        self.resize_shorter = resize_shorter
        self._stds = stds
        self._means = means
        self.fix_shape = fix_shape
        self.num_worker = num_devices
        self._size = self.pipelines[0].size()
        self.batch_size = self.pipelines[0].batch_size
        print ('dataloader size: {}, batch_size: {}'.format(self._size, self.batch_size))
        for pipeline in self.pipelines:
            pipeline.build()
        
        self.anchors_pool = {}
        self.count = 0
        
        self.mean = nd.array([0.485 * 255, 0.456 * 255, 0.406 * 255]).reshape(1, -1, 1, 1)
        self.std = nd.array([0.229 * 255, 0.224 * 255, 0.225 * 255]).reshape(1, -1, 1, 1)
        self.mean_list = [self.mean.as_in_context(mx.gpu(i)) for i in range(num_devices)]
        self.std_list = [self.std.as_in_context(mx.gpu(i)) for i in range(num_devices)]
        
    def __next__(self):
        if self.count >= self._size:
            self.reset()
            raise StopIteration
        
        images = []
        all_bboxes = []
        all_labels = []
        box_targets = []
        cls_targets = []
        all_image_ids = []
        for idx, pipe in enumerate(self.pipelines):
            ctx = mx.gpu(idx)
            batch_data, batch_bboxes, batch_labels, batch_image_ids = pipe.run()
            batch_images = []
            batch_xywh_bboxes = []
            batch_cls_ids = []
            batch_image_ids_ = []
            batch_box_targets, batch_cls_targets = [], []
            for i in range(self.batch_size):
                image = batch_data.at(i)
                image = self.feed_tensor_into_mx(image, ctx)
                image = nd.transpose(image, (2, 0, 1)).astype(np.float32)
                c, h, w = image.shape
                batch_images.append(image)

                bboxes = batch_bboxes.at(i)
                bboxes = self.feed_tensor_into_mx(bboxes, ctx)
                bboxes = self._normalized_ltrb_to_xywh(bboxes, h, w)
                batch_xywh_bboxes.append(bboxes)
                
                cls_ids = batch_labels.at(i)
                cls_ids = self.feed_tensor_into_mx(cls_ids, ctx)
                batch_cls_ids.append(cls_ids)

                image_ids = batch_image_ids.at(i)
                image_ids = self.feed_tensor_into_mx(image_ids, ctx)
                batch_image_ids_.append(image_ids)

            hs = [image.shape[1] for image in batch_images]
            ws = [image.shape[2] for image in batch_images]

            if not self.fix_shape:
                hw_ratio = hs[0]/ws[0]
                if hw_ratio >= 1:
                    image_h, image_w = self.max_size, self.resize_shorter
                    pad_hs = [image_h - image.shape[1] for image in batch_images]
                    pad_ws = [image_w - image.shape[2] for image in batch_images]
                else:
                    image_h, image_w = self.resize_shorter, self.max_size
                    pad_hs = [image_h - image.shape[1] for image in batch_images]
                    pad_ws = [image_w - image.shape[2] for image in batch_images]
            else:
                max_h = max(hs) + (32-max(hs)%32)%32  # in case max(hs)%32 == 0
                max_w = max(ws) + (32-max(ws)%32)%32
                pad_hs = [max_h - image.shape[1] for image in batch_images]
                pad_ws = [max_w - image.shape[2] for image in batch_images]
                image_w, image_h = max_w, max_h
                assert image_w == 512
                assert image_h == 512
            # nd.pad only support 4/5-dimentional data so expand then squeeze
            batch_images = [nd.expand_dims(image, axis=0) for image in batch_images]
            batch_images = [nd.pad(image, mode='constant', constant_value=0.0,
                                   pad_width=(0, 0, 0, 0, 0, pad_h, 0, pad_w))
                            for image, pad_h, pad_w in zip(batch_images, pad_hs, pad_ws)] 
            self.mean = self.mean_list[idx]
            self.std = self.std_list[idx]
            batch_images = [(image-self.mean)/self.std for image in batch_images]
            
            # no need to generate anchors each time
            anchors_name = '{}_{}_{}'.format(image_h, image_w, idx)
            if anchors_name in self.anchors_pool:
                anchors = self.anchors_pool[anchors_name]
            else:
                anchors = generate_retinanet_anchors(image_shape=(image_h, image_w))
                anchors = anchors.as_in_context(ctx)
                self.anchors_pool[anchors_name] = anchors

            for i in range(self.batch_size):
                image = batch_images[i]
                # c, h, w = image.shape
                # print ('image 2 shape: {}'.format(image.shape))

                bboxes = batch_xywh_bboxes[i]
                cls_ids = batch_cls_ids[i]
                
                box_ious = nd.contrib.box_iou(anchors, bboxes, format='center')
                ious, indices = nd.topk(box_ious, axis=-1, ret_typ='both', k=1)
                # print ('max ious: {}'.format(nd.max(ious)))

                box_target = nd.take(bboxes, indices).reshape((-1, 4))
                box_target = self.encode_box_target(box_target, anchors)
                
                cls_target = nd.take(cls_ids, indices).reshape((-1, 1))
     
                mask = nd.ones_like(ious)*-1
                mask = nd.where(ious<0.4, nd.zeros_like(ious), mask)
                mask = nd.where(ious>0.5, nd.ones_like(ious), mask)

                box_mask = nd.tile(mask, reps=(1, 4))
                box_target = nd.where(box_mask, box_target, nd.zeros_like(box_target))
                batch_box_targets.append(box_target)

                cls_target = nd.where(mask == 1.0, cls_target, mask)
                batch_cls_targets.append(cls_target)

            batch_box_targets = [nd.expand_dims(box_target, axis=0) for box_target in batch_box_targets]
            batch_cls_targets = [nd.expand_dims(cls_target, axis=0) for cls_target in batch_cls_targets]

            batch_images = nd.concat(*batch_images, dim=0)
            batch_box_targets = nd.concat(*batch_box_targets, dim=0)
            batch_cls_targets = nd.concat(*batch_cls_targets, dim=0).squeeze()
            # print ('batch_box_targets shape: {}'.format(batch_box_targets.shape))
            # print ('batch_cls_targets shape: {}'.format(batch_cls_targets.shape))

            images.append(batch_images)
            all_bboxes.append(batch_xywh_bboxes)
            all_labels.append(batch_cls_ids)
            box_targets.append(batch_box_targets)
            cls_targets.append(batch_cls_targets)
            all_image_ids.append(batch_image_ids_)

        self.count += self.num_worker*self.batch_size
        
        return images, box_targets, cls_targets, all_image_ids, all_bboxes, all_labels

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
            pipe.reset_iterator()
        self.count = 0
    
    @staticmethod
    def _normalized_ltrb_to_xywh(matrix, image_h, image_w):
        matrix[:, 0] *= image_w
        matrix[:, 1] *= image_h
        matrix[:, 2] *= image_w
        matrix[:, 3] *= image_h
        matrix_xywh = nd.zeros_like(matrix)
        matrix_xywh[:, 0] = (matrix[:, 0] + matrix[:, 2])/2
        matrix_xywh[:, 1] = (matrix[:, 1] + matrix[:, 3])/2
        matrix_xywh[:, 2] = matrix[:, 2] - matrix[:, 0]
        matrix_xywh[:, 3] = matrix[:, 3] - matrix[:, 1]
        
        return matrix_xywh

    def size(self):
        return self._size

if __name__ == '__main__':
    train_split = 'val2017'
    thread_batch_size = 2
    max_size = 1024
    resize_shorter = 640
    data_loader = RetinaNetTrainLoader(split=train_split, thread_batch_size=thread_batch_size,
                                       max_size=max_size, resize_shorter=resize_shorter,
                                       fix_shape=False)
    n = 0
    for i, data_batch in enumerate(iter(data_loader)):
        batch_images, batch_box_targets, batch_cls_targets, batch_image_ids, batch_bboxes, batch_labels = data_batch
        for thread_images, thread_box_targets, thread_cls_targets, thread_image_ids, thread_bboxes, thread_labels in \
                zip(batch_images, batch_box_targets, batch_cls_targets, batch_image_ids, batch_bboxes, batch_labels):
            for i in range(thread_images.shape[0]):
                n += 1
                image_id = thread_image_ids[i]
                print ('image_id: {}'.format(image_id))
                image = thread_images[i, :, :, :]
                print ('image shape: {}'.format(image.shape))

                """
                print ('sum image: {}'.format(nd.sum(image)))
                box_targets = thread_box_targets[i, :, :]
                print ('sum box_targets: {}'.format(nd.sum(box_targets)))
                cls_targets = thread_cls_targets[i, :]
                print ('sum cls_targets: {}'.format(nd.sum(cls_targets)))

                bboxes = thread_bboxes[i]
                print ('bboxes: {}'.format(bboxes))
                labels = thread_labels[i]
                    
                cls_targets_np = cls_targets.asnumpy()
                index = (cls_targets_np>0)
                pos_cls_targets = cls_targets_np[index]
                print ('pos_cls_targets: {}'.format(pos_cls_targets))

                box_targets_np = box_targets.asnumpy()
                pos_box_targets = box_targets_np[index.flatten(), :]
                print ('pos_box_targets: {}'.format(pos_box_targets))
                """
                # if n == 1:
                #     assert False


    print ('final n: {}'.format(n))
 
