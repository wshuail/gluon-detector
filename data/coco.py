import os
import sys
import numpy as np
from pycocotools.coco import COCO

class coco(object):
    classes = ['__background__',  # always index 0
           'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
           'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
           'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
           'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
           'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
           'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
           'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
           'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
           'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
           'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
           'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
           'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
           'scissors', 'teddy bear', 'hair drier', 'toothbrush']

    def __init__(self, image_set, root=os.path.expanduser('~/.mxnet/datasets/coco')):
        self._anno_file = os.path.join(root, 'annotations', 'instances_' + image_set + '.json')
        self.coco = COCO(self._anno_file)

        self.image_root = os.path.join(root, 'images', image_set)
        
        self.coco_id_to_class_id = self.get_cat_idx_dict()

        # self._roidb = self.load_gt_roidb()
        # self.filter_roidb()

    def filter_roidb(self):
        num_roidb = len(self._roidb)
        self._roidb = [roi_rec for roi_rec in self._roidb if len(roi_rec['gt_classes'])]
        num_after = len(self._roidb)
        print('filter roidb: {} -> {}'.format(num_roidb, num_after))

    def get_cat_idx_dict(self):
        cat_ids = self.coco.getCatIds()
        cats = [cat['name'] for cat in self.coco.loadCats(cat_ids)]
        coco_cats_ids = dict(zip(cats, cat_ids))

        class_cats_ids = dict(zip(self.classes, range(len(self.classes))))

        coco_id_to_class_id = dict([(coco_cats_ids[cls], class_cats_ids[cls]-1) for cls in self.classes[1: ]])
        
        return coco_id_to_class_id

    def load_image_annotation(self, index):
        img_info = self.coco.loadImgs(index)[0]
        # print ('img_info: {}'.format(img_info))

        file_name = img_info['file_name']
        file_path = os.path.join(self.image_root, file_name)
        assert os.path.exists(file_path), 'file_path {} doesn\'t exist.'.format(file_path)

        width = img_info['width']
        height = img_info['height']

        anno_ids = self.coco.getAnnIds(imgIds=index, iscrowd=False)
        objs = self.coco.loadAnns(anno_ids)
        # print ('objs: {}'.format(objs))

        # sanitize bboxes
        valid_objs = []
        for obj in objs:
            x, y, w, h = obj['bbox']
            x1 = np.max((0, x))
            y1 = np.max((0, y))
            x2 = np.min((width - 1, x1 + np.max((0, w - 1))))
            y2 = np.min((height - 1, y1 + np.max((0, h - 1))))
            if obj['area'] > 0 and x2 >= x1 and y2 >= y1:
                obj['clean_bbox'] = [x1, y1, x2, y2]
                valid_objs.append(obj)
        objs = valid_objs
        num_objs = len(objs)

        bboxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs,), dtype=np.int32)
        for idx, obj in enumerate(objs):
            coco_id = obj['category_id']
            cls_id = self.coco_id_to_class_id[coco_id]
            bboxes[idx, :] = obj['clean_bbox']
            gt_classes[idx] = cls_id

        roi_rec = {'index': index,
                   'image': file_path,
                   'height': height,
                   'width': width,
                   'boxes': bboxes,
                   'gt_classes': gt_classes,
                   'flipped': False}
        return roi_rec

    def load_gt_roidb(self):

        image_ids = self.coco.getImgIds()
        roidb = [self.load_image_annotation(image_id) for image_id in image_ids]

        return roidb
