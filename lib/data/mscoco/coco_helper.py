import os
import sys
import numpy as np
from pycocotools.coco import COCO


class COCOHelper(object):
    def __init__(self, split, root_dir='~/.mxnet/datasets/coco'):
        file_root = os.path.expanduser(os.path.join(root_dir, split))
        anno_file_name = 'instances_{}.json'.format(split)
        annotations_file = os.path.expanduser(os.path.join(root_dir, 'annotations', anno_file_name))
        self.coco = COCO(os.path.expanduser(annotations_file))
        self.image_dir = os.path.expanduser(os.path.join(root_dir, 'images', split))

    def img_id_to_path(self, img_ids):
        files_info = self.coco.loadImgs(ids=img_ids)
        images_info = []
        for file_info in files_info:
            img_info = {}
            height = file_info['height']
            width = file_info['width']
            file_name = file_info['file_name']
            img_path = os.path.join(self.image_dir, file_name)
            img_info['height'] = height
            img_info['width'] = width
            img_info['img_path'] = img_info
            images_info.append(img_info)
        return images_info

    def img_id_to_anns(self, img_ids):
        anns = []
        for img_id in img_ids:
            ann_ids = self.coco.getAnnIds(imgIds=[img_id])  # , iscrowd=True)
            ann = self.coco.loadAnns(ids=ann_ids)
            boxes = []
            for obj in ann:
                print (obj['bbox'])
                xmin, ymin, w, h = obj['bbox']
                xmax = xmin + w
                ymax = ymin + h
                cls = obj['category_id']
                box_info = [xmin, ymin, xmax, ymax, cls]
                boxes.append(box_info)
            boxes = np.array(boxes).reshape((-1, 5))
            anns.append(boxes)
        return anns


if __name__ == '__main__':
    split = 'train2017'
    coco = COCOHelper(split)
    image_ids = [475808]
    images_info = coco.img_id_to_path(image_ids)
    anns = coco.img_id_to_anns(image_ids)
    for image_info, ann in zip(images_info, anns):
        w = image_info['width']
        h = image_info['height']
        ann[:, 0] *= (512.0/w)
        ann[:, 1] *= (512.0/h)
        ann[:, 2] *= (512.0/w)
        ann[:, 3] *= (512.0/h)
        print (ann)
        print (ann.shape)

