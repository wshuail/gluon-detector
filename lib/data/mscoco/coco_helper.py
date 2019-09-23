import os
import sys
from pycocotools.coco import COCO


class COCOHelper(object):
    def __init__(self, split, root_dir='~/.mxnet/datasets/coco'):
        file_root = os.path.expanduser(os.path.join(root_dir, split))
        anno_file_name = 'instances_{}.json'.format(split)
        annotations_file = os.path.expanduser(os.path.join(root_dir, 'annotations', anno_file_name))
        self.coco = COCO(os.path.expanduser(annotations_file))
        self.image_dir = os.path.expanduser(os.path.join(root_dir, 'images', split))

    def img_id_to_path(self, img_ids):
        file_infos = self.coco.loadImgs(ids=img_ids)
        file_names = [file_info['file_name'] for file_info in file_infos]
        imgs_path = [os.path.join(self.image_dir, file_name) for file_name in file_names]
        return imgs_path

    def img_id_to_anns(self, img_ids):
        anns = []
        for img_id in img_ids:
            ann_ids = self.coco.getAnnIds(imgIds=[img_id])
            ann = self.coco.loadAnns(ids=ann_ids)
            boxes = []
            for obj in ann:
                bbox = obj['bbox']
                cls = obj['category_id']
                box_info = bbox + [cls]
                boxes.append(box_info)
            anns.append(boxes)
        return anns


if __name__ == '__main__':
    split = 'train2017'
    coco = COCOHelper(split)
    # img_ids = sorted(coco.coco.getImgIds()[0: 5])
    # imgs_path = coco.img_id_to_path(img_ids)
    # anns = coco.img_id_to_anns(img_ids)
    contiguous_id_to_json = {v: k for k, v in enumerate(coco.coco.getCatIds())}
    print (contiguous_id_to_json)
