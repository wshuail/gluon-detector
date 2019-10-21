import os
import sys
import numpy as np
import pickle

root_dir_1 = '/world/data-gpu-107/wangshuailong/data/retinanet/loader1'
root_dir_2 = '/world/data-gpu-107/wangshuailong/data/retinanet/loader2'

def get_image_ids(root_dir):
    image_ids = {}
    for pickle_name in os.listdir(root_dir):
        image_id = pickle_name.split('_')[0]
        if image_id not in image_ids:
            image_ids[image_id] = pickle_name
        else:
            print ('image_id {} already in.'.format(image_id))
    return image_ids

def load_pickle(pickle_path):
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    return data

image_ids_1 = get_image_ids(root_dir_1)
image_ids_2 = get_image_ids(root_dir_2)

image_ids_1_list = list(image_ids_1.keys())



no_count = 0
for i in range(1000):  # len(image_ids_1_list)):
    image_id = image_ids_1_list[i]
    if image_id in image_ids_2:
        pickle_name_1 = image_ids_1[image_id]
        pickle_1_path = os.path.join(root_dir_1, pickle_name_1)
        data_1 = load_pickle(pickle_1_path)
        image_1 = data_1['image']
        box_targets_1 = data_1['box_targets']
        cls_targets_1 = data_1['cls_targets']
        bboxes_1 = data_1['bboxes']
        labels_1 = data_1['labels']

        pickle_name_2 = image_ids_2[image_id]
        pickle_2_path = os.path.join(root_dir_2, pickle_name_2)
        data_2 = load_pickle(pickle_2_path)

        image_2 = data_2['image']
        box_targets_2 = data_2['box_targets']
        cls_targets_2 = data_2['cls_targets']
        bboxes_2 = data_2['bboxes']
        labels_2 = data_2['labels']

        if np.sum(bboxes_1[0: -1, :]) - np.sum(bboxes_2) >= 10:
            print (image_id)
            print (bboxes_1.shape)
            print (bboxes_1)
            print (bboxes_2.shape)
            print (bboxes_2)

    else:
        # print ('image id {} not result 2.'.format(image_id))
        no_count += 0
print ('no count: {}'.format(no_count))
