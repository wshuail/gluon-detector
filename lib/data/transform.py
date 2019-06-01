import numpy as np
from .utils import bbox as tbbox
from .utils import image as timage

def ssd_transform(img, label, target_shape=(512, 512),
                  rgb_mean=(0.485, 0.456, 0.406),
                  rgb_std=(0.229, 0.224, 0.225)):
    target_width, target_height = target_shape
    # random color jittering
    # img = experimental.image.random_color_distort(src)

    # random expansion with prob 0.5
    if np.random.uniform(0, 1) > 0.5:
        img, expand = timage.random_expand(img, fill=[m * 255 for m in rgb_mean])
        bbox = tbbox.translate(label, x_offset=expand[0], y_offset=expand[1])
    else:
        img, bbox = img, label

    # random cropping
    h, w, _ = img.shape
    bbox, crop = tbbox.random_crop_with_constraints(bbox, (w, h))
    x0, y0, w, h = crop
    img = img[y0: y0+h, x0: x0+w]

    # resize with random interpolation
    h, w, _ = img.shape
    interp = np.random.randint(0, 5)
    img = timage.imresize(img, target_width, target_height, interp=interp)
    bbox = tbbox.resize(bbox, (w, h), (target_width, target_height))

    # random horizontal flip
    h, w, _ = img.shape
    img, flips = timage.random_flip(img, px=0.5)
    bbox = tbbox.flip(bbox, (w, h), flip_x=flips[0])

    # normalize
    img = timage.normalize(img, rgb_mean, rgb_std)

    return img, bbox.astype(img.dtype)



