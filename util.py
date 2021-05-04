import numpy as np
import tensorflow as tf
from ops import np_free_form_mask

def f2uint(x):
    if x.__class__ == tf.Tensor:
        return tf.cast(tf.clip_by_value((x+1)*127.5, 0, 255), tf.uint8)
    else:
        return np.clip((x+1)*127.5, 0, 255).astype(np.uint8)


def generate_mask_rect(im_shapes, mask_shapes, rand=True):
    mask = np.zeros((im_shapes[0], im_shapes[1])).astype(np.float32)
    if rand:
        of0 = np.random.randint(0, im_shapes[0]-mask_shapes[0])
        of1 = np.random.randint(0, im_shapes[1]-mask_shapes[1])
    else:
        of0 = (im_shapes[0]-mask_shapes[0])//2
        of1 = (im_shapes[1]-mask_shapes[1])//2
    mask[of0:of0+mask_shapes[0], of1:of1+mask_shapes[1]] = 1
    mask = np.expand_dims(mask, axis=2)
    return mask


def generate_mask_stroke(im_size, parts=16, maxVertex=24, maxLength=100, maxBrushWidth=24, maxAngle=360):
    h, w = im_size[:2]
    mask = np.zeros((h, w, 1), dtype=np.float32)
    for _ in range(parts):
        mask = mask + np_free_form_mask(maxVertex, maxLength, maxBrushWidth, maxAngle, h, w)
    mask = np.minimum(mask, 1.0)
    return mask

def get_random_int(min=0, max=10, number=5):
    """Return a list of random integer by the given range and quantity.
    Examples
    ---------
    >>> r = get_random_int(min=0, max=10, number=5)
    ... [10, 2, 3, 3, 7]
    """
    return [random.randint(min,max) for p in range(0,number)]
