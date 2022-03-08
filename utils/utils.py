import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from PIL import Image


def get_classes(class_file_name):
    """
    loads class name from a file
    :param class_file_name: 
    :return: class names 
    """""
    class_names = {}
    with open(class_file_name, 'r') as data:
        for ID, name in enumerate(data):
            class_names[ID] = name.strip('\n')
    return class_names, len(class_names)


def get_anchors(anchors_path):
    """
    loads the anchors from a file
    """""
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    anchors = np.array(anchors).reshape(-1, 2)
    return anchors, len(anchors)


# ===================================================================
#   Convert image to RGB
#   Currently only support RGB, all the image should convert to RGB before send into model
# ===================================================================
def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image
    else:
        image = image.convert('RGB')
        return image


def resize_image(image, size, letterbox_image):
    iw, ih = image.size
    w, h = size
    if letterbox_image:
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)

        image = image.resize((nw, nh), Image.BICUBIC)
        new_image = Image.new('RGB', size, (128, 128, 128))
        new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
    else:
        new_image = image.resize((w, h), Image.BICUBIC)
    return new_image


def preprocess_input(image):
    image /= 255.0
    return image
