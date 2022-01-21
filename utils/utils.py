import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K


def read_class_names(class_file_name):
    """
    loads class name from a file
    :param class_file_name: 
    :return: class names 
    """""
    names = {}
    with open(class_file_name, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    return names


def get_anchors(anchors_path):
    """
    loads the anchors from a file
    """""
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    anchors = np.array(anchors).reshape(-1, 2)
    return anchors  #, len(anchors)
