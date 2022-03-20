import os
import struct
import numpy as np
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
def convert2rgb(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image
    else:
        image = image.convert('RGB')
        return image


def resize_image(image, size, letterbox_image):
    in_w, in_h = image.size
    net_w, net_h = size
    if letterbox_image:
        scale = min(net_w / in_w, net_h / in_h)
        int_w = int(in_w * scale)
        int_h = int(in_h * scale)

        image = image.resize((int_w, int_h), Image.BICUBIC)
        new_image = Image.new('RGB', size, (128, 128, 128))
        new_image.paste(image, ((net_w - int_w) // 2, (net_h - int_h) // 2))
    else:
        new_image = image.resize((net_w, net_h), Image.BICUBIC)
    return new_image


def normalize_input(image):
    image /= 255.0
    return image


# ===================================================================
#   Load weights to yolov3 model
#   Input: yolov3 keras model, yolov3.weights
#   Output: yolov3_model.h5
# ===================================================================
class WeightReader:
    def __init__(self, weight_file):
        with open(weight_file, 'rb') as w_f:
            major, = struct.unpack('i', w_f.read(4))
            minor, = struct.unpack('i', w_f.read(4))
            revision, = struct.unpack('i', w_f.read(4))

            if (major * 10 + minor) >= 2 and major < 1000 and minor < 1000:
                w_f.read(8)
            else:
                w_f.read(4)

            transpose = (major > 1000) or (minor > 1000)

            binary = w_f.read()

        self.offset = 0
        self.all_weights = np.frombuffer(binary, dtype='float32')

    def read_bytes(self, size):
        self.offset = self.offset + size
        return self.all_weights[self.offset - size:self.offset]

    def load_weights(self, model):
        for i in range(106):
            try:
                conv_layer = model.get_layer('conv_' + str(i))
                print("loading weights of convolution #" + str(i))

                if i not in [81, 93, 105]:
                    norm_layer = model.get_layer('bnorm_' + str(i))

                    size = np.prod(norm_layer.get_weights()[0].shape)

                    beta = self.read_bytes(size)  # bias
                    gamma = self.read_bytes(size)  # scale
                    mean = self.read_bytes(size)  # mean
                    var = self.read_bytes(size)  # variance

                    weights = norm_layer.set_weights([gamma, beta, mean, var])

                if len(conv_layer.get_weights()) > 1:
                    bias = self.read_bytes(np.prod(conv_layer.get_weights()[1].shape))
                    kernel = self.read_bytes(np.prod(conv_layer.get_weights()[0].shape))

                    kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
                    kernel = kernel.transpose([2, 3, 1, 0])
                    conv_layer.set_weights([kernel, bias])
                else:
                    kernel = self.read_bytes(np.prod(conv_layer.get_weights()[0].shape))
                    kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
                    kernel = kernel.transpose([2, 3, 1, 0])
                    conv_layer.set_weights([kernel])
            except ValueError:
                print("no convolution #" + str(i))

    def reset(self):
        self.offset = 0
