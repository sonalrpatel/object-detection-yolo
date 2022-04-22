"""
Implementation of YOLOv3 architecture
"""
import os
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Add, ZeroPadding2D, UpSampling2D, Concatenate
from tensorflow.keras.layers import LeakyReLU, BatchNormalization
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.regularizers import l2
from tensorflow.keras import Input, Model

from utils.utils import WeightReader

"""
Information about architecture config:
Tuple is structured by (filters, kernel_size, stride) 
Every conv is a same convolution. 
List is structured by "B" indicating a residual block followed by the number of repeats
"S" is for scale prediction block and computing the yolo loss
"U" is for upsampling the feature map and concatenating with a previous layer
"""


def DarknetConv2D_BN_Leaky(inputs, n_filters, kernel_size=(3, 3), down_sample=False, bn_act=True, layer_idx=None):
    strides = (2, 2) if down_sample else (1, 1)
    padding = 'valid' if down_sample else 'same'
    use_bias = False if bn_act else True

    x = Conv2D(n_filters, kernel_size=kernel_size, strides=strides, padding=padding, use_bias=use_bias,
               kernel_initializer = RandomNormal(stddev=0.02), kernel_regularizer = l2(5e-4),
               name='conv_' + str(layer_idx))(inputs)
    if bn_act:
        x = BatchNormalization(name='bnorm_' + str(layer_idx))(x)
        x = LeakyReLU(alpha=0.1, name='leaky_' + str(layer_idx))(x)

    return x


def ResidualBlock(inputs, n_filters, n_repeats=1, layer_idx=None):
    x = ZeroPadding2D(((1, 0), (1, 0)))(inputs)
    x = DarknetConv2D_BN_Leaky(x, n_filters, down_sample=True, layer_idx=layer_idx)
    for i in range(n_repeats):
        y = DarknetConv2D_BN_Leaky(x, n_filters // 2, kernel_size=(1, 1), layer_idx=layer_idx + 1 + (i * 3))
        y = DarknetConv2D_BN_Leaky(y, n_filters, layer_idx=layer_idx + 2 + (i * 3))
        x = Add()([x, y])  # layer_idx=layer_idx + 3 + (i*3)

    return x


def DarkNet53(inputs, layer_idx=0):
    # Layer 0
    x = DarknetConv2D_BN_Leaky(inputs, 32, layer_idx=layer_idx)
    # Layer 1 (+3*1) => 4
    x = ResidualBlock(x, 64, n_repeats=1, layer_idx=layer_idx + 1)
    # Layer 5 (+3*2) => 11
    x = ResidualBlock(x, 128, n_repeats=2, layer_idx=layer_idx + 5)
    # Layer 12 (+3*8) => 36
    x = ResidualBlock(x, 256, n_repeats=8, layer_idx=layer_idx + 12)
    skip_36 = x
    # Layer 37 (+3*8) => 61
    x = ResidualBlock(x, 512, n_repeats=8, layer_idx=layer_idx + 37)
    skip_61 = x
    # Layer 62 (+3*4) => 74
    x = ResidualBlock(x, 1024, n_repeats=4, layer_idx=layer_idx + 62)

    return skip_36, skip_61, x


def UpSampleConv(inputs, n_filters, layer_idx=0):
    x = inputs
    idx = 0
    if not tf.is_tensor(inputs):
        x = DarknetConv2D_BN_Leaky(inputs[0], n_filters, kernel_size=(1, 1), layer_idx=layer_idx)
        x = UpSampling2D(2)(x)                      # layer_idx=layer_idx + 1
        x = Concatenate(axis=-1)([x, inputs[1]])    # layer_idx=layer_idx + 2
        idx = 3
    x = DarknetConv2D_BN_Leaky(x, n_filters, kernel_size=(1, 1), layer_idx=layer_idx + idx)         # 512, 1x1
    x = DarknetConv2D_BN_Leaky(x, n_filters * 2, layer_idx=layer_idx + idx + 1)                     # 1024, 3x3
    x = DarknetConv2D_BN_Leaky(x, n_filters, kernel_size=(1, 1), layer_idx=layer_idx + idx + 2)     # 512, 1x1
    x = DarknetConv2D_BN_Leaky(x, n_filters * 2, layer_idx=layer_idx + idx + 3)                     # 1024, 3x3
    x = DarknetConv2D_BN_Leaky(x, n_filters, kernel_size=(1, 1), layer_idx=layer_idx + idx + 4)     # 512, 1x1

    return x


def ScalePrediction(inputs, n_filters, num_classes, layer_idx=0):
    x = DarknetConv2D_BN_Leaky(inputs, n_filters, layer_idx=layer_idx)  # 13x13x1024/26x26x512/52x52x256, 3x3
    x = DarknetConv2D_BN_Leaky(x, 3 * (num_classes + 5), kernel_size=(1, 1), bn_act=False, layer_idx=layer_idx + 1)
                                                                        # 13x13x255/26x26x255/52x52x255, 1x1

    return x


def YOLOv3(input_shape=(416, 416, 3), num_classes=80):
    x = Input(input_shape)

    def model(inputs, num_classes):
        # Layer 0 => 74
        skip_36, skip_61, x = DarkNet53(inputs, layer_idx=0)

        # Layer 75 => 79
        x = UpSampleConv(x, 512, layer_idx=75)
        # Layer 80 => 81
        y_lbbox = ScalePrediction(x, 1024, num_classes, layer_idx=80)

        # Layer 84 => 91
        x = UpSampleConv([x, skip_61], 256, layer_idx=84)
        # Layer 92 => 93
        y_mbbox = ScalePrediction(x, 512, num_classes, layer_idx=92)

        # Layer 96 => 103
        x = UpSampleConv([x, skip_36], 128, layer_idx=96)
        # Layer 104 => 105
        y_sbbox = ScalePrediction(x, 256, num_classes, layer_idx=104)

        return [y_lbbox, y_mbbox, y_sbbox]

    return Model(inputs=[x], outputs=model(x, num_classes))


if __name__ == "__main__":
    num_classes = 80
    image_size = 416
    image_shape = (image_size, image_size, 3)

    # define the model
    model = YOLOv3(image_shape, num_classes)

    # read the model weights
    rel_path = "../data/yolov3.weights"
    abs_file_path = os.path.join(os.path.dirname(__file__), rel_path)
    weight_reader = WeightReader(abs_file_path)

    # set/load the weights into the model
    weight_reader.load_weights(model)

    # save the model to .h5 file
    model.save('../data/yolo_weights.h5')

    model.load_weights(os.path.join(os.path.dirname(__file__), '../data/yolo_weights.h5'))

    input_tensor = Input(image_shape)
    output_tensor = model(input_tensor)

    print(model.summary())
