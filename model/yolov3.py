"""
Implementation of YOLOv3 architecture
"""
from abc import ABC
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Add, ZeroPadding2D, UpSampling2D, Concatenate, MaxPooling2D
from tensorflow.keras.layers import Layer, LeakyReLU, BatchNormalization
from tensorflow.keras import Input, Model
from tensorflow.keras.regularizers import l2

from loss.loss import YoloLoss


"""
Information about architecture config:
Tuple is structured by (filters, kernel_size, stride) 
Every conv is a same convolution. 
List is structured by "B" indicating a residual block followed by the number of repeats
"S" is for scale prediction block and computing the yolo loss
"U" is for upsampling the feature map and concatenating with a previous layer
"""


def DarknetConv2D(inputs, n_filters, kernel_size=(3, 3), down_sample=False):
    if down_sample:
        x = ZeroPadding2D(((1, 0), (1, 0)))(inputs)
        x = Conv2D(n_filters, kernel_size=kernel_size, strides=(2, 2), padding='valid')(x)
    else:
        x = Conv2D(n_filters, kernel_size=kernel_size, strides=(1, 1), padding='same')(inputs)

    return x


def DarknetConv2D_BN_Leaky(inputs, n_filters, kernel_size=(3, 3), down_sample=False, bn_act=True):
    x = DarknetConv2D(inputs, n_filters, kernel_size=kernel_size, down_sample=down_sample)
    if bn_act:
        x = BatchNormalization()(x)
        x = LeakyReLU(0.1)(x)

    return x


def ResidualBlock(inputs, n_filters, use_residual=True, n_repeats=1):
    x = inputs
    for i in range(n_repeats):
        y = DarknetConv2D_BN_Leaky(x, n_filters // 2, kernel_size=(1, 1))
        y = DarknetConv2D_BN_Leaky(y, n_filters)
        if use_residual:
            x = Add()([x, y])

    return x


def DarkNet53(inputs):
    x = DarknetConv2D_BN_Leaky(inputs, 32)

    x = DarknetConv2D_BN_Leaky(x, 64, down_sample=True)
    x = ResidualBlock(x, 64, n_repeats=1)

    x = DarknetConv2D_BN_Leaky(x, 128, down_sample=True)
    x = ResidualBlock(x, 128, n_repeats=2)

    x = DarknetConv2D_BN_Leaky(x, 256, down_sample=True)
    x = ResidualBlock(x, 256, n_repeats=8)
    skip1 = x

    x = DarknetConv2D_BN_Leaky(x, 512, down_sample=True)
    x = ResidualBlock(x, 512, n_repeats=8)
    skip2 = x

    x = DarknetConv2D_BN_Leaky(x, 1024, down_sample=True)
    x = ResidualBlock(x, 1024, n_repeats=4)

    return skip1, skip2, x


def UpSampleConv(inputs, n_filters):
    if not tf.is_tensor(inputs):
        x = DarknetConv2D_BN_Leaky(inputs[0], n_filters, kernel_size=(1, 1))
        x = UpSampling2D(2)(x)
        x = Concatenate(axis=-1)([x, inputs[1]])
    else:
        x = inputs
    x = DarknetConv2D_BN_Leaky(x, n_filters, kernel_size=(1, 1))    # 512, 1x1
    x = DarknetConv2D_BN_Leaky(x, n_filters * 2)                    # 1024, 3x3
    x = DarknetConv2D_BN_Leaky(x, n_filters, kernel_size=(1, 1))    # 512, 1x1
    x = DarknetConv2D_BN_Leaky(x, n_filters * 2)                    # 1024, 3x3
    x = DarknetConv2D_BN_Leaky(x, n_filters, kernel_size=(1, 1))    # 512, 1x1

    return x


def ScalePrediction(inputs, n_filters, num_classes):
    x = DarknetConv2D_BN_Leaky(inputs, n_filters)                   # 13x13x1024/26x26x512/52x52x256, 3x3
    x = DarknetConv2D_BN_Leaky(x, 3 * (num_classes + 5), \
                               kernel_size=(1, 1), bn_act=False)    # 13x13x255/26x26x255/52x52x255, 1x1

    return x


def YOLOv3(input_shape=(416, 416, 3), num_classes=80):
    x = Input(input_shape)

    def model(inputs, num_classes):
        skip1, skip2, x = DarkNet53(inputs)
        x = UpSampleConv(x, 512)
        y_lbbox = ScalePrediction(x, 1024, num_classes)
        x = UpSampleConv([x, skip2], 256)
        y_mbbox = ScalePrediction(x, 512, num_classes)
        x = UpSampleConv([x, skip1], 128)
        y_sbbox = ScalePrediction(x, 256, num_classes)

        return [y_lbbox, y_mbbox, y_sbbox]

    return Model(inputs=[x], outputs=model(x, num_classes))


if __name__ == "__main__":
    num_classes = 80
    image_size = 416
    image_shape = (image_size, image_size, 3)

    model = YOLOv3(image_shape, num_classes)

    input_tensor = Input(image_shape)
    output_tensor = model(input_tensor)

    print(model.summary())
