"""
Implementation of YOLOv3 architecture
Implementation of YOLOv3 architecture
"""
from abc import ABC
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Add, ZeroPadding2D, UpSampling2D, Concatenate, MaxPooling2D
from tensorflow.keras.layers import Layer, LeakyReLU, BatchNormalization
from tensorflow.keras import Input, Model
from tensorflow.keras.regularizers import l2


# References
# https://towardsdatascience.com/yolo-v3-object-detection-53fb7d3bfe6b
# https://towardsdatascience.com/model-sub-classing-and-custom-training-loop-from-scratch-in-tensorflow-2-cc1d4f10fb4e


"""
Information about architecture config:
Tuple is structured by (filters, kernel_size, stride) 
Every conv is a same convolution. 
List is structured by "B" indicating a residual block followed by the number of repeats
"S" is for scale prediction block and computing the yolo loss
"U" is for upsampling the feature map and concatenating with a previous layer
"""
config = [
    (32, 3, 1),
    (64, 3, 2),
    ["B", 1],
    (128, 3, 2),
    ["B", 2],
    (256, 3, 2),
    ["B", 8],
    (512, 3, 2),
    ["B", 8],
    (1024, 3, 2),
    ["B", 4],  # To this point is Darknet-53
    (512, 1, 1),
    (1024, 3, 1),
    "S",
    (256, 1, 1),
    "U",
    (256, 1, 1),
    (512, 3, 1),
    "S",
    (128, 1, 1),
    "U",
    (128, 1, 1),
    (256, 3, 1),
    "S",
]


class CNNBlock(Layer):
    def __init__(self, n_filters, kernel_size=(3, 3), bn_act=True, down_sample=False, **kwargs):
        super(CNNBlock, self).__init__()
        # conv layer, default setting is: strides=(1, 1) and padding='valid'
        self.conv_s1 = Conv2D(n_filters, kernel_size=kernel_size, strides=(1, 1), padding='same', **kwargs)
        self.conv_s2 = Conv2D(n_filters, kernel_size=kernel_size, strides=(2, 2), padding='valid', **kwargs)
        # zero padding, Darknet uses left and top padding instead of 'same' mode
        self.zp = ZeroPadding2D(((1, 0), (1, 0)))
        # batch norm layer
        self.bn = BatchNormalization()
        self.leaky = LeakyReLU(0.1)
        self.use_bn_act = bn_act
        self.do_down_sample = down_sample

    def call(self, inputs, training=False):
        if self.do_down_sample:
            x = self.zp(inputs)
            x = self.conv_s2(x)
        else:
            x = self.conv_s1(inputs)
        if self.use_bn_act:
            x = self.bn(x, training=training)
            x = self.leaky(x)

        return x


class ResidualBlock(Layer):
    def __init__(self, n_filters, use_residual=True, n_repeats=1):
        super(ResidualBlock, self).__init__()
        self.conv_k1 = CNNBlock(n_filters // 2, kernel_size=(1, 1))
        self.conv_k3 = CNNBlock(n_filters)
        self.use_residual = use_residual
        self.n_repeats = n_repeats

    def call(self, inputs, training=False):
        x = inputs
        for i in range(self.n_repeats):
            y = self.conv_k1(x, training=training)
            y = self.conv_k3(y, training=training)
            if self.use_residual:
                x = Add()([x, y])

        return x


class DarkNet53(Layer):
    def __init__(self):
        super(DarkNet53, self).__init__()
        self.DBL1 = CNNBlock(32)
        self.DBL2 = CNNBlock(64, down_sample=True)
        self.RES1 = ResidualBlock(64, n_repeats=1)
        self.DBL3 = CNNBlock(128, down_sample=True)
        self.RES2 = ResidualBlock(128, n_repeats=2)
        self.DBL4 = CNNBlock(256, down_sample=True)
        self.RES3 = ResidualBlock(256, n_repeats=8)
        self.DBL5 = CNNBlock(512, down_sample=True)
        self.RES4 = ResidualBlock(512, n_repeats=8)
        self.DBL6 = CNNBlock(1024, down_sample=True)
        self.RES5 = ResidualBlock(1024, n_repeats=4)

    def call(self, inputs, training=False):
        x = self.DBL1(inputs, training=training)
        x = self.DBL2(x, training=training)
        x = self.RES1(x, training=training)
        x = self.DBL3(x, training=training)
        x = self.RES2(x, training=training)
        x = self.DBL4(x, training=training)
        x = self.RES3(x, training=training)
        skip1 = x
        x = self.DBL5(x, training=training)
        x = self.RES4(x, training=training)
        skip2 = x
        x = self.DBL6(x, training=training)
        x = self.RES5(x, training=training)

        return skip1, skip2, x


class UpSampleConv(Layer):
    def __init__(self, n_filters):
        super(UpSampleConv, self).__init__()
        self.DBL1 = CNNBlock(n_filters, kernel_size=(1, 1))
        self.DBL2 = CNNBlock(n_filters, kernel_size=(1, 1))
        self.DBL3 = CNNBlock(n_filters * 2)
        self.UpSample = UpSampling2D(2)
        self.Concat = Concatenate(axis=-1)

    def call(self, inputs, training=False):
        if not tf.is_tensor(inputs):
            x = self.DBL1(inputs[0], training=training)
            x = self.UpSample(x)
            x = self.Concat([x, inputs[1]])
        else:
            x = inputs
        x = self.DBL2(x, training=training)         # 512, 1x1
        x = self.DBL3(x, training=training)         # 1024, 3x3
        x = self.DBL1(x, training=training)         # 512, 1x1
        x = self.DBL3(x, training=training)         # 1024, 3x3
        x = self.DBL1(x, training=training)         # 512, 1x1

        return x


class ScalePrediction(Layer):
    def __init__(self, n_filters, num_classes):
        super(ScalePrediction, self).__init__()
        self.num_classes = num_classes
        self.DBL = CNNBlock(n_filters)
        self.conv = CNNBlock(3 * (num_classes + 5), kernel_size=(1, 1), bn_act=False)

    def call(self, inputs, training=False):
        y = self.DBL(inputs, training=training)     # 13x13x1024/26x26x512/52x52x256, 3x3
        y = self.conv(y, training=training)         # 13x13x255/26x26x255/52x52x255, 1x1

        return y


class YOLOv3(Model):
    def __init__(self, num_classes=80):
        super(YOLOv3, self).__init__()
        self.num_classes = num_classes
        self.DN53 = DarkNet53()
        self.Conv512 = UpSampleConv(512)
        self.SPr13 = ScalePrediction(1024, num_classes)
        self.UpS1326Conv256 = UpSampleConv(256)
        self.SPr26 = ScalePrediction(512, num_classes)
        self.UpS2652Conv128 = UpSampleConv(128)
        self.SPr52 = ScalePrediction(256, num_classes)

    def call(self, inputs, training=False):
        skip1, skip2, x = self.DN53.call(inputs, training=training)
        x = self.Conv512(x, training=training)
        y_lbbox = self.SPr13(x, training=training)
        x = self.UpS1326Conv256([x, skip2], training=training)
        y_mbbox = self.SPr26(x, training=training)
        x = self.UpS2652Conv128([x, skip1], training=training)
        y_sbbox = self.SPr52(x, training=training)

        return [y_lbbox, y_mbbox, y_sbbox]

    def model(self, inputs):
        x = Input(inputs)
        return Model(inputs=[x], outputs=self.call(x))


if __name__ == "__main__":
    num_classes = 80
    IMAGE_SIZE = 416
    image_shape = (IMAGE_SIZE, IMAGE_SIZE, 3)

    model = YOLOv3(num_classes).model(image_shape)

    input_tensor = Input(image_shape)
    output = model(input_tensor)

    print(model.summary())
