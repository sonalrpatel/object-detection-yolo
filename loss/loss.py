"""
Implementation of Yolo Loss Function similar to the one in Yolov3 paper,
the difference from what I can tell is I use CrossEntropy for the classes
instead of BinaryCrossEntropy.
"""

import tensorflow.keras.losses as losses
from utils.utils import *


class YoloLoss(object):
    def __init__(self, num_classes, ignore_thresh=0.5):
        super(YoloLoss, self).__init__()
        self.mse = losses.MeanSquaredError()
        self.bce = losses.BinaryCrossentropy()
        self.entropy = losses.CategoricalCrossentropy()
        self.sigmoid = tf.keras.activations.sigmoid()

        # Constants signifying how much to pay for each respective part of the loss
        self.lambda_class = 1
        self.lambda_noobj = 10
        self.lambda_obj = 1
        self.lambda_box = 10

        self.num_classes = num_classes
        self.ignore_thresh = ignore_thresh

    def call(self, y_pred, y_true, anchors):
        # Check where obj and noobj (we ignore if target == -1)
        obj = y_true[..., 0] == 1  # in paper this is Iobj_i
        noobj = y_true[..., 0] == 0  # in paper this is Inoobj_i

        # ======================= #
        #   FOR NO OBJECT LOSS    #
        # ======================= #
        no_object_loss = self.bce(
            (y_pred[..., 0:1][noobj]), (y_true[..., 0:1][noobj]),
        )

        # ==================== #
        #   FOR OBJECT LOSS    #
        # ==================== #
        anchors = anchors.reshape(1, 3, 1, 1, 2)
        box_preds = tf.keras.layers.concatenate([self.sigmoid(y_pred[..., 1:3]), np.exp(y_pred[..., 3:5]) * anchors], dim=-1)
        ious = box_iou(box_preds[obj], y_true[..., 1:5][obj]).detach()
        object_loss = self.mse(self.sigmoid(y_pred[..., 0:1][obj]), ious * y_true[..., 0:1][obj])

        # ======================== #
        #   FOR BOX COORDINATES    #
        # ======================== #
        y_pred[..., 1:3] = self.sigmoid(y_pred[..., 1:3])  # x,y coordinates
        y_true[..., 3:5] = np.log(
            (1e-16 + y_true[..., 3:5] / anchors)
        )  # width, height coordinates
        box_loss = self.mse(y_pred[..., 1:5][obj], y_true[..., 1:5][obj])

        # ================== #
        #   FOR CLASS LOSS   #
        # ================== #
        class_loss = self.entropy(
            (y_pred[..., 5:][obj]), (y_true[..., 5][obj].long()),
        )

        return (
            self.lambda_box * box_loss
            + self.lambda_obj * object_loss
            + self.lambda_noobj * no_object_loss
            + self.lambda_class * class_loss
        )
