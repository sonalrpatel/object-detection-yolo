"""
Implementation of Yolo Loss Function similar to the one in Yolov3 paper,
the difference from what I can tell is I use CrossEntropy for the classes
instead of BinaryCrossEntropy.
"""

import tensorflow.keras.losses as losses
from utils.utils import *
from utils.utils_bbox import *
from utils.utils_metric import *


class YoloLoss(object):
    def __init__(self, anchors, anchors_mask, num_classes=80, ignore_thresh=0.5):
        super(YoloLoss, self).__init__()
        self.anchors = anchors
        self.anchors_mask = anchors_mask
        self.num_classes = num_classes
        self.ignore_thresh = ignore_thresh
        self.input_shape = (416, 416)
        self.num_layers = 3

    def loss(self, y_pred, y_true):
        # -----------------------------------------------------------#
        #   split predictions and ground truth, args is list contains [*model_body.output, *y_true]
        #   y_true is a list，contains 3 feature maps，shape are:
        #   (m,13,13,3,85)
        #   (m,26,26,3,85)
        #   (m,52,52,3,85)
        #   y_pred is a list，contains 3 feature maps，shape are:
        #   (m,13,13,3,85)
        #   (m,26,26,3,85)
        #   (m,52,52,3,85)
        # -----------------------------------------------------------#

        # -----------------------------------------------------------#
        #   input_shpae = (416, 416)
        # -----------------------------------------------------------#
        input_shape = K.cast(self.input_shape, K.dtype(y_true[0]))

        # -----------------------------------------------------------#
        #   grid shapes = [[13,13], [26,26], [52,52]]
        # -----------------------------------------------------------#
        grid_shapes = [K.cast(K.shape(y_pred[l])[1:3], K.dtype(y_true[0])) for l in range(self.num_layers)]

        # -----------------------------------------------------------#
        #   m = batch_size = number of images
        # -----------------------------------------------------------#
        m = K.shape(y_pred[0])[0]

        # ---------------------------------------------------------------#
        #   y_true is a list，contains 3 feature maps，shape are: (m,13,13,3,85), (m,26,26,3,85), (m,52,52,3,85)
        #   y_pred is a list，contains 3 feature maps，shape are: (m,13,13,3,85), (m,26,26,3,85), (m,52,52,3,85)
        # ---------------------------------------------------------------#
        loss = 0
        num_pos = 0
        for l in range(self.num_layers):
            # -----------------------------------------------------------#
            #   Here take fist feature map as example (m,13,13,3,85)
            #   Retrieve object score from last dim with shape => (m,13,13,3,1)
            # -----------------------------------------------------------#
            object_mask = y_true[l][..., 4:5]

            # -----------------------------------------------------------#
            #   Retrieve object category from last dim with shape => (m,13,13,3,80)
            # -----------------------------------------------------------#
            true_class_probs = y_true[l][..., 5:]

            # -----------------------------------------------------------#
            #   Retrieve Yolo predictions for bounding box
            #   And, decode those Yolo predictions to return 4 matrices as below
            #   grid        (13,13,3,2) grid coordinates
            #   raw_pred    (m,13,13,3,85) raw prediction
            #   pred_xy     (m,13,13,3,2) decode box center point x and y
            #   pred_wh     (m,13,13,3,2) decode width and height
            # -----------------------------------------------------------#
            grid, raw_pred, pred_xy, pred_wh = get_pred_boxes(y_pred[l],
                                                              self.anchors[self.anchors_mask[l]],
                                                              self.num_classes,
                                                              input_shape,
                                                              calc_loss=True)

            # -----------------------------------------------------------#
            #   concat predicted xy and wh to bounding box shape => (m,13,13,3,4)
            # -----------------------------------------------------------#
            pred_box = K.concatenate([pred_xy, pred_wh])

            # -----------------------------------------------------------#
            #   create a dynamic array to save negative samples
            # -----------------------------------------------------------#
            ignore_mask = tf.TensorArray(K.dtype(y_true[0]), size=1, dynamic_size=True)
            object_mask_bool = K.cast(object_mask, 'bool')

            # -----------------------------------------------------------#
            #   define a inner function to locate ignored samples
            # -----------------------------------------------------------#
            def loop_body(b, ignore_mask):
                # -----------------------------------------------------------#
                #   retrieve ground truth's bounding box's coordinates (x,y) and (w,h)=> shape (n, 4)
                # -----------------------------------------------------------#
                true_box = tf.boolean_mask(y_true[l][b, ..., 0:4], object_mask_bool[b, ..., 0])

                # -----------------------------------------------------------#
                #   calculate iou
                #   pred_box shape => (13,13,3,4)
                #   true_box shape => (n,4)
                #   iou between predicted box and true box, shape => (13,13,3,n)
                # -----------------------------------------------------------#
                iou = box_iou(pred_box[b], true_box)

                # -----------------------------------------------------------#
                #   best_iou shape => (13,13,3) means best iou for every anchor
                # -----------------------------------------------------------#
                best_iou = K.max(iou, axis=-1)

                # -----------------------------------------------------------#
                #   if best iou less than ignore threshold treat it as ignored sample
                #   add to ignore mask for calculate no object loss latter
                #   if best iou is greater or equal than ignore threshold, we think it's
                #   close to the true box will not treat it as a ignored sample
                # -----------------------------------------------------------#
                ignore_mask = ignore_mask.write(b, K.cast(best_iou < self.ignore_thresh, K.dtype(true_box)))
                return b + 1, ignore_mask

            # -----------------------------------------------------------#
            #   call while loop to find out ignored samples in every images one by one
            # -----------------------------------------------------------#
            _, ignore_mask = tf.while_loop(lambda b, *args: b < m, loop_body, [0, ignore_mask])

            # -----------------------------------------------------------#
            #   ignore_mask shape => (m,13,13,3)
            #   (m,13,13,3) =>  (m,13,13,3,1)
            # -----------------------------------------------------------#
            ignore_mask = ignore_mask.stack()
            ignore_mask = K.expand_dims(ignore_mask, -1)

            # -----------------------------------------------------------#
            #   normalize true xy and wh to align with predicted xy and wh to calculate loss later
            # -----------------------------------------------------------#
            raw_true_xy = y_true[l][..., :2] * grid_shapes[l][::-1] - grid
            raw_true_wh = K.log(y_true[l][..., 2:4] / self.anchors[self.anchors_mask[l]] * input_shape[::-1])

            # -----------------------------------------------------------#
            #   update raw_true_wh if no object update wh to 0 otherwise remain it as before
            # -----------------------------------------------------------#
            raw_true_wh = K.switch(object_mask, raw_true_wh, K.zeros_like(raw_true_wh))

            # -----------------------------------------------------------#
            #   calculate box loss scale, x and y both between 0-1
            #   if real box is bigger box_loss_scale will become smaller
            #   if real box is smaller box_loss_scale will become larger
            #   to make sure big box and smaller box to have almost same loss value
            # -----------------------------------------------------------#
            box_loss_scale = 2 - y_true[l][..., 2:3] * y_true[l][..., 3:4]

            # -----------------------------------------------------------#
            #   use binary_crossentropy calculate xy loss
            # -----------------------------------------------------------#
            xy_loss = object_mask * box_loss_scale * K.binary_crossentropy(raw_true_xy, raw_pred[..., 0:2],
                                                                           from_logits=True)

            # -----------------------------------------------------------#
            #   calculate wh_loss
            # -----------------------------------------------------------#
            wh_loss = object_mask * box_loss_scale * 0.5 * K.square(raw_true_wh - raw_pred[..., 2:4])

            # -----------------------------------------------------------#
            #   if there is true box，calculate cross entropy loss between predicted score and 1
            #   if there no box，calculate cross entropy loss between predicted score and 0
            #   and will ignore the samples if best_iou<ignore_thresh
            # -----------------------------------------------------------#
            confidence_loss = object_mask * K.binary_crossentropy(object_mask, raw_pred[..., 4:5], from_logits=True) + \
                              (1 - object_mask) * K.binary_crossentropy(object_mask, raw_pred[..., 4:5],
                                                                        from_logits=True) * ignore_mask

            class_loss = object_mask * K.binary_crossentropy(true_class_probs, raw_pred[..., 5:], from_logits=True)

            # -----------------------------------------------------------#
            #   sum up loss
            # -----------------------------------------------------------#
            xy_loss = K.sum(xy_loss)
            wh_loss = K.sum(wh_loss)
            confidence_loss = K.sum(confidence_loss)
            class_loss = K.sum(class_loss)

            # -----------------------------------------------------------#
            #   add up all the loss
            # -----------------------------------------------------------#
            num_pos += tf.maximum(K.sum(K.cast(object_mask, tf.float32)), 1)
            loss += xy_loss + wh_loss + confidence_loss + class_loss

        loss = loss / num_pos
        return loss


if __name__ == "__main__":
    outputs = tf.random.normal([4, 5, 5, 255])
    anchors = np.array([[10, 13], [16, 30], [33, 23], [30, 61], [62, 45], [59, 119], [116, 90], [156, 198], [373, 326]])
    anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    input_shape = (416, 416)

    # image_shape = K.reshape(outputs[-1], [-1])

    # y_true = [Input(shape=(input_shape[0] // {0: 32, 1: 16, 2: 8}[l], input_shape[1] // {0: 32, 1: 16, 2: 8}[l], len(anchors_mask[l]), 80 + 5)) for l in range(len(anchors_mask))]
    # y_pred = [Input(shape=(input_shape[0] // {0: 32, 1: 16, 2: 8}[l], input_shape[1] // {0: 32, 1: 16, 2: 8}[l], len(anchors_mask[l]), 80 + 5)) for l in range(len(anchors_mask))]

    y_true = [tf.random.normal([10, 13, 13, 3, 85]), tf.random.normal([10, 13, 13, 3, 85]), tf.random.normal([10, 13, 13, 3, 85])]
    y_pred = [tf.random.normal([10, 13, 13, 3, 85]), tf.random.normal([10, 13, 13, 3, 85]), tf.random.normal([10, 13, 13, 3, 85])]

    yolo_loss = YoloLoss(anchors, anchors_mask).loss(y_true, y_pred)

    print(input_shape.__len__(), outputs[-1].shape)
