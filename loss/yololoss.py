# coding=utf-8
"""
Implementation of Yolo Loss Function similar to the one in Yolov3 paper,
the difference from what I can tell is I use CrossEntropy for the classes
instead of BinaryCrossEntropy.
"""

import tensorflow.keras.losses as losses
from utils.utils import *
from utils.utils_bbox import *
from utils.utils_metric import *


def box_iou(pred_boxes, valid_true_boxes):
    """
    param:
        pred_boxes: [13, 13, 3, 4], (center_x, center_y, w, h)
        valid_true: [V, 4]
    """

    # [13, 13, 3, 2]
    pred_box_xy = pred_boxes[..., 0:2]
    pred_box_wh = pred_boxes[..., 2:4]

    # shape: [13, 13, 3, 1, 2]
    pred_box_xy = tf.expand_dims(pred_box_xy, -2)
    pred_box_wh = tf.expand_dims(pred_box_wh, -2)

    # [V, 2]
    true_box_xy = valid_true_boxes[:, 0:2]
    true_box_wh = valid_true_boxes[:, 2:4]

    # [13, 13, 3, 1, 2] & [V, 2] ==> [13, 13, 3, V, 2]
    intersect_mins = tf.maximum(pred_box_xy - pred_box_wh / 2., true_box_xy - true_box_wh / 2.)
    intersect_maxs = tf.minimum(pred_box_xy + pred_box_wh / 2., true_box_xy + true_box_wh / 2.)
    intersect_wh = tf.maximum(intersect_maxs - intersect_mins, 0.)

    # shape: [13, 13, 3, V]
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    # shape: [13, 13, 3, 1]
    pred_box_area = pred_box_wh[..., 0] * pred_box_wh[..., 1]
    # shape: [V]
    true_box_area = true_box_wh[..., 0] * true_box_wh[..., 1]
    # shape: [1, V]
    true_box_area = tf.expand_dims(true_box_area, axis=0)

    # [13, 13, 3, V]
    iou = intersect_area / (pred_box_area + true_box_area - intersect_area + 1e-10)

    return iou


def yolo_loss(args, input_shape, anchors, anchors_mask, num_classes):
    num_layers = len(anchors_mask)

    # -----------------------------------------------------------#
    #   y_pred is a list，contains 3 feature maps，shape are: (m,13,13,255), (m,26,26,255), (m,52,52,255)
    #   y_true is a list，contains 3 feature maps，shape are: (m,13,13,3,85), (m,26,26,3,85), (m,52,52,3,85)
    # -----------------------------------------------------------#
    y_pred = args[:num_layers]
    y_true = args[num_layers:]

    # -----------------------------------------------------------#
    #   input_shpae = (416, 416)
    # -----------------------------------------------------------#
    input_shape = K.cast(input_shape, K.dtype(y_true[0]))

    # -----------------------------------------------------------#
    #   grid shapes = [[13,13], [26,26], [52,52]]
    # -----------------------------------------------------------#
    grid_shapes = [K.cast(K.shape(y_pred[l])[1:3], K.dtype(y_true[0])) for l in range(num_layers)]

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
    for l in range(num_layers):
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
        #   pred_box    (m,13,13,3,4) decode box center point x & y and width & height
        #       pred_xy     (m,13,13,3,2) decode box center point x & y
        #       pred_wh     (m,13,13,3,2) decode width & height
        # -----------------------------------------------------------#
        grid, raw_pred, pred_box = get_pred_boxes(y_pred[l], anchors[anchors_mask[l]],
                                                  num_classes, input_shape, calc_loss=True)

        # -----------------------------------------------------------#
        #   create a dynamic array to save negative samples
        # -----------------------------------------------------------#
        ignore_mask = tf.TensorArray(tf.float32, size=0, dynamic_size=True)

        def loop_cond(idx, ignore_mask):
            return tf.less(idx, tf.cast(m, tf.int32))

        # idx: to loop over N images in a batch
        def loop_body(idx, ignore_mask):
            # shape: [13, 13, 3, 4] & [13, 13, 3]  ==>  [V, 4]
            # V: num of true gt box of each image in a batch
            valid_true_boxes = tf.boolean_mask(y_true[l][idx, ..., 0:4], tf.cast(object_mask[idx, ..., 0], 'bool'))

            # shape: [13, 13, 3, 4] & [V, 4] ==> [13, 13, 3, V]
            iou = box_iou(pred_box[idx], valid_true_boxes)

            # shape: [13, 13, 3]
            best_iou = tf.reduce_max(iou, axis=-1)

            # shape: [13, 13, 3]
            ignore_mask_tmp = tf.cast(best_iou < 0.5, tf.float32)

            # finally will be shape: [N, 13, 13, 3]
            ignore_mask = ignore_mask.write(idx, ignore_mask_tmp)

            return idx + 1, ignore_mask

        _, ignore_mask = tf.while_loop(cond=loop_cond, body=loop_body, loop_vars=[0, ignore_mask])

        # -----------------------------------------------------------#
        #   expand the dimension to [N, 13, 13, 3, 1]
        # -----------------------------------------------------------#
        ignore_mask = ignore_mask.stack()
        ignore_mask = K.expand_dims(ignore_mask, -1)

        # -----------------------------------------------------------#
        #   normalize true xy and wh to align with predicted xy and wh to calculate loss later
        # -----------------------------------------------------------#
        raw_true_xy = y_true[l][..., :2] * grid_shapes[l][::-1] - grid
        raw_true_wh = K.log(y_true[l][..., 2:4] / anchors[anchors_mask[l]] * input_shape[::-1])

        # -----------------------------------------------------------#
        #   update raw_true_wh if no object update wh to 0 otherwise remain it as before
        # -----------------------------------------------------------#
        raw_true_wh = tf.where(condition=tf.math.is_nan(raw_true_wh), x=tf.zeros_like(raw_true_wh), y=raw_true_wh)

        # -----------------------------------------------------------#
        #   calculate box loss scale, x and y both between 0-1
        #   if real box is bigger box_loss_scale will become smaller
        #   if real box is smaller box_loss_scale will become larger
        #   to make sure big box and smaller box to have almost same loss value
        # -----------------------------------------------------------#
        box_loss_scale = 2 - y_true[l][..., 2:3] * y_true[l][..., 3:4]

        # -----------------------------------------------------------#
        #   calculate xy loss using binary_crossentropy
        # -----------------------------------------------------------#
        xy_loss = object_mask * box_loss_scale * K.binary_crossentropy(raw_true_xy, K.sigmoid(raw_pred[..., 0:2]),
                                                                       from_logits=True)

        # -----------------------------------------------------------#
        #   calculate wh_loss
        # -----------------------------------------------------------#
        wh_loss = object_mask * box_loss_scale * 0.5 * K.square(raw_true_wh - raw_pred[..., 2:4])

        # -----------------------------------------------------------#
        #   confidence loss
        #   if there is true box，calculate cross entropy loss between predicted score and 1
        #   if there no box，calculate cross entropy loss between predicted score and 0
        #   and will ignore the samples if best_iou<ignore_thresh
        # -----------------------------------------------------------#
        conf_pos_mask = object_mask
        conf_neg_mask = (1 - object_mask) * ignore_mask
        conf_loss_pos = conf_pos_mask * K.binary_crossentropy(object_mask, raw_pred[..., 4:5], from_logits=True)
        conf_loss_neg = conf_neg_mask * K.binary_crossentropy(object_mask, raw_pred[..., 4:5], from_logits=True)
        confidence_loss = conf_loss_pos + conf_loss_neg

        # -----------------------------------------------------------#
        #   class loss
        # -----------------------------------------------------------#
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

    y_true = [Input(shape=(input_shape[0] // {0: 32, 1: 16, 2: 8}[l], input_shape[1] // {0: 32, 1: 16, 2: 8}[l], len(anchors_mask[l]), 25)) for l in range(len(anchors_mask))]
    y_pred = [Input(shape=(input_shape[0] // {0: 32, 1: 16, 2: 8}[l], input_shape[1] // {0: 32, 1: 16, 2: 8}[l], len(anchors_mask[l]) * 25)) for l in range(len(anchors_mask))]

    # y_true = [tf.random.normal([4, 13, 13, 3, 25]), tf.random.normal([4, 13, 13, 3, 25]), tf.random.normal([4, 13, 13, 3, 25])]
    # y_pred = [tf.random.normal([4, 13, 13, 3 * 25]), tf.random.normal([4, 13, 13, 3 * 25]), tf.random.normal([4, 13, 13, 3 * 25])]

    # yolo_loss = YoloLoss(input_shape, anchors, anchors_mask, num_classes=20).loss([*y_pred, *y_true])

    print(input_shape.__len__(), outputs[-1].shape)
