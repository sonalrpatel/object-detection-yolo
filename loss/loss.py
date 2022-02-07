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


class YoloLoss(object):
    def __init__(self, anchors, anchors_mask, num_classes=80, ignore_thresh=0.5):
        super(YoloLoss, self).__init__()
        self.use_label_smooth = False
        self.anchors = anchors
        self.anchors_mask = anchors_mask
        self.num_classes = num_classes
        self.ignore_thresh = ignore_thresh
        self.input_shape = (416, 416)
        self.img_size = (416, 416)
        self.num_layers = 3
        self.use_focal_loss = False
        self.use_static_shape = False
        self.class_num = 80

    def box_iou(self, pred_boxes, valid_true_boxes):
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

    def reorg_layer(self, feature_map, anchors):
        """
        feature_map: a feature_map from [feature_map_1, feature_map_2, feature_map_3] returned
            from `forward` function
        anchors: shape: [3, 2]
        """
        # NOTE: size in [h, w] format! don't get messed up!
        grid_size = feature_map.get_shape().as_list()[1:3] if self.use_static_shape else tf.shape(feature_map)[
                                                                                         1:3]  # [13, 13]
        # the downscale ratio in height and weight
        ratio = tf.cast(self.img_size / grid_size, tf.float32)
        # rescale the anchors to the feature_map
        # NOTE: the anchor is in [w, h] format!
        # rescaled_anchors = [(anchor[0] / ratio[1], anchor[1] / ratio[0]) for anchor in self.anchors]
        rescaled_anchors = K.reshape(K.constant(anchors) / [ratio[0], ratio[1]], [1, 1, 1, 3, 2])

        feature_map = tf.reshape(K.constant(feature_map), [-1, grid_size[0], grid_size[1], 3, 5 + self.class_num])

        # split the feature_map along the last dimension
        # shape info: take 416x416 input image and the 13*13 feature_map for example:
        # box_centers: [N, 13, 13, 3, 2] last_dimension: [center_x, center_y]
        # box_sizes: [N, 13, 13, 3, 2] last_dimension: [width, height]
        # conf_logits: [N, 13, 13, 3, 1]
        # prob_logits: [N, 13, 13, 3, class_num]
        box_centers, box_sizes, conf_logits, prob_logits = tf.split(feature_map, [2, 2, 1, self.class_num], axis=-1)
        box_centers = tf.nn.sigmoid(box_centers)

        # use some broadcast tricks to get the mesh coordinates
        grid_x = tf.range(grid_size[1], dtype=tf.int32)
        grid_y = tf.range(grid_size[0], dtype=tf.int32)
        grid_x, grid_y = tf.meshgrid(grid_x, grid_y)
        x_offset = tf.reshape(grid_x, (-1, 1))
        y_offset = tf.reshape(grid_y, (-1, 1))
        x_y_offset = tf.concat([x_offset, y_offset], axis=-1)
        # shape: [13, 13, 1, 2]
        x_y_offset = tf.cast(tf.reshape(x_y_offset, [grid_size[0], grid_size[1], 1, 2]), tf.float32)

        # get the absolute box coordinates on the feature_map
        box_centers = box_centers + x_y_offset
        # rescale to the original image scale
        box_centers = box_centers * ratio[::-1]

        # avoid getting possible nan value with tf.clip_by_value
        box_sizes = tf.exp(box_sizes) * rescaled_anchors
        # box_sizes = tf.clip_by_value(tf.exp(box_sizes), 1e-9, 100) * rescaled_anchors
        # rescale to the original image scale
        box_sizes = box_sizes * ratio[::-1]

        # shape: [N, 13, 13, 3, 4]
        # last dimension: (center_x, center_y, w, h)
        boxes = tf.concat([box_centers, box_sizes], axis=-1)

        # shape:
        # x_y_offset: [13, 13, 1, 2]
        # boxes: [N, 13, 13, 3, 4], rescaled to the original image scale
        # conf_logits: [N, 13, 13, 3, 1]
        # prob_logits: [N, 13, 13, 3, class_num]
        return x_y_offset, boxes, conf_logits, prob_logits

    def loss_layer(self, feature_map_i, yy_true, anchors):
        """
        calc loss function from a certain scale
        input:
            feature_map_i: feature maps of a certain scale. shape: [N, 13, 13, 3*(5 + num_class)] etc.
            y_true: y_ture from a certain scale. shape: [N, 13, 13, 3, 5 + num_class + 1] etc.
            anchors: shape [9, 2]
        """

        # size in [h, w] format! don't get messed up!
        grid_size = tf.shape(feature_map_i)[1:3]
        # the downscale ratio in height and weight
        ratio = tf.cast(self.img_size / grid_size, tf.float32)
        # N: batch_size
        N = tf.cast(tf.shape(feature_map_i)[0], tf.float32)

        x_y_offset, pred_boxes, pred_conf_logits, pred_prob_logits = self.reorg_layer(feature_map_i, anchors)

        ###########
        # get mask
        ###########

        # shape: take 416x416 input image and 13*13 feature_map for example:
        # [N, 13, 13, 3, 1]
        object_mask = yy_true[..., 4:5]

        # the calculation of ignore mask if referred from
        # https://github.com/pjreddie/darknet/blob/master/src/yolo_layer.c#L179
        ignore_mask = tf.TensorArray(tf.float32, size=0, dynamic_size=True)

        def loop_cond(idx, ignore_mask):
            return tf.less(idx, tf.cast(N, tf.int32))

        # idx: to loop over N images in a batch
        def loop_body(idx, ignore_mask):
            # shape: [13, 13, 3, 4] & [13, 13, 3]  ==>  [V, 4]
            # V: num of true gt box of each image in a batch
            valid_true_boxes = tf.boolean_mask(yy_true[idx, ..., 0:4], tf.cast(object_mask[idx, ..., 0], 'bool'))

            # shape: [13, 13, 3, 4] & [V, 4] ==> [13, 13, 3, V]
            iou = self.box_iou(pred_boxes[idx], valid_true_boxes)

            # shape: [13, 13, 3]
            best_iou = tf.reduce_max(iou, axis=-1)

            # shape: [13, 13, 3]
            ignore_mask_tmp = tf.cast(best_iou < 0.5, tf.float32)

            # finally will be shape: [N, 13, 13, 3]
            ignore_mask = ignore_mask.write(idx, ignore_mask_tmp)

            return idx + 1, ignore_mask

        _, ignore_mask = tf.while_loop(cond=loop_cond, body=loop_body, loop_vars=[0, ignore_mask])
        ignore_mask = ignore_mask.stack()

        # shape: [N, 13, 13, 3, 1]
        ignore_mask = tf.expand_dims(ignore_mask, -1)

        # shape: [N, 13, 13, 3, 2]
        pred_box_xy = pred_boxes[..., 0:2]
        pred_box_wh = pred_boxes[..., 2:4]

        # get xy coordinates in one cell from the feature_map
        # numerical range: 0 ~ 1
        # shape: [N, 13, 13, 3, 2]
        true_xy = yy_true[..., 0:2] / ratio[::-1] - x_y_offset
        pred_xy = pred_box_xy / ratio[::-1] - x_y_offset

        # get_tw_th
        # numerical range: 0 ~ 1
        # shape: [N, 13, 13, 3, 2]
        true_tw_th = yy_true[..., 2:4] / anchors
        pred_tw_th = pred_box_wh / anchors
        # for numerical stability
        true_tw_th = tf.where(condition=tf.equal(true_tw_th, 0),
                              x=tf.ones_like(true_tw_th), y=true_tw_th)
        pred_tw_th = tf.where(condition=tf.equal(pred_tw_th, 0),
                              x=tf.ones_like(pred_tw_th), y=pred_tw_th)
        true_tw_th = K.log(tf.clip_by_value(true_tw_th, 1e-9, 1e9))
        pred_tw_th = K.log(tf.clip_by_value(pred_tw_th, 1e-9, 1e9))

        # box size punishment:
        # box with smaller area has bigger weight. This is taken from the yolo darknet C source code.
        # shape: [N, 13, 13, 3, 1]
        box_loss_scale = 2. - (yy_true[..., 2:3] / tf.cast(self.img_size[1], tf.float32)) * (
                yy_true[..., 3:4] / tf.cast(self.img_size[0], tf.float32))

        ############
        # loss_part
        ############
        # mix_up weight
        # [N, 13, 13, 3, 1]
        mix_w = yy_true[..., -1:]

        # shape: [N, 13, 13, 3, 1]
        xy_loss = tf.reduce_sum(tf.square(true_xy - pred_xy) * object_mask * box_loss_scale * mix_w) / N
        wh_loss = tf.reduce_sum(tf.square(true_tw_th - pred_tw_th) * object_mask * box_loss_scale * mix_w) / N

        # shape: [N, 13, 13, 3, 1]
        conf_pos_mask = object_mask
        conf_neg_mask = (1 - object_mask) * ignore_mask
        conf_loss_pos = conf_pos_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=object_mask,
                                                                                logits=pred_conf_logits)
        conf_loss_neg = conf_neg_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=object_mask,
                                                                                logits=pred_conf_logits)
        # TODO: may need to balance the pos-neg by multiplying some weights
        conf_loss = conf_loss_pos + conf_loss_neg

        if self.use_focal_loss:
            alpha = 1.0
            gamma = 2.0
            # TODO: alpha should be a mask array if needed
            focal_mask = alpha * tf.pow(tf.abs(object_mask - tf.sigmoid(pred_conf_logits)), gamma)
            conf_loss *= focal_mask
        conf_loss = tf.reduce_sum(conf_loss * mix_w) / N

        # shape: [N, 13, 13, 3, 1]
        # whether to use label smooth
        if self.use_label_smooth:
            delta = 0.01
            label_target = (1 - delta) * yy_true[..., 5:-1] + delta * 1. / self.class_num
        else:
            label_target = yy_true[..., 5:]
        class_loss = object_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_target,
                                                                           logits=pred_prob_logits) * mix_w
        class_loss = tf.reduce_sum(class_loss) / N

        return xy_loss, wh_loss, conf_loss, class_loss

    def loss(self, y_true, y_pred):
        feature_map = y_pred
        y_pred = [tf.reshape(y_p, [-1, 13, 13, 3, 5 + self.class_num]) for y_p in y_pred]

        # -----------------------------------------------------------#
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
        loss1 = 0
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
            grid, raw_pred, pred_box = get_pred_boxes(y_pred[l], self.anchors[self.anchors_mask[l]],
                                                      self.num_classes, input_shape, calc_loss=True)

            # -----------------------------------------------------------#
            #   create a dynamic array to save negative samples
            # -----------------------------------------------------------#
            # ignore_mask = tf.TensorArray(K.dtype(y_true[0]), size=1, dynamic_size=True)
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
            raw_true_wh = K.log(y_true[l][..., 2:4] / self.anchors[self.anchors_mask[l]] * input_shape[::-1])

            # -----------------------------------------------------------#
            #   update raw_true_wh if no object update wh to 0 otherwise remain it as before
            # -----------------------------------------------------------#
            # raw_true_wh = K.switch(object_mask, raw_true_wh, K.zeros_like(raw_true_wh))

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
            xy_loss = object_mask * box_loss_scale * K.binary_crossentropy(raw_true_xy, raw_pred[..., 0:2],
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

            xy_loss1, wh_loss1, conf_loss1, class_loss1 = self.loss_layer(feature_map[l], y_true[l],
                                                                          self.anchors[self.anchors_mask[l]])
            loss1 += xy_loss1 + wh_loss1 + conf_loss1 + class_loss1

        loss = loss / num_pos
        return loss


if __name__ == "__main__":
    outputs = tf.random.normal([4, 5, 5, 255])
    anchors = np.array([[10, 13], [16, 30], [33, 23], [30, 61], [62, 45], [59, 119], [116, 90], [156, 198], [373, 326]])
    anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    input_shape = (416, 416)

    # y_true = [Input(shape=(input_shape[0] // {0: 32, 1: 16, 2: 8}[l], input_shape[1] // {0: 32, 1: 16, 2: 8}[l], len(anchors_mask[l]), 80 + 5)) for l in range(len(anchors_mask))]
    # y_pred = [Input(shape=(input_shape[0] // {0: 32, 1: 16, 2: 8}[l], input_shape[1] // {0: 32, 1: 16, 2: 8}[l], len(anchors_mask[l]), 80 + 5)) for l in range(len(anchors_mask))]

    y_true = [tf.random.normal([10, 13, 13, 3, 85]), tf.random.normal([10, 13, 13, 3, 85]),
              tf.random.normal([10, 13, 13, 3, 85])]
    y_pred = [tf.random.normal([10, 13, 13, 3 * 85]), tf.random.normal([10, 13, 13, 3 * 85]),
              tf.random.normal([10, 13, 13, 3 * 85])]

    # Dynamic implementation of conv dims for fully convolutional model.
    feats = y_pred[0]
    conv_dims = K.shape(feats)[1:3]  # assuming channels last
    # In YOLO the height index is the inner most iteration.
    conv_height_index = K.arange(0, stop=conv_dims[0])
    conv_width_index = K.arange(0, stop=conv_dims[1])
    conv_height_index = K.tile(conv_height_index, [conv_dims[1]])

    # TODO: Repeat_elements and tf.split doesn't support dynamic splits.
    # conv_width_index = K.repeat_elements(conv_width_index, conv_dims[1], axis=0)
    conv_width_index = K.tile(K.expand_dims(conv_width_index, 0), [conv_dims[0], 1])
    conv_width_index = K.flatten(K.transpose(conv_width_index))
    conv_index = K.transpose(K.stack([conv_height_index, conv_width_index]))
    conv_index = K.reshape(conv_index, [1, conv_dims[0], conv_dims[1], 1, 2])
    conv_index = K.cast(conv_index, K.dtype(feats))

    feats = K.reshape(feats, [-1, conv_dims[0], conv_dims[1], 3, 80 + 5])
    conv_dims = K.cast(K.reshape(conv_dims, [1, 1, 1, 1, 2]), K.dtype(feats))

    yolo_loss = YoloLoss(anchors, anchors_mask).loss(y_true, y_pred)

    print(input_shape.__len__(), outputs[-1].shape)
