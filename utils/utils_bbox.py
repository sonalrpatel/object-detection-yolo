import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input


# ==============================================================
#   Adjust predicted result to align with original image
#   Convert final layer features to bounding box parameters
# ==============================================================
def get_pred_boxes(final_layer_feats, anchors, num_classes, input_shape, calc_loss=False):
    # ==============================================================
    #   number of anchors
    # ==============================================================
    num_anchors = len(anchors)

    # ==============================================================
    #   (m is batch_size)
    #   final layer feature shape is (m, 13, 13, 255) or (m, 26, 26, 255) or (m, 52, 52, 255) as per scales
    #   grid_shape = (width, height) = (13, 13) or (26, 26) or (52, 52)
    # ==============================================================
    grid_shape = K.shape(final_layer_feats)[1:3]

    # ==============================================================
    #   generate grid with shape => (grid_shape[0], grid_shape[1], num_anchors, "2" for x & y) => e.g. (13, 13, 3, 2)
    # ==============================================================
    grid_x = K.tile(K.reshape(K.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]), [grid_shape[0], 1, num_anchors, 1])
    grid_y = K.tile(K.reshape(K.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]), [1, grid_shape[1], num_anchors, 1])
    grid = K.concatenate([grid_x, grid_y])
    grid = K.cast(grid, K.dtype(final_layer_feats))

    # ==============================================================
    #   adjust pre-defined anchors to shape (13, 13, num_anchors, 2)
    # ==============================================================
    anchors_tensor = K.reshape(K.constant(anchors), [1, 1, num_anchors, 2])
    anchors_tensor = K.tile(anchors_tensor, [grid_shape[0], grid_shape[1], 1, 1])

    # ==============================================================
    #   reshape prediction results (m, 13, 13, 255) to (m, 13, 13, 3, 85)
    #   85 = 4 + 1 + 80
    #   4 -> x offset, y offset, width and height
    #   1 -> confidence score
    #   80 -> 80 classes
    # ==============================================================
    final_layer_feats = K.reshape(final_layer_feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])

    # ==============================================================
    #   calculate bounding box center point bx, by and normalize by grid shape (13, 26 or 52)
    #   bx = sigmoid(tx) + cx
    #   by = sigmoid(tx) + cy
    #   calculate bounding box width(bw), height(bh) and normalize by input shape (416)
    #   bw = pw * exp(tw)
    #   bh = ph * exp(th)
    # ==============================================================
    box_xy = (K.sigmoid(final_layer_feats[..., :2]) + grid) / K.cast(grid_shape[::-1], K.dtype(final_layer_feats))
    box_wh = K.exp(final_layer_feats[..., 2:4]) * anchors_tensor / K.cast(input_shape[::-1], K.dtype(final_layer_feats))

    # ==============================================================
    #   if calc loss, then return -> grid, feats, box_xy, box_wh
    #   if during prediction, then return -> box_xy, box_wh, box_confidence, box_class_probs
    # ==============================================================
    if calc_loss:
        ret = [grid, final_layer_feats, box_xy, box_wh]
    else:
        # ==============================================================
        #   retrieve confidence score and class probabilities
        # ==============================================================
        box_confidence = K.sigmoid(final_layer_feats[..., 4:5])
        box_class_probs = K.sigmoid(final_layer_feats[..., 5:])

        ret = [box_xy, box_wh, box_confidence, box_class_probs]

    return ret


def yolo_head(feats, anchors, num_classes):
    """Convert final layer features to bounding box parameters.

    Parameters
    ----------
    feats : tensor
        Final convolutional layer features.
    anchors : array-like
        Anchor box widths and heights.
    num_classes : int
        Number of target classes.

    Returns
    -------
    box_xy : tensor
        x, y box predictions adjusted by spatial location in conv layer.
    box_wh : tensor
        w, h box predictions adjusted by anchors and conv spatial resolution.
    box_conf : tensor
        Probability estimate for whether each box contains any object.
    box_class_pred : tensor
        Probability distribution estimate for each box over class labels.
    """
    num_anchors = len(anchors)
    # Reshape to batch, height, width, num_anchors, box_params.
    anchors_tensor = K.reshape(K.variable(anchors), [1, 1, 1, num_anchors, 2])

    # Static implementation for fixed models.
    # TODO: Remove or add option for static implementation.
    # _, conv_height, conv_width, _ = K.int_shape(feats)
    # conv_dims = K.variable([conv_width, conv_height])

    # Dynamic implementation of conv dims for fully convolutional model.
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

    feats = K.reshape(feats, [-1, conv_dims[0], conv_dims[1], num_anchors, num_classes + 5])
    conv_dims = K.cast(K.reshape(conv_dims, [1, 1, 1, 1, 2]), K.dtype(feats))

    # Static generation of conv_index:
    # conv_index = np.array([_ for _ in np.ndindex(conv_width, conv_height)])
    # conv_index = conv_index[:, [1, 0]]  # swap columns for YOLO ordering.
    # conv_index = K.variable(
    #     conv_index.reshape(1, conv_height, conv_width, 1, 2))
    # feats = Reshape(
    #     (conv_dims[0], conv_dims[1], num_anchors, num_classes + 5))(feats)

    box_xy = K.sigmoid(feats[..., :2])
    box_wh = K.exp(feats[..., 2:4])
    box_confidence = K.sigmoid(feats[..., 4:5])
    box_class_probs = K.softmax(feats[..., 5:])

    # Adjust predictions to each spatial grid point and anchor size.
    # Note: YOLO iterates over height index before width index.
    box_xy = (box_xy + conv_index) / conv_dims
    box_wh = box_wh * anchors_tensor / conv_dims

    return box_xy, box_wh, box_confidence, box_class_probs


if __name__ == "__main__":
    outputs = tf.random.normal([2, 13, 13, 255])
    anchors = np.array([[10, 13], [16, 30], [33, 23], [30, 61], [62, 45], [59, 119], [116, 90], [156, 198], [373, 326]])
    anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    input_shape = (416, 416)

    # image_shape = K.reshape(outputs[-1], [-1])

    # y_true = [Input(shape=(
    # input_shape[0] // {0: 32, 1: 16, 2: 8}[l], input_shape[1] // {0: 32, 1: 16, 2: 8}[l], len(anchors_mask[l]), 80 + 5))
    #           for l in range(len(anchors_mask))]

    grid, raw_pred, pred_xy, pred_wh = get_pred_boxes(outputs, anchors[anchors_mask[0]], 80, input_shape)

    returns = yolo_head(outputs, anchors[anchors_mask[0]], 80)

    print(input_shape.__len__(), outputs[-1].shape)
