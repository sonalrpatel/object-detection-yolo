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
        returns = [grid, final_layer_feats, box_xy, box_wh]
    else:
        # ==============================================================
        #   retrieve confidence score and class probabilities
        # ==============================================================
        box_confidence = K.sigmoid(final_layer_feats[..., 4:5])
        box_class_probs = K.sigmoid(final_layer_feats[..., 5:])

        returns = [box_xy, box_wh, box_confidence, box_class_probs]

    return returns


if __name__ == "__main__":
    outputs = tf.random.normal([4, 5, 5, 255])
    anchors = np.array([[10, 13], [16, 30], [33, 23], [30, 61], [62, 45], [59, 119], [116, 90], [156, 198], [373, 326]])
    anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    input_shape = (416, 416)

    # image_shape = K.reshape(outputs[-1], [-1])

    y_true = [Input(shape=(input_shape[0] // {0: 32, 1: 16, 2: 8}[l], input_shape[1] // {0: 32, 1: 16, 2: 8}[l], len(anchors_mask[l]), 80 + 5)) for l in range(len(anchors_mask))]

    grid, raw_pred, pred_xy, pred_wh = get_pred_boxes(outputs, anchors[anchors_mask[0]], 80, input_shape)

    print(input_shape.__len__(), outputs[-1].shape)
