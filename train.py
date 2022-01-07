from PIL import Image
import numpy as np
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
import cv2


def rand(a=0, b=1):
    return np.random.rand() * (b - a) + a


def get_random_data(annotation_line, input_shape, max_boxes=20, proc_img=True):
    """random preprocessing for real-time data augmentation"""
    line = annotation_line.split()
    image = Image.open(line[0])
    iw, ih = image.size
    h, w = input_shape
    box = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])

    # resize image
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)
    dx = (w - nw) // 2
    dy = (h - nh) // 2
    image_data = 0
    if proc_img:
        image = image.resize((nw, nh), Image.BICUBIC)
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_image.paste(image, (dx, dy))
        image_data = np.array(new_image) / 255.

    # correct boxes
    box_data = np.zeros((max_boxes, 5))
    if len(box) > 0:
        np.random.shuffle(box)
        if len(box) > max_boxes: box = box[:max_boxes]
        box[:, [0, 2]] = box[:, [0, 2]] * scale + dx
        box[:, [1, 3]] = box[:, [1, 3]] * scale + dy
        box_data[:len(box)] = box

    return image_data, box_data


def preprocess_true_boxes(true_boxes, input_shape, anchors, num_classes):
    """
    Preprocess true boxes to training input format

    Parameters
    ----------
    true_boxes: array, shape=(m, T, 5)
        Absolute x_min, y_min, x_max, y_max, class_id relative to input_shape.
    input_shape: array-like, hw, multiples of 32
    anchors: array, shape=(N, 2), wh
    num_classes: integer

    Returns
    -------
    y_true: list of array, shape like yolo_outputs, xywh are relative value

    """
    assert (true_boxes[..., 4] < num_classes).all(), 'class id must be less than num_classes'
    num_scales = 3
    num_anchors = len(anchors)
    num_anchors_per_scale = num_anchors // num_scales  # default setting
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]

    input_shape = np.array(input_shape, dtype='int32')
    true_boxes = np.array(true_boxes, dtype='float32')
    # batch size
    batch_size = true_boxes.shape[0]

    # (x_min, y_min, x_max, y_max) is converted to (x_center, y_center, width, height) relative to input shape
    boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2
    boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]
    true_boxes[..., 0:2] = boxes_xy / input_shape[::-1]
    true_boxes[..., 2:4] = boxes_wh / input_shape[::-1]

    # grid shapes for 3 scales
    grid_shapes = [input_shape // {0: 32, 1: 16, 2: 8}[s] for s in range(num_scales)]

    # init y_true [num_scales][batch_size x (grid_shape_0 x grid_shape_1) x num_anchors_per_scale x (5 + num_classes)]
    y_true = [np.zeros((batch_size, grid_shapes[s][0], grid_shapes[s][1], len(anchor_mask[s]), 5 + num_classes),
                       dtype='float32') for s in range(num_scales)]

    # Expand dim to apply broadcasting
    anchors = np.expand_dims(anchors, 0)
    anchor_maxes = anchors / 2.
    anchor_mins = -anchor_maxes
    # find anchor area
    anchor_area = anchors[..., 0] * anchors[..., 1]

    # number of non zero boxes
    num_nz_boxes = (np.count_nonzero(boxes_wh, axis=1).sum(axis=1)/2).astype('int32')

    for batch_no in range(batch_size):
        # Discard zero rows
        box_wh = boxes_wh[batch_no, 0:num_nz_boxes[batch_no]]
        if len(box_wh) == 0:
            continue

        # Expand dim to apply broadcasting
        box_wh = np.expand_dims(box_wh, -2)
        box_maxes = box_wh / 2.
        box_mins = -box_maxes
        # find box area
        box_area = box_wh[..., 0] * box_wh[..., 1]

        # find intersection area
        intersect_mins = np.maximum(box_mins, anchor_mins)
        intersect_maxes = np.minimum(box_maxes, anchor_maxes)
        intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]

        # find iou
        iou_anchors = intersect_area / (box_area + anchor_area - intersect_area)

        # Find best anchor for each true box
        best_anchor_indices = np.argmax(iou_anchors, axis=-1)

        # y_true shape:
        # [num_scales][batch_size x (grid_shape_0 x grid_shape_1) x num_anchors_per_scale x (5 + num_classes)]
        for box_no, anchor_idx in enumerate(best_anchor_indices):
            # [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
            scale_idx, scale, anchor_on_scale = [(x, grid_shapes[x][0], anchor_mask[x].index(anchor_idx))
                                                 for x in range(len(anchor_mask)) if anchor_idx in anchor_mask[x]][0]

            # dimensions of a single box
            x, y, width, height, class_label = true_boxes[batch_no, box_no, 0:5]

            # index of the grid cell having the center of the bbox
            i = np.floor(y * scale).astype('int32')
            j = np.floor(x * scale).astype('int32')

            # fill y_true
            y_true[scale_idx][batch_no, i, j, anchor_on_scale, 0:4] = np.array([x, y, width, height])
            y_true[scale_idx][batch_no, i, j, anchor_on_scale, 4] = 1
            y_true[scale_idx][batch_no, i, j, anchor_on_scale, 5 + int(class_label)] = 1

    return y_true


def data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes):
    """data generator for fit_generator"""
    n = len(annotation_lines)
    i = 0

    image_data = []
    box_data = []
    for b in range(batch_size):
        if i == 0:
            np.random.shuffle(annotation_lines)
        image, box = get_random_data(annotation_lines[i], input_shape)
        box = np.floor(box / 2.5)
        image_data.append(image)
        box_data.append(box)
        i = (i + 1) % n

    image_data = np.array(image_data)
    box_data = np.array(box_data)
    y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)

    return y_true


def data_generator_wrapper(annotation_lines, batch_size, input_shape, anchors, num_classes):
    n = len(annotation_lines)
    if n == 0 or batch_size <= 0:
        return None
    return data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes)


def get_anchors(anchors_path):
    """loads the anchors from a file"""
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)


def get_classes(classes_path):
    """loads the classes"""
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names


def _main():
    annotation_path = 'data/annotations.txt'
    log_dir = 'logs/000/'
    classes_path = 'data/pascal_classes.txt'
    anchors_path = 'data/yolo_anchors.txt'
    class_names = get_classes(classes_path)
    num_classes = len(class_names)
    anchors = get_anchors(anchors_path)
    input_shape = (416, 416)  # multiple of 32, hw
    batch_size = 32

    val_split = 0.1
    with open(annotation_path) as f:
        lines = f.readlines()
    np.random.shuffle(lines)
    num_val = int(len(lines) * val_split)
    num_train = len(lines) - num_val

    data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes)

    return 0


if __name__ == '__main__':
    _main()
