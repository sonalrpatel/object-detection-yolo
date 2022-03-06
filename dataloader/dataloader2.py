import math
from random import shuffle

import cv2
import numpy as np
from PIL import Image
from tensorflow import keras
from configs import *
from utils.utils import *


def read_lines(annotation_path):
    with open(annotation_path) as f:
        annot_lines = f.readlines()
    return annot_lines


def load_img_bboxes_pairs(annotation_path):
    """
    Load annotations
    Customize this function as per your dataset
    :return:
        list of pairs of image path and corresponding bounding boxes
        example:
        [['.../00_Datasets/PASCAL_VOC/images/000007.jpg', [[0.639, 0.567, 0.718, 0.840, 6.0],
                                                           [0.529, 0.856, 0.125, 0.435, 4.0]]]
         ['.../00_Datasets/PASCAL_VOC/images/000008.jpg', [[0.369, 0.657, 0.871, 0.480, 3.0]]]]
    """""
    lines = read_lines(annotation_path)

    img_bboxes_pairs = [[line.split()[0], np.array([list(map(int, box.split(','))) for box in line.split()[1:]])]
                        for line in lines]

    return img_bboxes_pairs


class YoloDataGenerator(keras.utils.Sequence):
    def __init__(self, mode):
        self.img_bboxes_pairs = load_img_bboxes_pairs({'train': TRAIN_ANNOT_PATH, 'val': VAL_ANNOT_PATH}[mode])
        self.data_aug = {'train': TRAIN_DATA_AUG, 'val': VAL_DATA_AUG}[mode]
        self.batch_size = {'train': TRAIN_BATCH_SIZE, 'val': VAL_BATCH_SIZE}[mode]
        self.input_shape = IMAGE_SIZE
        self.anchors_mask = YOLO_ANCHORS_MASK
        self.num_scales = len(self.anchors_mask)
        self.num_images = len(self.img_bboxes_pairs)
        self.classes, self.num_classes = get_classes(PATH_CLASSES)
        self.anchors, self.num_anchors = get_anchors(PATH_ANCHORS)

    def __len__(self):
        """
        Denotes the number of batches per epoch
        """""
        return math.ceil(self.num_images / float(self.batch_size))

    def __getitem__(self, index):
        """
        Generate one batch of data when the batch corresponding to a given index
        is called, the generator executes the __getitem__ method to generate it.
        """""
        # Generate indexes for a batch
        batch_indexes = range(index * self.batch_size, (index + 1) * self.batch_size)

        # Generate data
        image_data, y_true = self.__data_generation(batch_indexes)

        return [image_data, *y_true], np.zeros(self.batch_size)

    def on_epoch_begin(self):
        """
        Shuffle indexes at start of each epoch
        """""
        # Shuffle the dataset
        if self.shuffle:
            np.random.shuffle(self.img_bboxes_pairs)

    def process_data(self, img_bboxes_pair, data_aug=False, max_boxes=20, proc_img=True):
        """
        Random preprocessing for real-time data augmentation
        """""
        image = Image.open(img_bboxes_pair[0])
        iw, ih = image.size
        h, w = self.input_shape
        box = img_bboxes_pair[1]

        # resize image
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)
        dx = (w - nw) // 2
        dy = (h - nh) // 2

        if data_aug:
            # todo augmentation
            image = image

        image_data = Image.new('RGB', (w, h), (128, 128, 128))
        if proc_img:
            image = image.resize((nw, nh), Image.BICUBIC)
            image_data.paste(image, (dx, dy))
            image_data = np.array(image_data) / 255.

        # correct boxes
        box_data = np.zeros((max_boxes, 5))
        if box.shape[0] > 0:
            np.random.shuffle(box)
            if box.shape[0] > max_boxes:
                box = box[:max_boxes]
            box[:, [0, 2]] = box[:, [0, 2]] * scale + dx
            box[:, [1, 3]] = box[:, [1, 3]] * scale + dy
            box_data[:box.shape[0]] = box

        return image_data, box_data

    def preprocess_true_boxes(self, true_boxes):
        """
        Preprocess true boxes to training input format
        :param
            true_boxes: array, shape=(m, T, 5)
            Absolute x_min, y_min, x_max, y_max, class_id relative to input_shape.
        :returns: 
            y_true: list of array, shape like yolo_outputs, xywh are relative value
        """""
        assert (true_boxes[..., 4] < self.num_classes).all(), 'class id must be less than num_classes'
        anchor_mask = YOLO_ANCHORS_MASK
        input_shape = np.array(self.input_shape, dtype='int32')

        # (x_min, y_min, x_max, y_max) is converted to (x_center, y_center, width, height) relative to input shape
        boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2
        boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]
        true_boxes[..., 0:2] = boxes_xy / input_shape[::-1]
        true_boxes[..., 2:4] = boxes_wh / input_shape[::-1]

        # grid shapes for 3 scales
        grid_shapes = [input_shape // {0: 32, 1: 16, 2: 8}[s] for s in range(self.num_scales)]

        # initialise y_true
        # [num_scales][batch_size x (grid_shape_0 x grid_shape_1) x num_anchors_per_scale x (5 + num_classes)]
        y_true = [np.zeros((self.batch_size, grid_shapes[s][0], grid_shapes[s][1], anchor_mask[s].__len__(),
                            5 + self.num_classes), dtype='float32') for s in range(self.num_scales)]

        # Expand dim to apply broadcasting
        anchors = np.expand_dims(self.anchors, 0)
        anchor_maxes = anchors / 2.
        anchor_mins = -anchor_maxes
        # find anchor area
        anchor_area = anchors[..., 0] * anchors[..., 1]

        # number of non zero boxes
        num_nz_boxes = (np.count_nonzero(boxes_wh, axis=1).sum(axis=1) / 2).astype('int32')

        for b_idx in range(self.batch_size):
            # Discard zero rows
            box_wh = boxes_wh[b_idx, 0:num_nz_boxes[b_idx]]
            if box_wh.shape[0] == 0:
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
                scale_idx, scale, anchor_on_scale = [(x, tuple(grid_shapes[x]), anchor_mask[x].index(anchor_idx))
                                                     for x in range(anchor_mask.__len__())
                                                     if anchor_idx in anchor_mask[x]][0]

                # dimensions of a single box
                x, y, width, height, class_label = true_boxes[b_idx, box_no, 0:5]

                # index of the grid cell having the center of the bbox
                i = np.floor(y * scale[0]).astype('int32')
                j = np.floor(x * scale[1]).astype('int32')

                # fill y_true
                y_true[scale_idx][b_idx, i, j, anchor_on_scale, 0:4] = np.array([x, y, width, height])
                y_true[scale_idx][b_idx, i, j, anchor_on_scale, 4] = 1
                y_true[scale_idx][b_idx, i, j, anchor_on_scale, 5 + int(class_label)] = 1

        return y_true

    def __data_generation(self, batch_indexes):
        """
        Generates data containing batch_size samples
        """""
        image_data = []
        box_data = []
        for i in batch_indexes:
            image, box = self.process_data(self.img_bboxes_pairs[i], self.data_aug)
            box = np.floor(box / 2.5)
            image_data.append(image)
            box_data.append(box)

        image_data = np.array(image_data)
        box_data = np.array(box_data)
        y_true = self.preprocess_true_boxes(box_data)

        return image_data, y_true


if __name__ == "__main__":

    ydg = YoloDataGenerator('train')
    num_batches = ydg.__len__()
    X0, y0 = ydg.__getitem__(0)
    X1, y1 = ydg.__getitem__(1)