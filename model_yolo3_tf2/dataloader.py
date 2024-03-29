import os
import math
import cv2
import numpy as np
from PIL import Image
from random import shuffle
from tensorflow import keras

from utils.utils import convert2rgb, preprocess_input


def read_lines(path):
    with open(path) as f:
        lines = f.readlines()
    return lines


def YoloAnnotationPairs(annotation_path):
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
    annotation_pairs = []
    for path in annotation_path:
        lines = read_lines(path)
        pairs = [[path.rsplit('/', 1)[0] + '/' + line.split()[0],
                  np.array([list(map(int, box.split(','))) for box in line.split()[1:]])]
                 for line in lines]
        annotation_pairs.extend(pairs)
    return annotation_pairs


class YoloDataGenerator(keras.utils.Sequence):
    def __init__(self, annotation_pairs, input_shape, anchors, batch_size, num_classes, anchors_mask, do_aug):
        self.annotation_pairs = annotation_pairs
        self.input_shape = input_shape
        self.anchors = anchors
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.num_samples = len(self.img_bboxes_pairs)
        self.num_batches = int(np.ceil(self.num_samples / self.batch_size))
        self.anchors_mask = anchors_mask
        self.do_aug = do_aug

    def __len__(self):
        return self.num_batches

    def __getitem__(self, index):
        image_data = []
        box_data = []
        for i in range(index * self.batch_size, (index + 1) * self.batch_size):
            i = i % self.num_samples
            # ===============================================
            #   Random data enhancement
            #   No random enhancement of data occurs during validation
            # ===============================================
            image, box = self.get_random_data(self.annotation_pairs[i], self.input_shape, random=self.do_aug)
            image_data.append(preprocess_input(np.array(image)))
            box_data.append(box)

        image_data = np.array(image_data)
        box_data = np.array(box_data)
        y_true = self.preprocess_true_boxes(box_data, self.input_shape, self.anchors, self.num_classes)
        return [image_data, *y_true], np.zeros(self.batch_size)

    def on_epoch_begin(self):
        shuffle(self.annotation_pairs)

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def get_random_data(self, img_bboxes_pair, input_shape, max_boxes=100, jitter=.3, hue=.1, sat=1.5, val=1.5,
                        random=True):
        # ===============================================
        #   Reads the image and converts it to an RGB image
        # ===============================================
        image = Image.open(img_bboxes_pair[0])
        image = convert2rgb(image)

        # ===============================================
        #   Gets the height width of the image and the target height
        # ===============================================
        iw, ih = image.size
        h, w = input_shape

        # ===============================================
        #   Gets the forecast box
        # ===============================================
        box = img_bboxes_pair[1]

        if not random:
            scale = min(w / iw, h / ih)
            nw = int(iw * scale)
            nh = int(ih * scale)
            dx = (w - nw) // 2
            dy = (h - nh) // 2

            # ===============================================
            #   Add a gray bar to the excess of the image
            # ===============================================
            image = image.resize((nw, nh), Image.BICUBIC)
            new_image = Image.new('RGB', (w, h), (128, 128, 128))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image, np.float32)

            # ===============================================
            #   Make adjustments to the real box
            # ===============================================
            box_data = np.zeros((max_boxes, 5))
            if len(box) > 0:
                np.random.shuffle(box)
                box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
                box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
                box[:, 0:2][box[:, 0:2] < 0] = 0
                box[:, 2][box[:, 2] > w] = w
                box[:, 3][box[:, 3] > h] = h
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                box = box[np.logical_and(box_w > 1, box_h > 1)]
                if len(box) > max_boxes:
                    box = box[:max_boxes]
                box_data[:len(box)] = box

            return image_data, box_data

        # ===============================================
        #   Scale the image and distort the length and width
        # ===============================================
        new_ar = w / h * self.rand(1 - jitter, 1 + jitter) / self.rand(1 - jitter, 1 + jitter)
        scale = self.rand(.25, 2)
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)
        image = image.resize((nw, nh), Image.BICUBIC)

        # ===============================================
        #   Add gray bars to the extra parts of the image
        # ===============================================
        dx = int(self.rand(0, w - nw))
        dy = int(self.rand(0, h - nh))
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_image.paste(image, (dx, dy))
        image = new_image

        # ===============================================
        #   Flip the image
        # ===============================================
        flip = self.rand() < .5
        if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)

        # ===============================================
        #   Gamut distortion
        # ===============================================
        hue = self.rand(-hue, hue)
        sat = self.rand(1, sat) if self.rand() < .5 else 1 / self.rand(1, sat)
        val = self.rand(1, val) if self.rand() < .5 else 1 / self.rand(1, val)
        x = cv2.cvtColor(np.array(image, np.float32) / 255, cv2.COLOR_RGB2HSV)
        x[..., 0] += hue * 360
        x[..., 0][x[..., 0] > 1] -= 1
        x[..., 0][x[..., 0] < 0] += 1
        x[..., 1] *= sat
        x[..., 2] *= val
        x[x[:, :, 0] > 360, 0] = 360
        x[:, :, 1:][x[:, :, 1:] > 1] = 1
        x[x < 0] = 0
        image_data = cv2.cvtColor(x, cv2.COLOR_HSV2RGB) * 255  # numpy array, 0 to 1

        # ===============================================
        #   Adjust the real box
        # ===============================================
        box_data = np.zeros((max_boxes, 5))
        if len(box) > 0:
            np.random.shuffle(box)
            box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
            box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
            if flip: box[:, [0, 2]] = w - box[:, [2, 0]]
            box[:, 0:2][box[:, 0:2] < 0] = 0
            box[:, 2][box[:, 2] > w] = w
            box[:, 3][box[:, 3] > h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w > 1, box_h > 1)]  # discard invalid box
            if len(box) > max_boxes: box = box[:max_boxes]
            box_data[:len(box)] = box

        return image_data, box_data

    def preprocess_true_boxes(self, true_boxes, input_shape, anchors, num_classes):
        """
        preprocess true boxes
        :param true_boxes: ground truth boxes with shape (m, n, 5)
                           m: stands for number of images
                           n: stands for number of boxes
                           5: stands for x_min, y_min, x_max, y_max and class_id
        :param input_shape: 416*416
        :param anchors: size of pre-defined 9 anchor boxes
        :param num_classes: number of classes
        :return:
        """
        assert (true_boxes[..., 4] < num_classes).all(), 'class id must be less than num_classes'

        true_boxes = np.array(true_boxes, dtype='float32')
        input_shape = np.array(input_shape, dtype='int32')

        # ===============================================
        #   3 feature layers in total
        # ===============================================
        num_layers = len(self.anchors_mask)

        # ===============================================
        #   m -> number of images，grid_shapes -> [[13,13], [26,26], [52,52]]
        # ===============================================
        m = true_boxes.shape[0]
        grid_shapes = [input_shape // {0: 32, 1: 16, 2: 8}[l] for l in range(num_layers)]

        # ===============================================
        #   y_true -> [(m,13,13,3,85),(m,26,26,3,85),(m,52,52,3,85)]
        # ===============================================
        y_true = [np.zeros((m, grid_shapes[l][0], grid_shapes[l][1], len(self.anchors_mask[l]), 5 + num_classes),
                           dtype='float32') for l in range(num_layers)]

        # ===============================================
        #   calculate center point xy, box width and box height
        #   boxes_xy shape -> (m,n,2)  boxes_wh -> (m,n,2)
        # ===============================================
        boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2
        boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]

        # ===============================================
        #   normalization
        # ===============================================
        true_boxes[..., 0:2] = boxes_xy / input_shape[::-1]
        true_boxes[..., 2:4] = boxes_wh / input_shape[::-1]

        # ===============================================
        #   [9,2] -> [1,9,2]
        # ===============================================
        anchors = np.expand_dims(anchors, 0)
        anchor_maxes = anchors / 2.
        anchor_mins = -anchor_maxes

        # ===============================================
        #   only retrieve image width > 0
        # ===============================================
        valid_mask = boxes_wh[..., 0] > 0

        # loop all the image
        for b in range(m):
            # ===============================================
            #   only retrieve image width > 0
            # ===============================================
            wh = boxes_wh[b, valid_mask[b]]
            if len(wh) == 0:
                continue

            # ===============================================
            #   [n,2] -> [n,1,2]
            # ===============================================
            wh = np.expand_dims(wh, -2)
            box_maxes = wh / 2.
            box_mins = - box_maxes

            # ===============================================
            #   Calculate IOU between true box and pre-defined anchors
            #   intersect_area  [n,9]
            #   box_area        [n,1]
            #   anchor_area     [1,9]
            #   iou             [n,9]
            # ===============================================
            intersect_mins = np.maximum(box_mins, anchor_mins)
            intersect_maxes = np.minimum(box_maxes, anchor_maxes)
            intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
            intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]

            box_area = wh[..., 0] * wh[..., 1]
            anchor_area = anchors[..., 0] * anchors[..., 1]

            iou = intersect_area / (box_area + anchor_area - intersect_area)
            best_anchor = np.argmax(iou, axis=-1)

            # loop all the best anchors, try to find it to which feature layer below
            # (m 13, 13, 3, 85), (m 26, 26, 3, 85),  (m 52, 52, 3, 85)
            for t, n in enumerate(best_anchor):
                # ===============================================
                #   Loop all the layers
                # ===============================================
                for l in range(num_layers):
                    if n in self.anchors_mask[l]:
                        # ===============================================
                        #   using floor true boxes' x、y coordinates
                        # ===============================================
                        i = np.floor(true_boxes[b, t, 0] * grid_shapes[l][1]).astype('int32')
                        j = np.floor(true_boxes[b, t, 1] * grid_shapes[l][0]).astype('int32')

                        # ===============================================
                        #   k -> index of pre-defined anchors
                        # ===============================================
                        k = self.anchors_mask[l].index(n)

                        # ===============================================
                        #   c -> the object category
                        # ===============================================
                        c = true_boxes[b, t, 4].astype('int32')

                        # ===============================================
                        #   y_true => shape => (m,13,13,3,85) or (m,26,26,3,85) or (m,52,52,3,85)
                        # ===============================================
                        y_true[l][b, j, i, k, 0:4] = true_boxes[b, t, 0:4]
                        y_true[l][b, j, i, k, 4] = 1
                        y_true[l][b, j, i, k, 5 + c] = 1

        return y_true
