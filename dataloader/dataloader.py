# ================================================================================
#   File name   : dataloader.py
#   Author      : sonalrpatel
#   Created date: 27-12-2021
#   GitHub      : https://github.com/sonalrpatel/object-detection-yolo
#   Description : functions used to prepare the dataset for custom training
# ================================================================================
# TODO: complete the dataloader file

import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from configs import *


class YOLODataset(object):
    # Dataset preprocess implementation
    def __init__(self, mode):
        self.mode = mode
        self.data_dir = DATA_DIR
        self.image_dir = IMAGE_DIR
        self.label_dir = LABEL_DIR
        self.image_size = IMAGE_SIZE
        self.batch_size = TRAIN_BATCH_SIZE
        self.data_aug = TRAIN_DATA_AUG

        self.train_input_sizes = TRAIN_INPUT_SIZE
        self.strides = np.array(YOLO_STRIDES)
        self.scales = np.array(YOLO_SCALES)
        self.classes = self.read_class_names(DATA_CLASSES)
        self.num_classes = len(self.classes)
        self.anchors = YOLO_ANCHORS
        self.anchor_per_scale = YOLO_ANCHOR_PER_SCALE
        self.max_bbox_per_scale = YOLO_MAX_BBOX_PER_SCALE

        self.img_bboxes_pairs = self.load_img_bboxes_pairs()
        self.num_samples = len(self.img_bboxes_pairs)
        self.num_batches = int(np.ceil(self.num_samples / self.batch_size))
        self.batch_count = 0

    def read_class_names(self, data_classes):
        names = {}
        return names

    def load_img_bboxes_pairs(self):
        """
        Customize this function as per your dataset
        # return: list of pairs of image path and corresponding bounding boxes
        # example:
            [['D:/01_PythonAIML/00_Datasets/PASCAL_VOC/images/000007.jpg',
             [[0.639, 0.5675675675675675, 0.718, 0.8408408408408409, 6.0]]]]
        """
        img_bboxes_pairs = []

        data_df = pd.read_csv(self.data_dir + self.mode + ".csv", header=None)
        data_df.columns = ['Image', 'label']

        for n in range(len(data_df)):
            img_path = os.path.join(self.image_dir, data_df['Image'][n])
            lbl_path = os.path.join(self.label_dir, data_df['label'][n])
            bboxes = np.roll(np.loadtxt(fname=lbl_path, delimiter=" ", ndmin=2), 4, axis=1).tolist()
            img_bboxes_pairs.append([img_path, bboxes])

        return img_bboxes_pairs

    def preprocess_true_boxes(self, bboxes):
        OUTPUT_LEVELS = len(self.strides)

        label = [np.zeros((self.train_output_sizes[i], self.train_output_sizes[i], self.anchor_per_scale,
                           5 + self.num_classes)) for i in range(OUTPUT_LEVELS)]
        bboxes_xywh = [np.zeros((self.max_bbox_per_scale, 4)) for _ in range(OUTPUT_LEVELS)]
        bbox_count = np.zeros((OUTPUT_LEVELS,))

        for bbox in bboxes:
            bbox_coor = bbox[:4]
            bbox_class_ind = bbox[4]

            onehot = np.zeros(self.num_classes, dtype=np.float)
            onehot[bbox_class_ind] = 1.0
            uniform_distribution = np.full(self.num_classes, 1.0 / self.num_classes)
            deta = 0.01
            smooth_onehot = onehot * (1 - deta) + deta * uniform_distribution

            bbox_xywh = np.concatenate([(bbox_coor[2:] + bbox_coor[:2]) * 0.5, bbox_coor[2:] - bbox_coor[:2]], axis=-1)
            bbox_xywh_scaled = 1.0 * bbox_xywh[np.newaxis, :] / self.strides[:, np.newaxis]

            iou = []
            exist_positive = False
            for i in range(OUTPUT_LEVELS):  # range(3):
                anchors_xywh = np.zeros((self.anchor_per_scale, 4))
                anchors_xywh[:, 0:2] = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32) + 0.5
                anchors_xywh[:, 2:4] = self.anchors[i]

                iou_scale = bbox_iou(bbox_xywh_scaled[i][np.newaxis, :], anchors_xywh)
                iou.append(iou_scale)
                iou_mask = iou_scale > 0.3

                if np.any(iou_mask):
                    xind, yind = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32)

                    label[i][yind, xind, iou_mask, :] = 0
                    label[i][yind, xind, iou_mask, 0:4] = bbox_xywh
                    label[i][yind, xind, iou_mask, 4:5] = 1.0
                    label[i][yind, xind, iou_mask, 5:] = smooth_onehot

                    bbox_ind = int(bbox_count[i] % self.max_bbox_per_scale)
                    bboxes_xywh[i][bbox_ind, :4] = bbox_xywh
                    bbox_count[i] += 1

                    exist_positive = True

            if not exist_positive:
                best_anchor_ind = np.argmax(np.array(iou).reshape(-1), axis=-1)
                best_detect = int(best_anchor_ind / self.anchor_per_scale)
                best_anchor = int(best_anchor_ind % self.anchor_per_scale)
                xind, yind = np.floor(bbox_xywh_scaled[best_detect, 0:2]).astype(np.int32)

                label[best_detect][yind, xind, best_anchor, :] = 0
                label[best_detect][yind, xind, best_anchor, 0:4] = bbox_xywh
                label[best_detect][yind, xind, best_anchor, 4:5] = 1.0
                label[best_detect][yind, xind, best_anchor, 5:] = smooth_onehot

                bbox_ind = int(bbox_count[best_detect] % self.max_bbox_per_scale)
                bboxes_xywh[best_detect][bbox_ind, :4] = bbox_xywh
                bbox_count[best_detect] += 1

        label_sbbox, label_mbbox, label_lbbox = label
        sbboxes, mbboxes, lbboxes = bboxes_xywh
        return label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes


if __name__ == "__main__":
    # annot = load_annotations()

    boxes = np.array(['302, 280, 572, 568, 25', '355, 509, 422, 624, 26', '406, 342, 640, 640, 0'])

    bboxes = np.array([list(map(int, box.split(','))) for box in boxes])

    # boxes = [box.reshape((-1, 5)) for box in boxes]
    boxes_xy = [0.5 * (box[:, 3:5] + box[:, 1:3]) for box in bboxes]
    boxes_wh = [box[:, 3:5] - box[:, 1:3] for box in bboxes]
    boxes_xy = [boxxy / orig_size for boxxy in boxes_xy]
    boxes_wh = [boxwh / orig_size for boxwh in boxes_wh]
    boxes = [np.concatenate((boxes_xy[i], boxes_wh[i], box[:, 0:1]), axis=1) for i, box in enumerate(bboxes)]

    print(len(annot))
