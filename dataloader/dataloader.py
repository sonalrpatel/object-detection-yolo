# ================================================================================
#   File name   : dataloader.py
#   Author      : sonalrpatel
#   Created date: 27-12-2021
#   GitHub      : https://github.com/sonalrpatel/object-detection-yolo
#   Description : functions used to prepare the dataset for custom training
# ================================================================================
# TODO: complete the dataloader file

import numpy as np
from configs import *


class YOLODataset(object):
    # Dataset preprocess implementation
    def __init__(self):
        self.img_dir = IMAGE_DIR
        self.label_dir = LABEL_DIR
        self.batch_size = TRAIN_BATCH_SIZE
        self.data_aug = TRAIN_DATA_AUG
        self.image_size = IMAGE_SIZE

        self.train_input_sizes = TRAIN_INPUT_SIZE
        self.strides = np.array(YOLO_STRIDES)
        self.scales = np.array(YOLO_SCALES)
        self.classes = self.read_class_names(DATA_CLASSES)
        self.num_classes = len(self.classes)
        self.anchors = YOLO_ANCHORS
        self.anchor_per_scale = YOLO_ANCHOR_PER_SCALE
        self.max_bbox_per_scale = YOLO_MAX_BBOX_PER_SCALE

        self.annotations = self.load_annotations()
        self.num_samples = len(self.annotations)
        self.num_batchs = int(np.ceil(self.num_samples / self.batch_size))
        self.batch_count = 0

    def read_class_names(self, data_classes):
        names = {}
        return names

    def load_annotations(self):
        final_annotations = []
        with open(annot_path, 'r') as f:
            txt = f.read().splitlines()
            annotations = [line.strip() for line in txt if len(line.strip().split()[1:]) != 0]
        np.random.shuffle(annotations)

        for annotation in annotations:
            # fully parse annotations
            line = annotation.split()
            image_path, index = "", 1
            for i, one_line in enumerate(line):
                if not one_line.replace(",", "").isnumeric():
                    if image_path != "":
                        image_path += " "
                    image_path += one_line
                else:
                    index = i
                    break
            # if not os.path.exists(image_path):
            #     raise KeyError("%s does not exist ... " % image_path)
            # if TRAIN_LOAD_IMAGES_TO_RAM:
            #     image = cv2.imread(image_path)
            # else:
            #     image = ''
            final_annotations.append([image_path, line[index:]])
        return final_annotations


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
