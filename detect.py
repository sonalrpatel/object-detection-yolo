import os
import cv2
import pandas as pd
from model_keras_yolo3.yolo import make_yolov3_model
from model_yolo3_tf2.yolo import yolo_body
from model.model_functional import YOLOv3
from utils.utils import *
from configs import *


# References
# This file is sourced from https://github.com/experiencor/keras-yolo3 repository
# yolov3 weights is downloaded from https://pjreddie.com/media/files/yolov3.weights

class BoundBox:
    def __init__(self, xmin, ymin, xmax, ymax, objness=None, classes=None):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax

        self.objness = objness
        self.classes = classes

        self.label = -1
        self.score = -1

    def get_label(self):
        if self.label == -1:
            self.label = np.argmax(self.classes)

        return self.label

    def get_score(self):
        if self.score == -1:
            self.score = self.classes[self.get_label()]

        return self.score


def _interval_overlap(interval_a, interval_b):
    x1, x2 = interval_a
    x3, x4 = interval_b

    if x3 < x1:
        if x4 < x1:
            return 0
        else:
            return min(x2, x4) - x1
    else:
        if x2 < x3:
            return 0
        else:
            return min(x2, x4) - x3


def _sigmoid(x):
    return 1. / (1. + np.exp(-x))


def bbox_iou(box1, box2):
    intersect_w = _interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
    intersect_h = _interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])

    intersect = intersect_w * intersect_h

    w1, h1 = box1.xmax - box1.xmin, box1.ymax - box1.ymin
    w2, h2 = box2.xmax - box2.xmin, box2.ymax - box2.ymin

    union = w1 * h1 + w2 * h2 - intersect

    return float(intersect) / union


def preprocess_input(image, net_h, net_w):
    new_h, new_w, _ = image.shape

    # determine the new size of the image
    if (float(net_w) / new_w) < (float(net_h) / new_h):
        new_h = int((new_h * net_w) / new_w)
        new_w = net_w
    else:
        new_w = int((new_w * net_h) / new_h)
        new_h = net_h

    # resize the image to the new size
    resized = cv2.resize(image[:, :, ::-1] / 255., (int(new_w), int(new_h)))

    # embed the image into the standard letter box
    new_image = np.ones((net_h, net_w, 3)) * 0.5
    new_image[int((net_h - new_h) // 2):int((net_h + new_h) // 2), int((net_w - new_w) // 2):int((net_w + new_w) // 2), :] = resized
    new_image = np.expand_dims(new_image, 0)

    return new_image


def decode_netout(yolos, anchors, anchors_mask, obj_thresh, nms_thresh, net_h, net_w):
    boxes_all = []
    for i in range(len(yolos)):
        netout = yolos[i][0]
        anchor_list = [int(k) for j in anchors[anchors_mask[i]] for k in j]
        grid_h, grid_w = netout.shape[:2]
        nb_box = 3
        netout = netout.reshape((grid_h, grid_w, nb_box, -1))
        nb_class = netout.shape[-1] - 5

        boxes = []

        netout[..., :2] = _sigmoid(netout[..., :2])
        netout[..., 4:] = _sigmoid(netout[..., 4:])
        netout[..., 5:] = netout[..., 4][..., np.newaxis] * netout[..., 5:]
        netout[..., 5:] *= netout[..., 5:] > obj_thresh

        for i in range(grid_h * grid_w):
            row = i / grid_w
            col = i % grid_w

            for b in range(nb_box):
                # 4th element is objectness score
                objectness = netout[int(row)][int(col)][b][4]
                # objectness = netout[..., :4]

                if objectness.all() <= obj_thresh:
                    continue

                # first 4 elements are x, y, w, and h
                x, y, w, h = netout[int(row)][int(col)][b][:4]

                x = (col + x) / grid_w  # center position, unit: image width
                y = (row + y) / grid_h  # center position, unit: image height
                w = anchor_list[2 * b + 0] * np.exp(w) / net_w  # unit: image width
                h = anchor_list[2 * b + 1] * np.exp(h) / net_h  # unit: image height

                # last elements are class probabilities
                classes = netout[int(row)][col][b][5:]

                box = BoundBox(x - w / 2, y - h / 2, x + w / 2, y + h / 2, objectness, classes)
                # box = BoundBox(x-w/2, y-h/2, x+w/2, y+h/2, None, classes)

                boxes.append(box)

        boxes_all += boxes
    return boxes_all


def correct_yolo_boxes(boxes, image_h, image_w, net_h, net_w):
    if (float(net_w) / image_w) < (float(net_h) / image_h):
        new_w = net_w
        new_h = (image_h * net_w) / image_w
    else:
        new_h = net_w
        new_w = (image_w * net_h) / image_h

    for i in range(len(boxes)):
        x_offset, x_scale = (net_w - new_w) / 2. / net_w, float(new_w) / net_w
        y_offset, y_scale = (net_h - new_h) / 2. / net_h, float(new_h) / net_h

        boxes[i].xmin = int((boxes[i].xmin - x_offset) / x_scale * image_w)
        boxes[i].xmax = int((boxes[i].xmax - x_offset) / x_scale * image_w)
        boxes[i].ymin = int((boxes[i].ymin - y_offset) / y_scale * image_h)
        boxes[i].ymax = int((boxes[i].ymax - y_offset) / y_scale * image_h)


def do_nms(boxes, nms_thresh):
    if len(boxes) > 0:
        nb_class = len(boxes[0].classes)
    else:
        return

    for c in range(nb_class):
        sorted_indices = np.argsort([-box.classes[c] for box in boxes])

        for i in range(len(sorted_indices)):
            index_i = sorted_indices[i]

            if boxes[index_i].classes[c] == 0: continue

            for j in range(i + 1, len(sorted_indices)):
                index_j = sorted_indices[j]

                if bbox_iou(boxes[index_i], boxes[index_j]) >= nms_thresh:
                    boxes[index_j].classes[c] = 0


def draw_boxes(image, boxes, labels, obj_thresh):
    for box in boxes:
        label_str = ''
        label = -1

        for i in range(len(labels)):
            if box.classes[i] > obj_thresh:
                label_str += labels[i]
                label = i
                print(labels[i] + ': ' + str(box.classes[i] * 100) + '%')

        if label >= 0:
            cv2.rectangle(image, (box.xmin, box.ymin), (box.xmax, box.ymax), (0, 255, 0), 3)
            cv2.putText(image,
                        label_str + ' ' + str(box.get_score()),
                        (box.xmin, box.ymin - 13),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1e-3 * image.shape[0],
                        (0, 255, 0), 2)

    return image


def _main():
    model_path = "data/yolo_weights.h5"
    weights_path = "data/yolov3.weights"
    image_path = "data/apple.jpg"

    model_path = os.path.join(os.path.dirname(__file__), model_path)
    weights_path = os.path.join(os.path.dirname(__file__), weights_path)
    image_path = os.path.join(os.path.dirname(__file__), image_path)

    # set some parameters
    net_h, net_w = 416, 416
    obj_thresh, nms_thresh = 0.5, 0.45

    # =======================================================
    #   Be sure to modify classes_path before training so that it corresponds to your own dataset
    # =======================================================
    classes_path = PATH_CLASSES

    # =======================================================
    #   Anchors_path represents the txt file corresponding to the a priori box, which is generally not modified
    #   Anchors_mask is used to help the code find the corresponding a priori box and is generally not modified
    # =======================================================
    anchors_path = PATH_ANCHORS
    anchors_mask = YOLO_ANCHORS_MASK
    labels, num_classes = get_classes(classes_path)
    anchors, num_anchors = get_anchors(anchors_path)
    # anchors = [[116, 90, 156, 198, 373, 326], [30, 61, 62, 45, 59, 119], [10, 13, 16, 30, 33, 23]]

    # make the yolov3 model to predict 80 classes on COCO
    yolov3 = YOLOv3((None, None, 3), num_classes)
    # yolov3 = make_yolov3_model((None, None, 3))
    # yolov3 = yolo_body((None, None, 3), anchors_mask, num_classes)

    # run model summary
    yolov3.summary()

    # load the weights trained on COCO into the model
    # option 1
    # weight_reader = WeightReader(weights_path)
    # weight_reader.load_weights(yolov3)
    # option 2
    yolov3.load_weights(model_path, by_name=True, skip_mismatch=True)

    # preprocess the image
    image = cv2.imread(image_path)
    image_h, image_w, _ = image.shape
    new_image = preprocess_input(image, net_h, net_w)

    # run the prediction
    yolos = yolov3.predict(new_image)

    # decode the output of the network
    boxes = decode_netout(yolos, anchors, anchors_mask, obj_thresh, nms_thresh, net_h, net_w)

    # correct the sizes of the bounding boxes
    correct_yolo_boxes(boxes, image_h, image_w, net_h, net_w)

    # suppress non-maximal boxes
    do_nms(boxes, nms_thresh)

    # draw bounding boxes on the image using labels
    draw_boxes(image, boxes, labels, obj_thresh)

    # write the image with bounding boxes to file
    cv2.imwrite(image_path.split('.')[0] + '_detected.jpg', image.astype('uint8'))

    print("Completed")


if __name__ == '__main__':
    _main()
