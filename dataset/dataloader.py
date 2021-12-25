import numpy as np
import cv2
import os

annot_path = "D:/01_PythonAIML/06_myProjects/TensorFlow-2.x-YOLOv3/model_data/coco/val2017.txt"

TRAIN_LOAD_IMAGES_TO_RAM = True


def load_annotations():
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
