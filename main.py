import numpy as np
from model.model_functional import YOLOv3


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
    img_bboxes_pairs = []
    for annot_path in annotation_path:
        lines = read_lines(annot_path)
        i_b_pairs = [[annot_path.rsplit('/', 1)[0] + '/' + line.split()[0],
                      np.array([list(map(int, box.split(','))) for box in line.split()[1:]])]
                     for line in lines]

        img_bboxes_pairs.extend(i_b_pairs)

    return img_bboxes_pairs


if __name__ == "__main__":
    DIR_DATA = ["data/fruits/", "data/demo/"]
    DIR_TRAIN = [d + "train/" for d in DIR_DATA]
    DIR_VALID = [d + "valid/" for d in DIR_DATA]
    DIR_TEST = [d + "test/" for d in DIR_DATA]
    PATH_CLASSES = DIR_TRAIN[0] + "_classes.txt"
    TRAIN_ANNOT_PATH = [d + "_annotations.txt" for d in DIR_TRAIN]

    pairs = load_img_bboxes_pairs(TRAIN_ANNOT_PATH)

    x = 0
