# ================================================================
#   File name   : configs.py
#   Author      : sonalrpatel
#   Created date: 27-12-2021
#   GitHub      : https://github.com/sonalrpatel/object-detection-yolo
#   Description : yolov3 configuration file
# ================================================================
# TODO: create such configs file for semantic segmentation project

# YOLO options
YOLO_TYPE = "yolov3"
YOLO_FRAMEWORK = "tf"
YOLO_V3_WEIGHTS = "data/yolov3.weights"
YOLO_CUSTOM_WEIGHTS = False
YOLO_IOU_LOSS_THRESH = 0.5
YOLO_STRIDES = [8, 16, 32]
YOLO_SCALES = [52, 26, 13]
YOLO_NUM_SCALES = 3
YOLO_ANCHOR_PER_SCALE = 3
YOLO_MAX_BBOX_PER_SCALE = 100
YOLO_INPUT_SIZE = 416
YOLO_ANCHORS = [[[10, 13], [16, 30], [33, 23]],         # 52x52 grids for small objects
                [[30, 61], [62, 45], [59, 119]],        # 26x26 grids for medium objects
                [[116, 90], [156, 198], [373, 326]]]    # 13x13 grids for large objects

# Dataset
# DATA_DIR = "D:/01_PythonAIML/00_Datasets/PASCAL_VOC/"
DATA_DIR = "D:/01_PythonAIML/06_myProjects/object-detection-yolo/data/"
CLASSES_PATH = DATA_DIR + "pascal_classes.txt"
ANCHORS_PATH = DATA_DIR + 'yolo_anchors.txt'
IMAGE_DIR = DATA_DIR + "images"
LABEL_DIR = DATA_DIR + "labels"
IMAGE_SIZE = (416, 416)

# TRAIN options
TRAIN_YOLO_TINY = False
TRAIN_SAVE_BEST_ONLY = True  # saves only best model according validation loss (True recommended)
TRAIN_SAVE_CHECKPOINT = False  # saves all best validated checkpoints in training process (may require a lot disk space) (False recommended)
TRAIN_LOGDIR = "log"
# TRAIN_ANNOT_PATH = DATA_DIR + "train.csv"
TRAIN_ANNOT_PATH = DATA_DIR + "annotations.txt"
TRAIN_CHECKPOINTS_FOLDER = "checkpoints"
TRAIN_MODEL_NAME = f"{YOLO_TYPE}_custom"
TRAIN_LOAD_IMAGES_TO_RAM = True  # With True faster training, but need more RAM
TRAIN_BATCH_SIZE = 16
TRAIN_INPUT_SIZE = 416
TRAIN_DATA_AUG = True
TRAIN_TRANSFER = True
TRAIN_FROM_CHECKPOINT = False  # "checkpoints/yolov3_custom"
TRAIN_LR_INIT = 1e-4
TRAIN_LR_END = 1e-6
TRAIN_WARMUP_EPOCHS = 2
TRAIN_EPOCHS = 100

# TEST options
TEST_ANNOT_PATH = DATA_DIR + "test.csv"
TEST_BATCH_SIZE = 8
TEST_INPUT_SIZE = 416
TEST_DATA_AUG = False
TEST_DECTECTED_IMAGE_PATH = ""
TEST_SCORE_THRESHOLD = 0.3
TEST_IOU_THRESHOLD = 0.45

