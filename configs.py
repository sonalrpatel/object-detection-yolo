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
YOLO_ANCHORS = [[[10, 13], [16, 30], [33, 23]],         # 52x52 grids for small objects
                [[30, 61], [62, 45], [59, 119]],        # 26x26 grids for medium objects
                [[116, 90], [156, 198], [373, 326]]]    # 13x13 grids for large objects
YOLO_ANCHORS_MASK = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
YOLO_LAYER_WITH_NAMES = True

# IMAGE size
IMAGE_SIZE = (416, 416)

# Dataset
# DATA_DIR = "data/demo/images/"
DIR_DATA = "data/"
DIR_IMAGE = DIR_DATA + "demo/images/"
DIR_LABEL = DIR_DATA + "demo/images/"
PATH_CLASSES = DIR_DATA + "coco_classes.txt"
PATH_ANCHORS = DIR_DATA + 'yolo_anchors.txt'
PATH_WEIGHT = DIR_DATA + 'yolov3.h5'
PATH_DARKNET_WEIGHT = DIR_DATA + 'yolov3.weights'

# TRAIN options
TRAIN_YOLO_TINY = False
TRAIN_SAVE_BEST_ONLY = True  # saves only best model according validation loss (True recommended)
TRAIN_SAVE_CHECKPOINT = False  # saves all best validated checkpoints in training process (may require a lot disk space) (False recommended)
TRAIN_LOGDIR = "log"
TRAIN_ANNOT_PATH = DIR_DATA + "demo/annotations.txt"
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

# VAL options
VAL_ANNOT_PATH = DIR_DATA + "annotations.txt"
VAL_BATCH_SIZE = 8
VAL_INPUT_SIZE = 416
VAL_DATA_AUG = False

# TEST options
TEST_ANNOT_PATH = DIR_DATA + "test.csv"
TEST_BATCH_SIZE = 8
TEST_INPUT_SIZE = 416
TEST_DATA_AUG = False
TEST_DECTECTED_IMAGE_PATH = ""
TEST_SCORE_THRESHOLD = 0.3
TEST_IOU_THRESHOLD = 0.5

# LOG directory
LOG_DIR = "D:/01_PythonAIML/06_myProjects/object-detection-yolo3/logs/weights/"
