# ================================================================
#   File name   : configs.py
#   Author      : sonalrpatel
#   Created date: 27-12-2021
#   GitHub      : https://github.com/sonalrpatel/object-detection-yolo
#   Description : yolov3 configuration file
# ================================================================

# YOLO options
YOLO_TYPE = "yolov3"
YOLO_FRAMEWORK = "tf"
YOLO_V3_WEIGHTS = "yolov3.weights"
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
# DIR_DATA is filled as a list to consider multiple dataset folders at the same time
DIR_DATA = "data/bdd100k/"
DIR_TRAIN = DIR_DATA + "train/"
DIR_VALID = DIR_DATA + "val/"
DIR_TEST = DIR_DATA + "test/"
PATH_CLASSES = DIR_DATA + "bdd_classes.txt"
PATH_ANCHORS = "model_data/yolo_anchors.txt"
PATH_WEIGHT = "model_data/yolov3_coco.h5"
PATH_DARKNET_WEIGHT = "model_data/yolov3.weights"

# TRAIN options
TRAIN_YOLO_TINY = False
TRAIN_SAVE_BEST_ONLY = True  # saves only best model according validation loss (True recommended)
TRAIN_SAVE_CHECKPOINT = False  # saves all validated checkpoints in training process (may require a lot disk space) (False recommended)
TRAIN_LOGDIR = "log"
TRAIN_ANNOT_PATH = DIR_DATA + "det_train.json"
TRAIN_CHECKPOINTS_FOLDER = "checkpoints"
TRAIN_MODEL_NAME = f"{YOLO_TYPE}_custom"
TRAIN_FROM_CHECKPOINT = False
TRAIN_TRANSFER = True
TRAIN_DATA_AUG = True
TRAIN_FREEZE_BODY = True
TRAIN_FREEZE_BATCH_SIZE = 32
TRAIN_UNFREEZE_BATCH_SIZE = 16  # note that more GPU memory is required after unfreezing the body
TRAIN_FREEZE_LR = 1e-3
TRAIN_UNFREEZE_LR = 1e-4
TRAIN_FREEZE_INIT_EPOCH = 0
TRAIN_FREEZE_END_EPOCH = 20
TRAIN_UNFREEZE_END_EPOCH = 40  # note that it is considered when TRAIN_FREEZE_BODY is True

# VAL options
VAL_ANNOT_PATH = DIR_DATA + "det_val.json"
VAL_DATA_AUG = False
VAL_BATCH_SIZE = 16
VAL_VALIDATION_USING = "TRAIN"  # note that when validation data does not exist, set it to TRAIN or None
VAL_VALIDATION_SPLIT = 0.2  # note that it will be used when VAL_VALIDATION_USING is TRAIN

# TEST options
TEST_ANNOT_PATH = DIR_DATA + "det_test.json"
TEST_BATCH_SIZE = 16
TEST_DATA_AUG = False
TEST_DETECTED_IMAGE_PATH = ""
TEST_SCORE_THRESHOLD = 0.3
TEST_IOU_THRESHOLD = 0.5

# LOG directory
LOG_DIR = "logs/"
