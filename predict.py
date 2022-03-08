# =========================================================================
#   predict.py integrates functions such as single image prediction, video/camera detection, FPS test and
#       directory traversal detection.
#   It is integrated into a py file, and the mode is modified by specifying the mode.
# =========================================================================
import time

import cv2
import time
import colorsys
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras.models import Model

from utils.utils import *
from utils.utils_bbox import *
from model.yolov3 import YOLOv3


class YoloResult(object):
    _defaults = {
        # =====================================================================
        #   To use your own trained model for prediction, you must modify model_path and classes_path!
        #   model_path points to the weights file under the logs folder,
        #       classes_path points to the txt under model_data
        #
        #   After training, there are multiple weight files in the logs folder,
        #       and you can select the validation set with lower loss.
        #   The lower loss of the validation set does not mean that the mAP is higher,
        #       it only means that the weight has better generalization performance on the validation set.
        #   If the shape does not match, pay attention to the modification of the model_path
        #       and classes_path parameters during training
        # =====================================================================
        "model_path": 'logs/self_trained_yolo_weights.h5',
        "classes_path": 'data/coco_classes.txt',

        # =====================================================================
        #   anchors_path represents the txt file corresponding to the a priori box, which is generally not modified.
        #   anchors_mask is used to help the code find the corresponding a priori box and is generally not modified.
        # =====================================================================
        "anchors_path": 'data/yolo_anchors.txt',
        "anchors_mask": [[6, 7, 8], [3, 4, 5], [0, 1, 2]],

        # =====================================================================
        #   The size of the input image, which must be a multiple of 32.
        # =====================================================================
        "input_shape": [416, 416],

        # =====================================================================
        #   Only prediction boxes with scores greater than confidence will be kept
        # =====================================================================
        "confidence": 0.5,

        # =====================================================================
        #   nms_iou size used for non-maximum suppression
        # =====================================================================
        "nms_iou": 0.3,
        "max_boxes": 100,

        # =====================================================================
        #   This variable is used to control whether to use letterbox_image
        #       to resize the input image without distortion,
        #   After many tests, it is found that the direct resize effect of closing letterbox_image is better
        # =====================================================================
        "letterbox_image": True,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    # =====================================================================
    #   Initialize yolo
    # =====================================================================
    def __init__(self, **kwargs):
        self.input_image_shape = Input([2, ], batch_size=1)
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)

        # =====================================================================
        #   Get the number of kinds and a priori boxes
        # =====================================================================
        self.class_names, self.num_classes = get_classes(self.classes_path)
        self.anchors, self.num_anchors = get_anchors(self.anchors_path)

        # =====================================================================
        #   Picture frame set different colors
        # =====================================================================
        hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))

        self.generate()

    # =====================================================================
    #   Load model
    # =====================================================================
    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        self.yolo_model = YOLOv3([None, None, 3], self.num_classes)
        self.yolo_model.load_weights(self.model_path)

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # =====================================================================
        #   In the DecodeBox function, we will post-process the prediction results
        #   The content of post-processing includes decoding, non-maximum suppression, threshold filtering, etc.
        # =====================================================================
        inputs = [*self.yolo_model.output, self.input_image_shape]

        outputs = Lambda(
            DecodeBox,
            output_shape=(1,),
            name='yolo_eval',
            arguments={
                'anchors': self.anchors,
                'num_classes': self.num_classes,
                'input_shape': self.input_shape,
                'anchor_mask': self.anchors_mask,
                'confidence': self.confidence,
                'nms_iou': self.nms_iou,
                'max_boxes': self.max_boxes,
                'letterbox_image': self.letterbox_image
            }
        )(inputs)

        self.yolo_model = Model([self.yolo_model.input, self.input_image_shape], outputs)

    @tf.function
    def get_pred(self, image_data, input_image_shape):
        out_boxes, out_scores, out_classes = self.yolo_model([image_data, input_image_shape], training=False)

        return out_boxes, out_scores, out_classes

    # =====================================================================
    #   Detect pictures
    # =====================================================================
    def detect_image(self, image):
        # =====================================================================
        #   Convert the image to an RGB image here to prevent an error in the prediction of the grayscale image.
        #   The code only supports prediction of RGB images, all other types of images will be converted to RGB
        # =====================================================================
        image = cvtColor(image)

        # =====================================================================
        #   Add gray bars to the image to achieve undistorted resize
        #   You can also directly resize for identification
        # =====================================================================
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)

        # =====================================================================
        #   Add the batch_size dimension and normalize it
        # =====================================================================
        image_data = np.expand_dims(preprocess_input(np.array(image_data, dtype='float32')), 0)

        # =====================================================================
        #   Feed the image into the network to make predictions!
        # =====================================================================
        input_image_shape = np.expand_dims(np.array([image.size[1], image.size[0]], dtype='float32'), 0)
        out_boxes, out_scores, out_classes = self.get_pred(image_data, input_image_shape)

        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

        # =====================================================================
        #   Set font and border thickness
        # =====================================================================
        font = ImageFont.truetype(font='model_data/simhei.ttf',
                                  size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = int(max((image.size[0] + image.size[1]) // np.mean(self.input_shape), 1))

        # =====================================================================
        #   Image drawing
        # =====================================================================
        for i, c in list(enumerate(out_classes)):
            predicted_class = self.class_names[int(c)]
            box = out_boxes[i]
            score = out_scores[i]

            top, left, bottom, right = box

            top = max(0, np.floor(top).astype('int32'))
            left = max(0, np.floor(left).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom).astype('int32'))
            right = min(image.size[0], np.floor(right).astype('int32'))

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            print(label, top, left, bottom, right)

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle([left + i, top + i, right - i, bottom - i], outline=self.colors[c])
            draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[c])
            draw.text(text_origin, str(label, 'UTF-8'), fill=(0, 0, 0), font=font)
            del draw

        return image

    def get_FPS(self, image, test_interval):
        # =====================================================================
        #   Convert the image to an RGB image here to prevent an error in the prediction of the grayscale image.
        #   The code only supports prediction of RGB images, all other types of images will be converted to RGB
        # =====================================================================
        image = cvtColor(image)

        # =====================================================================
        #   Add gray bars to the image to achieve undistorted resize
        #   You can also directly resize for identification
        # =====================================================================
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)

        # =====================================================================
        #   Add the batch_size dimension and normalize it
        # =====================================================================
        image_data = np.expand_dims(preprocess_input(np.array(image_data, dtype='float32')), 0)

        # =====================================================================
        #   Feed the image into the network to make predictions!
        # =====================================================================
        input_image_shape = np.expand_dims(np.array([image.size[1], image.size[0]], dtype='float32'), 0)
        out_boxes, out_scores, out_classes = self.get_pred(image_data, input_image_shape)

        t1 = time.time()
        for _ in range(test_interval):
            out_boxes, out_scores, out_classes = self.get_pred(image_data, input_image_shape)

        t2 = time.time()
        tact_time = (t2 - t1) / test_interval
        return tact_time

    # =====================================================================
    #   Detect pictures
    # =====================================================================
    def get_map_txt(self, image_id, image, class_names, map_out_path):
        f = open(os.path.join(map_out_path, "detection-results/" + image_id + ".txt"), "w")
        # =====================================================================
        #   Convert the image to an RGB image here to prevent the grayscale image from making errors during prediction.
        # =====================================================================
        image = cvtColor(image)

        # =====================================================================
        #   Add gray bars to the image to achieve undistorted resize
        #   You can also directly resize for identification
        # =====================================================================
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)

        # =====================================================================
        #   Add the batch_size dimension and normalize it
        # =====================================================================
        image_data = np.expand_dims(preprocess_input(np.array(image_data, dtype='float32')), 0)

        # =====================================================================
        #   Feed the image into the network to make predictions!
        # =====================================================================
        input_image_shape = np.expand_dims(np.array([image.size[1], image.size[0]], dtype='float32'), 0)
        out_boxes, out_scores, out_classes = self.get_pred(image_data, input_image_shape)

        for i, c in enumerate(out_classes):
            predicted_class = self.class_names[int(c)]
            try:
                score = str(out_scores[i].numpy())
            except:
                score = str(out_scores[i])
            top, left, bottom, right = out_boxes[i]
            if predicted_class not in class_names:
                continue

            f.write("%s %s %s %s %s %s\n" % (
                predicted_class, score[:6], str(int(left)), str(int(top)), str(int(right)), str(int(bottom))))

        f.close()
        return


if __name__ == "__main__":
    yolo = YoloResult()

    # =====================================================================
    #   mode is used to specify the mode of the test:
    #   'predict' means single image prediction. If you want to modify the prediction process, such as saving images,
    #       intercepting objects, etc., you can read the detailed notes below first.
    #   'video' means video detection, you can call the camera or video for detection, see the notes below for details.
    #   'fps' means test fps, the image used is street.jpg in img, see the notes below for details.
    #   'dir_predict' means to traverse the folder to detect and save. By default, the img folder is traversed and
    #       the img_out folder is saved. For details, see the notes below.
    # =====================================================================
    mode = "predict"

    # =====================================================================
    #   video_origin_path is used to specify the path of the video, when video_origin_path = 0, it means to detect
    #       the camera.
    #   If you want to detect the video, set it as video_origin_path = "xxx.mp4", which means to read the xxx.mp4 file
    #       in the root directory.
    #   video_save_path indicates the path where the video is saved, when video_save_path = "" it means not to save.
    #   If you want to save the video, set it as video_save_path = "yyy.mp4", which means that it will be saved as
    #       a yyy.mp4 file in the root directory.
    #   video_fps is the fps of the saved video.
    #   video_origin_path, video_save_path and video_fps are only valid when mode = 'video'
    #   When saving the video, you need ctrl+c to exit or run to the last frame to complete the complete save step.
    # =====================================================================
    video_origin_path = 0
    video_save_path = ""
    video_fps = 25.0

    # =====================================================================
    #   test_interval is used to specify the number of image detections when measuring fps
    #   In theory, the larger the test_interval, the more accurate the fps.
    # =====================================================================
    test_interval = 100

    # =====================================================================
    #   dir_origin_path specifies the folder path of the image used for detection
    #   dir_save_path specifies the save path of the detected image
    #   dir_origin_path and dir_save_path are only valid when mode = 'dir_predict'
    # =====================================================================
    dir_origin_path = "img/"
    dir_save_path = "img_out/"

    # =====================================================================
    #   If you want to save the detected image, use r_image.save("img.jpg") to save it, and modify it directly in
    #       predict.py.
    #   If you want to get the coordinates of the prediction frame, you can enter the yolo.detect_image function and
    #       read the four values of top, left, bottom, and right in the drawing part.
    #   If you want to use the prediction frame to intercept the target, you can enter the yolo.detect_image function,
    #       and use the obtained four values of top, left, bottom, and right in the drawing part.
    #   Use the matrix method to intercept the original image.
    #   If you want to write extra words on the prediction map, such as the number of specific targets detected,
    #       you can enter the yolo.detect_image function and judge the predicted_class in the drawing part,
    #       For example, judging if predicted_class == 'car': can judge whether the current target is a car,
    #       and then record the number. Use draw.text to write.
    # =====================================================================
    if mode == "predict":
        while True:
            img = input('Input image filename:')
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                r_image = yolo.detect_image(image)
                r_image.show()

    elif mode == "video":
        capture = cv2.VideoCapture(video_origin_path)
        if video_save_path != "":
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            out = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)

        ref, frame = capture.read()
        if not ref:
            raise ValueError("The camera (video) cannot be read correctly, please pay attention to whether the camera"
                             "is installed correctly (whether the video path is correctly filled in).")

        fps = 0.0
        while True:
            t1 = time.time()
            # read a frame
            ref, frame = capture.read()
            if not ref:
                break
            # format conversion, BGRtoRGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # convert to Image
            frame = Image.fromarray(np.uint8(frame))
            # test
            frame = np.array(yolo.detect_image(frame))
            # RGBtoBGR meets the opencv display format
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            fps = (fps + (1. / (time.time() - t1))) / 2
            print("fps= %.2f" % (fps))
            frame = cv2.putText(frame, "fps= %.2f" % (fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("video", frame)
            c = cv2.waitKey(1) & 0xff
            if video_save_path != "":
                out.write(frame)

            if c == 27:
                capture.release()
                break

        print("Video Detection Done!")
        capture.release()
        if video_save_path != "":
            print("Save processed video to the path :" + video_save_path)
            out.release()
        cv2.destroyAllWindows()

    elif mode == "fps":
        img = Image.open('img/street.jpg')
        tact_time = yolo.get_FPS(img, test_interval)
        print(str(tact_time) + ' seconds, ' + str(1 / tact_time) + 'FPS, @batch_size 1')

    elif mode == "dir_predict":
        import os
        from tqdm import tqdm

        img_names = os.listdir(dir_origin_path)
        for img_name in tqdm(img_names):
            if img_name.lower().endswith(
                    ('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                image_path = os.path.join(dir_origin_path, img_name)
                image = Image.open(image_path)
                r_image = yolo.detect_image(image)
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                r_image.save(os.path.join(dir_save_path, img_name))

    else:
        raise AssertionError("Please specify the correct mode: 'predict', 'video', 'fps' or 'dir_predict'.")
