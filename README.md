# object-detection-yolo

## Introduction

This repository is an implementation of YOLOv3 in Tensorflow. It contains complete pipeline for training and prediction on custom datasets. <br>
The primary references are [yolo3-tf2](https://github.com/Qucy/yolo3-tf2) and [keras-yolo3](https://github.com/qqwweee/keras-yolo3) which helped me to understand the implementations and bring up this repo.

## Predict

convert pre-trained Darknet weights to h5 format

```bash
# yolov3
wget https://pjreddie.com/media/files/yolov3.weights -O data/yolov3.weights
python convert.py --weights_path data/yolov3.weights --output_path data/yolov3_coco.h5
```

detection on a single image

```bash
# yolov3
python predict.py -w data/yolov3_coco.h5 -c data/coco_classes.txt -i data/sample/apple.jpg
```

## Training

pending to update 

## Benchmark / Result

pending to update

## References

The various resources referred are organised and listed below.

### YOLOv1

- [practical-guide-object-detection-yolo-framework](https://www.analyticsvidhya.com/blog/2018/12/practical-guide-object-detection-yolo-framewor-python/)
- [yolo-explained](https://medium.com/analytics-vidhya/yolo-explained-5b6f4564f31)

### YOLOv2

- [yolo2-walkthrough-with-examples](https://towardsdatascience.com/yolo2-walkthrough-with-examples-e40452ca265f)

model build - data loading - train - predict

- [YOLO_Explained](https://github.com/zzxvictor/YOLO_Explained)
- [YAD2K](https://github.com/allanzelener/YAD2K)

### YOLOv3

- [understanding-yolo-and-implementing-yolov3-for-object-detection](https://medium.com/analytics-vidhya/understanding-yolo-and-implementing-yolov3-for-object-detection-5f1f748cc63a)
- [all-you-need-to-know-about-yolo-v3-you-only-look-once](https://dev.to/afrozchakure/all-you-need-to-know-about-yolo-v3-you-only-look-once-e4m)
- [a-comprehensive-guide-to-yolov3](https://atharvamusale.medium.com/a-comprehensive-guide-to-yolov3-74029810ca81)
- [yolo-you-look-only-once](https://medium.com/analytics-vidhya/yolo-you-look-only-once-9af63cb143b7)
- [yolo-v3-explained](https://towardsdatascience.com/yolo-v3-explained-ff5b850390f)
- [yolo-v3-object-detection](https://towardsdatascience.com/yolo-v3-object-detection-53fb7d3bfe6b)
- [bounding-box-object-detectors-understanding-yolo](https://christopher5106.github.io/object/detectors/2017/08/10/bounding-box-object-detectors-understanding-yolo.html)
- [How YOLO V3 works?](https://www.youtube.com/watch?v=MKF1NHGgFfk)
- [YOLOv3 from Scratch](https://www.youtube.com/watch?v=Grir6TZbc1M)
- [Introduction into YOLO v3](https://www.youtube.com/watch?v=vRqSO6RsptU)

#### Implementation in pyTorch

model build - data loading - train - predict

- [implement-yolo-v3-object-detector-pytorch](https://www.kdnuggets.com/2018/05/implement-yolo-v3-object-detector-pytorch-part-1.html)
- [yolov3-implementation-with-training-setup](https://sannaperzon.medium.com/yolov3-implementation-with-training-setup-from-scratch-30ecb9751cb0)
- [pytorch/object_detection/YOLOv3](https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/object_detection/YOLOv3)

#### Implementation in Tensorflow

pre trained model - predict

- [yolov3-tensorflow](https://machinelearningspace.com/yolov3-tensorflow-2-part-1/)
- [implement_yolo](https://sheng-fang.github.io/2020-04-29-implement_yolo/) 

model build - data loading - train - predict

- [yolo3-tf2](https://github.com/Qucy/yolo3-tf2)
- [yolo3-tf2](https://pythonrepo.com/repo/Qucy-yolo3-tf2) (good details for concepts - model/loss/bbox)
- [keras-yolo3](https://github.com/experiencor/keras-yolo3) (good class organisation, generator, and augmentation)
- [qqwweee-keras-yolo3](https://github.com/qqwweee/keras-yolo3)
- [TensorFlow-2.x-YOLOv3](https://github.com/pythonlessons/TensorFlow-2.x-YOLOv3)
- [YOLOv3-object-detection-custom-training](https://github.com/pythonlessons/YOLOv3-object-detection-tutorial/tree/master/YOLOv3-custom-training)
- [YOLOv3-TF2-introduction](https://pylessons.com/YOLOv3-TF2-introduction/)
- [yolov3-tf2](https://github.com/zzh8829/yolov3-tf2)
- [tensorflow-yolo-v3](https://github.com/mystic123/tensorflow-yolo-v3)
- [keras-YOLOv3-model](https://github.com/david8862/keras-YOLOv3-model-set)

custom training

- [training-yolov3-with-a-custom-dataset](https://blog.roboflow.com/training-a-yolov3-object-detection-model-with-a-custom-dataset/)

transfer learning with yolo

- [types-of-deep-transfer-learning](https://hub.packtpub.com/5-types-of-deep-transfer-learning/)
- [transfer-learning-with-yolo-v3](https://medium.com/@cunhafh/transfer-learning-with-yolo-v3-darknet-and-google-colab-7f9a6f9c2afc)
- [yolov3-custom-object-detection-with-transfer-learning](https://tiwarinitin1999.medium.com/yolov3-custom-object-detection-with-transfer-learning-47186c8f166d)

load yolo weights

- [darknet-yolo](https://pjreddie.com/darknet/yolo/)
- [tensorflow-yolo-v3](https://github.com/mystic123/tensorflow-yolo-v3)
- [yolov3-tensorflow](https://machinelearningspace.com/yolov3-tensorflow-2-part-3/)
- [YOLOv3-custom-training](https://pylessons.com/YOLOv3-custom-training)
- [implementing-yolo-v3-in-tensorflow](https://itnext.io/implementing-yolo-v3-in-tensorflow-tf-slim-c3c55ff59dbe)
- [how-to-perform-object-detection-with-yolov3](https://machinelearningmastery.com/how-to-perform-object-detection-with-yolov3-in-keras/)
- [yolo3_one_file_to_detect_them_all](https://github.com/experiencor/keras-yolo3/blob/master/yolo3_one_file_to_detect_them_all.py)

### YOLOv4

load yolo weights

- [how-to-convert-your-yolov4-weights](https://dsbyprateekg.blogspot.com/2020/06/how-to-convert-your-yolov4-weights-to.html)
- [how-to-convert-yolov4-from-weights-to-h5](https://medium.com/@ravindrareddysiddam/how-to-convert-yolov4-from-weights-to-h5-format-b50b244b3298)
- [yolov4_to_tf2](https://github.com/dsbyprateekg/YOLOv4/blob/master/yolov4_to_tf2.ipynb)

### YOLOv5

- [how-to-use-yolo-v5-object-detection](https://www.analyticsvidhya.com/blog/2021/12/how-to-use-yolo-v5-object-detection-algorithm-for-custom-object-detection-an-example-use-case/)

### Model Subclassing

- [model-sub-classing](https://towardsdatascience.com/model-sub-classing-and-custom-training-loop-from-scratch-in-tensorflow-2-cc1d4f10fb4e)

### Custom Loss

- [keras-custom-loss-functions](https://cnvrg.io/keras-custom-loss-functions/)
- [kaggle-discussion](https://www.kaggle.com/c/pku-autonomous-driving/discussion/113928)

### Dataset / Format

- [coco-data-format-for-object-detection](https://towardsdatascience.com/coco-data-format-for-object-detection-a4c5eaf518c5)
- [cocodataset](https://cocodataset.org/#home)
- [pascal VOC](http://host.robots.ox.ac.uk/pascal/VOC/index.html)
- [2D Bounding box annotation formats for Object detection](https://medium.com/@sonalrpatel/various-object-detection-input-data-formats-5ea04667b778)

## Todo

- Update the training and benchmark / result sections