from model.model import YOLOv3


if __name__ == "__main__":
    num_classes = 80
    IMAGE_SIZE = 416
    image_shape = (IMAGE_SIZE, IMAGE_SIZE, 3)

    model = YOLOv3().model()

    # input_tensor = Input(image_shape)
    # output_tensor = model(input_tensor)

    print(model.summary())