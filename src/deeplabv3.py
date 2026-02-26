import tensorflow as tf
from src.general import load_config
from tensorflow.keras import layers, models

config = load_config("my_config.yaml")


def ASPP(x, filters=256):

    y1 = layers.Conv2D(filters, 1, padding="same", use_bias=False)(x)
    y1 = layers.BatchNormalization()(y1)
    y1 = layers.ReLU()(y1)

    y2 = layers.Conv2D(filters, 3, dilation_rate=6, padding="same", use_bias=False)(x)
    y2 = layers.BatchNormalization()(y2)
    y2 = layers.ReLU()(y2)

    y3 = layers.Conv2D(filters, 3, dilation_rate=12, padding="same", use_bias=False)(x)
    y3 = layers.BatchNormalization()(y3)
    y3 = layers.ReLU()(y3)

    y4 = layers.Conv2D(filters, 3, dilation_rate=18, padding="same", use_bias=False)(x)
    y4 = layers.BatchNormalization()(y4)
    y4 = layers.ReLU()(y4)

    y = layers.Concatenate()([y1, y2, y3, y4])
    y = layers.Conv2D(filters, 1, padding="same", use_bias=False)(y)
    y = layers.BatchNormalization()(y)
    y = layers.ReLU()(y)

    return y


def build_deeplabv3plus():

    IMG_HEIGHT = config["image_height"]
    IMG_WIDTH = config["image_width"]
    NUM_CLASSES = config["num_classes"]

    base_model = tf.keras.applications.ResNet50(
        weights="imagenet", include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)
    )

    # Low-level features
    low_level = base_model.get_layer("conv2_block3_out").output
    high_level = base_model.get_layer("conv4_block6_out").output

    # ASPP
    aspp = ASPP(high_level)

    # Upsample ASPP output
    aspp = layers.UpSampling2D(size=(4, 4), interpolation="bilinear")(aspp)

    # Reduce low-level features
    low_level = layers.Conv2D(48, 1, padding="same", use_bias=False)(low_level)
    low_level = layers.BatchNormalization()(low_level)
    low_level = layers.ReLU()(low_level)

    # Concatenate
    x = layers.Concatenate()([aspp, low_level])

    x = layers.Conv2D(256, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(256, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Final upsampling
    x = layers.UpSampling2D(size=(4, 4), interpolation="bilinear")(x)

    outputs = layers.Conv2D(NUM_CLASSES, 1, activation="softmax")(x)

    model = models.Model(inputs=base_model.input, outputs=outputs)

    return model
