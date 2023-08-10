import os

os.environ["SM_FRAMEWORK"] = "tf.keras"
from segmentation_models import Unet


def load_unet():
    NCLASSES = 21
    unet_model = Unet(
        "resnet50",
        encoder_weights="imagenet",
        classes=NCLASSES,
        activation="softmax",
        input_shape=(256, 256, 3),
        encoder_freeze=True,
        decoder_block_type="upsampling",
    )
    unet_model.load_weights("./models/UNET.h5")
    return unet_model
