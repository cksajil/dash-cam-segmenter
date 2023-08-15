import os
from src.general import load_config

os.environ["SM_FRAMEWORK"] = "tf.keras"
from segmentation_models import Unet

config = load_config("my_config.yaml")


def load_unet():
    """
    Function to load pretrained UNET model and return it
    """
    NCLASSES = 21

    unet_model = Unet(
        "resnet50",
        encoder_weights="imagenet",
        classes=NCLASSES,
        activation="softmax",
        input_shape=(config["image_height"], config["image_width"], 3),
        encoder_freeze=True,
        decoder_block_type="upsampling",
    )

    unet_model.load_weights("./models/UNET.h5")
    return unet_model
