import os
from src.general import load_config

os.environ["SM_FRAMEWORK"] = "tf.keras"
from segmentation_models import Unet

config = load_config("my_config.yaml")


def load_unet():
    """
    Function to load pretrained UNET model and return it
    """
    NCLASSES = config["num_classes"]

    unet_model = Unet(
        "resnet50",
        encoder_weights="imagenet",
        classes=NCLASSES,
        activation="softmax",
        input_shape=(config["image_height"], config["image_width"], 3),
        encoder_freeze=True,
        decoder_block_type="upsampling",
    )

    unet_model.load_weights(
        os.path.join(config["model_loc"], config["unet_model_name"])
    )
    return unet_model
