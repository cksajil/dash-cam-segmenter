import yaml
import os


def load_config(config_name):
    """
    A function to load and return config file in YAML format
    """
    CONFIG_PATH = "./config/"
    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        config = yaml.safe_load(file)

    return config


def create_folder(directory):
    """Function to create a folder in a location if it does not exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)


def get_list_of_seg_images():
    """
    Returns the list of segmented images in the ordinal order
    """
    print("Getting list of segmented frames")
    image_folder = "processed_frames"
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    frames = [int(image.split(".")[0].split("_")[1]) for image in images]
    frame_start = min(frames)
    frame_end = max(frames)
    images = ["frame_" + str(i) + ".png" for i in range(frame_start, frame_end)]
    return images
