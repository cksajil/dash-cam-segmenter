import os
import imageio
from tqdm import tqdm
import matplotlib.pyplot as plt
from src.general import load_config


config = load_config("my_config.yaml")


def plot_n_save(k, original, segmented):
    """
    A custom plotting function to display and save
    original and segmented images side by side
    """
    width = config["out_fig_width"]
    height = config["out_fig_height"]

    fig, ax = plt.subplots(1, 2)
    fig.tight_layout()
    ax[0].imshow(original)
    ax[0].set_xlabel("Original video")
    ax[0].set_xticks([])
    ax[0].set_yticks([])

    ax[1].imshow(segmented)
    ax[1].set_xlabel("Segmented video")
    ax[1].set_xticks([])
    ax[1].set_yticks([])

    fig.set_size_inches(width, height)

    fig.savefig(os.path.join(config["frames_loc"], "frame_{}.png".format(k)), dpi=200)
    plt.close()


def gif_creator(filenames):
    """
    A function to create segmented video
    from segmented images passed as parameters
    """
    print("Generaing segmented GIF video")
    with imageio.get_writer(
        os.path.join(config["video_loc"], config["out_gif_name"]), mode="I"
    ) as writer:
        for filename in tqdm(filenames):
            image = imageio.imread(os.path.join(config["frames_loc"], filename))
            writer.append_data(image)
