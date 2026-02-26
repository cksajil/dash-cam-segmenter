import os
import imageio
import matplotlib.pyplot as plt
from tqdm import tqdm
from src.general import load_config

config = load_config("my_config.yaml")


def plot_n_save(k: int, original, segmented) -> None:
    """Display and save original and segmented images side by side."""
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

    fig.set_size_inches(config["out_fig_width"], config["out_fig_height"])
    fig.savefig(os.path.join(config["frames_loc"], f"frame_{k}.png"), dpi=200)
    plt.close(fig)


def gif_creator(filenames: list[str]) -> None:
    """Create segmented GIF video from rendered frame images."""
    output_path = os.path.join("images", config["out_gif_name"])
    with imageio.get_writer(output_path, mode="I") as writer:
        for filename in tqdm(filenames, desc="write-gif"):
            image = imageio.imread(os.path.join(config["frames_loc"], filename))
            writer.append_data(image)
