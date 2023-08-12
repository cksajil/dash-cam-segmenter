import os
import imageio
from tqdm import tqdm
import matplotlib.pyplot as plt
from src.general import get_list_of_seg_images


def plot_n_save(k, original, segmented):
    width = 9
    height = 3

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
    fig.savefig("./processed_frames/frame_{}.png".format(k), dpi=200)
    plt.close()


def gif_creator(filenames):
    print("Generaing segmented GIF video")
    with imageio.get_writer(
        os.path.join("./video", "segmented_movie.gif"), mode="I"
    ) as writer:
        for filename in tqdm(filenames):
            image = imageio.imread(os.path.join("processed_frames", filename))
            writer.append_data(image)
