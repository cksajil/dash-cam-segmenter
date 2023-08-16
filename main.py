import os
import cv2
import ssl
import argparse
import requests
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from pytube import YouTube
from src.unet import load_unet
from src.visulaizer import plot_n_save, gif_creator
from src.general import load_config, create_folder, get_list_of_seg_images

ssl._create_default_https_context = ssl._create_stdlib_context

config = load_config("my_config.yaml")


def initialize_directories():
    """
    Create required directories if it does not exist
    """
    print("Creating folders if does not exist")
    folder_names = ["video", "processed_frames", "models"]
    for fol_name in folder_names:
        create_folder(fol_name)


def download_model():
    """
    Download pretrained model if not already downloaded
    """
    if not os.path.exists(os.path.join(config["model_loc"], config["unet_model_name"])):
        print("Downloading pretrained UNET model if not exists")
        doi = config["unet_model_doi"]
        response = requests.get(f"https://zenodo.org/api/records/{doi}")
        data = response.json()
        files = data["files"]
        model_files = [file for file in files if file["key"].endswith(".h5")]

        if len(model_files) == 0:
            print("No model files found.")
        else:
            model_file = model_files[0]
            file_url = model_file["links"]["self"]
            model_filename = model_file["key"]

            response = requests.get(file_url, stream=True)
            segments = response.iter_content()
            with open(
                os.path.join(config["model_loc"], config["unet_model_name"]), "wb"
            ) as file:
                for chunk in tqdm(segments):
                    file.write(chunk)

            print(f"Model downloaded as {model_filename}")


def on_progress(stream, chunk, bytes_remaining):
    """
    Function to track progress of download
    """
    total_size = stream.filesize
    bytes_downloaded = total_size - bytes_remaining
    percentage = (bytes_downloaded / total_size) * 100
    print(f"Downloading: {percentage:.2f}%")


def predict_with_model(model, image):
    """
    Function to predict segmented image
    from raw image using UNET model
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pred_image = model.predict(image[np.newaxis, :, :, :], verbose=0)
    pred_image = pred_image[0, :, :, :]
    pred_image = tf.argmax(pred_image, axis=-1)
    return pred_image


def process_video(video_path):
    """
    Function to process raw video and create segmented video output
    """
    print("Segmenting Frames")
    model = load_unet()
    cap = cv2.VideoCapture(video_path)
    N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    k = 0
    pbar = tqdm(total=N)
    while cap.isOpened():
        ret, frame_raw = cap.read()
        if not ret:
            break
        frame_raw = cv2.resize(
            frame_raw, (config["image_width"], config["image_height"])
        )
        frame_out = predict_with_model(model, frame_raw)
        plot_n_save(k, frame_raw, frame_out)
        pbar.update(k / N)
        k += 1
    pbar.close()

    cap.release()
    cv2.destroyAllWindows()


def download_youtube_video(youtube_url, output_dir):
    print("Downloads video if it does not exist locally")
    yt = YouTube(youtube_url, on_progress_callback=on_progress)
    stream = yt.streams.filter(res="360p", progressive=True).first()
    video_path = os.path.join(output_dir, config["out_video_name"])
    stream.download(output_path=output_dir, filename=config["out_video_name"])
    return video_path


def generate_video(images, FPS=30.0):
    print("Generates original and segmented comparison video")
    video_name = "segmented_video.mp4"
    image_folder = "processed_frames"
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape
    video = cv2.VideoWriter(
        filename=os.path.join("./video", video_name),
        fourcc=cv2.VideoWriter_fourcc("m", "p", "4", "v"),
        frameSize=(width, height),
        fps=FPS,
    )

    for image in tqdm(images):
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()


def main():
    initialize_directories()
    parser = argparse.ArgumentParser(
        description="Process YouTube videos or video files."
    )
    parser.add_argument("--youtube", type=str, help="YouTube video URL")
    parser.add_argument("--file", type=str, help="Path to the video file")
    args = parser.parse_args()

    if args.youtube:
        video_path = download_youtube_video(args.youtube, "video")
    elif args.file:
        video_path = args.file
    else:
        print("Please provide either a YouTube URL or a video file path.")
        return

    download_model()
    process_video(video_path)
    images = get_list_of_seg_images()
    # generate_video(images)
    gif_creator(images)


if __name__ == "__main__":
    main()
