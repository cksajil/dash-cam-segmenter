import os
import cv2
import ssl
import argparse
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from pytube import YouTube
from src.unet import load_unet
from src.visulaizer import plot_n_save
from src.general import load_config, create_folder

ssl._create_default_https_context = ssl._create_stdlib_context

config = load_config("my_config.yaml")

def initialize_directories():
    print("Creating folders")
    folder_names = ["video", "processed_frames", "models"]
    for fol_name in folder_names:
        create_folder(fol_name)


def on_progress(stream, chunk, bytes_remaining):
    total_size = stream.filesize
    bytes_downloaded = total_size - bytes_remaining
    percentage = (bytes_downloaded / total_size) * 100
    print(f"Downloading: {percentage:.2f}%")


def predict_with_model(model, image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pred_image = model.predict(image[np.newaxis, :, :, :], verbose=0)
    pred_image = pred_image[0, :, :, :]
    pred_image = tf.argmax(pred_image, axis=-1)
    return pred_image


def process_video(video_path):
    print("Segmenting Frames")
    model = load_unet()
    cap = cv2.VideoCapture(video_path)
    N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
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
        pbar.update(k/N)
        k+=1
    pbar.close()

    cap.release()
    cv2.destroyAllWindows()


def download_youtube_video(youtube_url, output_dir):
    print("Downloads video if it does not exist locally")
    yt = YouTube(youtube_url, on_progress_callback=on_progress)
    stream = yt.streams.filter(res="360p", progressive=True).first()
    video_path = os.path.join(output_dir, "demo_video.mp4")
    stream.download(output_path=output_dir, filename="demo_video.mp4")
    return video_path


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

    process_video(video_path)


if __name__ == "__main__":
    main()
