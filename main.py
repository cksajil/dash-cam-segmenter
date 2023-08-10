import os
import cv2
import ssl
import argparse
import numpy as np
import tensorflow as tf
from pytube import YouTube
from src.unet import load_unet
from src.general import load_config

ssl._create_default_https_context = ssl._create_stdlib_context

config = load_config("my_config.yaml")

color_mapping = [
    [255, 0, 0],  # Class 0 - Red
    [0, 255, 0],  # Class 1 - Green
    [0, 0, 255],  # Class 2 - Blue
    [255, 255, 0],  # Class 3 - Yellow
    [0, 255, 255],  # Class 4 - Cyan
    [255, 0, 255],  # Class 5 - Magenta
    [128, 0, 0],  # Class 6 - Maroon
    [0, 128, 0],  # Class 7 - Green (Dark)
    [0, 0, 128],  # Class 8 - Navy
    [128, 128, 0],  # Class 9 - Olive
    [0, 128, 128],  # Class 10 - Teal
    [128, 0, 128],  # Class 11 - Purple
    [255, 165, 0],  # Class 12 - Orange
    [210, 180, 140],  # Class 13 - Tan
    [70, 130, 180],  # Class 14 - Steel Blue
    [139, 69, 19],  # Class 15 - Saddle Brown
    [0, 255, 127],  # Class 16 - Spring Green
    [255, 255, 255],  # Class 17 - White
    [0, 0, 0],  # Class 18 - Black
    [128, 128, 128],  # Class 19 - Gray
    [255, 255, 128],  # Class 20 - Light Yellow
]


def on_progress(stream, chunk, bytes_remaining):
    total_size = stream.filesize
    bytes_downloaded = total_size - bytes_remaining
    percentage = (bytes_downloaded / total_size) * 100
    print(f"Downloading: {percentage:.2f}%")


def colorize_seg_image(original, segmented):
    segmented_color = np.zeros_like(original)
    for class_idx in range(segmented.shape[-1]):
        mask = segmented[:, :, class_idx] == 1
        segmented_color[mask] = color_mapping[class_idx]
    return segmented_color


def predict_with_model(model, image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pred_image = model.predict(image[np.newaxis, :, :, :])
    pred_image = pred_image[0, :, :, :]
    return pred_image


def process_video(video_path):
    model = load_unet()
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame_raw = cap.read()
        if not ret:
            break
        frame_raw = cv2.resize(
            frame_raw, (config["image_width"], config["image_height"])
        )
        frame_out = predict_with_model(model, frame_raw)
        frame_out = cv2.normalize(
            frame_out,
            None,
            alpha=0,
            beta=255,
            norm_type=cv2.NORM_MINMAX,
            dtype=cv2.CV_32F,
        ).astype(np.uint8)
        frame_out = colorize_seg_image(frame_raw, frame_out)
        stacked_frame = np.hstack((frame_raw, frame_out))
        cv2.imshow("Original Vs Segmented Video", stacked_frame)
        if cv2.waitKey(30) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def download_youtube_video(youtube_url, output_dir):
    yt = YouTube(youtube_url, on_progress_callback=on_progress)
    stream = yt.streams.filter(res="360p", progressive=True).first()
    video_path = os.path.join(output_dir, "demo_video.mp4")
    stream.download(output_path=output_dir, filename="demo_video.mp4")
    return video_path


def main():
    parser = argparse.ArgumentParser(
        description="Process YouTube videos or video files."
    )
    parser.add_argument("--youtube", type=str, help="YouTube video URL")
    parser.add_argument("--file", type=str, help="Path to the video file")
    args = parser.parse_args()

    if args.youtube:
        video_path = download_youtube_video(args.youtube, "downloaded_videos")
    elif args.file:
        video_path = args.file
    else:
        print("Please provide either a YouTube URL or a video file path.")
        return

    process_video(video_path)


if __name__ == "__main__":
    main()
