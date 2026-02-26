import argparse
import logging
import os
import ssl
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import requests
import tensorflow as tf
from pytubefix import YouTube
from requests import Response
from tqdm import tqdm

from src.general import create_folder, get_list_of_seg_images, load_config
from src.visulaizer import gif_creator, plot_n_save
from src.unet import build_unet
from src.deeplabv3 import build_deeplabv3plus

ssl._create_default_https_context = ssl._create_stdlib_context

LOGGER = logging.getLogger(__name__)
CONFIG = load_config("my_config.yaml")


def configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )


def initialize_directories() -> None:
    """Create required directories if they do not exist."""
    for folder_name in ("video", "processed_frames", "models"):
        create_folder(folder_name)


def _request_json(url: str, timeout: int = 30) -> dict:
    response: Response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return response.json()


def load_model() -> Path:
    model_name = CONFIG["deployed_model"]
    model_path = Path(CONFIG["model_loc"], model_name + ".keras")
    if model_name == "unet":
        model = build_unet()
    elif model_name == "deeplabv3":
        model = build_deeplabv3plus()
    model.load_weights(model_path)
    return model


def download_model() -> Path:
    """Download the pretrained model if not already available."""
    model_name = CONFIG["deployed_model"]
    model_path = Path(CONFIG["model_loc"]) / f"{model_name}.keras"

    if model_path.exists():
        LOGGER.info("Using existing model at %s", model_path)
        return model_path

    LOGGER.info("Downloading pretrained model '%s'", model_name)

    if CONFIG["deployed_model"] == "unet":
        doi = CONFIG["unet_model_doi"]
    elif CONFIG["deployed_model"] == "deeplabv3":
        doi = CONFIG["deeplabv3_model_doi"]

    download_url = f"https://zenodo.org/api/records/{doi}"
    record = _request_json(download_url)
    model_files = [
        file for file in record.get("files", []) if file["key"].endswith(".keras")
    ]

    if not model_files:
        raise RuntimeError("No .keras model files found in Zenodo record")

    file_url = model_files[0]["links"]["self"]
    response = requests.get(file_url, stream=True, timeout=60)
    response.raise_for_status()
    total = int(response.headers.get("content-length", 0))

    with model_path.open("wb") as handle, tqdm(
        total=total,
        unit="B",
        unit_scale=True,
        desc="model-download",
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                handle.write(chunk)
                pbar.update(len(chunk))

    LOGGER.info("Model downloaded to %s", model_path)
    return model_path


def on_progress(stream, _chunk, bytes_remaining):
    """Track YouTube download progress."""
    total_size = stream.filesize
    bytes_downloaded = total_size - bytes_remaining
    percentage = (bytes_downloaded / total_size) * 100
    LOGGER.info("YouTube download progress: %.2f%%", percentage)


def predict_with_model(model, image: np.ndarray) -> tf.Tensor:
    """Predict segmentation mask from a single normalized BGR image."""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (CONFIG["image_width"], CONFIG["image_height"]))
    pred_image = model.predict(image[np.newaxis, :, :, :], verbose=0)
    pred_image = pred_image[0, :, :, :]
    return tf.argmax(pred_image, axis=-1)


def process_video(video_path: str) -> None:
    """Process an input video and generate segmented frame images."""
    LOGGER.info("Segmenting video frames from %s", video_path)
    model = load_model()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Unable to open video file: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx = 0

    for old_frame in Path(CONFIG["frames_loc"]).glob("frame_*.png"):
        old_frame.unlink(missing_ok=True)

    with tqdm(total=total_frames, desc="segment-frames") as pbar:
        while cap.isOpened():
            ret, frame_raw = cap.read()
            if not ret:
                break

            frame_raw = cv2.resize(
                frame_raw, (CONFIG["image_width"], CONFIG["image_height"])
            )
            frame_raw = frame_raw.astype(np.float32) / 255.0
            frame_out = predict_with_model(model, frame_raw)
            plot_n_save(frame_idx, frame_raw, frame_out)

            frame_idx += 1
            pbar.update(1)

    LOGGER.info("Processed %s frames", frame_idx)
    cap.release()
    cv2.destroyAllWindows()


def download_youtube_video(youtube_url: str, output_dir: str) -> str:
    """Download YouTube video at 360p progressive stream."""
    LOGGER.info("Downloading YouTube video: %s", youtube_url)
    yt = YouTube(youtube_url, on_progress_callback=on_progress)
    stream = yt.streams.filter(res="360p", progressive=True).first()
    if stream is None:
        raise RuntimeError(
            "No compatible 360p progressive stream available for this URL"
        )

    output_path = os.path.join(output_dir, CONFIG["out_video_name"])
    stream.download(output_path=output_dir, filename=CONFIG["out_video_name"])
    LOGGER.info("Saved source video to %s", output_path)
    return output_path


def generate_video(images: Iterable[str], fps: float = 30.0) -> None:
    """Generate comparison video from segmented frame images."""
    images = list(images)
    if not images:
        raise ValueError("No frame images were found to create output video")

    LOGGER.info("Generating comparison video")
    image_folder = CONFIG["frames_loc"]
    first_frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, _ = first_frame.shape
    video = cv2.VideoWriter(
        filename=os.path.join(CONFIG["video_loc"], CONFIG["seg_video_name"]),
        fourcc=cv2.VideoWriter_fourcc("m", "p", "4", "v"),
        frameSize=(width, height),
        fps=fps,
    )

    for image_name in tqdm(images, desc="write-video"):
        video.write(cv2.imread(os.path.join(image_folder, image_name)))

    cv2.destroyAllWindows()
    video.release()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Segment dash-cam videos using pretrained UNET"
    )
    parser.add_argument("--youtube", type=str, help="YouTube video URL")
    parser.add_argument("--file", type=str, help="Path to a local video file")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    if bool(args.youtube) == bool(args.file):
        parser.error("Provide exactly one of --youtube or --file.")
    return args


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)
    initialize_directories()

    video_path = (
        download_youtube_video(args.youtube, CONFIG["video_loc"])
        if args.youtube
        else args.file
    )

    download_model()
    process_video(video_path)
    images = get_list_of_seg_images()
    if not images:
        raise RuntimeError("No segmented frames generated; cannot create output GIF")
    # Create a short preview of segmented output as gif
    gif_creator(images[:200])


if __name__ == "__main__":
    main()
