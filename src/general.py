import yaml
import numpy as np
from typing import Any
from pathlib import Path


CONFIG_PATH = Path(__file__).resolve().parent.parent / "config"


def load_config(config_name: str) -> dict[str, Any]:
    """Load a config file from the repository's config directory."""
    config_path = CONFIG_PATH / config_name
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def create_folder(directory: str) -> None:
    """Create a folder if it does not already exist."""
    Path(directory).mkdir(parents=True, exist_ok=True)


def get_list_of_seg_images() -> list[str]:
    """Return segmented frame images in numeric order."""
    config = load_config("my_config.yaml")
    image_folder = Path(config["frames_loc"])
    images = []

    for img in image_folder.glob("frame_*.png"):
        try:
            index = int(img.stem.split("_")[1])
        except (IndexError, ValueError):
            continue
        images.append((index, img.name))

    images.sort(key=lambda item: item[0])
    return [name for _, name in images]


def normalize_image(img: np.ndarray) -> np.ndarray:
    config = load_config("my_config.yaml")
    mean = np.array(config["mean"], dtype=np.float32)
    std = np.array(config["std"], dtype=np.float32)
    return (img - mean) / std
