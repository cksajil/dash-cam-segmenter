# Real-time segmentation of dash-cam videos

This project segments dash-cam video frames using a pretrained UNET model and renders the output as a GIF for quick review. It supports both local video files and YouTube URLs as input.

### Demo
- YouTube demo: https://youtu.be/U3R7oS2YvK4

---

## Production-grade improvements included
- Stronger CLI validation (`--youtube` XOR `--file` + helpful errors).
- Structured logging with optional `--verbose` mode.
- Safer model download flow with HTTP status validation and progress tracking.
- Deterministic frame ordering and cleanup of stale frames between runs.
- Improved path handling and config loading robustness.
- Typed function signatures and better runtime errors.

---

## Requirements
- Python 3.9+
- `pip install -r requirements.txt`

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Usage

```bash
# Segment a YouTube video
python main.py --youtube "https://youtu.be/h1u5OzTdbpc"

# Segment a local file
python main.py --file ./video/demo_video.mp4

# Optional: verbose logs
python main.py --file ./video/demo_video.mp4 --verbose
```

## Output
- Segmented frames are written to `processed_frames/`.
- Final GIF is written to `video/`.

