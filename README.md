# Real-time segmentation of dash-cam videos

This project segments dash-cam video frames using a pretrained UNET/DeepLabV3Plus models and renders the output as a GIF for quick review. It supports both local video files and YouTube URLs as input.

### Demo
![](https://github.com/cksajil/dash-cam-segmenter/blob/main/images/segmented_movie.gif?raw=true)


## Dataset
The dataset used for this project can be found at [Kaggle](https://www.kaggle.com/datasets/sajilck/road-segmentation-indian)


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

## Model Performance

[]('https://raw.githubusercontent.com/cksajil/dash-cam-segmenter/refs/heads/main/images/iou_curve.png')

## Output
- Segmented frames are written to `processed_frames/`.
- Final GIF is written to `images/`.

