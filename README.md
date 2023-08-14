# Real-time segmentation of dash cam videos
Pretrained computer vision models such as UNET and CANET are used to segment dash cam video specifically on Indian roads.

<img src="https://github.com/cksajil/dash-cam-segmenter/blob/cff58dd62d0e43fd500583ef806f16e5f02b7d5f/images/output_preview.png" width="1300" />

### [To see demo video, click here](https://youtu.be/U3R7oS2YvK4)

![Demo](https://github.com/cksajil/dash-cam-segmenter/blob/b7eb14a4f3137bcf4dc6d81407ad203d1976fd63/images/segmented_movie.gif)


**Python Version**
```
Python 3.9.1
```

### Setting up virtual environment

```console
# Installing Virtual Environment
python -m pip install --user virtualenv

# Creating New Virtual Environment
python -m venv envname

# Activating Virtual Environment
source envname/bin/activate

# Upgrade PIP
python -m pip install --upgrade pip

# Installing Packages
python -m pip install -r requirements.txt
```

### How to run

```console
python3 main.py --youtube "YouTube video URL"

python3 main.py --file <path_to_video_in_hard_disc>

# Example
python3 main.py --youtube "https://youtu.be/INcqJsGfBZU"

python3 main.py --file ./video/demo_video.mp4
```
