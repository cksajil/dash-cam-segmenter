# Real-time segmentation of dash cam videos (Work in progress)
Pretrained computer vision models such as UNET and CANET are used to segment dash cam video specifically on Indian roads.

<img src="https://i.ibb.co/hFkGkdd/output-preview.png" width="1300" />

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
