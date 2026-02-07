# Real-time segmentation of dash cam videos
Pretrained computer vision models such as UNET and CANET are used to segment dash cam video specifically on Indian roads.


<img src="https://media1.giphy.com/media/v1.Y2lkPTc5MGI3NjExY3plb2d2MWk3bnlheXY5eWRsNTBibWNtcmR1YTJ5ajhkc3hqc3R1ZCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/5e8MseZLO4dS6BK6N7/giphy.gif" width="800">

### [To see demo video, click here](https://youtu.be/U3R7oS2YvK4)


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
python3 main.py --youtube "https://youtu.be/h1u5OzTdbpc"

python3 main.py --file ./video/demo_video.mp4
```


### Output
An output video like the following comparing original and segmented frames side by side will get generated in the `video` folder.

<img src="https://media1.giphy.com/media/v1.Y2lkPTc5MGI3NjExY3plb2d2MWk3bnlheXY5eWRsNTBibWNtcmR1YTJ5ajhkc3hqc3R1ZCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/5e8MseZLO4dS6BK6N7/giphy.gif" width="800">