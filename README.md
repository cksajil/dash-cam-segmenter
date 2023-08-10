# Real-time segmentation of dash cam videos (Work in progress)
Pretrained computer vision models such as UNET and CANET are used to segment dash cam video specifically on Indian roads.

### Overview

### Dataset

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
python main.py --youtube "YouTube video URL"
```

### Testing
```console
python -m pytest --verbose
```

### Results
Original vs segmented video
