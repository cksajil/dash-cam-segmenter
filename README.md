# Real-time segmentation of dash cam videos (Work in progress)
Pretrained computer vision models such as UNET and CANET are used to segment dash cam video specifically on Indian roads.

![output_preview](https://i.ibb.co/hFkGkdd/output-preview.png)

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

# Example
python3 main.py --youtube "https://www.youtube.com/watch?v=BGqs0cXK0e8&t=13s&pp=ygUTdGhvZHVwdXpoYSBkYXNoIGNhbQ%3D%3D"
```

### Testing
```console
python -m pytest --verbose
```
