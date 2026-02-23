import os
import pandas as pd
import json
from tqdm import tqdm
from PIL import Image, ImageDraw
import numpy as np
import re


images_dir = os.path.join("data", "images")
mask_dir = os.path.join("data", "mask")
val_ext = (".json", ".jpg")


def get_file_list(file_dir):
    file_paths = []
    for root, _, files in os.walk(file_dir):
        for file in files:
            if file.startswith(".") or not file.lower().endswith(val_ext):
                continue
            filepath = os.path.join(root, file)
            file_paths.append(filepath)
    return file_paths


index_df = pd.DataFrame()
index_df["images"] = get_file_list(images_dir)
pfx = "_gtFine_polygons.json"
poly_list = []
for img_path in index_df["images"]:
    item = img_path.split("/")
    poly_list.append("/".join([item[0], "mask", item[2], item[3].split("_")[0] + pfx]))

index_df["masks"] = poly_list
index_df.to_csv(os.path.join("data", "index.csv"))

for idx, row in index_df.iterrows():
    if os.path.exists(row["images"]) and os.path.exists(row["masks"]):
        continue
    else:
        print("Some issue")


def get_poly(file):
    f = open(file, "r")
    content = json.loads(f.read())
    w = content["imgWidth"]
    h = content["imgHeight"]
    objects = content["objects"]
    label = []
    vertexlist = []
    for obj in objects:
        label.append(obj["label"])
        vertices = []
        for vertex in obj["polygon"]:
            vertices.append((vertex[0], vertex[1]))
        vertexlist.append(vertices)
    return w, h, label, vertexlist


label_clr = {
    "road": 10,
    "parking": 20,
    "drivable fallback": 20,
    "sidewalk": 30,
    "non-drivable fallback": 40,
    "rail track": 40,
    "person": 50,
    "animal": 50,
    "rider": 60,
    "motorcycle": 70,
    "bicycle": 70,
    "autorickshaw": 80,
    "car": 80,
    "truck": 90,
    "bus": 90,
    "vehicle fallback": 90,
    "trailer": 90,
    "caravan": 90,
    "curb": 100,
    "wall": 100,
    "fence": 110,
    "guard rail": 110,
    "billboard": 120,
    "traffic sign": 120,
    "traffic light": 120,
    "pole": 130,
    "polegroup": 130,
    "obs-str-bar-fallback": 130,
    "building": 140,
    "bridge": 140,
    "tunnel": 140,
    "vegetation": 150,
    "sky": 160,
    "fallback background": 160,
    "unlabeled": 0,
    "out of roi": 0,
    "ego vehicle": 170,
    "ground": 180,
    "rectification border": 190,
    "train": 200,
}


def compute_masks(index_df):
    os.makedirs(os.path.join("data", "maskimg"), exist_ok=True)
    for file in tqdm(index_df["masks"]):
        sub_fol_name = file.split("/")[2]
        os.makedirs(os.path.join("data", "maskimg", sub_fol_name), exist_ok=True)

        img_name = (
            re.search(r"(data)\/(mask)\/(\d+)\/(.*)(.json)", file).group(4) + ".png"
        )
        w, h, labels, vertexlist = get_poly(file)
        img = Image.new("RGB", (w, h))
        img1 = ImageDraw.Draw(img)
        for i in range(len(vertexlist)):
            if len(vertexlist[i]) < 2:
                continue
            else:
                img1.polygon(vertexlist[i], fill=label_clr[labels[i]])
        img = np.array(img)
        im = Image.fromarray(img[:, :, 0])
        im.save(os.path.join("data", "maskimg", sub_fol_name, img_name))


compute_masks(index_df)
