import os
import pandas as pd

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
