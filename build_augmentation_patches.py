import os
import itertools
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from skimage.io import imread, imsave
from PIL import Image
from tqdm import tqdm

# Directories configuration
home_path = os.path.expanduser("~")
data_path = os.path.join(home_path, "data", "cimat")
src_path = os.path.join(data_path, "dataset-cimat")
dst_path = os.path.join(data_path, "dataset-cimat", "segmentation")
# Initial configuration
image_path = "image_norm"
label_path = "mask_bin"
patch_size = 224

Image.MAX_IMAGE_PIXELS = None

# Get total oil and sea pixels
mask_totals_df = pd.read_csv(
    os.path.join(dst_path, "counts", "totals", "total_count.csv")
)
total_oil_pixels = mask_totals_df["oil_pixels"].iloc[0]
total_sea_pixels = mask_totals_df["sea_pixels"].iloc[0]
total_pixels = mask_totals_df["total_pixels"].iloc[0]
# Open list of patches counts per image
image_patches_dfs = []
for fname in os.listdir(os.path.join(src_path, "image_norm")):
    image_patches_df = pd.read_csv(
        os.path.join(dst_path, "counts", "images", fname.split(".")[0] + ".csv")
    )
    image_patches_dfs.append(image_patches_df)
# Join dataframe
mask_images_patches = pd.concat(image_patches_dfs)
# Sort by oil pixels (descending)
mask_images_patches = mask_images_patches.sort_values("oil_pixels", ascending=False)

print(
    f"Initial total pixel counts, oil: {total_oil_pixels}, sea: {total_sea_pixels}, total: {total_pixels}"
)
print(
    f"Percentage of pixel counts, oil: {round(total_oil_pixels/total_pixels,2)}, sea: {round(total_sea_pixels/total_pixels,2)}"
)
# Iterate over list of patches augmenting those with more than 10% of oil pixels until we have approximate the equal
augmented_oil_pixels = 0
augmented_sea_pixels = 0
for index, row in mask_images_patches.iterrows():
    patch_oil_pixels = row["oil_pixels"]
    patch_sea_pixels = row["sea_pixels"]
    patch_total_pixels = row["total_pixels"]
    if round(patch_oil_pixels / patch_total_pixels, 2) >= 10.0:
        augmented_oil_pixels += patch_oil_pixels
        augmented_sea_pixels += patch_sea_pixels

# total of oil and sea pixels
print(
    f"Augmented total pixel counts, oil: {augmented_oil_pixels}, sea: {augmented_sea_pixels}, total: {augmented_oil_pixels+augmented_sea_pixels}"
)
percentage_oil_pixels = round(
    augmented_oil_pixels / (augmented_oil_pixels + augmented_sea_pixels), 2
)
percentage_sea_pixels = round(
    augmented_sea_pixels / (augmented_oil_pixels + augmented_sea_pixels), 2
)
print(
    f"Percentage of augmented pixel counts, oil: {percentage_oil_pixels}, sea: {percentage_sea_pixels}"
)
print("Done!")
