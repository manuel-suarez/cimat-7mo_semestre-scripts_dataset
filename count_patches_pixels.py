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

# Definimos el modelo U-Net con un backbone preentrenado (ResNet)
slurm_array_task_id = os.getenv("SLURM_ARRAY_TASK_ID")
slurm_ntasks = os.getenv("SLURM_NTASKS")
slurm_procid = os.getenv("SLURM_PROCID")
slurm_task_pid = os.getenv("SLURM_TASK_PID")
print("SLURM_ARRAY_TASK_ID: ", slurm_array_task_id)
print("SLURM_NTASKS: ", slurm_ntasks)
print("SLURM_PROCID: ", slurm_procid)
print("SLURM_TASK_PID: ", slurm_task_pid)


def count_patch_pixels(
    src_path,
    img_dir,
    mask_dir,
    dst_path,
    img_name,
    patch_size,
):
    print(
        src_path,
        img_dir,
        mask_dir,
        dst_path,
        img_name,
        patch_size,
    )
    # In this case we are opening both image and mask to patchify at the same time
    # considering that we are removing outside regions pixels (SAR image) and separating
    # oil from not oil spill patches
    image = imread(os.path.join(src_path, img_dir, img_name + ".tif"), as_gray=True)
    mask = imread(os.path.join(src_path, mask_dir, img_name + ".png"), as_gray=True)
    # Scale image between 0 and 1
    min_image = np.min(image)
    max_image = np.max(image)
    image_scaled = (image - min_image) / (max_image - min_image)
    # Mark all pixels on the mask above 0 as oil
    mask[mask > 0] = 1

    # Verifying that image and mask have the same shape
    image_height, image_width = image.shape
    mask_height, mask_width = mask.shape
    if (image_height != mask_height) or (image_width != mask_width):
        print("Error, image and mask must have the same dimensions")
        exit(-1)

    count_x = int(image_width // patch_size) + 1
    count_y = int(image_height // patch_size) + 1

    total_patches = count_x * count_y
    patches_per_task = total_patches // int(slurm_ntasks)
    missing_patches_per_task = total_patches % int(slurm_ntasks)

    patches_indexes = [
        int(slurm_procid) * patches_per_task + index
        for index in range(patches_per_task)
    ]
    if int(slurm_procid) < missing_patches_per_task:
        additional_index = int(slurm_ntasks) * patches_per_task + int(slurm_procid)
        patches_indexes.append(additional_index)

    print("Patches indexes: ", patches_indexes)

    # Build patchex indexes to process considering the max patches, ntasks and task process id
    patches_positions = list(itertools.product(range(count_y), range(count_x)))
    for patch_index in patches_indexes:
        j, i = patches_positions[patch_index]

        print("Processing patch index: ", patch_index, i, j)
        # Get pixel positions for patch
        y = patch_size * j
        x = patch_size * i

        # Crop whenever patch size is outside image
        if x + patch_size > image_width:
            x = image_width - patch_size - 1
        if y + patch_size > image_height:
            y = image_height - patch_size - 1

        # Original image patch
        image_patch = image[y : y + patch_size, x : x + patch_size]
        # Scaled image patch
        image_scaled_patch = image_scaled[y : y + patch_size, x : x + patch_size]
        total_patches = total_patches + 1
        min_image_patch = np.min(image_patch)
        max_image_patch = np.max(image_patch)
        # We are checking if patch image values are 0, if so then continue next patch (we are in an invalid SAR image patch)
        # if min_image_patch == 0 and max_image_patch == 0:
        #    invalid_patches = invalid_patches + 1
        #    continue
        dst_img_name = img_name + f"_{patch_index:04d}_train"
        mask_patch = mask[y : y + patch_size, x : x + patch_size]

        # Count pixels
        oil_pixels = mask_patch[mask_patch == 1].sum()
        not_oil_pixels = mask_patch[mask_patch == 0].sum()

        # Save dataframe
        mask_patch_dict = {
            "patch_name": [dst_img_name],
            "oil_pixels": [oil_pixels],
            "not_oil_pixels": [not_oil_pixels],
        }
        mask_patch_df = pd.DataFrame.from_dict(mask_patch_dict)
        mask_patch_df.to_csv(
            os.path.join(dst_path, "counts", "oil_pixels", dst_img_name + ".csv")
        )


# Create output directories
os.makedirs(dst_path, exist_ok=True)
os.makedirs(os.path.join(dst_path, "features", "origin"), exist_ok=True)
os.makedirs(os.path.join(dst_path, "images"), exist_ok=True)
os.makedirs(os.path.join(dst_path, "labels"), exist_ok=True)
os.makedirs(os.path.join(dst_path, "figures"), exist_ok=True)

fname = os.listdir(os.path.join(src_path, "image_norm"))[int(slurm_array_task_id) - 1]
count_patch_pixels(
    src_path,
    "image_norm",
    "mask_bin",
    dst_path,
    fname.split(".")[0],
    patch_size,
)
print("Done!")
