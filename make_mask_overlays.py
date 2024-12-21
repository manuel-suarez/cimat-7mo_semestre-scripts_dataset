import os
import rasterio
import numpy as np
from PIL import Image
from skimage.io import imread
from matplotlib import pyplot as plt

Image.MAX_IMAGE_PIXELS = None

home_dir = os.path.expanduser("~")
data_dir = os.path.join(home_dir, "data", "cimat", "dataset-cimat")

print(os.listdir(os.path.join(data_dir, "image_tiff")))

os.makedirs("figures", exist_ok=True)


def plot_image_and_mask(basepath, filename):
    fname = filename.split(".")[0]

    image = rasterio.open(os.path.join(basepath, "image_tiff", fname + ".tif")).read(1)
    label = imread(
        os.path.join(basepath, "mask_bin", fname + ".png"), as_gray=True
    ).astype(np.float32)
    print("Label array: ", np.unique(label), label.shape, label.dtype)
    # Binarize
    binary_image = np.where(label > 0, 1, 0)
    print(fname, image.shape, label.shape, binary_image.shape)
    print(image.dtype, label.dtype, binary_image.dtype)
    # mask = np.ma.masked_values(label, 0.0)

    fig, ax = plt.subplots(1, 3, figsize=(16, 6))
    ax[0].set_title("SAR origin")
    im0 = ax[0].imshow(image, cmap="gray")
    fig.colorbar(im0, ax=ax[0])

    ax[1].set_title("Segmentation label")
    im1 = ax[1].imshow(label, cmap="gray", vmin=0.0, vmax=1.0, interpolation="none")
    fig.colorbar(im1, ax=ax[1])
    # Masked overlay
    ax[2].set_title("Mask")
    ax[2].imshow(image, cmap="gray")
    mask = np.ma.masked_where(binary_image == 0, binary_image, copy=True)
    im2 = ax[2].imshow(
        mask,
        cmap="viridis",
        alpha=0.1,
        vmin=0.0,
        vmax=1.0,
        interpolation="none",
    )
    fig.colorbar(im2, ax=ax[2])

    fig.suptitle(fname)
    plt.savefig(os.path.join("figures", fname + ".png"))
    plt.close()


slurm_ntasks = os.getenv("SLURM_NTASKS")
slurm_procid = os.getenv("SLURM_PROCID")
slurm_task_pid = os.getenv("SLURM_TASK_PID")
print("SLURM_NTASKS: ", slurm_ntasks)
print("SLURM_PROCID: ", slurm_procid)
print("SLURM_TASK_PID: ", slurm_task_pid)

# Open image according to slurm proc id
index = int(slurm_procid) - 1
print("Index of image: ", index)
fname = os.listdir(os.path.join(data_dir, "image_tiff"))[index]
print("Image: ", fname)

# Data
plot_image_and_mask(data_dir, fname)
