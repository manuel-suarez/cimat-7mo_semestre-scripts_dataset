import os
import rasterio
import PIL
import numpy as np
from skimage.io import imread
from matplotlib import pyplot as plt

PIL.Image.MAX_IMAGE_PIXELS = None

home_dir = os.path.expanduser("~")
data_dir = os.path.join(home_dir, "data")
sentinel_dir = os.path.join(data_dir, "cimat", "dataset-sentinel")
envisat_dir = os.path.join(data_dir, "cimat", "dataset-envisat")

print(os.listdir(os.path.join(sentinel_dir, "image_tiff")))
print(os.listdir(os.path.join(envisat_dir, "image_tiff")))

os.makedirs("figures", exist_ok=True)


def plot_image_and_mask(basepath, filename):
    fname = filename.split(".")[0]

    image = rasterio.open(os.path.join(basepath, "image_tiff", fname + ".tif")).read(1)
    label = imread(os.path.join(basepath, "mask_bin", fname + ".png"))
    print(fname, image.shape, label.shape)

    fig, ax = plt.subplots(1, 3, figsize=(16, 8))
    ax[0].imshow(image, cmap="gray")
    ax[0].set_title("SAR origin")
    ax[1].imshow(label, cmap="gray")
    ax[1].set_title("Segmentation label")
    # Masked overlay
    ax[2].imshow(image, cmap="gray")
    palette = plt.cm.viridis
    palette.set_bad(alpha=0.0)
    mask = np.ma.masked_where(label == 0.0, label)
    ax[2].imshow(label, cmap=palette, alpha=0.8, vmin=0.0, vmax=1.0)
    ax[2].set_title("Mask")

    plt.savefig(os.path.join("figures", fname + ".png"))
    plt.close()


# Sentinel
for fname in os.listdir(os.path.join(sentinel_dir, "image_tiff")):
    plot_image_and_mask(sentinel_dir, fname)
# Envisat
for fname in os.listdir(os.path.join(envisat_dir, "image_tiff")):
    plot_image_and_mask(envisat_dir, fname)
