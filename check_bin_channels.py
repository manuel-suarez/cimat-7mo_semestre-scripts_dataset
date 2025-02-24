import os
import numpy as np

from PIL import Image
from skimage.io import imread

Image.MAX_IMAGE_PIXELS = None
for mask_name in os.listdir("mask_bin"):
    mask = imread(os.path.join("mask_bin", mask_name))
    print(mask_name, mask.shape, mask.dtype, mask.min(), mask.max(), np.unique(mask))
