import os
import numpy as np

from tqdm import tqdm
from PIL import Image
from skimage.io import imread, imsave

input_path = "mask_png"
output_path = "mask_bin"
os.makedirs(output_path, exist_ok=True)

Image.MAX_IMAGE_PIXELS = None
for mask_name in tqdm(os.listdir(input_path)):
    mask = imread(os.path.join(input_path, mask_name))
    mask = mask // 255

    imsave(
        os.path.join(output_path, mask_name), mask.astype(np.int8), check_contrast=False
    )

print("Done!")
