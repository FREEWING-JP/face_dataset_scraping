import cv2
import glob
import sys
import shutil
import os

import numpy as np

args = sys.argv

image_dir = args[1]

image_files = glob.glob(image_dir+"/*.png")

os.makedirs("result",exist_ok=True)

for image_file in image_files:
    if (not (".png" in image_file)) and (not (".jpg" in image_file)):
        continue

    img = cv2.imread(image_file)
    img = cv2.resize(img, (64,64))
    img = cv2.resize(img, (128,128))

    cv2.imwrite("result/"+image_file.split("/")[-1],img)