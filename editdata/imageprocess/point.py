import cv2
from matplotlib import pyplot as plt
import numpy as np
import time
import glob
import os

sample_file = 'C:\\Users\\tlab\\Documents\\sss\\JPG\\*.jpg'
multiply_number = 3
r_dirname, r_filename = os.path.split(sample_file)
os.mkdir(f"{r_dirname}\\multiply{multiply_number}")
images_dir = glob.glob(sample_file)
######################################################################

for img_list in images_dir:
    dirname, filename = os.path.split(img_list)
    img = cv2.imread(img_list)
    custom_img = img * multiply_number
    save_dir = f"{dirname}\\multiply{multiply_number}\\multiply_{multiply_number}_{filename}"
    cv2.imwrite(save_dir, custom_img)
    print(save_dir)
