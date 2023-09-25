import cv2
import numpy as np
import glob
import os
from skimage import exposure
from skimage.exposure import match_histograms

sample_file = 'C:\\Users\\vole9\\OneDrive\\Documents\\data\\PNG\\*.png'
multi_number = 70
r_dirname, r_filename = os.path.split(sample_file)
os.mkdir(f"{r_dirname}\\unsharp{multi_number}")
images_dir = glob.glob(sample_file)

######################################################################
for img_list in images_dir:
    dirname, filename = os.path.split(img_list)
    img = cv2.imread(img_list)
    blurred_image = cv2.GaussianBlur(img, (0, 0), multi_number)
    custom_img = cv2.addWeighted(img, 2, blurred_image, -1, 0)
    save_dir = f"{dirname}\\unsharp{multi_number}\\unsharp{multi_number}_{filename}"
    cv2.imwrite(save_dir, custom_img)
    print(save_dir)
