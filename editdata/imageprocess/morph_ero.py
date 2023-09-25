import cv2
import numpy as np
import glob
import os
from skimage import exposure
from skimage.exposure import match_histograms

sample_file = 'C:\\Users\\vole9\\OneDrive\\Documents\\data\\PNG\\*.png'
multi_number = 2
r_dirname, r_filename = os.path.split(sample_file)
os.mkdir(f"{r_dirname}\\morph_erode_{multi_number}")
images_dir = glob.glob(sample_file)
############################Morphology################################
for img_list in images_dir:
    dirname, filename = os.path.split(img_list)
    img = cv2.imread(img_list)
    filterSize = np.ones((int(multi_number), int(multi_number)), np.uint8)
    custom_img = cv2.erode(img, filterSize, iterations=1)
    save_dir = f"{dirname}\\morph_erode_{multi_number}\\morph_erode{multi_number}_{filename}"
    cv2.imwrite(save_dir, custom_img)
    print(save_dir)
