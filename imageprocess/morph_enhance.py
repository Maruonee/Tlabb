import cv2
import numpy as np
import glob
import os
from skimage import exposure
from skimage.exposure import match_histograms

sample_file = 'C:\\Users\\vole9\\OneDrive\\Documents\\data\\PNG\\*.png'
multi_number = 320
r_dirname, r_filename = os.path.split(sample_file)
os.mkdir(f"{r_dirname}\\morph_conen_{multi_number}")
images_dir = glob.glob(sample_file)
############################Morphology################################
for img_list in images_dir:
    dirname, filename = os.path.split(img_list)
    img = cv2.imread(img_list)
    filterSize = (int(multi_number), int(multi_number))
    tophat_custom_img = cv2.morphologyEx(
        img, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,filterSize)
        )
    bothat_custom_img = cv2.morphologyEx(
        img,cv2.MORPH_BLACKHAT,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,filterSize)
        )
    k_img = tophat_custom_img + img
    custom_img = k_img - bothat_custom_img
    save_dir = f"{dirname}\\morph_conen_{multi_number}\\morph_conen{multi_number}_{filename}"
    cv2.imwrite(save_dir, custom_img)
    print(save_dir)
