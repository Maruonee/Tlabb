import cv2
import numpy as np
import glob
import os
from skimage import exposure
from skimage.exposure import match_histograms

sample_file = 'C:\\Users\\vole9\\OneDrive\\Documents\\data\\PNG\\*.png'
multiply_number = 3
r_dirname, r_filename = os.path.split(sample_file)
os.mkdir(f"{r_dirname}\\multiply{multiply_number}")
images_dir = glob.glob(sample_file)
##########################Specification##############################
reference_type = f"multiply{multiply_number}"
os.mkdir(f"{r_dirname}\\Specification_{reference_type}")
######################################################################
for img_list in images_dir:
    dirname, filename = os.path.split(img_list)
    img = cv2.imread(img_list)
    custom_img = img * multiply_number
    save_dir = f"{dirname}\\multiply{multiply_number}\\multiply_{multiply_number}_{filename}"
    cv2.imwrite(save_dir, custom_img)
    print(save_dir)
# ###################Specification###########################
    reference_img = custom_img
    dirname_s, filename_s = os.path.split(img_list)
    img_s = cv2.imread(img_list)
    custom_img_s = match_histograms(img_s, reference_img)    
    save_dir_s = f"{dirname_s}\\Specification_{reference_type}\\Specification_{reference_type}_{filename_s}"
    cv2.imwrite(save_dir_s, custom_img_s)
    print(save_dir_s)
