import cv2
import numpy as np
import glob
import os

sample_file = 'C:\\Users\\vole9\\OneDrive\\Documents\\data\\JPG\\*.jpg'
r_dirname, r_filename = os.path.split(sample_file)
images_dir = glob.glob(sample_file)
###################Clahe###########################
os.mkdir(f"{r_dirname}\\Clahe")

tileGridSize = (8,8)

for img_list in images_dir:
    dirname, filename = os.path.split(img_list)
    img = cv2.imread(img_list)
    clahe = cv2.createCLAHE(clipLimit = 2, tileGridSize = (8,8))
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    final_img = gray_img.astype('uint8')
    custom_img = clahe.apply(final_img)
    save_dir = f"{dirname}\\Clahe\\Clahe_{filename}"
    cv2.imwrite(save_dir, custom_img)
    print(save_dir)
