import cv2
import numpy as np
import glob
import os

sample_file = 'C:\\Users\\vole9\\OneDrive\\Documents\\data\\JPG\\*.jpg'
r_dirname, r_filename = os.path.split(sample_file)
images_dir = glob.glob(sample_file)
###################Gamma###########################
os.mkdir(f"{r_dirname}\\Gamma")

custom_number = 0.5

for img_list in images_dir:
    dirname, filename = os.path.split(img_list)
    img = cv2.imread(img_list)
    gamma_cvt = np.zeros((256,1),dtype = 'uint8')
    for i in range(256):
          gamma_cvt[i][0] = 255 * (float(i)/255) ** (1.0/float(custom_number))
    custom_img = cv2.LUT(img,gamma_cvt)
    save_dir = f"{dirname}\\Gamma\\Gamma_{filename}"
    cv2.imwrite(save_dir, custom_img)
    print(save_dir)
