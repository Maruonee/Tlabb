import cv2
import numpy as np
import glob
import os
from skimage import exposure
from skimage.exposure import match_histograms

sample_file = 'C:\\Users\\vole9\\OneDrive\\Documents\\data\\PNG\\*.png'
r_dirname, r_filename = os.path.split(sample_file)
images_dir = glob.glob(sample_file)
###################equal###########################
os.mkdir(f"{r_dirname}\\Equalization")

for img_list in images_dir:
    dirname, filename = os.path.split(img_list)
    img = cv2.imread(img_list)
    hist, bins = np.histogram(img.flatten(), 256,[0,256])
    cdf = hist.cumsum()
    cdf_m = np.ma.masked_equal(cdf,0)
    cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
    cdf = np.ma.filled(cdf_m,0).astype('uint8')
    custom_img = cdf[img]
    save_dir = f"{dirname}\\Equalization\\Equalization_{filename}"
    cv2.imwrite(save_dir, custom_img)
    print(save_dir)
    
# ###################Specification###########################
# reference = "C:\\Users\\tlab\\Documents\\sss\\JPG\\multiply3\\multiply_3_7_1.jpg"
# reference_type = "multiply3"
# os.mkdir(f"{r_dirname}\\Specification_{reference_type}")
# reference_img = cv2.imread(reference)

# for img_list in images_dir:
#     dirname, filename = os.path.split(img_list)
#     img = cv2.imread(img_list)
#     custom_img = match_histograms(img, reference_img,multichannel=True)    
#     save_dir = f"{dirname}\\Specification_{reference_type}\\Specification_{reference_type}_{filename}"
#     cv2.imwrite(save_dir, custom_img)
#     print(save_dir)

