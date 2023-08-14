import cv2
import numpy as np
import glob
import os

sample_file = 'C:\\Users\\tlab\\Documents\\sss\\JPG\\*.jpg'
r_dirname, r_filename = os.path.split(sample_file)
images_dir = glob.glob(sample_file)
###################retinex###########################
os.mkdir(f"{r_dirname}\\Retinex_Equalization")
sigma_number = 250

#single retinex 정의
def singleScaleRetinex(image, sigma):
    # 이미지를 0~1 범위로 변환
    image = image.astype(np.float32) / 255.0
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)
    retinex = np.log10(image + 1e-3) - np.log10(blurred + 1e-3)
    result = np.clip(retinex * 255.0, 0, 255).astype(np.uint8)
    return result


for img_list in images_dir:
    dirname, filename = os.path.split(img_list)
    img = cv2.imread(img_list)
    result_single_scale = singleScaleRetinex(img, sigma_number)
    img = result_single_scale
    hist, bins = np.histogram(img.flatten(), 256,[0,256])
    cdf = hist.cumsum()
    cdf_m = np.ma.masked_equal(cdf,0)
    cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
    cdf = np.ma.filled(cdf_m,0).astype('uint8')
    custom_img = cdf[img]
    save_dir = f"{dirname}\\Retinex_Equalization\\Retinex_Equalization_{filename}"
    cv2.imwrite(save_dir, custom_img)
    print(save_dir)
