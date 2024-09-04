import cv2
import numpy as np
from skimage.metrics import structural_similarity

# 테스트를 위한 이미지 로딩
original_image = cv2.imread('original_image.png', cv2.IMREAD_GRAYSCALE)
compressed_image = cv2.imread('compressed_image.png', cv2.IMREAD_GRAYSCALE)

#SNR 계산 코드
def calculate_snr(self, signal_roi, noise_roi):
    signal_mean = np.mean(signal_roi)
    noise_std = np.std(noise_roi)
    #분모 0일 경우
    if noise_std == 0:
        return np.inf
    else:
        snr = signal_mean / noise_std
        return snr
    
#CNR 계산 코드
def calculate_cnr(self, roi_a, roi_b):
    mean_a = np.mean(roi_a)
    mean_b = np.mean(roi_b)
    noise_std = np.std(roi_b) 
    #분모 0일 경우
    if noise_std == 0:
        return np.inf
    else:
        cnr = np.abs(mean_a - mean_b) / noise_std
        return cnr
#PSNR 계산 코드
def calculate_psnr(self, original_image, compared_image):
    mse = np.mean((original_image - compared_image) ** 2)
    if mse == 0:
        return np.inf
    max_pixel = np.max(original_image)
    psnr = (max_pixel ** 2) / mse
    return psnr
#SSIM 계산 코드
def calculate_ssim(self, original_image, compared_image):
    ssim_value = structural_similarity(original_image, compared_image, data_range=compared_image.max() - compared_image.min())
    return ssim_value
