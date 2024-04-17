import cv2
import os
import numpy as np

folder_path = "/home/tlab4090/Tlabb/segman/unet/files/Spine_PC_ap/Test/masks"
output_folder = "/home/tlab4090/Tlabb/segman/unet/files/Spine_PC_ap/Test/out"

# Ensure the output directory exists
if not os.path.exists(output_folder):
    os.mkdir(output_folder)
# 제로패딩
#2
# RGB영상 그레이로
def gray_to_binary(gray_img, threshold_value=1):
    _, binary = cv2.threshold(gray_img, threshold_value, 255, cv2.THRESH_BINARY)
    return binary

def rgb_to_gray(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray

for image_name in os.listdir(folder_path):
    if image_name.endswith(('png')):
        print(os.path.join(output_folder, image_name))
        img_path = os.path.join(folder_path, image_name)
        img = cv2.imread(img_path)
        gray_img = rgb_to_gray(img)
        binary_img = gray_to_binary(gray_img, threshold_value=1)
        cv2.imwrite(os.path.join(output_folder, image_name), binary_img)
        print(os.path.join(output_folder, image_name))
        
print("done")