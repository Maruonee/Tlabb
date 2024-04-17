import cv2
import os
import numpy as np

folder_path = "/home/tlab4090/Tlabb/segman/unet/files/images/val"
output_folder = "/home/tlab4090/Tlabb/segman/unet/files/images/out"


if not os.path.exists(output_folder):
    os.mkdir(output_folder)
    
# 제로패딩
def pad_image(img, target_width, target_height):
    height, width, _ = img.shape
    top_pad = (target_height - height) // 2
    bottom_pad = target_height - height - top_pad
    left_pad = (target_width - width) // 2
    right_pad = target_width - width - left_pad

    padded_img = cv2.copyMakeBorder(img, top_pad, bottom_pad, left_pad, right_pad, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return padded_img

# # RGB영상 그레이로
# def gray_to_binary(gray_img, threshold_value=1):
#     _, binary = cv2.threshold(gray_img, threshold_value, 255, cv2.THRESH_BINARY)
#     return binary
# #2
# def rgb_to_gray(img):
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     return gray

def replace_color(img, target_color, replacement_color):
    # Convert target and replacement colors from RGB to BGR (since OpenCV uses BGR format)
    target_color_bgr = (target_color[2], target_color[1], target_color[0])
    replacement_color_bgr = (replacement_color[2], replacement_color[1], replacement_color[0])
    # Create a mask for pixels that match the target color
    lower_bound = np.array(target_color_bgr, dtype=np.uint8)
    upper_bound = np.array(target_color_bgr, dtype=np.uint8)
    mask = cv2.inRange(img, lower_bound, upper_bound)
    # Replace the target color with the desired color
    img[mask != 0] = replacement_color_bgr
    return img

for image_name in os.listdir(folder_path):
    if image_name.endswith(('png')):
        print(os.path.join(output_folder, image_name))
        img_path = os.path.join(folder_path, image_name)
        img = cv2.imread(img_path)
        #제로패딩처리
        padded_img = pad_image(img, 3600, 3600)
        # #2
        # binary_img = gray_to_binary(img, threshold_value=1)
        # # 512변환
        # # recolor_img = replace_color(binary_img, (255,255,255), (255,255,55))
        # gray_img = rgb_to_gray(binary_img)
        final_img = cv2.resize(padded_img, (512, 512), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(os.path.join(output_folder, image_name), final_img)
        print(os.path.join(output_folder, image_name))
        
print("done")