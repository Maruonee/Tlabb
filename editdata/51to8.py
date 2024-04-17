import cv2
import os
import numpy as np

folder_path = "/home/tlab4090/Tlabb/segman/mmlab/data/spine/labels/val/ap"
output_folder = "/home/tlab4090/Tlabb/segman/mmlab/data/spine/labels/val/out"

if not os.path.exists(output_folder):
    os.mkdir(output_folder)
        
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
        final_img = replace_color(img, (255, 255, 255), (9, 9, 9))
        cv2.imwrite(os.path.join(output_folder, image_name), final_img)
        print(os.path.join(output_folder, image_name))
        
print("done")