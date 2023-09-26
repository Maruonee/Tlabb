import glob
import os

list_dir = "/home/tlab4090/datasets/list"
train_images_list = glob.glob("/home/tlab4090/datasets/Spine/images/train/Spine/*.png")
val_images_list = glob.glob("/home/tlab4090/datasets/Spine/images/val/Spine/*.png")
test_images_list = glob.glob("/home/tlab4090/datasets/Spine/images/test/Spine/*.png")

train_target_dir= os.path.dirname("/home/tlab4090/datasets/Spine/marks/train/Spine/")
val_target_dir = os.path.dirname("/home/tlab4090/datasets/Spine/marks/val/Spine/")
test_target_dir = os.path.dirname("/home/tlab4090/datasets/Spine/marks/test/Spine/")

for img_path in train_images_list:
    img_name = os.path.basename(img_path)
    path_mask = os.path.join(train_target_dir,img_name)
    full = img_path +"_change_"+ path_mask
    with open(os.path.join(list_dir,"train.lst"), "a") as a_file:
        a_file.write(full)
        a_file.write("\n")
        
for img_path in val_images_list:
    img_name = os.path.basename(img_path)
    path_mask = os.path.join(val_target_dir,img_name)
    full = img_path +"_change_"+ path_mask
    with open(os.path.join(list_dir,"val.lst"), "a") as a_file:
        a_file.write(full)
        a_file.write("\n")
        
for img_path in test_images_list:
    img_name = os.path.basename(img_path)
    full = img_path
    with open(os.path.join(list_dir,"test.lst"), "a") as a_file:
        a_file.write(full)
        a_file.write("\n")

