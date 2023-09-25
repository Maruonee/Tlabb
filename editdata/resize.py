import os
from PIL import Image

def resize_images_in_folder(input_folder, output_folder, target_size=(512, 512)):
    """
    Resize all images in the input folder and save them to the output folder with the target size.
    
    :param input_folder: Path to the directory containing images to be resized.
    :param output_folder: Path to the directory where resized images will be saved.
    :param target_size: A tuple indicating the target size (width, height) for the resized images.
    """
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)

        try:
            with Image.open(file_path) as img:
                img_resized = img.resize(target_size, Image.ANTIALIAS)
                save_path = os.path.join(output_folder, filename)
                img_resized.save(save_path, "PNG", quality = 100)
                print(f"Resized and saved {filename} to {save_path}")

        except Exception as e:
            print(f"Error processing {filename}. Error: {e}")



input_dir = "path/to/your/input/folder"
output_dir = "path/to/your/output/folder"
target_img_size = (512, 512)

resize_images_in_folder(input_dir, output_dir, target_img_size)