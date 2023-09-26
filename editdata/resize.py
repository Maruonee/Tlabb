import os
from PIL import Image, ImageOps

# Directory containing your images
input_folder = '/home/tlab4090/datasets/spine/masks/train/Spine'

# Output directory to save padded and resized images
output_folder = '/home/tlab4090/datasets/spine/masks/train/Spined'

# Padding size in pixels
padding = 20  # Adjust as needed

# Target size for the images
target_size = ((1024, 1024))

# Create the output directory if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Iterate through the files in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith(('.png')):  # Adjust the file extensions as needed
        # Open the image
        image_path = os.path.join(input_folder, filename)
        image = Image.open(image_path)

        # Calculate the aspect ratio of the original image
        aspect_ratio = image.width / image.height

        # Calculate the new width and height while preserving the aspect ratio
        if aspect_ratio > 1:
            new_width = target_size[0]
            new_height = int(target_size[0] / aspect_ratio)
        else:
            new_height = target_size[1]
            new_width = int(target_size[1] * aspect_ratio)

        # Resize the image to the target size
        image = image.resize((new_width, new_height))

        # Calculate the padding size to achieve a final size of 512x512
        left_padding = (target_size[0] - new_width) // 2
        top_padding = (target_size[1] - new_height) // 2
        right_padding = target_size[0] - new_width - left_padding
        bottom_padding = target_size[1] - new_height - top_padding

        # Add zero padding around the image
        padded_image = ImageOps.expand(image, border=(left_padding, top_padding, right_padding, bottom_padding), fill='white')

        # Save the padded and resized image to the output folder with the same filename
        output_path = os.path.join(output_folder, filename)
        padded_image.save(output_path,"PNG",quality = 100)
        print(f"{output_path}")

print("done")
