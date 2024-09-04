import numpy as np
import cv2
from skimage import exposure, morphology, restoration
from scipy.ndimage import gaussian_filter


# Example usage
image_path = 'path_to_your_ct_image.png'

# 1. Load the image (simulate loading a CT slice)
def load_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    return image

# 2. Segment the metal regions using a threshold
def segment_metal_regions(image, threshold=250):
    metal_mask = image > threshold
    # Perform morphological operations to refine the mask
    metal_mask = morphology.dilation(metal_mask, morphology.disk(5))
    return metal_mask

# 3. Apply Gaussian smoothing to the non-metal regions
def smooth_non_metal_regions(image, metal_mask, sigma=1.0):
    smoothed_image = gaussian_filter(image, sigma=sigma)
    # Only apply smoothing to non-metal regions
    image[~metal_mask] = smoothed_image[~metal_mask]
    return image

# 4. Inpaint the metal regions
def inpaint_metal_regions(image, metal_mask):
    # Normalize the image for inpainting
    normalized_image = (image - np.min(image)) / (np.max(image) - np.min(image))
    inpainted_image = restoration.inpaint_biharmonic(normalized_image, metal_mask, multichannel=False)
    return inpainted_image

# 5. Main function to perform metal artifact reduction
def perform_metal_artifact_reduction(image_path):
    # Load image
    image = load_image(image_path)
    
    # Enhance contrast to improve metal detection
    image = exposure.rescale_intensity(image, out_range=(0, 255))
    
    # Segment metal regions
    metal_mask = segment_metal_regions(image)
    
    # Smooth the image in non-metal regions
    smoothed_image = smooth_non_metal_regions(image.copy(), metal_mask)
    
    # Inpaint the metal regions
    final_image = inpaint_metal_regions(smoothed_image, metal_mask)
    
    # Convert back to 8-bit image for display
    final_image = (final_image * 255).astype(np.uint8)
    
    # Display results
    cv2.imshow("Original Image", image)
    cv2.imshow("Metal Mask", metal_mask.astype(np.uint8) * 255)
    cv2.imshow("Reduced Artifact Image", final_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

perform_metal_artifact_reduction(image_path)
