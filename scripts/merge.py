import os
import rasterio
import numpy as np
import matplotlib.pyplot as plt

def merge(rgb_folder, dsm_folder, hillshade_folder, output_folder, channels):
    """
    Merges specified channels from RGB, DSM, and Hillshade images into a new 3-channel image.

    Parameters:
        rgb_folder (str): Path to the folder containing RGB images.
        dsm_folder (str): Path to the folder containing DSM images.
        hillshade_folder (str): Path to the folder containing hillshade images.
        output_folder (str): Path to the folder to save the merged images.
        channels (list): List of 3 strings specifying the source for each channel ('r', 'g', 'b', 'dsm', 'hillshade').
    """
    # Ensure the output directory exists
    os.makedirs(output_folder, exist_ok=True)

    # List all files in the RGB folder
    file_names = [f for f in os.listdir(rgb_folder) if f.endswith('.tif')]

    for file_name in file_names:
        # Paths to corresponding files in each folder
        rgb_path = os.path.join(rgb_folder, file_name)
        dsm_path = os.path.join(dsm_folder, file_name)
        hillshade_path = os.path.join(hillshade_folder, file_name)

        # Read the images
        with rasterio.open(rgb_path) as src_rgb:
            rgb_data = src_rgb.read()
        with rasterio.open(dsm_path) as src_dsm:
            dsm_data = src_dsm.read(1)  # DSM is assumed to be single-channel
        with rasterio.open(hillshade_path) as src_hillshade:
            hillshade_data = src_hillshade.read(1)  # Hillshade is assumed to be single-channel

        # Prepare the output array
        merged_image = np.zeros((3, rgb_data.shape[1], rgb_data.shape[2]), dtype=np.float32)

        # Map channels to their respective data
        channel_map = {
            'r': rgb_data[0],
            'g': rgb_data[1],
            'b': rgb_data[2],
            'dsm': dsm_data,
            'hillshade': hillshade_data
        }

        # Populate the merged image with specified channels
        for i, channel in enumerate(channels):
            merged_image[i] = channel_map[channel]

        # Save the merged image
        output_path = os.path.join(output_folder, file_name)
        with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=merged_image.shape[1],
            width=merged_image.shape[2],
            count=3,
            dtype=merged_image.dtype,
            crs=src_rgb.crs,
            transform=src_rgb.transform
        ) as dst:
            dst.write(merged_image)



### Example usage ###
rgb_folder = r"C:\Users\matth\Documents\IPEO\LargeRocksDetectionDataset\swissImage_50cm_patches"
dsm_folder = r"C:\Users\matth\Documents\IPEO\LargeRocksDetectionDataset\swissSURFACE3D_patches"
hillshade_folder = r"C:\Users\matth\Documents\IPEO\LargeRocksDetectionDataset\swissSURFACE3D_hillshade_patches"
output_folder = r"C:\Users\matth\Documents\IPEO\merged_images"

# Specify channels in the order you want (e.g., Red from RGB, Green from RGB, and Hillshade)
channels = ['r', 'g', 'b']

merge(rgb_folder, dsm_folder, hillshade_folder, output_folder, channels)

# List all files in the folder and sort them
merged_files = sorted([f for f in os.listdir(output_folder) if f.endswith('.tif')])

# Display the first 3 images
for i, file_name in enumerate(merged_files[:3]):
    image_path = os.path.join(output_folder, file_name)
    with rasterio.open(image_path) as src:
        image_data = src.read()
    
    # Transpose the array for visualization if needed (channels last)
    image_data = np.transpose(image_data, (1, 2, 0))
    
    plt.figure(figsize=(8, 8))
    plt.imshow(image_data / np.max(image_data))  # Normalize for visualization
    plt.title(f"Image: {file_name}")
    plt.axis('off')
    plt.show()