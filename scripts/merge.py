import os
import numpy as np
import rasterio
import argparse

def merge(rgb_folder, dsm_folder, hillshade_folder, output_folder, channels):
    """
    Merges specified channels from RGB, DSM, and Hillshade images into a new 3-channel image.

    Parameters:
        rgb_folder (str): Path to the folder containing RGB images.
        dsm_folder (str): Path to the folder containing DSM images.
        hillshade_folder (str): Path to the folder containing hillshade images.
        output_folder (str): Path to the folder to save the merged images.
        channels (list): List of 5 integers (1 or 0) specifying whether to use 'r', 'g', 'b', 'dsm', 'hillshade' respectively.
                         At least 3 channels must be selected.
    """
    
    # Ensure the output directory exists
    os.makedirs(output_folder, exist_ok=True)

    # List all files in the RGB folder
    file_names = [f for f in os.listdir(rgb_folder) if f.endswith('.tif')]

    # Map channel indices to their respective names
    channel_names = ['r', 'g', 'b', 'dsm', 'hillshade']

    # Get the selected channels from the binary list
    selected_channels = [channel_names[i] for i, use in enumerate(channels) if use]

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

        # Populate the merged image with the selected channels
        for i, channel in enumerate(selected_channels):
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

parser = argparse.ArgumentParser(
    prog = "MergeMachine",
    description = "reads images (rgb, dsm, hillshade), splits rgb images into 3 channels and merges 3 new channels of choice",
    epilog = "what a nice function!")

parser.add_argument("rgb_folder", type=str)
parser.add_argument("dsm_folder", type=str)
parser.add_argument("hillshade_folder", type=str)
parser.add_argument("output_folder", type=str)
parser.add_argument("-c", "--channels", type=int, nargs=5, required=True)

args = parser.parse_args()

rgb_folder = args.rgb_folder
dsm_folder = args.dsm_folder
hillshade_folder = args.hillshade_folder
output_folder = args.output_folder
channels = args.channels

rgb_folder = os.path.normpath(rgb_folder)
dsm_folder = os.path.normpath(dsm_folder)
hillshade_folder = os.path.normpath(hillshade_folder)
output_folder = os.path.normpath(output_folder)

merge(rgb_folder, dsm_folder, hillshade_folder, output_folder, channels)