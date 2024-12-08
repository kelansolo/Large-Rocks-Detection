import os
import rasterio
import numpy as np
import argparse
import warnings

# Filter out NotGeoreferencedWarning warnings
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

rgb_exp_shape = (3,640,640)
hil_exp_shape = (1,640,640)
s3d_exp_shape = (1,640,640)

parser = argparse.ArgumentParser(
                    prog='BandMixer3000',
                    description='Mix the bands of RBG, S3D and HIL in a single images. Mix to keep 1 to 5 out of 5 bands',
                    epilog='')

parser.add_argument('input_dir', type=str, help="The input directory should containing the different images following the structure of the original dataset")
parser.add_argument("filter", type=int, nargs=5, help="Bands to keep. 1 to keep, 0 to remove. Bands order: [R G B S3D HIL]")
parser.add_argument("-o", "--output", type=str, default="", help="Output folder")

def mix(rgb_img, hil_img, s3d_img, out_path, filter):
    rgb_data = rgb_img.read()
    hil_data = hil_img.read()
    s3d_data = s3d_img.read() 

    # Ensure that all images have the expected shape
    if rgb_data.shape != rgb_exp_shape or hil_data.shape != hil_exp_shape or s3d_data.shape != s3d_exp_shape:
        print(f"Got unexpected image size, got rgb: {rgb_data.shape}, hil: {hil_data.shape} s3d: {s3d_data.shape}")
        return False

    # Concatenate all bands in a single array
    all_bands = np.concatenate((rgb_data, s3d_data, hil_data), axis=0)

    # Need to make sure the filter array is a numpy array
    filter = np.array(filter)

    # Filter to keep only the wanted bands
    mixed = all_bands[filter==1]

    # Save the new image 
    with rasterio.open(
        out_path, 
        "w",
        height=mixed.shape[1],
        width=mixed.shape[2],
        count=mixed.shape[0],
        dtype=rgb_data.dtype
        ) as file:
        file.write(mixed)

    return True

args = parser.parse_args()
in_dir = args.input_dir
filter = np.array(args.filter)
out_dir = args.output

print(f"Using input directory {in_dir} with filter {filter}")

# Verify the filter has at least one band
assert np.sum(filter) > 0, "Filter should at least keep one band, got 0"

rgb_dir = os.path.join(in_dir, 'swissImage_50cm_patches')
hil_dir = os.path.join(in_dir, 'swissSURFACE3D_hillshade_patches')
s3d_dir = os.path.join(in_dir, 'swissSURFACE3D_patches')

# Verify that the given input directory has the expected directories inside
assert os.path.exists(rgb_dir), "RGB images folder does not exist"
assert os.path.exists(hil_dir), "HIL images folder does not exist"
assert os.path.exists(s3d_dir), "S3D images folder does not exist"

if len(out_dir) == 0: # No output directory specified
    out_dir = os.path.join(in_dir, 'merged')

os.makedirs(out_dir, exist_ok=True)

# Iterate through all the images
images = os.listdir(rgb_dir)

count = 0
for file in images:
    if not ".tif" in file:
        print("Skipping non tiff file")
        continue

    rgb_img = os.path.join(rgb_dir, file)
    hil_img = os.path.join(hil_dir, file)
    s3d_img = os.path.join(s3d_dir, file)

    rgb = rasterio.open(rgb_img)
    hil = rasterio.open(hil_img)
    s3d = rasterio.open(s3d_img)

    out_path = os.path.join(out_dir, file)
    
    ok = mix(rgb, hil, s3d, out_path, filter)
    if ok:
        count+=1

print(f"Successfully mixed {count}/{len(images)} images")