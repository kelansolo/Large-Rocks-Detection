import rasterio, os
import numpy as np
from PIL import Image
import tifffile as tiff

input_dir = "./LargeRocksDetectionDataset"
output_dir = os.path.join(input_dir, "merged")

os.makedirs(output_dir)

rgb_dir= os.path.join(input_dir, "swissImage_50cm_patches")
S3D_dir= os.path.join(input_dir, "swissSURFACE3D_patches")
HS_dir= os.path.join(input_dir, "swissSURFACE3D_hillshade_patches")

images = [rgb_dir, S3D_dir, HS_dir]

for img in os.listdir(rgb_dir):


    rgb_img = os.path.join(rgb_dir, img)
    s3d_img = os.path.join(S3D_dir, img)
    hil_img = os.path.join(HS_dir, img)

    im_rgb = tiff.imread(rgb_img)
    im_s3d = tiff.imread(s3d_img)
    im_hil = tiff.imread(hil_img)

    arr_rgb = np.array(im_rgb)
    arr_s3d = np.array(im_s3d)
    arr_s3d = np.expand_dims(arr_s3d, axis=2)
    arr_hil = np.array(im_hil)
    arr_hil = np.expand_dims(arr_hil, axis=2)

    merged_img = np.concatenate((arr_rgb, arr_s3d, arr_hil), axis=2)

    assert merged_img.shape[2] == 5, "Yeah not that's not correct mate"
    output_path = os.path.join(output_dir, img)

    tiff.imwrite(output_path, merged_img)
    break

    
    