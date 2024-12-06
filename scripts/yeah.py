import os 
import rasterio
import numpy as np
from PIL import Image
import tifffile as tiff
import shutil

input_dir = './data_5b_split_all'
output_dir = './output'

in_train = os.path.join(input_dir, "train")
in_test = os.path.join(input_dir, "test")
in_val = os.path.join(input_dir, "val")

out_train = os.path.join(output_dir, "train")
out_test = os.path.join(output_dir, "test")
out_val = os.path.join(output_dir, "val")

os.makedirs(out_train)
os.makedirs(out_test)
os.makedirs(out_val)

mix = [1,1,1,0,0] # rgb

def transfer(in_dir, out_dir, filter):
    for file in os.listdir(in_dir):
        if ".txt" in file:
            # Skip annotation files
            continue

        im_path = os.path.join(in_dir, file)
        
        im = tiff.imread(im_path)
        im_arr = np.array(im)

        print(im_arr.shape)

        filter = np.array(filter)

        keep_bands = im_arr[filter==1]

        assert keep_bands.shape == (3, 640,640), f"Image should have 3 bands, got shape {keep_bands.shape} using filter {filter} on image {im_arr.shape}"

        out_path = os.path.join(out_dir, file)

        tiff.imwrite(out_path, keep_bands)


        txt_in_path = im_path.replace(".tif", ".txt")
        txt_out_path = out_path.replace(".tif", ".txt")

        shutil.copy(txt_in_path, txt_out_path)

transfer(in_train, out_train, mix)
print("Transfer trainin done")
transfer(in_test, out_test, mix)
print("Transfer test done")
transfer(in_val, out_val, mix)
print("Transfer val done")





