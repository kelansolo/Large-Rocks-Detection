import os 
import rasterio
import numpy as np
import shutil
import argparse

parser = argparse.ArgumentParser(
                    prog='Testbench dataset maker',
                    description='Create a dataset from the 5b dataset with 3 selected bands',
                    epilog='')

parser.add_argument('input_dir', type=str, help="The input directory should containing the different images following the structure of the original dataset")
parser.add_argument("output_dir", type=str, help="Output folder")
parser.add_argument("filter", type=int, nargs=5, help="Bands to keep. 1 to keep, 0 to remove. Bands order: [R G B S3D HIL]. Must be exactly 3 bands")

args = parser.parse_args()

input_dir = args.input_dir
output_dir = args.output_dir
filter = np.array(args.filter)

# Check the filter
assert np.sum(filter) == 3, f"Filter should keep exactly 3 bands, got {np.sum(filter)} (filter: {filter})"

in_train = os.path.join(input_dir, "train")
in_test = os.path.join(input_dir, "test")
in_val = os.path.join(input_dir, "val")

# Check that the input folder exist
assert os.path.exists(in_train), "Train images folder does not exist"
assert os.path.exists(in_test), "Test images folder does not exist"
assert os.path.exists(in_val), "Val images folder does not exist"


out_train = os.path.join(output_dir, "train")
out_test = os.path.join(output_dir, "test")
out_val = os.path.join(output_dir, "val")

os.makedirs(out_train)
os.makedirs(out_test)
os.makedirs(out_val)


def transfer(in_dir, out_dir, filter):
    for file in os.listdir(in_dir):
        if ".txt" in file:
            # Skip annotation files
            continue

        im_path = os.path.join(in_dir, file)
        
        im_data = rasterio.open(im_path).read()

        keep_bands = im_data[filter==1]

        assert keep_bands.shape == (3, 640,640), f"Image should have 3 bands, got shape {keep_bands.shape} using filter {filter} on image {im_data.shape}"

        out_path = os.path.join(out_dir, file)

        with rasterio.open(
            out_path, 
            "w",
            height=keep_bands.shape[1],
            width=keep_bands.shape[2],
            count=keep_bands.shape[0],
            dtype=im_data.dtype
            ) as file:
            file.write(keep_bands)


        txt_in_path = im_path.replace(".tif", ".txt")
        txt_out_path = out_path.replace(".tif", ".txt")

        shutil.copy(txt_in_path, txt_out_path)

transfer(in_train, out_train, filter)
print("Transfer trainin done")
transfer(in_test, out_test, filter)
print("Transfer test done")
transfer(in_val, out_val, filter)
print("Transfer val done")





