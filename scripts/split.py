import numpy as np
import os
import shutil
import json
import argparse


def make_split(path, in_folder_name, out_folder_name, train_val_test_fractions == [0.6,0.2,0.2]):



# Create command line parser
parser = argparse.ArgumentParser(
                    prog='SplitMaker',
                    description='Makes split with train and validation images using provided percentage of validation and using  a given fraction of the dataset (full set by default)',
                    epilog='Fuck You')


parser.add_argument('input_dir', type=str)
parser.add_argument('output_dir', type=str)
parser.add_argument('-f', '--fractions', default= [0.6,0.2,0.2], type=float,  nargs=3)

args = parser.parse_args()

fractions = args.fractions
input_dir = args.input_dir
output_dir = args.output_dir

# Format directories
if input_dir[-1] != "/":
    input_dir = input_dir + "/"
if output_dir[-1] != "/":
    output_dir = output_dir + "/"



files = os.listdir(input_dir)

# Create the directories
os.makedirs(output_dir + "train/", exist_ok=True)
os.makedirs(output_dir + "val/", exist_ok=True)
os.makedirs(output_dir + "test/", exist_ok=True)


random.seed(seed)
random.shuffle(files)

# Randomly select the validation images according to the percentage
nb = int(len(data) * perc)
indices = np.random.choice(len(data), int(nb), replace=False)

print(f"Processing {len(data)} images ({lightness*100}% of the original dataset) using {nb} validation images ({perc*100}% of the dataset)")

# Load the labels
with open("LargeRocksDetectionDataset/large_rock_dataset.json", 'r') as f:
    labels = json.load(f)


for i, file in enumerate(data):

    out = output_dir + "val/" if i in indices else output_dir + "train/" 
    #print(f"{i}, {j}")
    # Move the image to the validation directory
    shutil.copy(input_dir + files[i], out + files[i])
    # Create the annotation file
    create_annotations(labels, files[i], out)
        
