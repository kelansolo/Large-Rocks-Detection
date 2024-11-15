import numpy as np
import os
import shutil
import json
import argparse

def convert_to_txt(annotations):
    out = ""
    for annotation in annotations:
        c = 0
        center_x, center_y = annotation['relative_within_patch_location']
        width = 0.0015
        out += f"{c} {center_x} {center_y} {width} {width}\n"

    return out

def create_annotations(labels, file, to):
    for data in labels['dataset']:
        if data['file_name'] == file:
            annotations = data['rocks_annotations']
            l = convert_to_txt(annotations)
            name = file.split('.')[0]
            with open(f"{to}/{name}.txt", 'w') as f:
                f.write(l)

# Create command line parser
parser = argparse.ArgumentParser(
                    prog='SplitMaker',
                    description='Makes split with train and validation images using provided percentage of validation and using  a given fraction of the dataset (full set by default)',
                    epilog='Fuck You')

parser.add_argument('percentage')
parser.add_argument('input_dir', type=str)
parser.add_argument('output_dir', type=str)
parser.add_argument('-l', '--lightness', default=1.0, type=float, action='store')

args = parser.parse_args()

perc = float(args.percentage)
lightness = (args.lightness)
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

# Randomly choose a fraction of the dataset according to the lightness parameter
fraction = int(len(files) * lightness)
data = np.random.choice(files, fraction, replace=False)

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
        
