import argparse
import os
import json

def get_annotation(labels, file):
    for data in labels['dataset']:
        if data['file_name'] == file:
            return data['rocks_annotations'], True
    
    return None, False

def make_annotation(image_path, labels,  box_width_pix=10):
    width = box_width_pix / 640 # default value is 10pix i.e. 5x5m
    c = 0 # All object are of the same class i.e.
    annotations, found = get_annotation(labels, image_path)
    if not found:
        print(f"Could not find annotation for image {image_path}")
        return None, False

    out = ""
    for annotation in annotations:
        center_x, center_y = annotation['relative_within_patch_location']

        out += f"{c} {center_x} {center_y} {width} {width}\n"

    return out, True

parser = argparse.ArgumentParser(
                    prog='Annotater3000',
                    description='Annotates the images with the provided labels',
                    epilog='Fuck You')

parser.add_argument('input_dir', type=str, help="The input directory should contain the images directly inside")
parser.add_argument('labels_file', type=str)

args = parser.parse_args()

input_dir = args.input_dir
labels_file = args.labels_file

input_dir = os.path.normpath(input_dir)


# Check that the input path is a folder
assert os.path.isdir(input_dir), f"Input dir {input_dir} should be a folder"


# Check the labels file is a file and a json file
assert os.path.isfile(labels_file), f"Provided labels file should be a file. Got  {labels_file}"

# Load the labels
with open(labels_file, 'r') as f:
    labels = json.load(f)

# Iterate through all the images and create the annotation file
images = os.listdir(input_dir)
succ = 0
tot = len(images)

for image_path in images:

    annotations, ok = make_annotation(image_path, labels)

    if not ok:
        print(f"Could not create annotation for image {image_path}")
        continue

    image_name = image_path.split(".")[0]

    lbl_path = os.path.join(input_dir, f'{image_name}.txt')
    with open(lbl_path, 'w') as f:
        f.write(annotations)
        succ += 1

print(f"Successfully created {succ}/{tot} annotations")