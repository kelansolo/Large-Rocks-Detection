import argparse
import os
import json

def get_annotation(labels, file):
    for data in labels['dataset']:
        if data['file_name'] == file:
            return data['rocks_annotations'], True
    
    return None, False

def make_annotation(image_path, labels):
    annotations, found = get_annotation(labels, image_path)
    if not found:
        print(f"Could not find annotation for image {image_path}")
        return None, False

    out = ""
    for annotation in annotations:
        c = 0 # Class, hardcoded to 0 because we only have one class
        center_x, center_y = annotation['relative_within_patch_location']
        width = 0.0015 # Width is hardcoded to 0.0015 -> 10 pixels, this is the same for the height

        out += f"{c} {center_x} {center_y} {width} {width}\n"

    return out, True

parser = argparse.ArgumentParser(
                    prog='Annotater3000',
                    description='Annotates the images with the provided labels',
                    epilog='Fuck You')

parser.add_argument('input_dir', type=str, help="The input directory should contain a directory named 'images' inside")
parser.add_argument('labels_file', type=str)

args = parser.parse_args()

input_dir = args.input_dir
labels_file = args.labels_file

input_dir = os.path.normpath(input_dir)


images_dir = os.path.join(input_dir, "images")
labels_dir = os.path.join(input_dir, "labels")

# Check that the input path is a folder
assert os.path.isdir(input_dir), "Input dir should be a folder. It should contain a folder 'images' inside with the images"

# Check that the input path has a "image" folder inside
assert os.path.exists(images_dir), f"Input dir should contains an 'image' direction inside of it. {images_dir} doens't exist"

# Check the labels file is a file and a json file
assert os.path.isfile(labels_file), f"Provided labels file should be a file. Got  {labels_file}"

# Load the labels
with open(labels_file, 'r') as f:
    labels = json.load(f)

# Create the output folder for the labels

os.makedirs(labels_dir, exist_ok=True)

# Iterate through all the images and create the annotation file


images = os.listdir(images_dir)
succ = 0
tot = len(images)

for image_path in images:

    annotations, ok = make_annotation(image_path, labels)

    if not ok:
        print(f"Could not create annotation for image {image_path}")
        continue

    image_name = image_path.split(".")[0]

    lbl_path = os.path.join(labels_dir, f'{image_name}.txt')
    with open(lbl_path, 'w') as f:
        f.write(annotations)
        succ += 1

print(f"Successfully created {succ}/{tot} annotations")







