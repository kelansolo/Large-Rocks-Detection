import os
import json
import argparse
import shutil

def create_annotation(annotations, pix_box_size):
    width = pix_box_size / 640
    c = 0

    txt = ""
    for annotation in annotations:
        center_x, center_y = annotation['relative_within_patch_location']
        txt += f"{c} {center_x} {center_y} {width} {width}\n"

    return txt


parser = argparse.ArgumentParser(
                    prog='MakeDefaultDataset',
                    description='Create the default reproducible dataset from the large rock dataset',
                    epilog='')

parser.add_argument('dir', type=str, help="The 'LargeRocksDetectionDataset' directory")
parser.add_argument('out_dir', type=str, help="Ouptput directory to put the dataset in")
parser.add_argument('-n', default= [-1,-1], type=int,  nargs=2, help = 'Number of images to use for train and validation. Default is all')
parser.add_argument('-b', default=10, type=int, help = 'Bbox size in pixels. Default is 10 i.e. 5m')

args = parser.parse_args()

data_dir = args.dir

json_path = os.path.join(data_dir, 'large_rock_dataset.json')
assert os.path.isfile(json_path) and os.path.exists(json_path)

images_path = os.path.join(data_dir, 'swissImage_50cm_patches')
assert os.path.isdir(images_path) and os.path.exists(images_path)

output_dir = args.out_dir

train_dir = os.path.join(output_dir, 'train')
os.makedirs(train_dir)

val_dir = os.path.join(output_dir, 'val')
os.makedirs(val_dir)

images = os.listdir(images_path)

with open(json_path, 'r') as f:
    labels = json.load(f)


bbox_width_pix = int(args.b)

# mercredi bad

images_paths_train = []
labels_txt_train =[]

images_paths_val = []
labels_txt_val =[]

for annotation in labels['dataset']:
    
    # Create the annotation txt file content for the image
    txt = create_annotation(annotation['rocks_annotations'], bbox_width_pix)

    output_path = train_dir if annotation['split'] == "train" else val_dir

    img_name = annotation['file_name'].split('.')[0]

    # Write the annotation file content to a txt file
    #txt_path = os.path.join(output_path, img_name+".txt")
    #with open(txt_path, 'w') as f:
    #    f.write(txt)
        
    original_img_path = os.path.join(images_path, annotation['file_name'])
    new_img_path = os.path.join(output_path, annotation['file_name'])

    if annotation['split'] == "train":
        labels_txt_train.append(txt)
        images_paths_train.append([original_img_path, new_img_path])
    else:
        labels_txt_val.append(txt)
        images_paths_val.append([original_img_path, new_img_path])

    

    #shutil.copy(original_img_path, new_img_path)

nb_train = args.n[0]
nb_train = len(images_paths_train) if nb_train == -1 or nb_train > len(images_paths_train) else nb_train

nb_val = args.n[1]
nb_val = len(images_paths_val) if nb_val == -1 or nb_val > len(images_paths_val) else nb_val

print(f"Using {nb_train} out of {len(images_paths_train)} train images")
for i in range(0, nb_train):
    old, new = images_paths_train[i]
    shutil.copy(old, new)
    with open(new.replace('tif', 'txt'), 'w') as f:
        f.write(labels_txt_train[i])

print(f"Using {nb_val} out of {len(images_paths_val)} validation images")
for i in range(0, nb_val):
    old, new = images_paths_val[i]
    shutil.copy(old, new)
    with open(new.replace('tif', 'txt'), 'w') as f:
        f.write(labels_txt_val[i])
    
print("Done.")