import yaml
import os
import argparse


dataset_base = './'

parser = argparse.ArgumentParser(
                    prog='Yeah2')

parser.add_argument('dataset', type=str, help="Dataset")
parser.add_argument('name', type=str, help="Name for the yaml file")
parser.add_argument('-a', action='store_true', help="Add to turn on data augmentation")

args = parser.parse_args()


dataset = args.dataset
name = args.name

augment_off_dic = {
    "hsv_h": 0.00,
    "hsv_s": 0.0,
    "hsv_v": 0.0,
    "degrees": 0.0,
    "translate": 0.0,
    "scale": 0.0,
    "shear": 0.0,
    "perspective": 0.0000,
    "flipud": 0.0,
    "fliplr": 0.0,
    "mosaic": 0.0,
    "mixup": 0.0,
    "copy_paste": 0.0,
}

augment_on_dic = {
    "hsv_h": 0.00,
    "hsv_s": 0.0,
    "hsv_v": 0.0,
    "degrees": 360.0,
    "translate": 0.2,
    "scale": 0.0,
    "shear": 0.0,
    "perspective": 0.0000,
    "flipud": 0.5,
    "fliplr": 0.5,
    "mosaic": 1.0,
    "mixup": 0.0,
    "copy_paste": 0.0,
}


yaml_doc = {
    "path": './dataset',
    "train": 'train',
    "val": 'val',
    "names": {
        0: "gros cailloux"
    }
}

data_path = os.path.join(dataset_base, dataset)
yaml_doc['path'] = data_path

if args.a:
    yaml_doc['augment'] = augment_on_dic
else:
    yaml_doc['augment'] = augment_off_dic

with open(name, "w") as f:
    yaml.dump(yaml_doc, f)