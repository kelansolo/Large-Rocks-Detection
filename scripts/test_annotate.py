import matplotlib.patches as patches
import matplotlib.pyplot as plt
import os
import numpy as np

def get_bboxes(data_dir, img_name):
    file = f'{data_dir}/labels/{img_name}.txt'
    with open(file, 'r') as f:
        lines = f.readlines()
    bboxes = []
    for line in lines:
        bbox = line.split()
        bboxes.append([float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])])
    return bboxes

def plot16(img_names, data_dir):
    f, ax = plt.subplots(4,4, sharex=True, sharey=True)

    f.set_figheight(20)
    f.set_figwidth(20)
    f.subplots_adjust(wspace=0, hspace=0)

    for i in range(4):
        for j in range(4):
            index = i*4 + j
            img = plt.imread(f'{data_dir}/images/{img_names[index]}.tif')
            ax[i, j].imshow(img)      

            bboxes = get_bboxes(data_dir, img_names[index])
            for bbox in bboxes:
                x = int(bbox[1] * 640)
                y = int(bbox[2] * 640)
                rect = patches.Circle((x, y), radius=20, linewidth=2, edgecolor='r', facecolor='none')
                ax[i, j].add_patch(rect)

            ax[i, j].set_title(img_names[index])

    plt.show()

data_dir = './../data/train'

images_dir = f'{data_dir}/images'

# Pick 16 random images
img_path = os.listdir(images_dir)
random_imgs = np.random.choice(img_path, 16, replace=False)
img_names = [img.split('.')[0] for img in random_imgs]

plot16(img_names, data_dir)